"""W&B-backed reports: all-metric run comparison, sweep summary, history CSV.

W&B is the source of truth for every training result, so all reporting reads
from there and can be (re)run locally at any time — nothing depends on files
left behind on a pod.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ci.config import WANDB_PROJECT
from ci.stats import mean, paired_t_test, stdev, welch_t_test
from ci.wandb_metric import read_metric

# define_metric(summary=...) stores its statistic under one of these keys
# (kept in sync with ci/wandb_metric.py).
_SUMMARY_STAT_KEYS = ("max", "min", "last", "mean", "value")

# Substring heuristics for "which direction is better" — used only to attach a
# verdict to significant differences; metrics matching neither still get a
# delta and p-value, just no better/worse call.
_HIGHER_IS_BETTER = ("thput", "acc", "reward", "score", "sps", "explained_variance")
_LOWER_IS_BETTER = ("loss", "rmse", "seconds", "runtime")


def metric_direction(name: str) -> Optional[str]:
    low = name.lower()
    if any(tok in low for tok in _HIGHER_IS_BETTER):
        return "higher"
    if any(tok in low for tok in _LOWER_IS_BETTER):
        return "lower"
    return None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def flatten_summary(summary: dict, prefix: str = "") -> dict[str, float]:
    """Flatten a W&B run summary into {slash/path: float}.

    Handles the two representations that defeat naive access (see
    ci/wandb_metric.py): slash-namespaced metrics stored as nested dicts, and
    define_metric(summary=...) values stored as {"max": 0.4}-style stat dicts.
    Keys starting with "_" (W&B internals) are skipped.
    """
    flat: dict[str, float] = {}
    for key, val in summary.items():
        if key.startswith("_"):
            continue
        path = f"{prefix}{key}"
        if _is_number(val):
            flat[path] = float(val)
        elif isinstance(val, dict):
            stat = next((k for k in _SUMMARY_STAT_KEYS if k in val), None)
            if stat is not None and _is_number(val[stat]):
                flat[path] = float(val[stat])
            else:
                flat.update(flatten_summary(val, prefix=path + "/"))
    return flat


@dataclass
class MetricRow:
    name: str
    base_mean: float
    base_std: float
    test_mean: float
    test_std: float
    n_base: int
    n_test: int
    p_value: Optional[float]  # None when either side has < 2 values
    paired: bool

    @property
    def delta(self) -> float:
        return self.test_mean - self.base_mean

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value is None or self.p_value > alpha:
            return ""
        direction = metric_direction(self.name)
        if direction is None:
            return "significant"
        improved = self.delta > 0 if direction == "higher" else self.delta < 0
        return "better" if improved else "worse"


def compare_metric_rows(
    base: dict[int, dict[str, float]], test: dict[int, dict[str, float]]
) -> list[MetricRow]:
    """Compare every metric across two {seed: flat_summary} sides.

    Uses a paired t-test on seeds present in both sides (>= 2), otherwise
    Welch's t-test on the unpaired values. Rows are sorted most-significant
    first, so the interesting differences top the table.
    """
    names = sorted({m for fs in list(base.values()) + list(test.values()) for m in fs})
    rows = []
    for name in names:
        b = {s: fs[name] for s, fs in base.items() if name in fs}
        t = {s: fs[name] for s, fs in test.items() if name in fs}
        if not b or not t:
            continue  # metric only exists on one side; nothing to compare
        common = sorted(set(b) & set(t))
        paired = len(common) >= 2
        if paired:
            b_vals = [b[s] for s in common]
            t_vals = [t[s] for s in common]
            _, _, p = paired_t_test(t_vals, b_vals)
        else:
            b_vals, t_vals = list(b.values()), list(t.values())
            p = welch_t_test(t_vals, b_vals)[2] if min(len(b_vals), len(t_vals)) >= 2 else None
        rows.append(
            MetricRow(
                name=name,
                base_mean=mean(b_vals),
                base_std=stdev(b_vals),
                test_mean=mean(t_vals),
                test_std=stdev(t_vals),
                n_base=len(b_vals),
                n_test=len(t_vals),
                p_value=p,
                paired=paired,
            )
        )
    rows.sort(key=lambda r: (r.p_value is None, r.p_value if r.p_value is not None else 1.0, r.name))
    return rows


def render_compare_markdown(
    rows: list[MetricRow],
    base_label: str,
    test_label: str,
    alpha: float = 0.05,
) -> str:
    icons = {"better": "&#x2705;", "worse": "&#x274C;", "significant": "&#x26A0;&#xFE0F;", "": ""}
    lines = [
        f"## SFT comparison: `{test_label}` vs `{base_label}`",
        "",
        f"Every numeric W&B summary metric, paired by seed where possible "
        f"(paired t-test; Welch's otherwise). Significance level: {alpha}.",
        "",
        f"| Metric | {base_label} | {test_label} | Δ | p | |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        p_str = "-" if r.p_value is None else (f"{r.p_value:.3f}" if r.p_value >= 0.001 else f"{r.p_value:.1e}")
        v = r.verdict(alpha)
        lines.append(
            f"| `{r.name}` | {r.base_mean:.4g} ± {r.base_std:.2g} (n={r.n_base}) "
            f"| {r.test_mean:.4g} ± {r.test_std:.2g} (n={r.n_test}) "
            f"| {r.delta:+.4g} | {p_str} | {icons[v]} {v} |"
        )
    if not rows:
        lines.append("| _no comparable metrics found_ | | | | | |")
    return "\n".join(lines)


def _fetch_group(api, project_path: str, group: str) -> dict[int, dict[str, float]]:
    """Fetch a W&B group's runs as {seed: flattened summary}."""
    out: dict[int, dict[str, float]] = {}
    for run in api.runs(project_path, filters={"group": group}):
        seed = run.config.get("seed")
        if seed is None:
            continue
        summary = getattr(run.summary, "_json_dict", None) or dict(run.summary)
        out[int(seed)] = flatten_summary(summary)
    return out


def _project_path(api) -> str:
    return f"{api.default_entity}/{WANDB_PROJECT}"


def compare_report(base_group: str, test_group: str) -> str:
    """Markdown comparison of every metric across two W&B run groups."""
    import wandb

    api = wandb.Api()
    path = _project_path(api)
    base = _fetch_group(api, path, base_group)
    test = _fetch_group(api, path, test_group)
    if not base or not test:
        return (
            f"No runs found for comparison: {base_group} has {len(base)} run(s), "
            f"{test_group} has {len(test)} run(s)."
        )
    rows = compare_metric_rows(base, test)
    return render_compare_markdown(rows, base_label=base_group, test_label=test_group)


def sweep_report(sweep_path: str, top_n: int = 5) -> str:
    """Markdown summary of a finished (or in-flight) W&B sweep."""
    import wandb

    api = wandb.Api()
    sweep = api.sweep(sweep_path)

    metric_cfg = sweep.config.get("metric", {})
    metric_name = metric_cfg.get("name", "eval/thput_eot")
    metric_goal = metric_cfg.get("goal", "maximize")
    sweep_params = sweep.config.get("parameters", {})
    reverse = metric_goal == "maximize"

    entity, project, sweep_id = sweep_path.split("/")
    sweep_url = f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}"

    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        return (
            "## W&B Sweep Results\n\n**No completed runs found in sweep.**\n\n"
            f"[View sweep on W&B]({sweep_url})"
        )

    missing = float("-inf") if reverse else float("inf")

    def get_metric(run):
        return read_metric(run.summary, metric_name, missing)

    runs.sort(key=get_metric, reverse=reverse)
    best_run = runs[0]
    best_metric = get_metric(best_run)
    param_names = sorted(sweep_params.keys())

    lines = ["## W&B Sweep Results\n", "| | |", "|---|---|"]
    lines.append(f"| **Sweep** | [{sweep_path}]({sweep_url}) |")
    lines.append(f"| **Completed runs** | {len(runs)} |")
    lines.append(f"| **Metric** | `{metric_name}` ({metric_goal}) |")
    lines.append(f"| **Best value** | **{best_metric:.4f}** |")
    lines.append("")

    lines.append("### Best Hyperparameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    for param in param_names:
        value = best_run.config.get(param)
        lines.append(f"| `{param}` | `{value:.6g}` |" if isinstance(value, float) else f"| `{param}` | `{value}` |")
    lines.append(f"\n[View best run on W&B]({best_run.url})\n")

    top_n = min(top_n, len(runs))
    lines.append(f"### Top {top_n} Runs\n")
    header = ["Rank", *[f"`{p}`" for p in param_names], f"`{metric_name}`", "W&B"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for i, run in enumerate(runs[:top_n]):
        metric_val = get_metric(run)
        row = [str(i + 1)]
        for param in param_names:
            v = run.config.get(param)
            row.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        row.append(f"**{metric_val:.4f}**" if i == 0 else f"{metric_val:.4f}")
        row.append(f"[View]({run.url})")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# Headline columns for the history CSV: one stable, plottable row per CI run.
HISTORY_METRICS = [
    "val/thput",
    "val/thput_eot",
    "val/acc",
    "eval/thput",
    "eval/thput_eot",
    "rollout/thput",
    "moving_avg_throughput",
]


def history_csv(out: str, limit: int = 500) -> int:
    """Write a per-run metric history (sha, kind, date, duration, metrics).

    Pulls every W&B run tagged `ci`, oldest first, so the CSV is a
    chronological record suitable for plotting metrics over time. Returns the
    number of rows written.
    """
    import csv

    import wandb

    api = wandb.Api()
    runs = api.runs(_project_path(api), filters={"tags": "ci"}, order="+created_at")

    rows = []
    for run in runs:
        tags = set(run.tags or [])
        sha = next((t.removeprefix("sha:") for t in tags if t.startswith("sha:")), "")
        kind = next((t.removeprefix("fci:") for t in tags if t.startswith("fci:")), "")
        summary = getattr(run.summary, "_json_dict", None) or dict(run.summary)
        row = {
            "created_at": run.created_at,
            "sha": sha,
            "kind": kind,
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "duration_s": summary.get("_runtime", ""),
        }
        for m in HISTORY_METRICS:
            v = read_metric(summary, m, None)
            row[m] = "" if v is None else v
        rows.append(row)
        if len(rows) >= limit:
            break

    fieldnames = ["created_at", "sha", "kind", "run_id", "name", "state", "duration_s", *HISTORY_METRICS]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
