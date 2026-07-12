"""W&B-backed reports: all-metric run comparison, sweep summary, history CSV.

W&B is the source of truth for every training result, so all reporting reads
from there and can be (re)run locally at any time — nothing depends on files
left behind on a pod.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    main_mean: float
    main_std: float
    pr_mean: float
    pr_std: float
    n_main: int
    n_pr: int
    p_value: Optional[float]  # None when either side has < 2 values
    paired: bool

    @property
    def delta(self) -> float:
        return self.pr_mean - self.main_mean

    def verdict(self, alpha: float = 0.05) -> str:
        if self.p_value is None or self.p_value > alpha:
            return ""
        direction = metric_direction(self.name)
        if direction is None:
            return "significant"
        improved = self.delta > 0 if direction == "higher" else self.delta < 0
        return "better" if improved else "worse"


def compare_metric_rows(
    main: dict[int, dict[str, float]], pr: dict[int, dict[str, float]]
) -> list[MetricRow]:
    """Compare every metric across two {seed: flat_summary} sides.

    Uses a paired t-test on seeds present in both sides (>= 2), otherwise
    Welch's t-test on the unpaired values. Rows are sorted most-significant
    first, so the interesting differences top the table.
    """
    names = sorted({m for fs in list(main.values()) + list(pr.values()) for m in fs})
    rows = []
    for name in names:
        m_by_seed = {s: fs[name] for s, fs in main.items() if name in fs}
        p_by_seed = {s: fs[name] for s, fs in pr.items() if name in fs}
        if not m_by_seed or not p_by_seed:
            continue  # metric only exists on one side; nothing to compare
        common = sorted(set(m_by_seed) & set(p_by_seed))
        paired = len(common) >= 2
        if paired:
            m_vals = [m_by_seed[s] for s in common]
            p_vals = [p_by_seed[s] for s in common]
            _, _, p = paired_t_test(p_vals, m_vals)
        else:
            m_vals, p_vals = list(m_by_seed.values()), list(p_by_seed.values())
            p = welch_t_test(p_vals, m_vals)[2] if min(len(m_vals), len(p_vals)) >= 2 else None
        rows.append(
            MetricRow(
                name=name,
                main_mean=mean(m_vals),
                main_std=stdev(m_vals),
                pr_mean=mean(p_vals),
                pr_std=stdev(p_vals),
                n_main=len(m_vals),
                n_pr=len(p_vals),
                p_value=p,
                paired=paired,
            )
        )
    rows.sort(key=lambda r: (r.p_value is None, r.p_value if r.p_value is not None else 1.0, r.name))
    return rows


# Metrics surfaced OUTSIDE the <details> block of a report (the full
# every-metric table sits inside it). Ordered regexes; a metric is headline
# when any matches, and headline rows sort by first-matching pattern. Lesson
# names are matched, never hardcoded (`val/{LESSON}/thput_eot` covers every
# current and future lesson). SFT (val/) and PPO (eval/, rollout/, ...) key
# spaces are disjoint, so one list serves both kinds.
HEADLINE_PATTERNS = [
    # SFT: throughput first (overall then per-lesson), then accuracies
    # (overall then per-head; the head names come from the [a-z]+_acc shape,
    # which deliberately excludes per-lesson accs like val/{LESSON}/acc).
    r"^val/thput_eot$",
    r"^val/[A-Z0-9_]+/thput_eot$",
    r"^val/acc$",
    r"^val/[a-z]+_acc$",
    # PPO: on-policy rollout health (overall then per-lesson), then speed.
    r"^rollout/thput$",
    r"^rollout/reward$",
    r"^rollout/length$",
    r"^rollout/invalid_frac$",
    r"^rollout/[A-Z0-9_]+/thput$",
    r"^perf/update_seconds$",
    r"^perf/rollout_seconds$",
    r"^perf/eval_seconds$",
    r"^perf/sps$",
    # Speed tail for SFT (train/val seconds).
    r"^perf/",
]
_HEADLINE_RES = [re.compile(p) for p in HEADLINE_PATTERNS]


def select_headline(names) -> list[str]:
    """Headline subset of metric names, ordered by pattern then name."""
    out = []
    for pattern in _HEADLINE_RES:
        out.extend(sorted(n for n in names if pattern.match(n) and n not in out))
    return out


def _row_line(r: MetricRow, alpha: float) -> str:
    icons = {"better": "&#x2705;", "worse": "&#x274C;", "significant": "&#x26A0;&#xFE0F;", "": ""}
    p_str = "-" if r.p_value is None else (f"{r.p_value:.3f}" if r.p_value >= 0.001 else f"{r.p_value:.1e}")
    v = r.verdict(alpha)
    return (
        f"| `{r.name}` | {r.main_mean:.4g} ± {r.main_std:.2g} (n={r.n_main}) "
        f"| {r.pr_mean:.4g} ± {r.pr_std:.2g} (n={r.n_pr}) "
        f"| {r.delta:+.4g} | {p_str} | {icons[v]} {v} |"
    )


def render_compare_markdown(
    rows: list[MetricRow],
    main_label: str,
    pr_label: str,
    alpha: float = 0.05,
    headline: Optional[list[str]] = None,
) -> str:
    """Headline metrics up front; the full every-metric table in <details>.

    headline: explicit metric names, or None for the HEADLINE_PATTERNS match.
    """
    header = [
        f"| Metric | {main_label} | {pr_label} | Δ | p | |",
        "|---|---|---|---|---|---|",
    ]
    by_name = {r.name: r for r in rows}
    if headline is None:
        headline = select_headline(by_name)
    headline_rows = [by_name[m] for m in headline if m in by_name]

    lines = [
        f"## Compare: `{pr_label}` vs `{main_label}`",
        "",
        f"Paired by seed where possible (paired t-test; Welch's otherwise). "
        f"Significance level: {alpha}.",
        "",
    ]
    if headline_rows:
        lines += header + [_row_line(r, alpha) for r in headline_rows] + [""]
    significant = sum(1 for r in rows if r.verdict(alpha))
    lines.append(
        f"<details><summary>All {len(rows)} metrics "
        f"({significant} significant, sorted by p-value)</summary>"
    )
    lines.append("")
    lines += header
    if rows:
        lines += [_row_line(r, alpha) for r in rows]
    else:
        lines.append("| _no comparable metrics found_ | | | | | |")
    lines += ["", "</details>"]
    return "\n".join(lines)


# ── Assertions ─────────────────────────────────────────────────────
# e.g. "pr:val/thput > main:val/thput" — evaluated on group means. Sides:
# pr: = the PR's commit, main: = the baseline (test:/base: kept as aliases);
# a bare number is a constant threshold. == and ~= mean approximately equal
# (|lhs - rhs| <= tolerance): exact float equality on run means would
# never hold, so a tolerance is built in and overridable with a trailing
# "+- 0.01" (or "+/- 0.01").

_ASSERT_RE = re.compile(
    r"^\s*(\S+)\s*(<=|>=|==|~=|<|>)\s*(\S+?)\s*(?:\+/?-\s*(\S+)\s*)?$"
)
_SIDE_ALIASES = {"pr": "pr", "test": "pr", "main": "main", "base": "main"}
_OPS = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
}
_APPROX_OPS = ("==", "~=")
APPROX_TOLERANCE_DEFAULT = 1e-3


@dataclass
class AssertionResult:
    expression: str
    passed: bool
    detail: str


def _resolve_operand(token: str, means: dict[str, dict[str, float]]) -> tuple[float, str]:
    """Returns (value, human label). `means` maps side ("pr"/"main") to
    {metric: mean}. Raises ValueError for unknown sides/metrics."""
    if ":" in token:
        side_raw, metric = token.split(":", 1)
        side = _SIDE_ALIASES.get(side_raw.lower())
        if side is None:
            raise ValueError(f"unknown side '{side_raw}' (use pr: or main:)")
        if metric not in means[side]:
            raise ValueError(f"metric '{metric}' not found on the {side} side")
        return means[side][metric], f"{token}={means[side][metric]:.4g}"
    return float(token), token


def evaluate_assertion(expression: str, rows: list[MetricRow]) -> AssertionResult:
    means = {
        "pr": {r.name: r.pr_mean for r in rows},
        "main": {r.name: r.main_mean for r in rows},
    }
    m = _ASSERT_RE.match(expression)
    if m is None:
        return AssertionResult(
            expression, False, "could not parse (want: LHS <op> RHS [+- TOL])"
        )
    lhs_tok, op, rhs_tok, tol_tok = m.groups()
    if tol_tok is not None and op not in _APPROX_OPS:
        return AssertionResult(
            expression, False, f"a '+- tolerance' only applies to == / ~=, not {op}"
        )
    try:
        lhs, lhs_label = _resolve_operand(lhs_tok, means)
        rhs, rhs_label = _resolve_operand(rhs_tok, means)
        tolerance = float(tol_tok) if tol_tok is not None else APPROX_TOLERANCE_DEFAULT
    except ValueError as e:
        return AssertionResult(expression, False, str(e))
    if op in _APPROX_OPS:
        diff = abs(lhs - rhs)
        passed = diff <= tolerance
        return AssertionResult(
            expression,
            passed,
            f"{lhs_label} {op} {rhs_label}: |Δ| = {diff:.4g} vs tolerance {tolerance:g}",
        )
    passed = _OPS[op](lhs, rhs)
    return AssertionResult(expression, passed, f"{lhs_label} {op} {rhs_label}")


def render_assertions_markdown(results: list[AssertionResult]) -> str:
    if not results:
        return ""
    lines = ["### Assertions", ""]
    for r in results:
        icon = "&#x2705;" if r.passed else "&#x274C;"
        lines.append(f"- {icon} `{r.expression}` — {r.detail}")
    lines.append("")
    return "\n".join(lines)


def _fetch_group(api, project_path: str, group: str) -> dict[int, dict[str, float]]:
    """Fetch a W&B group's FINISHED runs as {seed: flattened summary}.

    Two finished runs on one seed (a rerun into the same group) would
    silently shadow each other in the dict, so the newest wins and the
    collision is logged — a seed's numbers must never be a lottery over
    which run iterated last.
    """
    newest_by_seed: dict[int, Any] = {}
    for run in api.runs(project_path, filters={"group": group}):
        seed = run.config.get("seed")
        if seed is None or run.state != "finished":
            continue
        seed = int(seed)
        prev = newest_by_seed.get(seed)
        if prev is not None:
            print(
                f"warning: group {group} has multiple finished runs for seed "
                f"{seed}; keeping the newest",
                flush=True,
            )
        if prev is None or (run.created_at or "") > (prev.created_at or ""):
            newest_by_seed[seed] = run
    out: dict[int, dict[str, float]] = {}
    for seed, run in newest_by_seed.items():
        summary = getattr(run.summary, "_json_dict", None) or dict(run.summary)
        out[seed] = flatten_summary(summary)
    return out


def _project_path(api) -> str:
    return f"{api.default_entity}/{WANDB_PROJECT}"


def wait_for_groups(
    main_group: str,
    pr_group: str,
    expect_each: int,
    timeout_seconds: int,
    poll_seconds: int = 120,
    pod_ids: Optional[list[str]] = None,
    on_poll: Optional[Callable[[], None]] = None,
) -> None:
    """Block until both W&B groups have `expect_each` finished runs.

    Ends early when every launched pod is gone (pass `pod_ids`): a vanished
    pod can never add a run, so whatever exists at that point is final — one
    grace poll for W&B to settle, then report. A pod past its name-encoded
    deadline counts as gone even while RunPod still lists it (a pod that
    never started a container sits idle until the watchdog reaps it — the
    report must not wait on that corpse). Falls back to the timeout when
    pods can't be checked.
    """
    import wandb

    deadline = time.time() + timeout_seconds
    pods_all_gone_since = None
    while time.time() < deadline:
        api = wandb.Api()  # fresh client: avoid cached run listings
        path = _project_path(api)
        counts = {}
        for group in (main_group, pr_group):
            runs = api.runs(path, filters={"group": group})
            counts[group] = sum(1 for r in runs if r.state == "finished")
        print(f"finished runs: {counts} (want {expect_each} each)", flush=True)
        if on_poll is not None:
            on_poll()  # e.g. refresh the PR launch comment's live statuses
        if all(c >= expect_each for c in counts.values()):
            return

        if pod_ids:
            try:
                from ci import runpod_api
                from ci.config import parse_pod_name

                now = time.time()
                live = set()
                for p in runpod_api.list_ci_pods():
                    meta = parse_pod_name(p.get("name") or "")
                    if meta is not None and meta.deadline <= now:
                        continue  # dead man walking: can never add a run
                    live.add(p["id"])
                alive = sorted(set(pod_ids) & live)
            except Exception as e:
                alive = None
                print(f"could not check pod liveness: {e}", flush=True)
            if alive == []:
                if pods_all_gone_since is None:
                    pods_all_gone_since = time.time()
                    print("all launched pods are gone; one grace poll for W&B to settle", flush=True)
                else:
                    print("pods gone and counts settled; reporting what exists", flush=True)
                    return
            elif alive:
                pods_all_gone_since = None
                print(f"still waiting on pod(s): {alive}", flush=True)

        time.sleep(poll_seconds)
    print(f"wait_for_groups timed out after {timeout_seconds}s; reporting what exists")


def compare_report(
    main_group: str,
    pr_group: str,
    assertions: Optional[list[str]] = None,
) -> tuple[str, bool]:
    """Markdown comparison of every metric across two W&B run groups.

    Returns (markdown, ok): ok is False when any assertion fails or when
    either group has no finished runs to compare.
    """
    import wandb

    api = wandb.Api()
    path = _project_path(api)
    main = _fetch_group(api, path, main_group)
    pr = _fetch_group(api, path, pr_group)
    if not main or not pr:
        return (
            f"No runs found for comparison: {main_group} has {len(main)} run(s), "
            f"{pr_group} has {len(pr)} run(s).",
            False,
        )
    rows = compare_metric_rows(main, pr)
    entity = api.default_entity
    md = render_compare_markdown(rows, main_label=main_group, pr_label=pr_group)
    if entity:
        for group in (main_group, pr_group):
            md = md.replace(
                f"`{group}`",
                f"[`{group}`](https://wandb.ai/{entity}/{WANDB_PROJECT}/groups/{group})",
                1,
            )
    ok = True
    if assertions:
        results = [evaluate_assertion(a, rows) for a in assertions]
        ok = all(r.passed for r in results)
        md = render_assertions_markdown(results) + "\n" + md
    return md, ok


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


# ── Per-run PR summaries (posted by the reporter cron) ─────────────
# Long training runs outlive GitHub's 6h job limit, so the workflow that
# launched them can't wait around. Instead ci-reporter.yml runs every 30 min:
# any finished W&B run tagged pr:<N> that has no summary comment yet gets one.
# An invisible marker in each comment makes the sweep idempotent.

RUN_MARKER_TEMPLATE = "<!-- factorion-ci-run:{run_id} -->"


def run_summary_markdown(
    run_id: str,
    name: str,
    state: str,
    url: str,
    kind: str,
    sha7: str,
    summary_flat: dict[str, float],
    headline: Optional[list[str]] = None,
) -> str:
    """Summary comment for a single finished CI run."""
    icon = "&#x2705;" if state == "finished" else "&#x274C;"
    duration = summary_flat.get("_runtime")
    import os

    repo = os.environ.get("GITHUB_REPOSITORY")
    commit = f"[`{sha7}`](https://github.com/{repo}/commit/{sha7})" if repo else f"`{sha7}`"
    lines = [
        RUN_MARKER_TEMPLATE.format(run_id=run_id),
        f"## {icon} CI {kind} run `{name}` {state}",
        "",
        f"Commit {commit} &middot; [view on W&B]({url})"
        + (f" &middot; {int(duration) // 60} min" if duration else ""),
        "",
    ]
    if headline is None:
        headline = select_headline(summary_flat)
    shown = [(m, summary_flat[m]) for m in headline if m in summary_flat]
    if shown:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| `{m}` | {v:.4g} |" for m, v in shown]
    others = {
        m: v
        for m, v in sorted(summary_flat.items())
        if m not in dict(shown) and not m.startswith("_")
    }
    if others:
        lines += [
            "",
            f"<details><summary>All {len(shown) + len(others)} metrics</summary>",
            "",
            "| Metric | Value |",
            "|---|---|",
            *[f"| `{m}` | {v:.4g} |" for m, v in others.items()],
            "",
            "</details>",
        ]
    return "\n".join(lines)


def boot_failure_markdown(
    run_id: str,
    url: str,
    kind: str,
    sha7: str,
    log_tail: str = "",
) -> str:
    """PR comment for a pod that died before it could start the run.

    Boot failures never reach the training command (no metrics to report), so
    this is a purpose-built message: what happened, the likely cause, and the
    tail of the boot log for immediate diagnosis.
    """
    import os

    repo = os.environ.get("GITHUB_REPOSITORY")
    commit = f"[`{sha7}`](https://github.com/{repo}/commit/{sha7})" if repo else f"`{sha7}`"
    lines = [
        RUN_MARKER_TEMPLATE.format(run_id=run_id),
        f"## &#x274C; CI {kind} pod failed to boot",
        "",
        f"The pod for commit {commit} never started the run — it failed while "
        "cloning the repo, building the Rust extension, or during setup, so no "
        "training happened.",
        "",
        f"[Boot log on W&B]({url}) &middot; a common cause is the commit having "
        "left every branch (PR merged and its branch deleted, or force-pushed) "
        "after the pod was launched.",
    ]
    if log_tail.strip():
        lines += ["", "```", log_tail.strip()[-1500:], "```"]
    return "\n".join(lines)


def select_unreported(
    candidates: list[dict], existing_bodies: list[str]
) -> list[dict]:
    """Pure core of the reporter: candidates (dicts with a "run_id" key) whose
    marker doesn't appear in any existing PR comment body."""
    joined = "\n".join(existing_bodies)
    return [
        c
        for c in candidates
        if RUN_MARKER_TEMPLATE.format(run_id=c["run_id"]) not in joined
    ]


def post_pending_reports(window_days: int = 3, dry_run: bool = False) -> int:
    """Post summary comments for finished PR-linked runs that lack one.

    Scans W&B runs tagged `ci` from the last `window_days`, keeps terminal
    runs carrying a pr:<N> tag (skipping compare fan-out runs — those are
    reported as one comparison by the compare workflow), and comments on the
    PR unless the run's marker is already present. Returns #posted.
    """
    from datetime import datetime, timedelta, timezone

    import wandb

    from ci import github_api

    api = wandb.Api()
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    by_pr: dict[int, list[dict]] = {}
    for run in api.runs(_project_path(api), filters={"tags": "ci"}, order="-created_at"):
        created = datetime.fromisoformat(str(run.created_at).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created < cutoff:
            break  # runs are newest-first; everything after this is older
        if run.state not in ("finished", "crashed", "failed"):
            continue
        tags = set(run.tags or [])
        if any(t.startswith("cmp:") for t in tags):
            continue
        if not any(t.startswith("kind:") for t in tags):
            continue  # pre-rework runs also carried ci + pr:<N> tags
        pr = next((t.removeprefix("pr:") for t in tags if t.startswith("pr:")), None)
        if pr is None or not pr.isdigit():
            continue
        summary = getattr(run.summary, "_json_dict", None) or dict(run.summary)
        flat = flatten_summary(summary)
        if "_runtime" in summary and _is_number(summary["_runtime"]):
            flat["_runtime"] = float(summary["_runtime"])
        by_pr.setdefault(int(pr), []).append(
            {
                "run_id": run.id,
                "name": run.name,
                "state": run.state,
                "url": run.url,
                "kind": next(
                    (t.removeprefix("kind:") for t in tags if t.startswith("kind:")), "?"
                ),
                "sha7": next(
                    (t.removeprefix("sha:") for t in tags if t.startswith("sha:")), "?"
                ),
                "summary_flat": flat,
                "boot_failure": "boot-failure" in tags,
                "boot_log_tail": str(summary.get("boot_log_tail", "")),
            }
        )

    posted = 0
    for pr_number, candidates in by_pr.items():
        bodies = github_api.list_pr_comment_bodies(pr_number)
        for c in select_unreported(candidates, bodies):
            if c["boot_failure"]:
                md = boot_failure_markdown(
                    run_id=c["run_id"],
                    url=c["url"],
                    kind=c["kind"],
                    sha7=c["sha7"],
                    log_tail=c["boot_log_tail"],
                )
            else:
                md = run_summary_markdown(
                    run_id=c["run_id"],
                    name=c["name"],
                    state=c["state"],
                    url=c["url"],
                    kind=c["kind"],
                    sha7=c["sha7"],
                    summary_flat=c["summary_flat"],
                )
            if dry_run:
                print(f"[dry-run] would comment on PR #{pr_number} for run {c['run_id']}")
            else:
                github_api.post_pr_comment(pr_number, md)
                print(f"Commented on PR #{pr_number} for run {c['run_id']}")
            posted += 1
    if not posted:
        print("No pending run reports.")
    return posted
