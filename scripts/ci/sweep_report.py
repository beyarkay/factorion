#!/usr/bin/env python3
"""Generate a markdown report from a completed W&B sweep.

Queries the W&B API for all runs in a sweep, sorts by the target metric,
and produces a markdown summary suitable for posting as a GitHub PR comment.

Required env vars:
    WANDB_API_KEY  - W&B API key

Usage:
    python scripts/ci/sweep_report.py \
        --sweep-path entity/project/sweep_id \
        --output /tmp/sweep_summary.md \
        --pr-number 42 \
        --commit-sha abc1234
"""

import argparse
import sys

import wandb


def main():
    parser = argparse.ArgumentParser(
        description="Generate markdown report from a W&B sweep"
    )
    parser.add_argument(
        "--sweep-path",
        required=True,
        help="Full sweep path (entity/project/sweep_id)",
    )
    parser.add_argument(
        "--output",
        default="/tmp/sweep_summary.md",
        help="Output markdown file path",
    )
    parser.add_argument("--pr-number", default="unknown")
    parser.add_argument("--commit-sha", default="unknown")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top runs to show in the table",
    )
    args = parser.parse_args()

    api = wandb.Api()
    try:
        sweep = api.sweep(args.sweep_path)
    except Exception as e:
        print(f"ERROR: Could not fetch sweep {args.sweep_path}: {e}")
        sys.exit(1)

    metric_cfg = sweep.config.get("metric", {})
    metric_name = metric_cfg.get("name", "moving_avg/throughput")
    metric_goal = metric_cfg.get("goal", "maximize")
    sweep_params = sweep.config.get("parameters", {})
    reverse = metric_goal == "maximize"

    # Fetch all runs, keep only finished ones
    runs = [r for r in sweep.runs if r.state == "finished"]

    # sweep_path is entity/project/sweep_id — W&B URLs need /sweeps/ before the ID
    parts = args.sweep_path.split("/")
    sweep_url = f"https://wandb.ai/{parts[0]}/{parts[1]}/sweeps/{parts[2]}"

    if not runs:
        md = (
            "## W&B Hyperparameter Sweep Results\n\n"
            "**No completed runs found in sweep.**\n\n"
            f"[View sweep on W&B]({sweep_url})\n\n"
            f"<sub>Commit {args.commit_sha[:8]} "
            f"\u00b7 PR #{args.pr_number}</sub>\n"
        )
        with open(args.output, "w") as f:
            f.write(md)
        print(md)
        return

    # Sort runs by the sweep metric
    def get_metric(run):
        val = run.summary.get(metric_name)
        if val is None:
            return float("-inf") if reverse else float("inf")
        return val

    runs.sort(key=get_metric, reverse=reverse)

    best_run = runs[0]
    best_metric = get_metric(best_run)
    param_names = sorted(sweep_params.keys())

    md_lines = []

    # ── Header ────────────────────────────────────────────────────
    md_lines.append("## W&B Hyperparameter Sweep Results\n")
    md_lines.append("| | |")
    md_lines.append("|---|---|")
    md_lines.append(
        f"| **Sweep** | [{args.sweep_path}]({sweep_url}) |"
    )
    md_lines.append(f"| **Completed runs** | {len(runs)} |")
    md_lines.append(
        f"| **Metric** | `{metric_name}` ({metric_goal}) |"
    )
    md_lines.append(f"| **Best value** | **{best_metric:.4f}** |")
    md_lines.append("")

    # ── Best hyperparameters ──────────────────────────────────────
    md_lines.append("### Best Hyperparameters\n")
    md_lines.append("| Parameter | Value |")
    md_lines.append("|-----------|-------|")
    for param in param_names:
        value = best_run.config.get(param)
        if isinstance(value, float):
            md_lines.append(f"| `{param}` | `{value:.6g}` |")
        else:
            md_lines.append(f"| `{param}` | `{value}` |")
    md_lines.append(
        f"| **{metric_name}** | **{best_metric:.4f}** |"
    )
    best_run_url = best_run.url
    md_lines.append(f"\n[View best run on W&B]({best_run_url})\n")

    # ── Top N runs table ──────────────────────────────────────────
    top_n = min(args.top_n, len(runs))
    md_lines.append(f"### Top {top_n} Runs\n")

    # Build header row
    header_parts = ["Rank"]
    for p in param_names:
        header_parts.append(f"`{p}`")
    header_parts.append(f"`{metric_name}`")
    header_parts.append("W&B")
    md_lines.append("| " + " | ".join(header_parts) + " |")
    md_lines.append(
        "|" + "|".join("---" for _ in header_parts) + "|"
    )

    for i, run in enumerate(runs[:top_n]):
        metric_val = get_metric(run)
        row_parts = [str(i + 1)]
        for param in param_names:
            v = run.config.get(param)
            if isinstance(v, float):
                row_parts.append(f"{v:.4g}")
            else:
                row_parts.append(str(v))
        if i == 0:
            row_parts.append(f"**{metric_val:.4f}**")
        else:
            row_parts.append(f"{metric_val:.4f}")
        run_url = run.url
        row_parts.append(f"[View]({run_url})")
        md_lines.append("| " + " | ".join(row_parts) + " |")

    md_lines.append("")

    # ── Parameter ranges (collapsed) ─────────────────────────────
    md_lines.append(
        "<details><summary>Sweep parameter ranges</summary>\n"
    )
    md_lines.append("```yaml")
    for param in param_names:
        cfg = sweep_params[param]
        if "values" in cfg:
            md_lines.append(f"{param}: {cfg['values']}")
        elif "min" in cfg and "max" in cfg:
            dist = cfg.get("distribution", "uniform")
            md_lines.append(
                f"{param}: [{cfg['min']}, {cfg['max']}] ({dist})"
            )
    md_lines.append("```")
    md_lines.append("</details>\n")

    md_lines.append(
        f"<sub>Commit {args.commit_sha[:8]} "
        f"\u00b7 PR #{args.pr_number}</sub>"
    )

    report = "\n".join(md_lines)
    with open(args.output, "w") as f:
        f.write(report)
    print(f"Report written to {args.output}")
    print(report)


if __name__ == "__main__":
    main()
