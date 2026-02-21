#!/usr/bin/env python3
"""Check W&B metrics from a CI smoke test run against baseline thresholds.

Queries the W&B API for a completed run, verifies it finished successfully,
and compares key metrics against configurable baseline thresholds.

Required env vars:
    WANDB_API_KEY  - W&B API key

Usage:
    python scripts/ci/check_wandb_metrics.py \
        --project factorion-ci \
        --run-name ci-pr42-abc1234 \
        --baseline-file scripts/ci/baseline.json
"""

import argparse
import json
import sys

import wandb


def find_run(project: str, run_name: str, entity: str | None = None) -> wandb.apis.public.Run:
    """Find a W&B run by its display name (exp_name in ppo.py).

    ppo.py sets the run name to: {env_id}__{exp_name}__{seed}__{iso8601}
    We match on runs whose name contains the run_name as the exp_name component.
    """
    api = wandb.Api()
    full_project = f"{entity}/{project}" if entity else project

    # Try exact match on config.exp_name first (most reliable)
    runs = api.runs(
        full_project,
        filters={"config.exp_name": run_name},
        order="-created_at",
    )
    runs_list = list(runs)
    if runs_list:
        return runs_list[0]

    # Fallback: match on display name containing the run_name
    runs = api.runs(
        full_project,
        filters={"display_name": {"$regex": f".*{run_name}.*"}},
        order="-created_at",
    )
    runs_list = list(runs)
    if runs_list:
        return runs_list[0]

    print(f"ERROR: No run found matching '{run_name}' in project '{full_project}'")
    sys.exit(1)


def check_run_completed(run: wandb.apis.public.Run) -> bool:
    """Verify the run finished without crashing."""
    if run.state == "finished":
        return True
    if run.state == "running":
        print(f"WARNING: Run {run.id} is still running (state={run.state})")
        return True  # allow it, might still be finalizing
    print(f"ERROR: Run {run.id} did not finish successfully (state={run.state})")
    return False


def compare_metrics(
    metrics: dict, baseline: dict
) -> tuple[bool, list[str], list[str]]:
    """Compare run metrics against baseline thresholds.

    Baseline format (per metric):
        {
            "min": <float>,           # fail if metric < min
            "max": <float>,           # fail if metric > max
            "baseline_value": <float>,# reference baseline value
            "direction": "higher"|"lower",  # which direction is better
            "required": true|false    # if false, missing metric is a warning not failure
        }

    Returns (passed, failures, warnings).
    """
    failures = []
    warnings = []

    for metric_name, constraint in baseline.items():
        required = constraint.get("required", True)

        if metric_name not in metrics:
            msg = f"Metric '{metric_name}' not found in run summary"
            if required:
                failures.append(msg)
            else:
                warnings.append(msg)
            continue

        value = metrics[metric_name]

        # Check absolute bounds
        if "min" in constraint and value < constraint["min"]:
            failures.append(
                f"{metric_name}: {value:.6g} < minimum threshold {constraint['min']:.6g}"
            )
        if "max" in constraint and value > constraint["max"]:
            failures.append(
                f"{metric_name}: {value:.6g} > maximum threshold {constraint['max']:.6g}"
            )

        # Check against baseline value (informational + optional gate)
        if "baseline_value" in constraint:
            baseline_val = constraint["baseline_value"]
            direction = constraint.get("direction", "higher")
            regression = (
                (direction == "higher" and value < baseline_val)
                or (direction == "lower" and value > baseline_val)
            )
            if regression:
                msg = (
                    f"{metric_name}: {value:.6g} regressed vs baseline "
                    f"{baseline_val:.6g} ({direction} is better)"
                )
                if constraint.get("gate", False):
                    failures.append(msg)
                else:
                    warnings.append(msg)

    return len(failures) == 0, failures, warnings


def main():
    parser = argparse.ArgumentParser(
        description="Check W&B metrics from CI smoke test against baseline"
    )
    parser.add_argument(
        "--project",
        default="factorion-ci",
        help="W&B project name",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity (team/user). Uses default if not set.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="W&B run ID (takes precedence over --run-name)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="W&B run name / exp_name to search for",
    )
    parser.add_argument(
        "--baseline-file",
        default="scripts/ci/baseline.json",
        help="Path to baseline metrics JSON",
    )
    args = parser.parse_args()

    if not args.run_id and not args.run_name:
        print("ERROR: Must specify --run-id or --run-name")
        sys.exit(1)

    # Find the run
    api = wandb.Api()
    if args.run_id:
        full_project = f"{args.entity}/{args.project}" if args.entity else args.project
        run = api.run(f"{full_project}/{args.run_id}")
    else:
        run = find_run(args.project, args.run_name, args.entity)

    print(f"Found run: {run.name} (id={run.id}, state={run.state})")
    print(f"  URL: {run.url}")

    # Check run completed
    if not check_run_completed(run):
        sys.exit(1)

    # Load baseline
    with open(args.baseline_file) as f:
        baseline = json.load(f)

    # Get run summary metrics
    metrics = dict(run.summary)

    print("\nMetrics vs baseline:")
    print(f"{'Metric':<40} {'Value':>12} {'Threshold':>12} {'Status':>8}")
    print("-" * 76)

    for metric_name, constraint in baseline.items():
        value = metrics.get(metric_name)
        value_str = f"{value:.6g}" if value is not None else "MISSING"

        threshold_parts = []
        if "min" in constraint:
            threshold_parts.append(f">={constraint['min']:.6g}")
        if "max" in constraint:
            threshold_parts.append(f"<={constraint['max']:.6g}")
        threshold_str = ", ".join(threshold_parts) if threshold_parts else "-"

        status = "OK" if value is not None else "MISS"
        if value is not None:
            if "min" in constraint and value < constraint["min"]:
                status = "FAIL"
            elif "max" in constraint and value > constraint["max"]:
                status = "FAIL"

        print(f"  {metric_name:<38} {value_str:>12} {threshold_str:>12} {status:>8}")

    # Compare
    passed, failures, warnings = compare_metrics(metrics, baseline)

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")

    if passed:
        print("\nAll metric checks passed")
    else:
        print("\nMetric checks FAILED:")
        for f_msg in failures:
            print(f"  - {f_msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
