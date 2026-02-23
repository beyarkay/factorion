#!/usr/bin/env python3
"""Compare multi-seed benchmark results between a PR branch and a baseline.

Performs Welch's t-test on key RL metrics and generates a markdown report
suitable for posting as a GitHub PR comment.

Usage:
    python scripts/ci/compare_runs.py \
        --pr-results results/pr/all_results.json \
        --baseline-results results/baseline/all_results.json \
        --pr-label "pr" \
        --pr-number 42 \
        --commit-sha abc1234 \
        --output summary.md
"""

import argparse
import json
import math
import sys


# ── Statistical helpers (no scipy dependency) ──────────────────────


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def welch_t_test(
    a: list[float], b: list[float]
) -> tuple[float, float, float]:
    """Welch's t-test for two independent samples with unequal variances.

    Returns (t_statistic, degrees_of_freedom, p_value).
    p_value is two-tailed, computed via the regularized incomplete beta function.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 0.0, 1.0

    mean_a, mean_b = mean(a), mean(b)
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, n_a + n_b - 2, 1.0

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2

    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed
    p_value = min(p_value, 1.0)

    return t_stat, df, p_value


def _t_distribution_p_value(t: float, df: float) -> float:
    """One-tailed p-value from the t-distribution using the regularized
    incomplete beta function.  P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2).
    """
    x = df / (df + t * t)
    return 0.5 * _regularized_incomplete_beta(x, df / 2.0, 0.5)


def _regularized_incomplete_beta(
    x: float, a: float, b: float, max_iter: int = 200, tol: float = 1e-12
) -> float:
    """Regularized incomplete beta function I_x(a, b) via continued fraction
    (Lentz's algorithm)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use the symmetry relation when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_incomplete_beta(1.0 - x, b, a, max_iter, tol)

    # Front factor: x^a * (1-x)^b / (a * B(a,b))
    ln_front = (
        a * math.log(x)
        + b * math.log(1.0 - x)
        - math.log(a)
        - _ln_beta(a, b)
    )
    front = math.exp(ln_front)

    # Continued fraction (modified Lentz)
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return front * f


def _ln_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled_std


# ── Metric extraction ─────────────────────────────────────────────

METRICS_TO_COMPARE = [
    {
        "key": "moving_avg_throughput",
        "label": "Throughput (moving avg)",
        "direction": "higher",
        "fmt": ".4f",
    },
    {
        "key": "max_missing_entities",
        "label": "Curriculum level",
        "direction": "higher",
        "fmt": ".1f",
    },
    {
        "key": "sps",
        "label": "Training speed (SPS)",
        "direction": "higher",
        "fmt": ",.0f",
    },
]


def extract_metric(results: list[dict], key: str) -> list[float]:
    values = []
    for r in results:
        if key in r:
            values.append(float(r[key]))
    return values


def verdict_str(
    p_value: float,
    direction: str,
    pr_mean: float,
    baseline_mean: float,
    alpha: float = 0.05,
) -> str:
    if p_value > alpha:
        return "No significant difference"
    if direction == "higher":
        if pr_mean > baseline_mean:
            return "Significantly better"
        else:
            return "Significantly worse"
    else:  # lower is better
        if pr_mean < baseline_mean:
            return "Significantly better"
        else:
            return "Significantly worse"


def verdict_icon(verdict: str) -> str:
    if "better" in verdict.lower():
        return "&#x2705;"  # green check
    elif "worse" in verdict.lower():
        return "&#x274C;"  # red X
    else:
        return "&#x2796;"  # dash


# ── Report generation ──────────────────────────────────────────────


def generate_report(
    pr_results: list[dict],
    baseline_results: list[dict],
    pr_label: str,
    pr_number: str,
    commit_sha: str,
    alpha: float = 0.05,
) -> str:
    n_pr = len(pr_results)
    n_base = len(baseline_results)

    lines = []
    lines.append("## GPU Benchmark Results")
    lines.append("")
    lines.append(f"**PR #{pr_number}** (`{commit_sha[:8]}`) vs **main** baseline")
    lines.append(f"- Seeds: {n_pr} (PR) vs {n_base} (baseline)")
    lines.append(f"- Significance level: {alpha}")

    gpu_name = pr_results[0].get("gpu", "") if pr_results else ""
    if not gpu_name:
        # Try to infer from wandb_url or other fields
        gpu_name = "unknown"
    timesteps = pr_results[0].get("total_timesteps", "?") if pr_results else "?"
    lines.append(f"- Timesteps per seed: {timesteps:,}" if isinstance(timesteps, int) else f"- Timesteps per seed: {timesteps}")
    lines.append("")

    # Summary table
    lines.append("### Comparison")
    lines.append("")
    lines.append(f"| Metric | main (n={n_base}) | PR (n={n_pr}) | Change | p-value | Verdict |")
    lines.append("|--------|-------------------|---------------|--------|---------|---------|")

    comparison_data = []

    for metric in METRICS_TO_COMPARE:
        key = metric["key"]
        label = metric["label"]
        direction = metric["direction"]
        fmt = metric["fmt"]

        pr_vals = extract_metric(pr_results, key)
        base_vals = extract_metric(baseline_results, key)

        if len(pr_vals) < 2 or len(base_vals) < 2:
            lines.append(f"| **{label}** | insufficient data | insufficient data | - | - | - |")
            continue

        pr_m, pr_s = mean(pr_vals), stdev(pr_vals)
        base_m, base_s = mean(base_vals), stdev(base_vals)

        _, _, p_val = welch_t_test(pr_vals, base_vals)
        d = cohens_d(pr_vals, base_vals)

        if base_m != 0:
            pct_change = (pr_m - base_m) / abs(base_m) * 100
            change_str = f"{pct_change:+.1f}%"
        else:
            change_str = "N/A"

        v = verdict_str(p_val, direction, pr_m, base_m, alpha)
        icon = verdict_icon(v)

        base_str = f"{base_m:{fmt}} +/- {base_s:{fmt}}"
        pr_str = f"{pr_m:{fmt}} +/- {pr_s:{fmt}}"
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else f"{p_val:.1e}"

        lines.append(
            f"| **{label}** | {base_str} | {pr_str} | {change_str} | {p_str} | {icon} {v} |"
        )

        comparison_data.append({
            "metric": label,
            "baseline_mean": base_m,
            "pr_mean": pr_m,
            "p_value": p_val,
            "cohens_d": d,
            "verdict": v,
        })

    # Per-seed detail table
    lines.append("")
    lines.append("<details><summary>Per-seed details</summary>")
    lines.append("")

    # Throughput per seed
    lines.append("#### Throughput per seed")
    lines.append("")
    lines.append("| Seed | Baseline | PR |")
    lines.append("|------|----------|----|")

    base_thputs = extract_metric(baseline_results, "moving_avg_throughput")
    pr_thputs = extract_metric(pr_results, "moving_avg_throughput")
    max_len = max(len(base_thputs), len(pr_thputs))

    for i in range(max_len):
        base_val = f"{base_thputs[i]:.4f}" if i < len(base_thputs) else "-"
        pr_val = f"{pr_thputs[i]:.4f}" if i < len(pr_thputs) else "-"
        seed_num = i + 1
        lines.append(f"| {seed_num} | {base_val} | {pr_val} |")

    # W&B links
    lines.append("")
    lines.append("#### W&B run links")
    lines.append("")
    for i, r in enumerate(pr_results):
        url = r.get("wandb_url")
        if url:
            lines.append(f"- PR seed {i+1}: [{url}]({url})")
    for i, r in enumerate(baseline_results):
        url = r.get("wandb_url")
        if url:
            lines.append(f"- Baseline seed {i+1}: [{url}]({url})")

    lines.append("")
    lines.append("</details>")
    lines.append("")
    lines.append(f"<sub>Commit {commit_sha[:8]} | PR #{pr_number}</sub>")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare multi-seed benchmark results (PR vs baseline)"
    )
    parser.add_argument(
        "--pr-results",
        required=True,
        help="Path to PR branch all_results.json",
    )
    parser.add_argument(
        "--baseline-results",
        required=True,
        help="Path to baseline all_results.json",
    )
    parser.add_argument("--pr-label", default="pr")
    parser.add_argument("--pr-number", default="unknown")
    parser.add_argument("--commit-sha", default="unknown")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output",
        default="summary.md",
        help="Path to write markdown report",
    )
    args = parser.parse_args()

    with open(args.pr_results) as f:
        pr_results = json.load(f)
    with open(args.baseline_results) as f:
        baseline_results = json.load(f)

    if not pr_results:
        print("ERROR: No PR results found")
        sys.exit(1)
    if not baseline_results:
        print("ERROR: No baseline results found")
        sys.exit(1)

    print(f"PR results:       {len(pr_results)} seeds")
    print(f"Baseline results: {len(baseline_results)} seeds")

    report = generate_report(
        pr_results=pr_results,
        baseline_results=baseline_results,
        pr_label=args.pr_label,
        pr_number=args.pr_number,
        commit_sha=args.commit_sha,
        alpha=args.alpha,
    )

    with open(args.output, "w") as f:
        f.write(report)

    print(f"\nReport written to {args.output}")

    # Also print to stdout for CI logs
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Exit with non-zero if any metric is significantly worse
    for metric in METRICS_TO_COMPARE:
        key = metric["key"]
        direction = metric["direction"]
        pr_vals = extract_metric(pr_results, key)
        base_vals = extract_metric(baseline_results, key)
        if len(pr_vals) >= 2 and len(base_vals) >= 2:
            _, _, p_val = welch_t_test(pr_vals, base_vals)
            v = verdict_str(p_val, direction, mean(pr_vals), mean(base_vals), args.alpha)
            # Only gate on throughput regression, not SPS
            if key == "moving_avg_throughput" and "worse" in v.lower():
                print(f"\nFAILED: {metric['label']} significantly regressed")
                sys.exit(1)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
