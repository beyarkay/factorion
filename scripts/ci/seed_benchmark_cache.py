#!/usr/bin/env python3
"""One-time script to seed the benchmark cache with results from PR #18.

Run once from the repo root with gh CLI authenticated:
    python scripts/ci/seed_benchmark_cache.py

This reconstructs all_results.json from the per-seed data posted in
https://github.com/beyarkay/factorion/pull/18#issuecomment-3957201151
and uploads both the PR and baseline results to the benchmark-cache release.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

# PR #18 benchmark metadata (10 seeds, 100k timesteps, A100 PCIe)
# PR: https://github.com/beyarkay/factorion/pull/18#issuecomment-3957201151
# Baseline: https://github.com/beyarkay/factorion/commit/d4f8c1e9be4bf60af69784544a2313c8864a7ec0
PR_COMMIT = "cd411e9a768d85c8c5cdefad149fbb34c8a4e0f9"
BASELINE_COMMIT = "d4f8c1e9be4bf60af69784544a2313c8864a7ec0"
NUM_SEEDS = 10
TIMESTEPS = 100000

# Per-seed curriculum scores extracted from the PR comment
PR_CURRICULUM_SCORES = [
    0.8820, 1.8500, 0.9400, 1.9040, 0.9220,
    1.8560, 1.9140, 1.7300, 1.8020, 1.6820,
]
BASELINE_CURRICULUM_SCORES = [
    0.8740, 0.7780, 0.9180, 0.8960, 1.8160,
    1.8520, 1.8280, 1.8160, 1.8700, 0.9020,
]

# W&B run URLs
PR_WANDB_URLS = [
    "https://wandb.ai/beyarkay/factorion/runs/npegvyjx",
    "https://wandb.ai/beyarkay/factorion/runs/tz2zjv2v",
    "https://wandb.ai/beyarkay/factorion/runs/3744n7xf",
    "https://wandb.ai/beyarkay/factorion/runs/m9dc17d2",
    "https://wandb.ai/beyarkay/factorion/runs/vrxwezls",
    "https://wandb.ai/beyarkay/factorion/runs/fti7nrib",
    "https://wandb.ai/beyarkay/factorion/runs/7b6b6ilm",
    "https://wandb.ai/beyarkay/factorion/runs/iu4g80ul",
    "https://wandb.ai/beyarkay/factorion/runs/d65yp55u",
    "https://wandb.ai/beyarkay/factorion/runs/iucd4reo",
]
BASELINE_WANDB_URLS = [
    "https://wandb.ai/beyarkay/factorion/runs/mgfp0iqq",
    "https://wandb.ai/beyarkay/factorion/runs/xuslp0ht",
    "https://wandb.ai/beyarkay/factorion/runs/nfpkozhx",
    "https://wandb.ai/beyarkay/factorion/runs/ngu7bqjd",
    "https://wandb.ai/beyarkay/factorion/runs/nca89dj1",
    "https://wandb.ai/beyarkay/factorion/runs/3g0wm8qk",
    "https://wandb.ai/beyarkay/factorion/runs/xgzs5cov",
    "https://wandb.ai/beyarkay/factorion/runs/4s584ulx",
    "https://wandb.ai/beyarkay/factorion/runs/9v1eo02n",
    "https://wandb.ai/beyarkay/factorion/runs/d7zyoaw1",
]

# Summary stats from the PR comment
PR_SPS_MEAN = 187
BASELINE_SPS_MEAN = 226


def score_to_fields(score: float) -> dict:
    """Derive max_missing_entities and moving_avg_throughput from curriculum_score."""
    import math
    level = math.floor(score) + 1
    throughput = round(score - math.floor(score), 4)
    return {"max_missing_entities": level, "moving_avg_throughput": throughput}


def build_results(
    scores: list[float],
    wandb_urls: list[str],
    sps_mean: int,
) -> list[dict]:
    results = []
    for i, (score, url) in enumerate(zip(scores, wandb_urls), start=1):
        fields = score_to_fields(score)
        results.append({
            "global_step": TIMESTEPS,
            "total_timesteps": TIMESTEPS,
            "moving_avg_throughput": fields["moving_avg_throughput"],
            "curriculum_score": score,
            "max_missing_entities": fields["max_missing_entities"],
            "sps": sps_mean,
            "seed": i,
            "seed_file": str(i),
            "wandb_url": url,
        })
    return results


def main() -> None:
    print(f"PR commit:       {PR_COMMIT[:12]}")
    print(f"Baseline commit: {BASELINE_COMMIT[:12]}")

    pr_results = build_results(PR_CURRICULUM_SCORES, PR_WANDB_URLS, PR_SPS_MEAN)
    baseline_results = build_results(
        BASELINE_CURRICULUM_SCORES, BASELINE_WANDB_URLS, BASELINE_SPS_MEAN,
    )
    baseline_sha = BASELINE_COMMIT

    with tempfile.TemporaryDirectory() as tmpdir:
        pr_path = os.path.join(tmpdir, "pr_results.json")
        baseline_path = os.path.join(tmpdir, "baseline_results.json")

        with open(pr_path, "w") as f:
            json.dump(pr_results, f, indent=2)
        with open(baseline_path, "w") as f:
            json.dump(baseline_results, f, indent=2)

        print(f"\nPR results ({len(pr_results)} seeds):")
        print(json.dumps(pr_results[0], indent=2))

        print(f"\nBaseline results ({len(baseline_results)} seeds):")
        print(json.dumps(baseline_results[0], indent=2))

        # Upload via benchmark_cache.py
        cache_script = os.path.join(
            os.path.dirname(__file__), "benchmark_cache.py",
        )

        print(f"\n>>> Saving PR results for {PR_COMMIT[:12]}...")
        subprocess.run(
            [sys.executable, cache_script, "save",
             "--sha", PR_COMMIT,
             "--seeds", str(NUM_SEEDS),
             "--timesteps", str(TIMESTEPS),
             "--input", pr_path],
            check=True,
        )

        print(f">>> Saving baseline results for {baseline_sha[:12]}...")
        subprocess.run(
            [sys.executable, cache_script, "save",
             "--sha", baseline_sha,
             "--seeds", str(NUM_SEEDS),
             "--timesteps", str(TIMESTEPS),
             "--input", baseline_path],
            check=True,
        )

    print("\nCache seeded successfully!")


if __name__ == "__main__":
    main()
