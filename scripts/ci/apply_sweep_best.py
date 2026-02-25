#!/usr/bin/env python3
"""Apply the best hyperparameters from a W&B sweep to ppo.py and open a PR.

Reads the best run from a completed sweep, updates the default values in the
Args dataclass in ppo.py, and creates a pull request with the changes.

Required env vars:
    WANDB_API_KEY  - W&B API key
    GH_TOKEN       - GitHub token for creating PRs

Usage:
    python scripts/ci/apply_sweep_best.py \
        --sweep-path entity/project/sweep_id \
        --ppo-file ppo.py \
        --base-branch main
"""

import argparse
import re
import subprocess
import sys

import wandb

# Hyperparameters in the Args dataclass that sweeps can tune.
# Maps sweep param name -> (python type formatter).
# Only these will be updated; other sweep config keys are ignored.
TUNABLE_PARAMS = {
    "adam_epsilon": "float",
    "chan1": "int",
    "chan2": "int",
    "chan3": "int",
    "clip_coef": "float",
    "coeff_throughput": "float",
    "coeff_shaping_direction": "float",
    "coeff_shaping_entity": "float",
    "coeff_shaping_location": "float",
    "ent_coef_start": "float",
    "ent_coef_end": "float",
    "flat_dim": "int",
    "gae_lambda": "float",
    "gamma": "float",
    "learning_rate": "float",
    "max_grad_norm": "float",
    "tile_head_std": "float",
    "vf_coef": "float",
}


def format_value(name: str, value) -> str:
    """Format a hyperparameter value for Python source code."""
    typ = TUNABLE_PARAMS.get(name, "float")
    if typ == "int":
        return str(int(value))
    # Use general format to avoid unnecessary trailing zeros
    # but keep enough precision for small values like adam_epsilon
    return f"{float(value):.6g}"


def update_ppo_defaults(ppo_path: str, best_config: dict, sweep_url: str) -> list:
    """Update Args dataclass defaults in ppo.py. Returns list of changed params."""
    with open(ppo_path) as f:
        content = f.read()

    changed = []

    for param_name, value in sorted(best_config.items()):
        if param_name not in TUNABLE_PARAMS:
            continue

        formatted = format_value(param_name, value)
        type_hint = "int" if TUNABLE_PARAMS[param_name] == "int" else "float"

        # Match lines like:  learning_rate: float = 2.5e-4
        pattern = rf"^(\s*{param_name}\s*:\s*{type_hint}\s*=\s*)(.+)$"
        match = re.search(pattern, content, re.MULTILINE)
        if not match:
            print(f"  WARNING: could not find {param_name} in {ppo_path}")
            continue

        old_val = match.group(2).strip()
        if old_val == formatted:
            continue

        content = content[: match.start(2)] + formatted + content[match.end(2) :]
        changed.append((param_name, old_val, formatted))
        print(f"  {param_name}: {old_val} -> {formatted}")

    # Add sweep URL comment above the class if not already present
    sweep_comment = f"# Best hyperparameters from W&B sweep: {sweep_url}"
    # Replace existing sweep comment if present, otherwise insert before "class Args"
    existing_pattern = r"^# Best hyperparameters from W&B sweep: .+\n"
    if re.search(existing_pattern, content, re.MULTILINE):
        content = re.sub(existing_pattern, sweep_comment + "\n", content)
    else:
        content = content.replace("class Args:", sweep_comment + "\nclass Args:")

    with open(ppo_path, "w") as f:
        f.write(content)

    return changed


def run(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(
        description="Apply best sweep hyperparameters to ppo.py and open a PR"
    )
    parser.add_argument(
        "--sweep-path",
        required=True,
        help="Full sweep path (entity/project/sweep_id)",
    )
    parser.add_argument(
        "--ppo-file", default="ppo.py", help="Path to ppo.py"
    )
    parser.add_argument(
        "--base-branch", default="main", help="Base branch for the PR"
    )
    parser.add_argument(
        "--pr-number",
        default=None,
        help="PR number that triggered the sweep (for cross-linking)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only update the file, don't create a branch/PR",
    )
    parser.add_argument(
        "--output-pr-url",
        default="/tmp/sweep_pr_url.txt",
        help="File to write the created PR URL to",
    )
    args = parser.parse_args()

    # ── Fetch best run from sweep ────────────────────────────────
    api = wandb.Api()
    try:
        sweep = api.sweep(args.sweep_path)
    except Exception as e:
        print(f"ERROR: Could not fetch sweep {args.sweep_path}: {e}")
        sys.exit(1)

    metric_cfg = sweep.config.get("metric", {})
    metric_name = metric_cfg.get("name", "moving_avg/curriculum_score")
    metric_goal = metric_cfg.get("goal", "maximize")
    reverse = metric_goal == "maximize"

    runs = [r for r in sweep.runs if r.state == "finished"]
    if not runs:
        print("No finished runs in sweep. Skipping.")
        sys.exit(0)

    def get_metric(run):
        val = run.summary.get(metric_name)
        if val is None:
            return float("-inf") if reverse else float("inf")
        return val

    runs.sort(key=get_metric, reverse=reverse)
    best_run = runs[0]
    best_metric = get_metric(best_run)

    parts = args.sweep_path.split("/")
    sweep_id = parts[2]
    sweep_url = f"https://wandb.ai/{parts[0]}/{parts[1]}/sweeps/{parts[2]}"

    print(f"Best run: {best_run.name} ({metric_name}={best_metric:.4f})")
    print(f"Sweep URL: {sweep_url}")

    # ── Update ppo.py ────────────────────────────────────────────
    print(f"\nUpdating {args.ppo_file}...")
    changed = update_ppo_defaults(args.ppo_file, best_run.config, sweep_url)

    if not changed:
        print("No hyperparameters changed. Skipping PR creation.")
        sys.exit(0)

    print(f"\nUpdated {len(changed)} hyperparameter(s)")

    if args.dry_run:
        print("Dry run — not creating branch or PR.")
        return

    # ── Create branch, commit, push, open PR ─────────────────────
    branch_name = f"sweep/apply-best-{sweep_id}"

    # Configure git for CI
    run(["git", "config", "user.name", "github-actions[bot]"])
    run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"])

    # Create and switch to new branch from base
    result = run(["git", "checkout", "-b", branch_name], check=False)
    if result.returncode != 0:
        # Branch might already exist
        run(["git", "checkout", branch_name], check=False)

    run(["git", "add", args.ppo_file])

    # Build commit message
    commit_lines = [
        f"Update hyperparameters from sweep {sweep_id}",
        "",
        f"Applied best hyperparameters from W&B sweep:",
        f"  {sweep_url}",
        f"",
        f"Best run: {best_run.name} ({metric_name}={best_metric:.4f})",
        f"",
        "Changes:",
    ]
    for param, old, new in changed:
        commit_lines.append(f"  {param}: {old} -> {new}")

    commit_msg = "\n".join(commit_lines)
    run(["git", "commit", "-m", commit_msg])
    run(["git", "push", "-u", "origin", branch_name])

    # Build PR body
    pr_body_lines = [
        "## Apply best hyperparameters from sweep",
        "",
        f"Automatically generated from W&B sweep [{sweep_id}]({sweep_url}).",
        "",
        f"**Best metric:** `{metric_name}` = **{best_metric:.4f}**",
        "",
        "### Parameter changes",
        "",
        "| Parameter | Old | New |",
        "|-----------|-----|-----|",
    ]
    for param, old, new in changed:
        pr_body_lines.append(f"| `{param}` | `{old}` | `{new}` |")

    pr_body_lines.append("")
    if args.pr_number:
        pr_body_lines.append(f"Sweep triggered from PR #{args.pr_number}.")
    pr_body_lines.append(f"\n[View sweep on W&B]({sweep_url})")
    pr_body = "\n".join(pr_body_lines)

    pr_title = f"Update hyperparameters from sweep {sweep_id}"

    result = run(
        [
            "gh", "pr", "create",
            "--title", pr_title,
            "--body", pr_body,
            "--base", args.base_branch,
            "--head", branch_name,
        ],
        check=False,
    )

    if result.returncode == 0:
        pr_url = result.stdout.strip()
        print(f"\nPR created: {pr_url}")
        with open(args.output_pr_url, "w") as f:
            f.write(pr_url)
    else:
        print(f"WARNING: Failed to create PR: {result.stderr}")
        # Maybe PR already exists
        if "already exists" in result.stderr:
            print("PR already exists, skipping.")


if __name__ == "__main__":
    main()
