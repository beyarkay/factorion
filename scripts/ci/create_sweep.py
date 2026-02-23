#!/usr/bin/env python3
"""Create a W&B sweep from a YAML config file.

Outputs sweep info (ID, path, URL) for use in CI pipelines.
Writes a JSON file with sweep details and prints key=value pairs
for GitHub Actions output parsing.

Required env vars:
    WANDB_API_KEY  - W&B API key

Usage:
    python scripts/ci/create_sweep.py \
        --config sweep.yaml \
        --project factorion \
        --output-file /tmp/sweep_info.json
"""

import argparse
import json
import sys

import wandb
import yaml


def main():
    parser = argparse.ArgumentParser(description="Create a W&B sweep")
    parser.add_argument(
        "--config", default="sweep.yaml", help="Path to sweep YAML config"
    )
    parser.add_argument(
        "--project", default="factorion", help="W&B project name"
    )
    parser.add_argument(
        "--entity", default=None, help="W&B entity (team or user)"
    )
    parser.add_argument(
        "--output-file",
        default="/tmp/sweep_info.json",
        help="Path to write sweep info JSON",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        sweep_config = yaml.safe_load(f)

    print(f"Creating sweep from {args.config}...", flush=True)
    print(f"  Project: {args.project}", flush=True)
    print(f"  Metric: {sweep_config.get('metric', {})}", flush=True)
    print(
        f"  Parameters: {list(sweep_config.get('parameters', {}).keys())}",
        flush=True,
    )

    try:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.project,
            entity=args.entity,
        )
    except Exception as e:
        print(f"ERROR: Failed to create sweep: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve entity (may have been auto-detected by wandb)
    api = wandb.Api()
    entity = args.entity or api.default_entity
    sweep_path = f"{entity}/{args.project}/{sweep_id}"
    sweep_url = f"https://wandb.ai/{entity}/{args.project}/sweeps/{sweep_id}"

    info = {
        "sweep_id": sweep_id,
        "sweep_path": sweep_path,
        "sweep_url": sweep_url,
        "entity": entity,
        "project": args.project,
    }

    # Print key=value pairs for CI output parsing
    for key, value in info.items():
        print(f"{key}={value}")

    with open(args.output_file, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSweep info written to {args.output_file}")


if __name__ == "__main__":
    main()
