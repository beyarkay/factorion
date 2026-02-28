#!/usr/bin/env python3
"""Destroy a RunPod pod, optionally recording cost data first.

Used as a cleanup step in CI - always runs, even if previous steps failed.
Queries pod cost before termination so the PR comment can include spend info.

Required env vars:
    RUNPOD_API_KEY  - RunPod API key

Usage:
    python scripts/ci/runpod_destroy.py --pod-info-file /tmp/pod_info.json
    python scripts/ci/runpod_destroy.py --pod-id <pod_id>
    python scripts/ci/runpod_destroy.py --pod-info-file /tmp/pod_info.json \
        --append-to-summary /tmp/summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import runpod


def format_uptime(seconds: float) -> str:
    """Convert seconds to a human-readable duration like '20m 55s' or '1h 5m 30s'."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def query_pod_cost(pod_id: str, created_at: float | None = None) -> dict:
    """Query RunPod for pod cost info. Returns dict with cost fields.

    The top-level ``uptimeSeconds`` field in the RunPod API is deprecated and
    returns 0.  We prefer ``runtime.uptimeInSeconds``; when neither is
    available we fall back to computing elapsed time from *created_at* (a
    Unix timestamp recorded when the pod was provisioned).
    """
    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    pod = runpod.get_pod(pod_id)

    cost_per_hr = pod["costPerHr"]
    gpu_name = pod.get("machine", {}).get("gpuDisplayName", pod.get("gpuDisplayName", "GPU"))

    # Try runtime.uptimeInSeconds first (current API), then the legacy
    # top-level uptimeSeconds, then compute from the creation timestamp.
    runtime = pod.get("runtime") or {}
    uptime_seconds = runtime.get("uptimeInSeconds") or pod.get("uptimeSeconds") or 0
    if not uptime_seconds and created_at is not None:
        uptime_seconds = max(0, time.time() - created_at)

    total_cost = cost_per_hr * (uptime_seconds / 3600)

    return {
        "pod_id": pod_id,
        "cost_per_hr": cost_per_hr,
        "uptime_seconds": uptime_seconds,
        "total_cost": round(total_cost, 2),
        "gpu_name": gpu_name,
        "uptime_human": format_uptime(uptime_seconds),
    }


def terminate_with_retry(pod_id: str, max_retries: int = 4) -> None:
    """Terminate a pod with exponential backoff retry."""
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    for attempt in range(max_retries):
        try:
            print(f"Terminating pod {pod_id} (attempt {attempt + 1}/{max_retries})...")
            runpod.terminate_pod(pod_id)
            print(f"Pod {pod_id} terminated successfully")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"ERROR: Could not terminate pod {pod_id} after {max_retries} attempts: {e}")
                sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Destroy a RunPod pod")
    parser.add_argument("--pod-info-file", default="/tmp/pod_info.json")
    parser.add_argument("--pod-id", default=None, help="Pod ID (overrides --pod-info-file)")
    parser.add_argument(
        "--cost-output-file",
        default="/tmp/pod_cost.json",
        help="Write cost data as JSON to this file",
    )
    parser.add_argument(
        "--append-to-summary",
        default=None,
        help="Append formatted cost line to this markdown file",
    )
    args = parser.parse_args()

    pod_id = args.pod_id
    created_at = None
    if not pod_id:
        try:
            with open(args.pod_info_file) as f:
                pod_info = json.load(f)
            pod_id = pod_info["pod_id"]
            created_at = pod_info.get("created_at")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not read pod info from {args.pod_info_file}: {e}")
            print("Nothing to clean up.")
            return

    # Query cost before termination (best-effort)
    try:
        cost = query_pod_cost(pod_id, created_at=created_at)
        print(f"Pod cost: ${cost['total_cost']:.2f} ({cost['uptime_human']} @ ${cost['cost_per_hr']}/hr)")

        with open(args.cost_output_file, "w") as f:
            json.dump(cost, f, indent=2)
        print(f"Cost data written to {args.cost_output_file}")

        if args.append_to_summary:
            cost_line = (
                f"\n**Cost:** ${cost['total_cost']:.2f} "
                f"({cost['uptime_human']} on {cost['gpu_name']} "
                f"@ ${cost['cost_per_hr']}/hr via RunPod)\n"
            )
            try:
                existing = ""
                if os.path.exists(args.append_to_summary):
                    with open(args.append_to_summary) as f:
                        existing = f.read()
                with open(args.append_to_summary, "w") as f:
                    f.write(existing + cost_line)
                print(f"Cost line appended to {args.append_to_summary}")
            except OSError as e:
                print(f"::warning::Could not append cost to {args.append_to_summary}: {e}")
    except Exception as e:
        print(f"::warning::Could not query pod cost: {e}")

    # Always terminate, even if cost query failed
    terminate_with_retry(pod_id)


if __name__ == "__main__":
    main()
