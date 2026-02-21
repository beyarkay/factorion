#!/usr/bin/env python3
"""Destroy a RunPod pod.

Used as a cleanup step in CI - always runs, even if previous steps failed.

Required env vars:
    RUNPOD_API_KEY  - RunPod API key

Usage:
    python scripts/ci/runpod_destroy.py --pod-info-file /tmp/pod_info.json
    python scripts/ci/runpod_destroy.py --pod-id <pod_id>
"""

import argparse
import json
import os
import sys
import time

import runpod


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
    args = parser.parse_args()

    pod_id = args.pod_id
    if not pod_id:
        try:
            with open(args.pod_info_file) as f:
                pod_info = json.load(f)
            pod_id = pod_info["pod_id"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not read pod info from {args.pod_info_file}: {e}")
            print("Nothing to clean up.")
            return

    terminate_with_retry(pod_id)


if __name__ == "__main__":
    main()
