#!/usr/bin/env python3
"""Create a RunPod GPU pod for CI.

Provisions an A100 (or specified GPU) pod, waits for it to reach a running
state, and writes connection info to an output JSON file.

Required env vars:
    RUNPOD_API_KEY  - RunPod API key

Usage:
    python scripts/ci/runpod_create.py --name-prefix ci-smoke --output-file /tmp/pod_info.json
"""

import argparse
import json
import os
import sys
import time

import runpod

# Ordered fallback list: try A100 first, then fall back to other GPUs
GPU_FALLBACKS = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA RTX A6000",
    "NVIDIA H100 80GB HBM3",
]

DOCKER_IMAGE = "beyarkay/factorion-ci-gpu:latest"
CONTAINER_DISK_GB = 40
POD_START_TIMEOUT = 600  # seconds (large Docker image needs time for first pull)


def create_pod(gpu_type: str, name_prefix: str = "ci", timeout: int = POD_START_TIMEOUT) -> dict:
    """Create a RunPod pod and wait for it to be ready.

    Returns dict with pod_id, ssh_host, ssh_port, gpu_type, status.
    """
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    gpu_types_to_try = (
        [gpu_type] if gpu_type not in GPU_FALLBACKS
        else GPU_FALLBACKS[GPU_FALLBACKS.index(gpu_type):]
    )

    pod = None
    for gpu in gpu_types_to_try:
        print(f"Trying GPU: {gpu}", flush=True)
        try:
            pod = runpod.create_pod(
                name=f"{name_prefix}-{int(time.time())}",
                image_name=DOCKER_IMAGE,
                gpu_type_id=gpu,
                gpu_count=1,
                volume_in_gb=0,
                container_disk_in_gb=CONTAINER_DISK_GB,
                ports="22/tcp",
                support_public_ip=True,
                env={
                    "RUNPOD_API_KEY": os.environ["RUNPOD_API_KEY"],
                },
            )
            if pod and pod.get("id"):
                print(f"Pod created: {pod['id']} ({gpu})", flush=True)
                break
        except Exception as e:
            print(f"  Failed: {e}", flush=True)
            pod = None
            continue

    if not pod or not pod.get("id"):
        print("ERROR: Could not create pod with any available GPU type", flush=True)
        sys.exit(1)

    pod_id = pod["id"]

    # Wait for pod to reach running state
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = runpod.get_pod(pod_id)
        desired = status.get("desiredStatus", "")
        runtime = status.get("runtime")
        elapsed = int(time.time() - start_time)

        if desired == "RUNNING" and runtime:
            # Extract public SSH IP and port from runtime ports
            ssh_host = None
            ssh_port = 22
            ports = runtime.get("ports", []) or []
            for p in ports:
                if p.get("privatePort") == 22 and p.get("isIpPublic"):
                    ssh_host = p["ip"]
                    ssh_port = p["publicPort"]
                    break
            if not ssh_host:
                print(f"  Running but no public SSH port yet ({elapsed}s)", flush=True)
                time.sleep(10)
                continue
            print(f"Pod ready. SSH: root@{ssh_host} -p {ssh_port}", flush=True)
            return {
                "pod_id": pod_id,
                "ssh_host": ssh_host,
                "ssh_port": ssh_port,
                "gpu_type": status.get("machine", {}).get("gpuDisplayName", "unknown"),
                "status": "running",
            }

        print(f"  Waiting... ({elapsed}s, desired={desired})", flush=True)
        time.sleep(10)

    # Timeout - clean up the pod
    print(f"ERROR: Pod {pod_id} did not start within {timeout}s. Terminating.")
    try:
        runpod.terminate_pod(pod_id)
    except Exception as e:
        print(f"WARNING: Failed to terminate timed-out pod {pod_id}: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create a RunPod GPU pod for CI")
    parser.add_argument(
        "--gpu-type",
        default="NVIDIA A100 80GB PCIe",
        help="GPU type to request (falls back to cheaper GPUs if unavailable)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=POD_START_TIMEOUT,
        help="Timeout in seconds waiting for pod to start",
    )
    parser.add_argument(
        "--name-prefix",
        default="ci",
        help="Pod name prefix (e.g. ci-smoke, ci-bench, ci-sweep)",
    )
    parser.add_argument(
        "--output-file",
        default="/tmp/pod_info.json",
        help="Path to write pod info JSON",
    )
    args = parser.parse_args()

    pod_info = create_pod(gpu_type=args.gpu_type, name_prefix=args.name_prefix, timeout=args.timeout)

    with open(args.output_file, "w") as f:
        json.dump(pod_info, f, indent=2)
    print(f"Pod info written to {args.output_file}")


if __name__ == "__main__":
    main()
