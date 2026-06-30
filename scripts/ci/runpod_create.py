#!/usr/bin/env python3
"""Create a RunPod GPU pod for CI.

Provisions an RTX 2000 Ada (or specified GPU) pod on a host new enough for our
torch build (see ALLOWED_CUDA_VERSIONS), waits for it to reach a running state,
and writes connection info to an output JSON file.

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

# Preferred first, then availability fallbacks (each pricier/bigger). The default
# is the RTX 2000 Ada — the GPU the speed benchmarks were tuned on; these models
# are tiny (122k params) and the workload is single-core-CPU-bound, so a bigger
# GPU is overkill (see tests/benchmarks/EXPERIMENT_LOG.md).
GPU_FALLBACKS = [
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]

# Only schedule on hosts whose driver supports a CUDA version new enough for our
# torch build. uv.lock pins torch 2.12.x+cu130, which needs a CUDA 13.0 runtime
# (host driver >= 580); an older host (e.g. CUDA 12.x) leaves torch.cuda
# unavailable. Keep in sync with the torch cuXXX build in pyproject/uv.lock.
ALLOWED_CUDA_VERSIONS = ["13.0"]

DOCKER_IMAGE = "beyarkay/factorion-ci-gpu:latest"
CONTAINER_DISK_GB = 40
POD_START_TIMEOUT = 600  # seconds (large Docker image needs time for first pull)

# GPU availability is transient — RunPod often replies "no longer any instances
# available, please refresh and try again" when a machine was just snapped up.
# Retrying the same GPU a few seconds later usually succeeds, so we exhaust
# ``max_retries`` attempts on each GPU before falling back to the next one.
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 5.0  # seconds between attempts on the same GPU


def create_pod(
    gpu_type: str,
    name_prefix: str = "ci",
    timeout: int = POD_START_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> dict:
    """Create a RunPod pod and wait for it to be ready.

    Tries each GPU in the fallback chain, retrying the same GPU up to
    ``max_retries`` times (waiting ``retry_delay`` seconds between attempts)
    before falling back to the next GPU in the lineup.

    Returns dict with pod_id, ssh_host, ssh_port, gpu_type, status.
    """
    runpod.api_key = os.environ["RUNPOD_API_KEY"]

    gpu_types_to_try = (
        [gpu_type] if gpu_type not in GPU_FALLBACKS
        else GPU_FALLBACKS[GPU_FALLBACKS.index(gpu_type):]
    )

    pod = None
    for gpu in gpu_types_to_try:
        for attempt in range(1, max_retries + 1):
            print(f"Trying GPU: {gpu} (attempt {attempt}/{max_retries})", flush=True)
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
                    allowed_cuda_versions=ALLOWED_CUDA_VERSIONS,
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

            if attempt < max_retries:
                print(f"  Retrying {gpu} in {retry_delay:.0f}s...", flush=True)
                time.sleep(retry_delay)

        if pod and pod.get("id"):
            break

    if not pod or not pod.get("id"):
        print(
            f"ERROR: Could not create pod with any available GPU type "
            f"({max_retries} attempts each)",
            flush=True,
        )
        sys.exit(1)

    pod_id = pod["id"]

    start_time = time.time()
    while time.time() - start_time < timeout:
        status = runpod.get_pod(pod_id)
        desired = status.get("desiredStatus", "")
        runtime = status.get("runtime")
        elapsed = int(time.time() - start_time)

        if desired == "RUNNING" and runtime:
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
                "created_at": time.time(),
            }

        print(f"  Waiting... ({elapsed}s, desired={desired})", flush=True)
        time.sleep(10)

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
        default=GPU_FALLBACKS[0],
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Attempts per GPU before falling back to the next GPU in the lineup",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Seconds to wait between attempts on the same GPU",
    )
    args = parser.parse_args()

    pod_info = create_pod(
        gpu_type=args.gpu_type,
        name_prefix=args.name_prefix,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
    )

    with open(args.output_file, "w") as f:
        json.dump(pod_info, f, indent=2)
    print(f"Pod info written to {args.output_file}")


if __name__ == "__main__":
    main()
