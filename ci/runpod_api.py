"""Thin wrappers around the RunPod SDK: create / list / terminate / cost.

Ported from the old scripts/ci/runpod_create.py + runpod_destroy.py, minus the
SSH plumbing — pods are now fire-and-forget (the job command ships in the pod's
docker args), so nothing here ever needs a shell on the pod.

Required env var: RUNPOD_API_KEY.
"""

from __future__ import annotations

import os
import time

import runpod

from ci.config import (
    ALLOWED_CUDA_VERSIONS,
    CONTAINER_DISK_GB,
    DOCKER_IMAGE,
    GPU_FALLBACKS,
    POD_PREFIX,
)

POD_START_TIMEOUT = 900  # seconds (large Docker image needs time for first pull)

# GPU availability is transient — RunPod often replies "no longer any instances
# available" when a machine was just snapped up. Retry the same GPU a few times
# before falling back to the next one in the lineup.
MAX_RETRIES_PER_GPU = 5
RETRY_DELAY_SECONDS = 5.0


def _init() -> None:
    runpod.api_key = os.environ["RUNPOD_API_KEY"]


def create_pod(name: str, gpu_type: str, docker_args: str, env: dict) -> dict:
    """Create a pod, walking the GPU fallback chain. Returns the pod dict.

    Raises RuntimeError when no GPU in the lineup could be provisioned.
    """
    _init()
    gpus = (
        [gpu_type]
        if gpu_type not in GPU_FALLBACKS
        else GPU_FALLBACKS[GPU_FALLBACKS.index(gpu_type) :]
    )

    for gpu in gpus:
        for attempt in range(1, MAX_RETRIES_PER_GPU + 1):
            print(f"Trying GPU: {gpu} (attempt {attempt}/{MAX_RETRIES_PER_GPU})", flush=True)
            try:
                pod = runpod.create_pod(
                    name=name,
                    image_name=DOCKER_IMAGE,
                    gpu_type_id=gpu,
                    gpu_count=1,
                    volume_in_gb=0,
                    container_disk_in_gb=CONTAINER_DISK_GB,
                    ports="22/tcp",
                    support_public_ip=True,
                    allowed_cuda_versions=ALLOWED_CUDA_VERSIONS,
                    docker_args=docker_args,
                    env=env,
                )
            except Exception as e:
                print(f"  Failed: {e}", flush=True)
                pod = None
            if pod and pod.get("id"):
                print(f"Pod created: {pod['id']} ({gpu})", flush=True)
                return pod
            if attempt < MAX_RETRIES_PER_GPU:
                time.sleep(RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"Could not create pod with any available GPU type "
        f"({MAX_RETRIES_PER_GPU} attempts each): {gpus}"
    )


def wait_until_running(pod_id: str, timeout: int = POD_START_TIMEOUT) -> dict:
    """Poll until the pod is RUNNING with a live runtime; return its status.

    Raises TimeoutError if the pod doesn't come up — the caller decides whether
    to terminate (a never-booted pod would otherwise sit idle until a watchdog
    reaps it).
    """
    _init()
    start = time.time()
    while time.time() - start < timeout:
        status = runpod.get_pod(pod_id)
        if status.get("desiredStatus") == "RUNNING" and status.get("runtime"):
            return status
        elapsed = int(time.time() - start)
        print(
            f"  Waiting for pod... ({elapsed}s, desired={status.get('desiredStatus', '?')})",
            flush=True,
        )
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} not running within {timeout}s")


def list_pods() -> list[dict]:
    _init()
    # The SDK annotates get_pods() -> dict but returns a list at runtime.
    return list(runpod.get_pods())


def list_ci_pods() -> list[dict]:
    return [p for p in list_pods() if (p.get("name") or "").startswith(POD_PREFIX)]


def terminate_with_retry(pod_id: str, max_retries: int = 4) -> None:
    """Terminate a pod with exponential backoff retry."""
    _init()
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
                raise RuntimeError(
                    f"Could not terminate pod {pod_id} after {max_retries} attempts"
                ) from e


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
    _init()
    pod = runpod.get_pod(pod_id)

    cost_per_hr = pod["costPerHr"]
    gpu_name = pod.get("machine", {}).get("gpuDisplayName", pod.get("gpuDisplayName", "GPU"))

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
