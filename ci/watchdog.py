"""Reap leaked CI pods. Run 6-hourly by .github/workflows/pod-watchdog.yml.

Third safety layer (after the pod's own EXIT trap and in-pod deadline timer):
works even for pods that never booted far enough to start the first two. Every
CI pod encodes its creation time and kill-by deadline in its name (see
ci/config.pod_name), so the decision needs nothing but `runpod.get_pods()`.

Pods not named `fci-*` are never touched — manually created pods are safe.

Usage:
    python -m ci.watchdog [--dry-run]

Required env var: RUNPOD_API_KEY. Only needs `pip install runpod`.
"""

from __future__ import annotations

import argparse
import time

from ci.config import MAX_POD_AGE_SECONDS, POD_PREFIX, parse_pod_name


def decide_terminations(
    pods: list[dict], now: float, max_age_seconds: int = MAX_POD_AGE_SECONDS
) -> list[tuple[dict, str]]:
    """Pure decision core: which pods to kill and why. Testable offline."""
    to_kill = []
    for pod in pods:
        name = pod.get("name") or ""
        if not name.startswith(POD_PREFIX):
            continue
        meta = parse_pod_name(name)
        if meta is None:
            # A renamed/foreign fci-* pod: fall back to observed uptime.
            uptime = (pod.get("runtime") or {}).get("uptimeInSeconds") or 0
            if uptime > max_age_seconds:
                to_kill.append(
                    (pod, f"unparseable fci name, uptime {int(uptime)}s > {max_age_seconds}s")
                )
        elif now > meta.deadline:
            to_kill.append((pod, f"past its deadline (epoch {meta.deadline})"))
        elif now > meta.created + max_age_seconds:
            to_kill.append((pod, f"older than the {max_age_seconds}s absolute cap"))
    return to_kill


def run(dry_run: bool = False) -> int:
    """Sweep once; returns the number of pods (to be) terminated."""
    import runpod  # lazy: the decision core stays importable without it

    import os

    runpod.api_key = os.environ["RUNPOD_API_KEY"]
    # The SDK annotates get_pods() -> dict but returns a list at runtime.
    pods = list(runpod.get_pods())
    print(f"{len(pods)} pod(s) total, {sum(1 for p in pods if (p.get('name') or '').startswith(POD_PREFIX))} CI pod(s)")

    doomed = decide_terminations(pods, now=time.time())
    for pod, reason in doomed:
        label = f"{pod.get('name')} ({pod.get('id')})"
        if dry_run:
            print(f"[dry-run] would terminate {label}: {reason}")
        else:
            print(f"Terminating {label}: {reason}")
            runpod.terminate_pod(pod["id"])
    if not doomed:
        print("Nothing to reap.")
    return len(doomed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report, don't terminate")
    args = parser.parse_args()
    run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
