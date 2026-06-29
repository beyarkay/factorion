#!/usr/bin/env python3
"""Append one benchmark result row to results.csv.

Reads a hyperfine --export-json file (timing stats) and the ppo.py summary JSON
(rollout/update breakdown), and appends a single row to RESULTS_CSV, writing the
header first if the file does not yet exist. Invoked by measure.sh; not meant to
be run by hand. All inputs come from environment variables so measure.sh does
not have to quote paths.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

FIELDS = [
    "timestamp_utc",
    "branch",
    "commit",
    "dirty",
    "mean_s",
    "stddev_s",
    "median_s",
    "min_s",
    "max_s",
    "runs",
    "rollout_total_s",
    "update_total_s",
    "rollout_steady_s",
    "update_steady_s",
    "total_timesteps",
    "note",
]


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    hf = _load(os.environ["HF_JSON"])["results"][0]
    summary = _load(os.environ["SUMMARY_JSON"])
    results_csv = os.environ["RESULTS_CSV"]

    row = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": os.environ.get("BRANCH", ""),
        "commit": os.environ.get("COMMIT", ""),
        "dirty": os.environ.get("DIRTY", ""),
        "mean_s": round(hf["mean"], 3),
        "stddev_s": round(hf.get("stddev", 0.0), 3),
        "median_s": round(hf["median"], 3),
        "min_s": round(hf["min"], 3),
        "max_s": round(hf["max"], 3),
        "runs": len(hf.get("times", [])),
        "rollout_total_s": summary.get("rollout_seconds_total", ""),
        "update_total_s": summary.get("update_seconds_total", ""),
        "rollout_steady_s": summary.get("rollout_seconds_steady_mean", ""),
        "update_steady_s": summary.get("update_seconds_steady_mean", ""),
        "total_timesteps": summary.get("total_timesteps", ""),
        "note": os.environ.get("NOTE", ""),
    }

    exists = os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"Logged: branch={row['branch']} mean={row['mean_s']}s "
        f"±{row['stddev_s']} (min {row['min_s']}, max {row['max_s']}, "
        f"{row['runs']} runs) -> {results_csv}"
    )


if __name__ == "__main__":
    main()
