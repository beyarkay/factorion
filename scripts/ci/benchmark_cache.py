#!/usr/bin/env python3
"""Benchmark result cache backed by a GitHub Release.

Stores per-commit benchmark results as release assets on a special
``benchmark-cache`` release so that expensive GPU benchmarks can be
skipped when the same commit+parameters have already been evaluated.

Asset naming: ``v1-{sha12}-s{seeds}-t{timesteps}.json``

Usage:
    # Check whether cached results exist (exit 0 = hit, exit 1 = miss)
    python benchmark_cache.py check \\
        --sha abc123def456 --seeds 10 --timesteps 100000 \\
        --output /tmp/cached_results.json

    # Upload results to the cache
    python benchmark_cache.py save \\
        --sha abc123def456 --seeds 10 --timesteps 100000 \\
        --input /path/to/all_results.json

    # List cached assets
    python benchmark_cache.py list
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

RELEASE_TAG = "benchmark-cache"
CACHE_VERSION = "v1"


def asset_name(sha: str, seeds: int, timesteps: int) -> str:
    """Deterministic asset name for a given commit + benchmark config."""
    return f"{CACHE_VERSION}-{sha[:12]}-s{seeds}-t{timesteps}.json"


def _run_gh(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a ``gh`` CLI command, returning the CompletedProcess."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=check,
    )


def _ensure_release() -> None:
    """Create the benchmark-cache release if it doesn't already exist."""
    result = _run_gh(["release", "view", RELEASE_TAG], check=False)
    if result.returncode == 0:
        return
    print(f"Creating release '{RELEASE_TAG}'...")
    _run_gh([
        "release", "create", RELEASE_TAG,
        "--title", "Benchmark Result Cache",
        "--notes",
        "Auto-managed cache of GPU benchmark results.\n"
        "Assets are keyed by commit SHA and benchmark parameters.\n"
        "Do not delete this release — it is used by CI.",
        "--latest=false",
    ])
    print(f"Release '{RELEASE_TAG}' created.")


# ── Subcommands ──────────────────────────────────────────────────


def cmd_check(args: argparse.Namespace) -> None:
    """Check cache and download results if available."""
    name = asset_name(args.sha, args.seeds, args.timesteps)
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    result = _run_gh(
        ["release", "download", RELEASE_TAG,
         "--pattern", name,
         "--dir", out_dir],
        check=False,
    )

    downloaded = os.path.join(out_dir, name)
    if result.returncode == 0 and os.path.exists(downloaded):
        # Validate JSON before declaring a hit
        try:
            with open(downloaded) as f:
                data = json.load(f)
            if not isinstance(data, list) or len(data) == 0:
                raise ValueError("expected non-empty JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"miss (corrupt cache asset: {exc})")
            os.remove(downloaded)
            sys.exit(1)

        # Move to the requested output path
        if downloaded != os.path.abspath(args.output):
            shutil.move(downloaded, args.output)
        print(f"hit ({name} -> {args.output}, {len(data)} seeds)")
        sys.exit(0)
    else:
        print(f"miss ({name})")
        sys.exit(1)


def cmd_save(args: argparse.Namespace) -> None:
    """Upload results to the cache release."""
    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    # Validate input JSON
    with open(args.input) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        print("ERROR: input must be a non-empty JSON array")
        sys.exit(1)

    _ensure_release()

    name = asset_name(args.sha, args.seeds, args.timesteps)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, name)
        shutil.copy2(args.input, tmp_path)
        _run_gh([
            "release", "upload", RELEASE_TAG,
            tmp_path, "--clobber",
        ])
    print(f"saved ({name}, {len(data)} seeds)")


def cmd_list(args: argparse.Namespace) -> None:
    """List all cached benchmark assets."""
    result = _run_gh(
        ["release", "view", RELEASE_TAG, "--json", "assets"],
        check=False,
    )
    if result.returncode != 0:
        print("No benchmark-cache release found.")
        sys.exit(0)

    info = json.loads(result.stdout)
    assets = info.get("assets", [])
    if not assets:
        print("Cache is empty (no assets).")
        return

    print(f"Cached benchmark results ({len(assets)} assets):")
    for a in sorted(assets, key=lambda x: x.get("name", "")):
        size_kb = a.get("size", 0) / 1024
        print(f"  {a['name']}  ({size_kb:.1f} KB)")


# ── CLI ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU benchmark result cache (GitHub Release backend)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- check
    p_check = sub.add_parser("check", help="Check if cached results exist")
    p_check.add_argument("--sha", required=True, help="Git commit SHA")
    p_check.add_argument("--seeds", type=int, required=True)
    p_check.add_argument("--timesteps", type=int, required=True)
    p_check.add_argument(
        "--output", required=True,
        help="Path to write cached results if found",
    )

    # -- save
    p_save = sub.add_parser("save", help="Upload results to cache")
    p_save.add_argument("--sha", required=True, help="Git commit SHA")
    p_save.add_argument("--seeds", type=int, required=True)
    p_save.add_argument("--timesteps", type=int, required=True)
    p_save.add_argument(
        "--input", required=True,
        help="Path to all_results.json to upload",
    )

    # -- list
    sub.add_parser("list", help="List cached assets")

    args = parser.parse_args()

    commands = {
        "check": cmd_check,
        "save": cmd_save,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
