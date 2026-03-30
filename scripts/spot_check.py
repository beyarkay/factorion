"""Spot-check script for verifying lesson generation.

Generates complete factories for each scenario type, verifies throughput,
renders to HTML, and prints summary statistics.

Usage:
    uv run python scripts/spot_check.py --scenario all --count 10 --size 8
    uv run python scripts/spot_check.py --scenario splitter_split --count 5 --size 12 --output-dir /tmp/factories
"""

import os
import sys
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import tyro

# Ensure factorion is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

import factorion  # noqa: E402
import factorion_rs  # noqa: E402

_, _objs = factorion.datatypes.run()
_, _fns = factorion.functions.run()

LessonKind = _objs["LessonKind"]
generate_lesson = _fns["generate_lesson"]
world2html = _fns["world2html"]
str2ent = _fns["str2ent"]


class Scenario(Enum):
    belt = "belt"
    inserter = "inserter"
    splitter_split = "splitter_split"
    splitter_merge = "splitter_merge"
    all = "all"


SCENARIO_TO_KIND = {
    Scenario.belt: LessonKind.MOVE_ONE_ITEM,
    Scenario.inserter: LessonKind.INSERTER_TRANSFER,
    Scenario.splitter_split: LessonKind.SPLITTER_SPLIT,
    Scenario.splitter_merge: LessonKind.SPLITTER_MERGE,
}


@dataclasses.dataclass
class Args:
    scenario: Scenario = Scenario.all
    """Which scenario to generate."""
    count: int = 10
    """Number of factories to generate per scenario."""
    size: int = 8
    """Grid size (NxN)."""
    output_dir: str = "spot_check_output"
    """Directory to write HTML files."""
    seed_start: int = 0
    """Starting seed."""


def run_scenario(kind: LessonKind, name: str, args: Args) -> dict:
    """Generate factories for a scenario, verify, and render to a single HTML page."""
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generated = 0
    passed = 0
    failed = 0
    throughputs = []
    html_fragments = []

    for i in range(args.count):
        seed = args.seed_start + i
        try:
            world_cwh, min_ent = generate_lesson(
                size=args.size,
                kind=kind,
                num_missing_entities=0,
                seed=seed,
            )
            world_whc = world_cwh.permute(1, 2, 0)
            tp, unreachable = factorion_rs.simulate_throughput(
                world_whc.numpy().astype(np.int64)
            )
            generated += 1

            status = "PASS" if tp > 0 else "FAIL"
            if tp > 0:
                passed += 1
                throughputs.append(tp)
            else:
                failed += 1
                print(f"  [{name}] seed={seed}: throughput=0 (FAIL)")

            html_obj = world2html(world_WHC=world_whc)
            color = "#d4edda" if tp > 0 else "#f8d7da"
            html_fragments.append(
                f'<div style="border:1px solid #ccc; padding:10px; margin:10px 0; '
                f'background:{color}; border-radius:6px;">'
                f'<h3 style="margin:0 0 6px 0;">[{status}] seed={seed} | '
                f'size={args.size} | kind={name} | '
                f'tp={tp:.4f} | unreachable={unreachable} | min_ent={min_ent}</h3>'
                f'<code style="font-size:12px; color:#555;">generate_lesson('
                f'size={args.size}, kind=LessonKind.{kind.name}, '
                f'num_missing_entities=0, seed={seed})</code>'
                f'<div style="margin-top:8px;">{html_obj.text}</div>'
                f'</div>'
            )

        except Exception as e:
            failed += 1
            print(f"  [{name}] seed={seed}: EXCEPTION: {e}")
            html_fragments.append(
                f'<div style="border:1px solid #ccc; padding:10px; margin:10px 0; '
                f'background:#f8d7da; border-radius:6px;">'
                f'<h3 style="margin:0;">[ERROR] seed={seed} | size={args.size} | kind={name}</h3>'
                f'<code style="font-size:12px; color:#555;">generate_lesson('
                f'size={args.size}, kind=LessonKind.{kind.name}, '
                f'num_missing_entities=0, seed={seed})</code>'
                f'<pre style="color:red; margin-top:8px;">{e}</pre>'
                f'</div>'
            )

    # Write single HTML page per scenario
    tps = throughputs
    summary = (
        f"{passed}/{generated} passed"
        + (f", tp range [{min(tps):.2f}, {max(tps):.2f}], avg {sum(tps)/len(tps):.2f}" if tps else "")
    )
    html_page = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{name} — spot check</title>
<style>body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}</style>
</head><body>
<h1>{name}</h1>
<p><strong>{summary}</strong> | size={args.size} | seeds {args.seed_start}..{args.seed_start + args.count - 1}</p>
<hr>
{''.join(html_fragments)}
</body></html>"""
    html_path = output_path / f"{name}.html"
    html_path.write_text(html_page)
    print(f"  Wrote {html_path}")

    return {
        "name": name,
        "generated": generated,
        "passed": passed,
        "failed": failed,
        "throughputs": throughputs,
    }


def main():
    args = tyro.cli(Args)

    if args.scenario == Scenario.all:
        scenarios = [
            (Scenario.belt, "belt"),
            (Scenario.inserter, "inserter"),
            (Scenario.splitter_split, "splitter_split"),
            (Scenario.splitter_merge, "splitter_merge"),
        ]
    else:
        scenarios = [(args.scenario, args.scenario.value)]

    print(f"Spot-check: {len(scenarios)} scenario(s), {args.count} each, size={args.size}")
    print(f"Output: {args.output_dir}/")
    print()

    results = []
    for scenario, name in scenarios:
        kind = SCENARIO_TO_KIND[scenario]
        print(f"--- {name} ---")
        result = run_scenario(kind, name, args)
        results.append(result)

    # Print summary
    print()
    print("=" * 60)
    print(f"{'Scenario':<20} {'Gen':>5} {'Pass':>5} {'Fail':>5} {'Min TP':>8} {'Max TP':>8} {'Avg TP':>8}")
    print("-" * 60)
    for r in results:
        tps = r["throughputs"]
        min_tp = f"{min(tps):.2f}" if tps else "N/A"
        max_tp = f"{max(tps):.2f}" if tps else "N/A"
        avg_tp = f"{sum(tps)/len(tps):.2f}" if tps else "N/A"
        print(f"{r['name']:<20} {r['generated']:>5} {r['passed']:>5} {r['failed']:>5} {min_tp:>8} {max_tp:>8} {avg_tp:>8}")
    print("=" * 60)

    total_failed = sum(r["failed"] for r in results)
    if total_failed > 0:
        print(f"\nWARNING: {total_failed} total failures!")
        sys.exit(1)
    else:
        print(f"\nAll factories passed!")


if __name__ == "__main__":
    main()
