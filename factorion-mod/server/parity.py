"""Engine ↔ Factorio parity harness (issue #261).

Builds known-good factories with `factorion.build_factory`, asks the
Factorion engine what each sink should receive
(`factorion_rs.py_sink_deliveries`), then replays the same factory inside
a real Factorio instance via the mod's `parity_start` RCON interface and
compares the measured per-sink items/s against the engine's prediction.

Run from the repo root with a Factorio instance hosting a save that has
the factorion mod enabled and RCON reachable (see factorion-mod/README.md):

    uv run python factorion-mod/server/parity.py \
        --rcon-port 64502 --rcon-password <pw> \
        --lessons all --seeds 5 --size 11

`--dry-run` prints the specs and engine expectations without needing
Factorio at all — useful for checking the tensor→spec conversion.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make the repo root importable when this script is run via uv from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    LessonKind,
    Misc,
    build_factory,
    entities,
    items,
    render_factory,
    str2ent,
)

from blueprint import _DIR_MODEL_TO_BP, _hyphenate  # noqa: E402
from rcon import RconClient  # noqa: E402


# --------------------------------------------------------------------------- #
# Factory tensor → parity spec.
# --------------------------------------------------------------------------- #

# Timing / convergence knobs shared by single and batch specs. Kept in one
# place so world_to_parity_spec and build_batch_spec can't drift.
_DEFAULT_TIMING = dict(
    game_speed=32.0,
    sample_every=15,
    warmup_max=1800,
    measure_max=36000,
    warmup_min=300,
    measure_min=600,
    measure_min_items=25,
    check_every=300,
    converge_rel=0.02,
    converge_hits=3,
    converge_floor=0.02,
)

# Substation-grid pitch: a substation supplies an 18x18 area and its wire
# reaches 18 tiles, so an 18-tile grid of substations both powers and
# connects everything. Each factory sits centred in one 18x18 cell, clear
# of the substations at the cell corners.
_PITCH = 18


def _convert_world(world_CWH) -> tuple[list[dict], list[dict], list[dict], int]:
    """Tensor → (entities, sources, sinks, grid_size) in factory-local tiles.

    Mirrors blueprint.py's conventions: entity positions are Factorio
    centers (1x1 at tile (3,5) → 3.5,5.5; E/W-facing multi-tile entities
    swap their width/height), blueprint direction is the 16-step enum, and
    inserters get the model→blueprint +8 pickup/drop flip. Source and sink
    tiles are emitted as separate lists (the mod places scripted belts on
    them), with their tile coordinates and the flowing item.
    """
    world = np.asarray(world_CWH)
    assert world.ndim == 3, f"expected (C, W, H), got {world.shape}"
    _, W, H = world.shape

    ent_ch = world[Channel.ENTITIES.value]
    dir_ch = world[Channel.DIRECTION.value]
    item_ch = world[Channel.ITEMS.value]
    misc_ch = world[Channel.MISC.value]

    src_id = str2ent("source").value
    snk_id = str2ent("sink").value

    spec_entities: list[dict] = []
    sources: list[dict] = []
    sinks: list[dict] = []
    seen = np.zeros((W, H), dtype=bool)

    for x in range(W):
        for y in range(H):
            if seen[x, y]:
                continue
            ent_id = int(ent_ch[x, y])
            if ent_id == 0:
                continue
            ent_meta = entities.get(ent_id)
            if ent_meta is None or not ent_meta.is_placeable:
                continue

            dir_model = int(dir_ch[x, y])
            dir_bp = _DIR_MODEL_TO_BP.get(dir_model) or 0
            item_id = int(item_ch[x, y])
            item_meta = items.get(item_id)
            item_name = (
                _hyphenate(item_meta.name)
                if item_meta is not None and item_meta.name != "empty"
                else None
            )

            if ent_id in (src_id, snk_id):
                entry = {
                    "x": x,
                    "y": y,
                    "direction": dir_bp,
                    "item": item_name,
                }
                (sources if ent_id == src_id else sinks).append(entry)
                seen[x, y] = True
                continue

            name = _hyphenate(ent_meta.name)
            direction = dir_bp
            # Blueprint/create_entity direction for inserters points at the
            # drop tile; the model's Direction is the pickup→drop flow. Same
            # flip blueprint.py applies.
            if "inserter" in name:
                direction = (direction + 8) % 16

            entry = {
                "name": name,
                "tile_x": x,
                "tile_y": y,
                "direction": direction,
            }

            # Center of an N×M entity is the geometric midpoint of its tile
            # span; E/W-facing entities swap their footprint dims.
            w, h = ent_meta.width, ent_meta.height
            if dir_model in (2, 4):
                w, h = h, w
            entry["x"] = x + w / 2.0
            entry["y"] = y + h / 2.0

            if name == "assembling-machine-1" and item_name is not None:
                entry["recipe"] = item_name
            if name == "underground-belt":
                misc = int(misc_ch[x, y])
                if misc == Misc.UNDERGROUND_DOWN.value:
                    entry["type"] = "input"
                elif misc == Misc.UNDERGROUND_UP.value:
                    entry["type"] = "output"
            spec_entities.append(entry)

            for dx in range(w):
                for dy in range(h):
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < W and 0 <= ty < H:
                        seen[tx, ty] = True

    return spec_entities, sources, sinks, max(W, H)


def world_to_parity_spec(world_CWH, *, run_id: str, **timing) -> dict:
    """Single-factory spec (entities/sources/sinks at top level). The mod
    treats this as a one-factory batch. `timing` overrides `_DEFAULT_TIMING`.
    """
    ents, sources, sinks, grid_size = _convert_world(world_CWH)
    spec = {**_DEFAULT_TIMING, **timing}
    spec.update({
        "run_id": run_id,
        "grid_size": grid_size,
        "entities": ents,
        "sources": sources,
        "sinks": sinks,
    })
    return spec


def build_batch_spec(factory_worlds, *, run_id: str = "batch", **timing) -> dict:
    """Assemble many factories into one spec, each tiled in its own cell of
    an 18-tile substation grid.

    `factory_worlds` is a list of (run_id, world_CWH). Factories are laid
    out in a near-square col×row grid; factory k sits centred in cell
    (k%cols, k//cols). Substations are placed on every cell corner (an
    (cols+1)×(rows+1) grid), which both powers and wires the whole field.
    Returns a spec whose `factories` list the mod builds and measures in
    parallel.
    """
    n = len(factory_worlds)
    cols = math.ceil(math.sqrt(n)) if n else 1
    rows = math.ceil(n / cols) if n else 1

    factories = []
    for idx, (fid, world) in enumerate(factory_worlds):
        ents, sources, sinks, grid_size = _convert_world(world)
        margin = max((_PITCH - grid_size) // 2, 0)
        col, row = idx % cols, idx // cols
        factories.append({
            "run_id": fid,
            "grid_size": grid_size,
            "offset_x": col * _PITCH + margin,
            "offset_y": row * _PITCH + margin,
            "entities": ents,
            "sources": sources,
            "sinks": sinks,
        })

    substations = [
        [i * _PITCH, j * _PITCH]
        for i in range(cols + 1)
        for j in range(rows + 1)
    ]

    spec = {**_DEFAULT_TIMING, **timing}
    spec.update({
        "run_id": run_id,
        "extent_x": cols * _PITCH + _PITCH,
        "extent_y": rows * _PITCH + _PITCH,
        "substations": substations,
        "factories": factories,
    })
    return spec


def expected_sink_rates(world_CWH) -> dict[tuple[int, int], tuple[str | None, float]]:
    """Engine-predicted {(x, y): (factorio_item_name, items_per_sec)} per sink."""
    world = np.asarray(world_CWH)
    world_WHC = np.ascontiguousarray(world.transpose(1, 2, 0)).astype(np.int64)
    out = {}
    for x, y, item_name, achieved in factorion_rs.py_sink_deliveries(world_WHC):
        out[(x, y)] = (
            _hyphenate(item_name) if item_name is not None else None,
            achieved,
        )
    return out


# --------------------------------------------------------------------------- #
# Driving a run over RCON.
# --------------------------------------------------------------------------- #

def _remote_call(rcon: RconClient, method: str, *args: str) -> str:
    quoted = ",".join("'" + a + "'" for a in args)
    sep = "," if quoted else ""
    cmd = (
        "/silent-command rcon.print(remote.call('factorion','"
        + method + "'" + sep + quoted + "))"
    )
    return rcon.exec(cmd).strip()


def run_parity(
    rcon: RconClient,
    spec: dict,
    *,
    poll_interval: float = 0.25,
    timeout: float = 600.0,
) -> dict:
    """Start a parity run and poll until it finishes. Returns the result dict."""
    payload = json.dumps(spec, separators=(",", ":"))
    # The payload rides inside a single-quoted Lua string; our item /
    # prototype names can't contain quotes or backslashes, but guard anyway.
    assert "'" not in payload and "\\" not in payload, "spec not RCON-safe"

    started = json.loads(_remote_call(rcon, "parity_start", payload))
    if started.get("status") != "running":
        raise RuntimeError(f"parity_start failed: {started}")

    t0 = time.time()
    last_progress = None
    while True:
        time.sleep(poll_interval)
        raw = _remote_call(rcon, "parity_poll")
        status = json.loads(raw)
        if status.get("status") == "done":
            return status
        if status.get("status") == "error":
            raise RuntimeError(f"parity run failed: {status}")
        if status.get("status") == "running":
            # Mirror the in-game announcements: one line per phase change
            # or 25% step, so long runs show a heartbeat on this side too.
            done = status.get("ticks_done", 0)
            total = max(status.get("total_ticks", 1), 1)
            progress = (status.get("phase"), int(4 * done / total))
            if progress != last_progress:
                last_progress = progress
                print(f"  ... {status.get('phase')} "
                      f"{done}/{total} ticks ({done / total:.0%})")
        if time.time() - t0 > timeout:
            _remote_call(rcon, "parity_abort")
            raise TimeoutError(
                f"parity run timed out after {timeout:.0f}s: {status}"
            )


# --------------------------------------------------------------------------- #
# Comparison + report.
# --------------------------------------------------------------------------- #

@dataclass
class SinkComparison:
    x: int
    y: int
    item: str | None
    engine_rate: float
    factorio_rate: float
    passed: bool
    note: str = ""


@dataclass
class ParityReport:
    lesson: str
    seed: int
    sinks: list[SinkComparison] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.sinks)


def compare_sinks(
    expected: dict[tuple[int, int], tuple[str | None, float]],
    measured: list[dict],
    *,
    rel_tol: float,
    abs_tol: float,
) -> list[SinkComparison]:
    out: list[SinkComparison] = []
    measured_by_pos = {(m["x"], m["y"]): m for m in measured}
    for (x, y), (item, engine_rate) in sorted(expected.items()):
        m = measured_by_pos.pop((x, y), None)
        if m is None:
            out.append(SinkComparison(
                x, y, item, engine_rate, 0.0, False,
                note="sink missing from Factorio result",
            ))
            continue
        factorio_rate = float(m["rate"])
        err = abs(factorio_rate - engine_rate)
        tol = max(abs_tol, rel_tol * max(engine_rate, factorio_rate))
        passed = np.isfinite(engine_rate) and err <= tol
        note = ""
        if not np.isfinite(engine_rate):
            note = "engine rate non-finite"
        elif m.get("all_items") and set(m["all_items"]) - {item}:
            extras = {k: v for k, v in m["all_items"].items() if k != item}
            note = f"sink also received {extras}"
        out.append(SinkComparison(
            x, y, item, engine_rate, factorio_rate, passed, note,
        ))
    for (x, y), m in sorted(measured_by_pos.items()):
        out.append(SinkComparison(
            x, y, m.get("item"), 0.0, float(m["rate"]), False,
            note="sink not predicted by engine",
        ))
    return out


def format_report(report: ParityReport, result: dict | None, world=None) -> str:
    lines = [f"=== {report.lesson} seed={report.seed}: "
             f"{'PASS' if report.passed else 'FAIL'} ==="]
    if report.error:
        lines.append(f"  error: {report.error}")
        return "\n".join(lines)
    if result is not None:
        conv = "converged" if result.get("converged") else "HIT TICK CAP"
        lines[0] += (f"  [{conv}, {result.get('warmup_ticks', 0)} warmup"
                     f" + {result.get('measure_ticks', 0)} measure ticks]")
    for s in report.sinks:
        rel = (
            abs(s.factorio_rate - s.engine_rate)
            / max(abs(s.engine_rate), 1e-9)
        )
        lines.append(
            f"  sink ({s.x},{s.y}) {s.item}: engine {s.engine_rate:.3f}/s, "
            f"factorio {s.factorio_rate:.3f}/s (err {rel:.1%}) "
            f"{'ok' if s.passed else 'MISMATCH'}"
            + (f" [{s.note}]" if s.note else "")
        )
    if not report.passed and world is not None:
        lines.append("  factory:")
        for row in render_factory(world).splitlines():
            lines.append("    " + row)
    if not report.passed and result is not None:
        lines.append("  factorio-side diagnostics:")
        for src in result.get("sources", []):
            lines.append(
                f"    source ({src['x']},{src['y']}) {src['item']}: "
                f"fed {src['rate']:.3f}/s"
            )
        for e in result.get("entities", []):
            desc = f"    {e['name']} ({e['x']},{e['y']})"
            if e["kind"] == "belt":
                lanes = e.get("avg_line_items") or {}
                if isinstance(lanes, dict):
                    lanes = [lanes[k] for k in sorted(lanes)]
                desc += " lanes=" + ",".join(f"{v:.2f}" for v in lanes)
            else:
                statuses = e.get("status_counts") or {}
                if statuses:
                    top = sorted(statuses.items(), key=lambda kv: -kv[1])
                    desc += " status=" + ",".join(
                        f"{k}:{v}" for k, v in top[:3]
                    )
                if "products_finished" in e:
                    desc += f" crafted={e['products_finished']}"
                if "held_frac" in e:
                    desc += f" held={e['held_frac']:.0%}"
            lines.append(desc)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def _parse_lessons(arg: str) -> list[LessonKind]:
    if arg == "all":
        return list(LessonKind)
    out = []
    for name in arg.split(","):
        name = name.strip().upper()
        try:
            out.append(LessonKind[name])
        except KeyError:
            valid = ", ".join(k.name for k in LessonKind)
            raise SystemExit(f"unknown lesson {name!r}; valid: {valid}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=27015)
    ap.add_argument("--rcon-password", default="factorion")
    ap.add_argument("--lessons", default="all",
                    help="'all' or comma-separated LessonKind names.")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Seeds 0..N-1 per lesson.")
    ap.add_argument("--size", type=int, default=11)
    ap.add_argument("--batch-size", type=int, default=40,
                    help="Factories built + measured together on one surface "
                         "per batch (they run in parallel, so wall-time is "
                         "the slowest factory, not the sum).")
    ap.add_argument("--warmup-max", type=int, default=1800,
                    help="Max warmup ticks before forcing the measure phase.")
    ap.add_argument("--measure-max", type=int, default=36000,
                    help="Max measure ticks (cap if the rate never converges).")
    ap.add_argument("--measure-min-items", type=int, default=25,
                    help="Every sink needs >=N items before convergence "
                         "is allowed (rate resolution ~ 1/N).")
    ap.add_argument("--converge-rel", type=float, default=0.02,
                    help="Rate plateau: stop when within this rel. change.")
    ap.add_argument("--converge-hits", type=int, default=3,
                    help="Consecutive stable checks required to converge.")
    ap.add_argument("--game-speed", type=float, default=32.0,
                    help="game.speed while a run is active.")
    ap.add_argument("--rel-tol", type=float, default=0.10,
                    help="Relative tolerance on per-sink items/s.")
    ap.add_argument("--abs-tol", type=float, default=0.25,
                    help="Absolute tolerance on per-sink items/s.")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="Per-run wall-clock timeout (s).")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Dump every spec/result/comparison to this JSON file.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print specs + engine expectations; no Factorio.")
    args = ap.parse_args()

    lessons = _parse_lessons(args.lessons)
    reports: list[ParityReport] = []
    dumps: list[dict] = []
    timing = dict(
        game_speed=args.game_speed,
        warmup_max=args.warmup_max,
        measure_max=args.measure_max,
        measure_min_items=args.measure_min_items,
        converge_rel=args.converge_rel,
        converge_hits=args.converge_hits,
    )

    # Build every factory up front so we can tile them into batches.
    jobs: list[dict] = []
    for lesson in lessons:
        for seed in range(args.seeds):
            factory = build_factory(args.size, lesson, seed=seed)
            if factory is None:
                print(f"=== {lesson.name} seed={seed}: SKIP "
                      "(build_factory returned None) ===")
                continue
            world = factory.world_CWH
            jobs.append({
                "run_id": f"{lesson.name}-s{seed}",
                "lesson": lesson.name,
                "seed": seed,
                "world": world,
                "expected": expected_sink_rates(world),
            })

    if args.dry_run:
        for job in jobs:
            world = job["world"]
            ents, srcs, snks, _ = _convert_world(world)
            print(f"=== {job['run_id']} (dry run) ===")
            for row in render_factory(world).splitlines():
                print("  " + row)
            for (x, y), (item, rate) in sorted(job["expected"].items()):
                print(f"  engine: sink ({x},{y}) {item} ← {rate:.3f}/s")
            print(f"  spec: {len(ents)} entities, {len(srcs)} sources, "
                  f"{len(snks)} sinks")
            dumps.append({
                "run_id": job["run_id"],
                "expected": {f"{k[0]},{k[1]}": v
                             for k, v in job["expected"].items()},
            })
    else:
        rcon = RconClient(args.rcon_host, args.rcon_port, args.rcon_password)
        rcon.connect()
        ping = _remote_call(rcon, "ping")
        print(f"mod ping: {ping or '(no response — is the save loaded?)'}")

        batches = [jobs[i:i + args.batch_size]
                   for i in range(0, len(jobs), args.batch_size)]
        for bi, batch in enumerate(batches):
            print(f"\n--- batch {bi + 1}/{len(batches)} "
                  f"({len(batch)} factories) ---")
            spec = build_batch_spec(
                [(j["run_id"], j["world"]) for j in batch],
                run_id=f"batch{bi}", **timing,
            )
            by_id: dict[str, dict] = {}
            batch_error: str | None = None
            try:
                result = run_parity(rcon, spec, timeout=args.timeout)
                by_id = {f["run_id"]: f for f in result.get("factories", [])}
            except (RuntimeError, TimeoutError, OSError) as e:
                batch_error = str(e)

            for j in batch:
                report = ParityReport(lesson=j["lesson"], seed=j["seed"])
                fres = by_id.get(j["run_id"])
                if batch_error is not None:
                    report.error = batch_error
                elif fres is None:
                    report.error = "factory missing from batch result"
                else:
                    report.sinks = compare_sinks(
                        j["expected"], fres.get("sinks", []),
                        rel_tol=args.rel_tol, abs_tol=args.abs_tol,
                    )
                reports.append(report)
                print(format_report(report, fres, j["world"]))
                dumps.append({
                    "run_id": j["run_id"],
                    "expected": {f"{k[0]},{k[1]}": v
                                 for k, v in j["expected"].items()},
                    "result": fres,
                    "passed": report.passed,
                })

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(dumps, indent=2))
        print(f"wrote {args.json_out}")

    if not args.dry_run:
        n_pass = sum(r.passed for r in reports)
        # A sink is "exact" only at literally 0 error; anything above is
        # called out, so a genuinely clean run reads 0 imperfect sinks
        # rather than hiding small drift inside the pass tolerance.
        imperfect = [
            (r.lesson, r.seed, s)
            for r in reports for s in r.sinks
            if abs(s.factorio_rate - s.engine_rate) > 1e-9
        ]
        print(f"\n{n_pass}/{len(reports)} factories match "
              f"(rel_tol={args.rel_tol:.0%}, abs_tol={args.abs_tol})")
        print(f"{len(imperfect)} sink(s) with >0 error:")
        for lesson, seed, s in sorted(
            imperfect,
            key=lambda t: abs(t[2].factorio_rate - t[2].engine_rate)
            / max(abs(t[2].engine_rate), 1e-9),
            reverse=True,
        ):
            rel = (abs(s.factorio_rate - s.engine_rate)
                   / max(abs(s.engine_rate), 1e-9))
            print(f"  {rel:6.1%}  {lesson} s{seed} sink({s.x},{s.y}) "
                  f"{s.item}: engine {s.engine_rate:.3f} vs "
                  f"factorio {s.factorio_rate:.3f}")
        sys.exit(0 if n_pass == len(reports) else 1)


if __name__ == "__main__":
    main()
