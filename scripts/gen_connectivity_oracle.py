#!/usr/bin/env python3
"""Parse the hand-authored connectivity ground-truth table and emit a Rust
oracle that the exhaustive connectivity test asserts against.

The table (scripts/connectivity_ground_truth.txt) is the canonical adjacency
"B directly East of A", indexed by entity x facing. It uses two shorthands the
human filled in, which this script resolves to concrete edges:

  '_'  redundant lower-triangle cell. By 180-degree rotational invariance,
       cell(rowEnt=X/dx, colEnt=Y/dy) == flip( cell(Y/flip(dy), X/flip(dx)) ),
       where flip(dir) is the 180-degree turn and flip(edge) swaps '>'/'<'.
  'b'  the source/sink in this column behaves exactly like a belt, so the value
       is whatever the same row gives against a belt at the same facing.

Run:  uv run python scripts/gen_connectivity_oracle.py
"""

from __future__ import annotations

import pathlib
import sys

ENTS = ["belt", "inserter", "ug_up", "ug_down", "source", "sink"]
DIRS = ["N", "E", "S", "W"]
N = len(ENTS) * len(DIRS)  # 24
BELT = ENTS.index("belt")

HERE = pathlib.Path(__file__).resolve().parent
TABLE_PATH = HERE / "connectivity_ground_truth.txt"
CASES_PATH = HERE.parent / "factorion_rs" / "tests" / "connectivity_cases.txt"

# Grid glyph per entity. NOTE: the human legend has no inserter; we use 'i'.
GLYPH = {
    "belt": "b",
    "inserter": "i",
    "ug_up": "u",
    "ug_down": "d",
    "source": "S",
    "sink": "K",
}
# Facing suffix in the grid.
DIRG = {"N": "^", "E": ">", "S": "v", "W": "<"}


def idx(ent: int, d: int) -> int:
    return ent * len(DIRS) + d


def flip_dir(d: int) -> int:
    # N<->S (0<->2), E<->W (1<->3): a 180-degree turn.
    return d ^ 2


def flip_edge(c: str) -> str:
    return {">": "<", "<": ">"}.get(c, c)


def parse_table(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    labels: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("row\\col"):
            continue
        tok = line.split()
        label, cells = tok[0], tok[1:]
        if len(cells) != N:
            sys.exit(f"row {label!r}: expected {N} cells, got {len(cells)}")
        labels.append(label)
        rows.append(cells)
    expected = [f"{e}/{d}" for e in ENTS for d in DIRS]
    if labels != expected:
        sys.exit(f"row labels/order mismatch:\n got={labels}\n want={expected}")
    if len(rows) != N:
        sys.exit(f"expected {N} rows, got {len(rows)}")
    return rows


def resolve(grid: list[list[str]]) -> list[list[str]]:
    g = [row[:] for row in grid]

    def concrete(v: str) -> bool:
        return v in (".", ">", "<", "x")

    # 'b' and '_' can depend on each other (a 'b' points at a belt column that
    # is itself a '_', whose mirror is another 'b', ...), but every chain
    # bottoms out at a concrete core cell. Iterate to a fixpoint: each round,
    # resolve any cell whose source is already concrete.
    #   'b' at (r,c)  -> g[r][belt column, same facing]
    #   '_' at (r,c)  -> flip( g[mirror] ), mirror = 180deg-rotated transpose
    changed = True
    while changed:
        changed = False
        for r in range(N):
            for c in range(N):
                cell = g[r][c]
                if cell == "b":
                    v = g[r][idx(BELT, c % len(DIRS))]
                    if concrete(v):
                        g[r][c] = v
                        changed = True
                elif cell == "_":
                    er, dr = divmod(r, len(DIRS))
                    ec, dc = divmod(c, len(DIRS))
                    v = g[idx(ec, flip_dir(dc))][idx(er, flip_dir(dr))]
                    if concrete(v):
                        g[r][c] = flip_edge(v)
                        changed = True

    for r in range(N):
        for c in range(N):
            if not concrete(g[r][c]):
                sys.exit(f"unresolved cell ({r},{c})={g[r][c]!r} (dependency cycle?)")

    # Validate: fully resolved, and globally consistent under the mirror
    # (so the human's upper triangle has no internal contradictions).
    for r in range(N):
        for c in range(N):
            if g[r][c] not in (".", ">", "<", "x"):
                sys.exit(f"cell ({r},{c}) still unresolved: {g[r][c]!r}")
            er, dr = divmod(r, len(DIRS))
            ec, dc = divmod(c, len(DIRS))
            mr, mc = idx(ec, flip_dir(dc)), idx(er, flip_dir(dr))
            if g[r][c] != flip_edge(g[mr][mc]):
                sys.exit(
                    f"mirror inconsistency: ({r},{c})={g[r][c]!r} but "
                    f"flip(({mr},{mc})={g[mr][mc]!r}) disagrees"
                )
    return g


def _edges(conn: str, ag: str, bg: str) -> list[str]:
    """Graph edges for one canonical cell: A is @0,0, B is @1,0."""
    a2b = f"{ag}@0,0 --> {bg}@1,0"
    b2a = f"{bg}@1,0 --> {ag}@0,0"
    return {".": [], ">": [a2b], "<": [b2a], "x": [a2b, b2a]}[conn]


def emit_cases(g: list[list[str]]) -> str:
    """One FactorySpec text case per oracle cell (B directly East of A).

    Each case asserts the exact edge set build_graph must produce for the two
    placed entities. throughput/items are intentionally absent — these cases
    test connectivity only.
    """
    out = [
        "# AUTO-GENERATED by scripts/gen_connectivity_oracle.py from",
        "# scripts/connectivity_ground_truth.txt -- do not edit by hand.",
        "#",
        "# Exhaustive 1x1 connectivity, canonical adjacency: B is directly East",
        "# of A (rotation invariance, separately proven, covers the other three).",
        "# Cases are split on '=== <label> ===' lines; the text between a label",
        "# and the next label is one FactorySpec for parse().",
        "#",
        "# Grid glyphs: b belt, i inserter, u ug_up, d ug_down, S source, K sink;",
        "# facing ^ N, > E, v S, < W; '..' empty. NOTE: 'i' (inserter) is not in",
        "# the original legend.",
    ]
    for r in range(N):
        er, dr = divmod(r, len(DIRS))
        for c in range(N):
            ec, dc = divmod(c, len(DIRS))
            a_name, a_dir = ENTS[er], DIRS[dr]
            b_name, b_dir = ENTS[ec], DIRS[dc]
            ag, bg = GLYPH[a_name], GLYPH[b_name]
            a_cell = f"{ag}{DIRG[a_dir]}"
            b_cell = f"{bg}{DIRG[b_dir]}"
            out.append(f"=== {a_name}/{a_dir} vs {b_name}/{b_dir} ===")
            out.append("graph: |")
            out.extend(f"  {e}" for e in _edges(g[r][c], ag, bg))
            out.append("---")
            out.append(f"{a_cell} {b_cell}")
            out.append("")
    return "\n".join(out)


def main() -> None:
    grid = parse_table(TABLE_PATH.read_text())
    resolved = resolve(grid)

    # Human-readable resolved matrix (sanity check).
    print("=== resolved matrix (B East of A) ===", file=sys.stderr)
    code = {
        "belt": "B",
        "inserter": "I",
        "ug_up": "U",
        "ug_down": "D",
        "source": "S",
        "sink": "K",
    }
    header = "row\\col   " + "".join(
        f"{code[ENTS[e]]}{d}" for e in range(len(ENTS)) for d in DIRS
    )
    print(header, file=sys.stderr)
    for r in range(N):
        e, d = divmod(r, len(DIRS))
        print(
            f"{ENTS[e]:>8}/{DIRS[d]} " + "".join(f"{c:>2}" for c in resolved[r]),
            file=sys.stderr,
        )

    corpus = emit_cases(resolved)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.write_text(corpus + "\n")
    n_cases = corpus.count("\n=== ") + corpus.startswith("=== ")
    n_edges = sum(
        c in (">", "<") for row in resolved for c in row
    ) + 2 * sum(c == "x" for row in resolved for c in row)
    print(f"wrote {n_cases} cases ({n_edges} asserted edges) to {CASES_PATH}")


if __name__ == "__main__":
    main()
