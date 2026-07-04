"""Characterization tests for factory graph construction (issue #178).

These pin down the behaviour the rest of the codebase relies on when it
turns a world tensor into a connection graph: the exact node set, the
exact edge set, and source→sink reachability. Every test goes through
``helpers.build_factory_graph`` — the single indirection point for graph
construction, backed by the Rust engine (``factorion_rs.py_build_graph``).
These snapshots were first frozen against the legacy Python graph builder
for the clean worlds where the two engines agree, so they doubled as the
migration's safety net and remain the permanent characterization of the
Rust graph builder.

Two layers of coverage:

* **Handcrafted golden snapshots** — small worlds whose full (node, edge)
  sets are asserted literally. Ground truth derived from the current
  engine and cross-checked against the Rust unit tests in
  ``factorion_rs/src/graph.rs``.
* **Known-correct generated worlds** — every ``LessonKind`` is a generator
  of factories that ``build_factory`` has already verified simulate to
  positive throughput, so the graph *must* connect each source to a sink.
"""

import networkx as nx
import pytest

from factorion import (
    Direction,
    LessonKind,
    Misc,
    blank_entities,
    build_factory,
)
from helpers import (
    build_factory_graph,
    make_world,
    set_assembler,
    set_entity,
    set_splitter,
)


def _n(name, x, y):
    """Graph node label: ``f"{name}\\n@{x},{y}"`` (engine convention)."""
    return f"{name}\n@{x},{y}"


def _lanes(name, x, y):
    """Both lane-node labels of a belt-ish entity tile: ``name:L`` and
    ``name:R`` (lanes are named relative to the tile's facing)."""
    return [f"{name}:L\n@{x},{y}", f"{name}:R\n@{x},{y}"]


def _summary(graph):
    """(set of node labels, set of (src, dst) edge tuples)."""
    return set(graph.nodes), set(graph.edges)


# ── Handcrafted golden snapshots ─────────────────────────────────────────────


class TestHandcraftedGraphSnapshots:
    """Full node+edge sets for unambiguous, hand-laid worlds."""

    def test_belt_chain(self):
        # source → belt → belt → sink, all facing east on one row. Each belt
        # tile is two lane nodes: the source fills both lanes, belt→belt is
        # lane-preserving, and the sink drains both lanes.
        w = make_world(6)
        set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(w, 1, 0, "transport_belt", Direction.EAST)
        set_entity(w, 2, 0, "transport_belt", Direction.EAST)
        set_entity(w, 3, 0, "sink", Direction.EAST, "copper_cable")
        nodes, edges = _summary(build_factory_graph(w))
        b1l, b1r = _lanes("transport_belt", 1, 0)
        b2l, b2r = _lanes("transport_belt", 2, 0)
        assert nodes == {
            _n("stack_inserter", 0, 0),
            b1l,
            b1r,
            b2l,
            b2r,
            _n("bulk_inserter", 3, 0),
        }
        assert edges == {
            (_n("stack_inserter", 0, 0), b1l),
            (_n("stack_inserter", 0, 0), b1r),
            (b1l, b2l),
            (b1r, b2r),
            (b2l, _n("bulk_inserter", 3, 0)),
            (b2r, _n("bulk_inserter", 3, 0)),
        }

    def test_inserter_chain(self):
        # source → inserter → belt → inserter → sink.
        w = make_world(6)
        set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(w, 1, 0, "inserter", Direction.EAST)
        set_entity(w, 2, 0, "transport_belt", Direction.EAST)
        set_entity(w, 3, 0, "inserter", Direction.EAST)
        set_entity(w, 4, 0, "sink", Direction.EAST, "copper_cable")
        nodes, edges = _summary(build_factory_graph(w))
        bl, br = _lanes("transport_belt", 2, 0)
        assert nodes == {
            _n("stack_inserter", 0, 0),
            _n("inserter", 1, 0),
            bl,
            br,
            _n("inserter", 3, 0),
            _n("bulk_inserter", 4, 0),
        }
        # The drop lands on the belt's RIGHT lane only (belt parallel to the
        # inserter — wiki rule); pickup draws from both lanes.
        assert edges == {
            (_n("stack_inserter", 0, 0), _n("inserter", 1, 0)),
            (_n("inserter", 1, 0), br),
            (bl, _n("inserter", 3, 0)),
            (br, _n("inserter", 3, 0)),
            (_n("inserter", 3, 0), _n("bulk_inserter", 4, 0)),
        }

    def test_long_handed_inserter_reaches_two_tiles(self):
        # A long-handed inserter reaches TWO tiles: it picks up from the source
        # two cells behind it (0,0) and drops onto the sink two cells ahead
        # (4,0), skipping the empty cells (1,0) and (3,0) in between.
        w = make_world(6)
        set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(w, 2, 0, "long_handed_inserter", Direction.EAST)
        set_entity(w, 4, 0, "sink", Direction.EAST, "copper_cable")
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == {
            _n("stack_inserter", 0, 0),
            _n("long_handed_inserter", 2, 0),
            _n("bulk_inserter", 4, 0),
        }
        assert edges == {
            (_n("stack_inserter", 0, 0), _n("long_handed_inserter", 2, 0)),
            (_n("long_handed_inserter", 2, 0), _n("bulk_inserter", 4, 0)),
        }

    def test_long_handed_inserter_skips_adjacent_tiles(self):
        # Belts sit directly behind/ahead of the long inserter (distance 1). A
        # long inserter only ever touches tiles at distance 2, so neither belt
        # connects and the graph has no edges.
        w = make_world(6)
        set_entity(w, 0, 0, "transport_belt", Direction.EAST)
        set_entity(w, 1, 0, "long_handed_inserter", Direction.EAST)
        set_entity(w, 2, 0, "transport_belt", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == {
            *_lanes("transport_belt", 0, 0),
            _n("long_handed_inserter", 1, 0),
            *_lanes("transport_belt", 2, 0),
        }
        assert edges == set()

    def test_underground_belt_run(self):
        # belt → underground(down) … underground(up) → belt.
        w = make_world(6)
        set_entity(w, 0, 0, "transport_belt", Direction.EAST)
        set_entity(w, 1, 0, "underground_belt", Direction.EAST,
                   misc=Misc.UNDERGROUND_DOWN.value)
        set_entity(w, 4, 0, "underground_belt", Direction.EAST,
                   misc=Misc.UNDERGROUND_UP.value)
        set_entity(w, 5, 0, "transport_belt", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        b0l, b0r = _lanes("transport_belt", 0, 0)
        udl, udr = _lanes("underground_belt", 1, 0)
        uul, uur = _lanes("underground_belt", 4, 0)
        b5l, b5r = _lanes("transport_belt", 5, 0)
        assert nodes == {b0l, b0r, udl, udr, uul, uur, b5l, b5r}
        # Lanes persist through the tunnel: Left→Left, Right→Right.
        assert edges == {
            (b0l, udl),
            (b0r, udr),
            (udl, uul),
            (udr, uur),
            (uul, b5l),
            (uur, b5r),
        }

    def test_single_belt_has_no_edges(self):
        w = make_world(3)
        set_entity(w, 1, 1, "transport_belt", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == set(_lanes("transport_belt", 1, 1))
        assert edges == set()

    def test_empty_world_has_no_nodes(self):
        nodes, edges = _summary(build_factory_graph(make_world(5)))
        assert nodes == set()
        assert edges == set()

    def test_adjacent_inserters_do_not_connect(self):
        # #122: an inserter may never pick up from / drop onto another inserter.
        w = make_world(6)
        set_entity(w, 1, 1, "inserter", Direction.EAST)
        set_entity(w, 2, 1, "inserter", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == {_n("inserter", 1, 1), _n("inserter", 2, 1)}
        assert edges == set()

    def test_opposing_belts_do_not_connect(self):
        # A belt never feeds a belt pointing straight back at it.
        w = make_world(6)
        set_entity(w, 1, 0, "transport_belt", Direction.EAST)
        set_entity(w, 2, 0, "transport_belt", Direction.WEST)
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == {
            *_lanes("transport_belt", 1, 0),
            *_lanes("transport_belt", 2, 0),
        }
        assert edges == set()

    def test_source_drops_onto_belt_ahead(self):
        w = make_world(5)
        set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(w, 1, 0, "transport_belt", Direction.EAST)
        _, edges = _summary(build_factory_graph(w))
        bl, br = _lanes("transport_belt", 1, 0)
        # A source fills BOTH lanes of the belt it points into.
        assert edges == {
            (_n("stack_inserter", 0, 0), bl),
            (_n("stack_inserter", 0, 0), br),
        }

    def test_sink_pulls_from_belt_behind(self):
        w = make_world(5)
        set_entity(w, 0, 0, "transport_belt", Direction.EAST)
        set_entity(w, 1, 0, "sink", Direction.EAST, "copper_cable")
        _, edges = _summary(build_factory_graph(w))
        bl, br = _lanes("transport_belt", 0, 0)
        # A sink drains BOTH lanes of the belt feeding it.
        assert edges == {
            (bl, _n("bulk_inserter", 1, 0)),
            (br, _n("bulk_inserter", 1, 0)),
        }


class TestSplitterGraphSnapshots:
    """A splitter is two belts side by side, each with two lanes: four graph
    nodes across its two tiles. Inputs feed the tile they touch, lane-
    preservingly; every tile-lane pool fans out to the same lane of the
    receivers ahead of BOTH tiles (that fan-out is the splitting)."""

    def test_merge_both_input_belts_feed_their_tiles(self):
        # Two east-facing belts, one behind each splitter tile, feed in.
        w = make_world(8)
        set_splitter(w, 2, 2, Direction.EAST)  # tiles (2,2) anchor, (2,3)
        set_entity(w, 1, 2, "transport_belt", Direction.EAST)
        set_entity(w, 1, 3, "transport_belt", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        b2l, b2r = _lanes("transport_belt", 1, 2)
        b3l, b3r = _lanes("transport_belt", 1, 3)
        y2l, y2r = _lanes("splitter", 2, 2)
        y3l, y3r = _lanes("splitter", 2, 3)
        assert nodes == {b2l, b2r, b3l, b3r, y2l, y2r, y3l, y3r}
        assert edges == {
            (b2l, y2l),
            (b2r, y2r),
            (b3l, y3l),
            (b3r, y3r),
        }

    def test_split_every_tile_lane_feeds_both_output_belts(self):
        # One belt feeds the splitter, which fans out to both output belts.
        w = make_world(8)
        set_splitter(w, 2, 2, Direction.EAST)  # tiles (2,2) anchor, (2,3)
        set_entity(w, 1, 2, "transport_belt", Direction.EAST)
        set_entity(w, 3, 2, "transport_belt", Direction.EAST)
        set_entity(w, 3, 3, "transport_belt", Direction.EAST)
        nodes, edges = _summary(build_factory_graph(w))
        inl, inr = _lanes("transport_belt", 1, 2)
        y2l, y2r = _lanes("splitter", 2, 2)
        y3l, y3r = _lanes("splitter", 2, 3)
        o2l, o2r = _lanes("transport_belt", 3, 2)
        o3l, o3r = _lanes("transport_belt", 3, 3)
        assert nodes == {inl, inr, y2l, y2r, y3l, y3r, o2l, o2r, o3l, o3r}
        assert edges == {
            (inl, y2l),
            (inr, y2r),
            # Left pools route to left lanes of both outputs…
            (y2l, o2l),
            (y2l, o3l),
            (y3l, o2l),
            (y3l, o3l),
            # …and right pools to right lanes. No lane leakage.
            (y2r, o2r),
            (y2r, o3r),
            (y3r, o2r),
            (y3r, o3r),
        }


class TestAssemblerGraphSnapshots:
    """A 3×3 assembler connects only through inserters on its perimeter, and
    all of its edges remap onto the (4,4) anchor."""

    def test_in_and_out_inserters_route_through_anchor(self):
        w = make_world(11)
        ax, ay = 4, 4
        set_assembler(w, ax, ay, "copper_cable")
        # input inserter west of the body, facing into the assembler
        set_entity(w, ax - 1, ay + 1, "inserter", Direction.EAST)
        set_entity(w, ax - 2, ay + 1, "source", Direction.EAST, "copper_plate")
        # output inserter east of the body, facing away from the assembler
        set_entity(w, ax + 3, ay + 1, "inserter", Direction.EAST)
        set_entity(w, ax + 4, ay + 1, "sink", Direction.EAST, "copper_cable")
        nodes, edges = _summary(build_factory_graph(w))
        assert nodes == {
            _n("assembling_machine_1", 4, 4),
            _n("stack_inserter", 2, 5),
            _n("inserter", 3, 5),
            _n("inserter", 7, 5),
            _n("bulk_inserter", 8, 5),
        }
        assert edges == {
            (_n("stack_inserter", 2, 5), _n("inserter", 3, 5)),
            (_n("inserter", 3, 5), _n("assembling_machine_1", 4, 4)),
            (_n("assembling_machine_1", 4, 4), _n("inserter", 7, 5)),
            (_n("inserter", 7, 5), _n("bulk_inserter", 8, 5)),
        }


# ── Known-correct generated worlds (the lesson generators) ───────────────────


def _graph_of_generated(kind, seed, size=10):
    """Build a lesson factory and return its graph, or ``None`` if the
    randomized layout search failed for this seed (caller skips)."""
    factory = build_factory(size=size, kind=kind, seed=seed)
    if factory is None:
        return None
    world_CWH, _ = blank_entities(factory, num_missing_entities=0)
    return build_factory_graph(world_CWH.permute(1, 2, 0))


# Every lesson kind whose generator wires a source → … → sink chain. These
# are "definitely correct" worlds: build_factory only returns layouts that
# simulate to positive Rust throughput, so the graph must connect them.
_CONNECTIVITY_KINDS = [
    LessonKind.MOVE_ONE_ITEM,
    LessonKind.MOVE_VIA_UG_BELT,
    LessonKind.SPLITTER_SPLIT,
    LessonKind.SPLITTER_MERGE,
    # LessonKind.ASSEMBLE_1IN_1OUT,
    # LessonKind.ASSEMBLE_2IN_1OUT,
]

class TestGeneratedWorldConnectivity:
    """For known-correct factories, every source reaches some sink and every
    sink is reached by some source through the constructed graph. Holds for
    *all* lesson kinds — including the underground lessons the legacy Python
    graph builder mis-wired — now that build_factory_graph runs on the Rust
    engine (issue #178)."""

    @pytest.mark.parametrize("kind", _CONNECTIVITY_KINDS, ids=lambda k: k.name)
    @pytest.mark.parametrize("seed", range(12))
    def test_every_source_reaches_a_sink(self, kind, seed):
        graph = _graph_of_generated(kind, seed)
        if graph is None:
            pytest.skip(f"{kind.name} seed={seed}: layout search returned None")
        sources = [n for n in graph.nodes if "stack_inserter" in n]
        sinks = [n for n in graph.nodes if "bulk_inserter" in n]
        assert sources, f"{kind.name} seed={seed}: no source node in graph"
        assert sinks, f"{kind.name} seed={seed}: no sink node in graph"

        for src in sources:
            assert any(nx.has_path(graph, src, snk) for snk in sinks), (
                f"{kind.name} seed={seed}: source {src!r} reaches no sink"
            )
        for snk in sinks:
            assert any(nx.has_path(graph, src, snk) for src in sources), (
                f"{kind.name} seed={seed}: sink {snk!r} unreachable from any source"
            )

    @pytest.mark.parametrize("kind", _CONNECTIVITY_KINDS, ids=lambda k: k.name)
    @pytest.mark.parametrize("seed", range(6))
    def test_graph_nodes_match_placed_entities(self, kind, seed):
        """Every node corresponds to a real placed-entity tile, and the node
        count never exceeds two per non-empty tile (belt-ish tiles own one
        node per lane; everything else one per entity)."""
        factory = build_factory(size=10, kind=kind, seed=seed)
        if factory is None:
            pytest.skip(f"{kind.name} seed={seed}: layout search returned None")
        world_CWH, _ = blank_entities(factory, num_missing_entities=0)
        graph = build_factory_graph(world_CWH.permute(1, 2, 0))
        ent_ch = world_CWH[0]
        non_empty = int((ent_ch != 0).sum().item())
        assert 0 < graph.number_of_nodes() <= 2 * non_empty, (
            f"{kind.name} seed={seed}: {graph.number_of_nodes()} nodes vs "
            f"{non_empty} non-empty tiles"
        )


# ── Stale-extension guard ────────────────────────────────────────────────────


class TestFactorionRsCurrencyGuard:
    """`build_graph_nx` reaches into `factorion_rs.py_build_graph` at call time,
    so a Rust extension built before the graph migration (#178) imports cleanly
    (it still has `py_items`) but then crashes deep in the web UI's auto-graph
    with a cryptic `AttributeError: module 'factorion_rs' has no attribute
    'py_build_graph'`. The import-time guard turns that into an actionable
    error. See `factorion._assert_factorion_rs_current`."""

    def test_stale_build_raises_actionable_error(self):
        import types

        import factorion

        # A wheel built before #178: has py_items (so plain import succeeds) but
        # not py_build_graph — exactly the state that produced the crash.
        stale = types.SimpleNamespace(
            simulate_throughput=lambda *a, **k: None,
            py_entity_tiles=lambda *a, **k: None,
            py_items=lambda *a, **k: {},
            py_recipes=lambda *a, **k: {},
        )
        assert not hasattr(stale, "py_build_graph")
        with pytest.raises(ImportError, match="py_build_graph"):
            factorion._assert_factorion_rs_current(stale)
        with pytest.raises(ImportError, match="maturin develop"):
            factorion._assert_factorion_rs_current(stale)

    def test_installed_build_passes_guard(self):
        """The real, freshly-built extension satisfies the guard — also guards
        against a typo'd name in the required-functions list."""
        import factorion
        import factorion_rs

        factorion._assert_factorion_rs_current(factorion_rs)  # must not raise
