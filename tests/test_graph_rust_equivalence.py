"""Equivalence/divergence characterization: Rust ``build_graph`` vs the legacy
Python ``world2graph`` (issue #178).

This is the keystone of the migration. Before deleting ``world2graph`` we pin
down *exactly* how the Rust-built graph relates to the Python-built one, so the
swap is provably safe:

* **Node sets are always identical.** Both engines emit one node per placeable
  entity (anchor tile only for multi-tile units), so connectivity questions are
  asked over the same vertices.

* **Edges differ only in known, correct ways**, every one a bug the Rust engine
  fixes and the Python engine still has:
    - *Python-only* edges always point **into a source** (``stack_inserter``).
      A source is a pure producer; ``world2graph`` spuriously gives it an input
      edge from whatever sits behind it. Rust drops these (#174: source/sink
      connect like belts, never to each other).
    - *Rust-only* edges always start at an **underground belt** (the #173
      underground rewrite that only landed in Rust) or end at a **sink**
      (``bulk_inserter`` — the #174 sink-connects-like-a-belt delivery).

* **Rust connects every known-correct factory** source→sink, including the
  underground lessons ``world2graph`` mis-wires.

When ``world2graph`` is removed this file goes with it — the permanent safety
net is the golden snapshots in ``test_graph_construction.py`` (which these
tests confirm Rust reproduces exactly for clean worlds).
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
    make_world,
    rust_factory_graph,
    set_assembler,
    set_entity,
    set_splitter,
    world2graph,  # the legacy Python path under test
)


def _entity_type(label):
    """Entity name out of a ``"{name}\\n@{x},{y}"`` node label."""
    return label.split("\n")[0]


def _sets(graph):
    return set(graph.nodes), set(graph.edges)


# All lesson kinds, including the two whose generators place underground belts.
_ALL_KINDS = list(LessonKind)


def _generated(kind, seed, size=10):
    factory = build_factory(size=size, kind=kind, seed=seed)
    if factory is None:
        return None
    return blank_entities(factory, num_missing_entities=0)[0].permute(1, 2, 0)


# ── Clean handcrafted worlds: Rust reproduces Python exactly ──────────────────


def _clean_worlds():
    """Hand-laid worlds with no divergence triggers (no source with an entity
    behind it, no long/odd undergrounds). Rust and Python must agree to the
    edge."""
    worlds = {}

    w = make_world(6)
    set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
    set_entity(w, 1, 0, "transport_belt", Direction.EAST)
    set_entity(w, 2, 0, "transport_belt", Direction.EAST)
    set_entity(w, 3, 0, "sink", Direction.EAST, "copper_cable")
    worlds["belt_chain"] = w

    w = make_world(6)
    set_entity(w, 0, 0, "source", Direction.EAST, "copper_cable")
    set_entity(w, 1, 0, "inserter", Direction.EAST)
    set_entity(w, 2, 0, "transport_belt", Direction.EAST)
    set_entity(w, 3, 0, "inserter", Direction.EAST)
    set_entity(w, 4, 0, "sink", Direction.EAST, "copper_cable")
    worlds["inserter_chain"] = w

    w = make_world(6)
    set_entity(w, 0, 0, "transport_belt", Direction.EAST)
    set_entity(w, 1, 0, "underground_belt", Direction.EAST,
               misc=Misc.UNDERGROUND_DOWN.value)
    set_entity(w, 4, 0, "underground_belt", Direction.EAST,
               misc=Misc.UNDERGROUND_UP.value)
    set_entity(w, 5, 0, "transport_belt", Direction.EAST)
    worlds["underground"] = w

    w = make_world(8)
    set_splitter(w, 2, 2, Direction.EAST)
    set_entity(w, 1, 2, "transport_belt", Direction.EAST)
    set_entity(w, 1, 3, "transport_belt", Direction.EAST)
    set_entity(w, 3, 2, "transport_belt", Direction.EAST)
    set_entity(w, 3, 3, "transport_belt", Direction.EAST)
    worlds["splitter"] = w

    w = make_world(11)
    set_assembler(w, 4, 4, "copper_cable")
    set_entity(w, 3, 5, "inserter", Direction.EAST)
    set_entity(w, 2, 5, "source", Direction.EAST, "copper_plate")
    set_entity(w, 7, 5, "inserter", Direction.EAST)
    set_entity(w, 8, 5, "sink", Direction.EAST, "copper_cable")
    worlds["assembler"] = w

    return worlds


@pytest.mark.parametrize("name", sorted(_clean_worlds()))
def test_rust_matches_python_on_clean_worlds(name):
    w = _clean_worlds()[name]
    py_nodes, py_edges = _sets(world2graph(w))
    rs_nodes, rs_edges = _sets(rust_factory_graph(w))
    assert rs_nodes == py_nodes
    assert rs_edges == py_edges


# ── Generated worlds: node parity, explained edge divergence, connectivity ────


class TestNodeSetsAlwaysMatch:
    @pytest.mark.parametrize("kind", _ALL_KINDS, ids=lambda k: k.name)
    @pytest.mark.parametrize("seed", range(20))
    def test_node_sets_identical(self, kind, seed):
        whc = _generated(kind, seed)
        if whc is None:
            pytest.skip(f"{kind.name} seed={seed}: layout search returned None")
        py_nodes = set(world2graph(whc).nodes)
        rs_nodes = set(rust_factory_graph(whc).nodes)
        assert rs_nodes == py_nodes


class TestEdgeDivergenceIsExplained:
    """Every edge that differs between the engines is accounted for by a known
    Rust fix; no unexplained divergence slips through."""

    @pytest.mark.parametrize("kind", _ALL_KINDS, ids=lambda k: k.name)
    @pytest.mark.parametrize("seed", range(20))
    def test_only_known_fixes_differ(self, kind, seed):
        whc = _generated(kind, seed)
        if whc is None:
            pytest.skip(f"{kind.name} seed={seed}: layout search returned None")
        py_edges = set(world2graph(whc).edges)
        rs_edges = set(rust_factory_graph(whc).edges)

        # Python-only edges: spurious input edges into a source.
        for u, v in py_edges - rs_edges:
            assert _entity_type(v) == "stack_inserter", (
                f"{kind.name} seed={seed}: unexplained Python-only edge "
                f"{_entity_type(u)} -> {_entity_type(v)} (expected a phantom "
                f"edge into a source)"
            )

        # Rust-only edges: underground exits, or deliveries into a sink.
        for u, v in rs_edges - py_edges:
            assert (
                _entity_type(u) == "underground_belt"
                or _entity_type(v) == "bulk_inserter"
            ), (
                f"{kind.name} seed={seed}: unexplained Rust-only edge "
                f"{_entity_type(u)} -> {_entity_type(v)} (expected an "
                f"underground-exit or sink-delivery edge)"
            )


class TestRustConnectsKnownCorrectFactories:
    """The property ``world2graph`` fails on undergrounds: in the Rust graph
    every source reaches a sink and every sink is reached, for *all* lesson
    kinds. build_factory only emits layouts with positive Rust throughput, so
    this must hold."""

    @pytest.mark.parametrize("kind", _ALL_KINDS, ids=lambda k: k.name)
    @pytest.mark.parametrize("seed", range(20))
    def test_source_reaches_sink_in_rust_graph(self, kind, seed):
        whc = _generated(kind, seed)
        if whc is None:
            pytest.skip(f"{kind.name} seed={seed}: layout search returned None")
        graph = rust_factory_graph(whc)
        sources = [n for n in graph.nodes if "stack_inserter" in n]
        sinks = [n for n in graph.nodes if "bulk_inserter" in n]
        assert sources and sinks
        for src in sources:
            assert any(nx.has_path(graph, src, snk) for snk in sinks), (
                f"{kind.name} seed={seed}: source {src!r} reaches no sink in "
                f"the Rust graph"
            )
        for snk in sinks:
            assert any(nx.has_path(graph, src, snk) for src in sources), (
                f"{kind.name} seed={seed}: sink {snk!r} unreachable in the "
                f"Rust graph"
            )
