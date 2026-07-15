"""Tests for the MINE_ORE lesson: an electric mining drill on an ore patch
routed to a sink. The ore lives on the ORES terrain channel — never blanked,
never counted as a removable entity."""

import pytest

from helpers import (
    Channel,
    LessonKind,
    blank_entities,
    build_factory,
    rs_throughput,
    str2ent,
    str2item,
)

ORE_VALUES = {
    str2item(name).value for name in ("copper_ore", "iron_ore", "coal", "stone")
}


def _build(seed, size=10):
    return build_factory(size=size, kind=LessonKind.MINE_ORE, seed=seed)


def _first_built(size=10):
    for seed in range(20):
        factory = _build(seed, size=size)
        if factory is not None:
            return factory
    pytest.fail("no MINE_ORE factory built in 20 seeds")


class TestMineOreGeneration:
    def test_generates_drill_ore_and_sink(self):
        factory = _first_built()
        ent = factory.world_CWH[Channel.ENTITIES.value]
        ores = factory.world_CWH[Channel.ORES.value]
        drill_id = str2ent("electric_mining_drill").value
        assert int((ent == drill_id).sum()) == 9, "exactly one 3x3 drill"
        assert int((ent == str2ent("bulk_inserter").value).sum()) == 1, "one sink"
        painted = {int(v) for v in ores.unique().tolist()} - {0}
        assert len(painted) == 1, "single-type ore patch"
        assert painted <= ORE_VALUES

    def test_solved_factory_is_drill_limited(self):
        factory = _first_built()
        thput, unreachable = rs_throughput(factory.world_CWH.permute(1, 2, 0))
        assert thput == pytest.approx(0.5), "flat drill rate reaches the sink"
        assert unreachable == 0

    def test_blanking_clears_entities_but_never_ore(self):
        factory = _first_built()
        ores_before = factory.world_CWH[Channel.ORES.value].clone()
        world, removed = blank_entities(
            factory, num_missing_entities=factory.total_entities
        )
        assert removed == factory.total_entities
        drill_id = str2ent("electric_mining_drill").value
        assert int((world[Channel.ENTITIES.value] == drill_id).sum()) == 0, (
            "the drill is a removable (blankable) unit"
        )
        assert bool((world[Channel.ORES.value] == ores_before).all()), (
            "ore is terrain: blanking must not touch the ORES channel"
        )

    def test_drill_counts_as_one_removable_unit(self):
        factory = _first_built()
        belts = int(
            (
                factory.world_CWH[Channel.ENTITIES.value]
                == str2ent("transport_belt").value
            ).sum()
        )
        assert factory.total_entities == 1 + belts
