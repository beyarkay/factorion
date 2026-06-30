"""What the env state looks like after the agent places a 3x3 assembling machine.

An assembling machine is a multi-tile (3x3) entity, so a single placement
action at one anchor tile fans out across nine tiles. These tests pin exactly
how `FactorioEnv.step` writes that footprint into the state tensor, and what
the agent sees on its next observation.

Scenario (matches the worked example in the issue):

    S> b> i> .. .. ..        ->        S> b> i> aa aa aa
    .. .. .. .. .. ..                  .. .. .. aa aa aa
    .. .. .. .. .. ..                  .. .. .. aa aa aa

The asm anchor is tile (x=0, y=3); placing it fills the 3x3 block spanning
rows 0-2, cols 3-5 with the assembling-machine entity id.
"""

import os
import sys

import numpy as np
import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from ppo import FactorioEnv  # noqa: E402
from helpers import Channel, Direction, str2ent  # noqa: E402

ENT = Channel.ENTITIES.value
DIR = Channel.DIRECTION.value
ITEMS = Channel.ITEMS.value
MISC = Channel.MISC.value
FOOT = Channel.FOOTPRINT.value

EMPTY = str2ent("empty").value
SOURCE = str2ent("stack_inserter").value
SINK = str2ent("bulk_inserter").value
BELT = str2ent("transport_belt").value
INSERTER = str2ent("inserter").value
ASM = str2ent("assembling_machine_1").value
RECIPE = str2ent("electronic_circuit").value  # a valid asm recipe ingredient


@pytest.fixture()
def env():
    """A 6x6 env whose world we overwrite with the worked-example layout.

    `reset` installs a real (random-lesson) factory; we blow that away and
    plant the `S> b> i>` line ourselves so the test is deterministic and
    matches the issue's example exactly.
    """
    env = FactorioEnv(size=6, max_steps=30, idx=0)
    env.reset(seed=0)

    w = env._world_CWH
    w.zero_()
    w[FOOT] = 1  # everything buildable
    east = Direction.EAST.value
    # row index = x (first spatial dim), col index = y (second spatial dim)
    w[ENT, 0, 0], w[DIR, 0, 0] = SOURCE, east
    w[ENT, 0, 1], w[DIR, 0, 1] = BELT, east
    w[ENT, 0, 2], w[DIR, 0, 2] = INSERTER, east
    # Keep the env's notion of source/sink consistent with the world we planted.
    env._source_id = SOURCE
    env._sink_id = SINK
    yield env
    env.close()


def _place_asm(env, x, y, direction=Direction.NONE.value, item=RECIPE):
    action = {
        "xy": np.array([x, y]),
        "entity": ASM,
        "direction": direction,
        "item": item,
        "misc": 0,
        "eot": 0,
    }
    return env.step(action)


# The nine tiles a 3x3 asm anchored at (0, 3) occupies (rows 0-2, cols 3-5).
# Verified against factorion_rs.py_entity_tiles in test_footprint_is_nine_tiles.
ASM_FOOTPRINT = tuple((x, y) for x in (0, 1, 2) for y in (3, 4, 5))


class TestAsmFootprintWrite:
    def test_footprint_is_nine_tiles(self):
        """The anchor fans out to a 3x3 block at rows 0-2, cols 3-5, exactly as
        the Rust footprint helper computes it."""
        tiles = factorion_rs.py_entity_tiles(0, 3, Direction.NONE.value, 3, 3)
        assert tiles is not None
        assert set(map(tuple, tiles)) == set(ASM_FOOTPRINT)

    def test_entity_id_written_to_all_nine_tiles(self, env):
        """Every footprint tile gets the assembling-machine entity id."""
        _place_asm(env, 0, 3)
        ent = env._world_CWH[ENT].numpy().astype(int)
        for tx, ty in ASM_FOOTPRINT:
            assert ent[tx, ty] == ASM, f"tile ({tx},{ty}) is {ent[tx, ty]}, want {ASM}"

    def test_resulting_entities_channel_matches_worked_example(self, env):
        """The full ENTITIES channel equals the issue's expected layout."""
        _place_asm(env, 0, 3)
        ent = env._world_CWH[ENT].numpy().astype(int)
        expected = np.zeros((6, 6), dtype=int)
        expected[0, 0] = SOURCE
        expected[0, 1] = BELT
        expected[0, 2] = INSERTER
        for x in (0, 1, 2):
            for y in (3, 4, 5):
                expected[x, y] = ASM
        np.testing.assert_array_equal(ent, expected)

    def test_original_line_is_untouched(self, env):
        """Placing the asm does not disturb the S> b> i> line."""
        _place_asm(env, 0, 3)
        ent = env._world_CWH[ENT].numpy().astype(int)
        assert (ent[0, 0], ent[0, 1], ent[0, 2]) == (SOURCE, BELT, INSERTER)

    def test_direction_written_to_all_nine_tiles(self, env):
        """Direction is written at every footprint tile (not just the anchor).

        A 3x3 asm is square, so the footprint is rotation-invariant, but the
        env still stamps the chosen direction into all nine DIRECTION cells.
        """
        _place_asm(env, 0, 3, direction=Direction.EAST.value)
        d = env._world_CWH[DIR].numpy().astype(int)
        for tx, ty in ASM_FOOTPRINT:
            assert d[tx, ty] == Direction.EAST.value

    def test_recipe_written_only_at_anchor(self, env):
        """The recipe (ITEMS channel) lives on the anchor tile alone, not the
        whole footprint."""
        _place_asm(env, 0, 3)
        it = env._world_CWH[ITEMS].numpy().astype(int)
        assert it[0, 3] == RECIPE
        for tx, ty in ASM_FOOTPRINT:
            if (tx, ty) != (0, 3):
                assert it[tx, ty] == EMPTY, (
                    f"recipe leaked to non-anchor tile ({tx},{ty})"
                )

    def test_next_observation_reflects_placement(self, env):
        """The observation handed back is the mutated world the agent sees next
        turn — the 3x3 asm block is present in the returned obs."""
        obs, _, _, _, _ = _place_asm(env, 0, 3)
        obs_ent = obs[ENT].astype(int)
        for tx, ty in ASM_FOOTPRINT:
            assert obs_ent[tx, ty] == ASM
        np.testing.assert_array_equal(obs_ent, env._world_CWH[ENT].numpy().astype(int))

    def test_placement_counts_as_one_entity(self, env):
        """A multi-tile asm is one placed entity unit, not nine."""
        before = env._num_placed_entities
        _, _, _, _, info = _place_asm(env, 0, 3)
        assert env._num_placed_entities == before + 1
        assert not any(info["invalid_reason"].values())


class TestAsmPlacementValidity:
    def test_asm_without_recipe_is_invalid_and_world_unchanged(self, env):
        """Placing an asm with no recipe is rejected; nothing is written."""
        before = env._world_CWH[ENT].numpy().astype(int).copy()
        _, _, _, _, info = _place_asm(env, 0, 3, item=EMPTY)
        assert info["invalid_reason"]["place_asm_mach_wo_recipe"]
        np.testing.assert_array_equal(
            env._world_CWH[ENT].numpy().astype(int), before
        )

    def test_asm_over_plain_entity_overwrites(self, env):
        """The only occupancy the env protects is source/sink and masked/
        out-of-bounds tiles — a footprint tile holding a plain belt/inserter is
        simply overwritten. Anchor (0,1) covers the belt at (0,1) and the
        inserter at (0,2); both become asm."""
        _, _, _, _, info = _place_asm(env, 0, 1)
        assert not any(info["invalid_reason"].values())
        ent = env._world_CWH[ENT].numpy().astype(int)
        assert ent[0, 1] == ASM and ent[0, 2] == ASM

    def test_asm_over_source_tile_is_rejected(self, env):
        """A footprint that would replace the source (or sink) is rejected and
        the world is left untouched."""
        before = env._world_CWH[ENT].numpy().astype(int).copy()
        # Anchor (0,0): footprint covers the source at (0,0).
        _, _, _, _, info = _place_asm(env, 0, 0)
        assert info["invalid_reason"]["replaced_source_or_sink"]
        np.testing.assert_array_equal(
            env._world_CWH[ENT].numpy().astype(int), before
        )

    def test_asm_running_off_grid_is_rejected(self, env):
        """An anchor too close to the edge for a 3x3 footprint is out of
        bounds and rejected."""
        before = env._world_CWH[ENT].numpy().astype(int).copy()
        # Anchor (0, 5): footprint would need cols 5,6,7 — off a size-6 grid.
        _, _, _, _, info = _place_asm(env, 0, 5)
        assert info["invalid_reason"]["too_wide"]
        np.testing.assert_array_equal(
            env._world_CWH[ENT].numpy().astype(int), before
        )
