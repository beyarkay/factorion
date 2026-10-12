"""Tests for the whole-factory denoiser (denoise.py)."""

import os
import sys
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import LessonKind, blank_entities, build_factory, str2ent  # noqa: E402
from factorion import FULL_FACTORY_KINDS, LESSON_IS_TRIAL, Factory  # noqa: E402
from denoise import (  # noqa: E402
    STATE_CHANNELS,
    DenoiserUNet,
    RECIPES,
    TYPES,
    corrupt,
    free_mask,
    generate,
    legalize,
    run_denoise_eval,
    score_world,
    state_to_tokens,
    tokens_to_state,
    world_to_state,
)
from training_config import SharedArgs  # noqa: E402

TINY: dict[str, Any] = dict(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4)
_ASM = str2ent("assembling_machine_1").value


def _factory(size, kind, seed) -> Factory:
    factory = build_factory(size=size, kind=kind, seed=seed)
    assert factory is not None
    return factory


def _tokens(world_CWH):
    state = torch.from_numpy(world_to_state(world_CWH))[None]
    return state, [t[0].numpy() for t in state_to_tokens(state)]


@pytest.mark.parametrize("kind", [k for k in FULL_FACTORY_KINDS if not LESSON_IS_TRIAL[k]], ids=lambda k: k.name)
def test_every_training_factory_is_a_reachable_target(kind):
    """Every solved factory is expressible in tokens, and decoding its tokens
    onto its blank markers rebuilds it exactly — and it flows."""
    for seed in range(2):
        factory = _factory(size=SharedArgs.size, kind=kind, seed=seed)
        state, (typ, rec, part) = _tokens(factory.world_CWH)
        free = free_mask(state)
        assert torch.equal(tokens_to_state(state, free, *(torch.from_numpy(t)[None] for t in (typ, rec, part))), state.long())
        cond = blank_entities(factory)[0].numpy().astype(np.int64)
        world = legalize(cond, typ, rec, part, np.ones(typ.shape))
        assert np.array_equal(world, factory.world_CWH.numpy()), seed
        assert score_world(world, factory.max_throughput, 0.0)["thput_raw"] > 0


@pytest.mark.parametrize("io", ["token", "factored"])
def test_a_confident_prediction_decodes_to_the_factory(io):
    """Logits that put all their mass on a solved factory's tokens decode,
    through `pick` and `legalize`, to exactly that factory — assemblers and
    splitters included, re-anchored by raster scan when the heads carry no
    footprint part (`factored`)."""
    model = DenoiserUNet(**TINY, io=io)
    for kind in (LessonKind.SPLITTER_1IN, LessonKind.MIRRORED_2IN):
        factory = _factory(15, kind, 0)
        state, toks = _tokens(factory.world_CWH)
        typ, rec, part = (torch.from_numpy(t)[None] for t in toks)
        if io == "token":
            targets = [typ, rec, part]
        else:
            edm = torch.from_numpy(np.array(TYPES))[typ]
            targets = [edm[..., 0], edm[..., 1], torch.from_numpy(np.array(RECIPES))[rec], edm[..., 2]]
        logits = torch.cat(
            [100 * torch.nn.functional.one_hot(t, n).permute(0, 3, 1, 2).float() for t, n in zip(targets, model.head_sizes)],
            dim=1,
        )
        picked = [t[0].numpy() for t in model.pick(logits)]
        cond = blank_entities(factory)[0].numpy().astype(np.int64)
        world = legalize(cond, picked[0], picked[1], picked[2] if io == "token" else None, np.ones(picked[0].shape))
        assert np.array_equal(world, factory.world_CWH.numpy()), kind


class TestLegalize:
    def _asm_factory(self):
        return _factory(size=15, kind=LessonKind.SAME_SIDE_1IN, seed=0)

    def test_assembler_fragment_is_dropped(self):
        """A lone assembler tile must not reach the simulator, which would
        score it as a whole machine."""
        factory = self._asm_factory()
        _, (typ, rec, part) = _tokens(factory.world_CWH)
        x, y = map(int, np.argwhere(factory.world_CWH.numpy()[0] == _ASM)[0])
        frag = np.zeros_like(typ)
        frag[x, y] = typ[x, y]
        cond = blank_entities(factory)[0].numpy().astype(np.int64)
        world = legalize(cond, frag, rec, part, np.ones(typ.shape))
        assert not (world[0] == _ASM).any()

    def test_majority_of_a_footprint_places_the_whole_machine(self):
        factory = self._asm_factory()
        solved = factory.world_CWH.numpy()
        _, (typ, rec, part) = _tokens(solved)
        x, y = map(int, np.argwhere(solved[0] == _ASM)[0])  # an anchor (min corner)
        holes = typ.copy()
        holes[x, y] = holes[x + 1, y + 1] = holes[x + 2, y] = 0
        cond = blank_entities(factory)[0].numpy().astype(np.int64)
        world = legalize(cond, holes, rec, part, np.ones(typ.shape))
        assert np.array_equal(world, solved)


class TestCorrupt:
    def test_levels_zero_and_one(self):
        state = torch.stack([torch.from_numpy(world_to_state(_factory(size=9, kind=LessonKind.MOVE_ONE_ITEM, seed=s).world_CWH)) for s in range(4)])
        free = free_mask(state)
        clean = state_to_tokens(state)
        prior = torch.full((len(TYPES),), 1.0 / len(TYPES))
        kept = corrupt(clean, free, torch.zeros(4), prior)
        assert all(torch.equal(a, b) for a, b in zip(kept, clean))
        noisy = corrupt(clean, free, torch.ones(4), prior)
        noisy_state = tokens_to_state(state, free, *noisy)
        assert torch.equal(noisy_state[~free.unsqueeze(1).expand_as(noisy_state)], state.long()[~free.unsqueeze(1).expand_as(noisy_state)])
        assert (noisy[0][free] != clean[0][free]).any()


@pytest.mark.parametrize("size", [7, 15, 22])
def test_model_is_fully_convolutional(size):
    state = torch.zeros(2, STATE_CHANNELS, size, size, dtype=torch.long)
    state[:, 4] = 1
    logits = DenoiserUNet(**TINY)(state)
    assert logits.shape[2:] == (size, size)


@pytest.mark.parametrize("seed", range(4))
def test_legalized_noise_is_buildable_through_the_env(seed):
    """Whatever token grid it gets, `legalize` returns a world the env's own
    placement actions rebuild exactly, scored at the same throughput: the
    denoiser is never credited for a factory the agent couldn't place."""
    from ppo import FactorioEnv, _CH_DIR, _CH_ENT, _CH_ITEMS, _CH_MISC
    from denoise import noise_tokens, part_map, score_world

    kind = LessonKind.SPLITTER_1IN
    factory = _factory(15, kind, seed)
    cond = blank_entities(factory)[0].numpy().astype(np.int64)
    prior = torch.full((len(TYPES),), 1.0 / len(TYPES))
    typ, rec, part = (t[0].numpy() for t in noise_tokens(prior, (1, 15, 15), torch.Generator().manual_seed(seed)))
    world = legalize(cond, typ, rec, part, np.ones(typ.shape))

    env = FactorioEnv(size=15)
    env.reset(seed=seed, options={"kind": kind, "num_missing_entities": 225})
    anchors = (world[_CH_ENT] != 0) & (cond[_CH_ENT] == 0) & (part_map(world) == 0)
    info = {}
    for x, y in zip(*np.nonzero(anchors)):
        c = world[:, x, y]
        *_, info = env.step({"xy": np.array([x, y]), "entity": int(c[_CH_ENT]), "direction": int(c[_CH_DIR]),
                             "item": int(c[_CH_ITEMS]), "misc": int(c[_CH_MISC])})
        assert not any(info["invalid_reason"].values())
    assert np.array_equal(env._world_CWH.numpy().astype(np.int64), world)
    assert info["thput_normed"] == score_world(world, factory.max_throughput, 0.0)["thput"]


def test_generate_never_edits_the_markers():
    factory = _factory(size=11, kind=LessonKind.MOVE_ONE_ITEM, seed=1)
    cond = torch.from_numpy(world_to_state(blank_entities(factory)[0])).long()[None].repeat(3, 1, 1, 1)
    toks, conf, converged, n_refine = generate(DenoiserUNet(**TINY).eval(), cond, steps=2, refine=2)
    state = tokens_to_state(cond, free_mask(cond), *toks)
    fixed = ~free_mask(cond)
    assert torch.equal(state[:, 0][fixed], cond[:, 0][fixed])
    assert conf.shape == (3, 11, 11) and converged.shape == n_refine.shape == (3,)


def test_eval_pools_trials_apart_and_records_each_factory():
    kinds = [LessonKind.MOVE_ONE_ITEM, LessonKind.TRIAL_RECIPE_TREE_DEPTH_1]
    factories = [(s, k, _factory(size=15, kind=k, seed=s)) for k in kinds for s in range(2)]
    records: list[dict] = []
    out = run_denoise_eval(
        DenoiserUNet(**TINY), factories, torch.device("cpu"), best_of=3, steps=1, refine=1,
        temperature=1.0, cost_scale=0.001, seed=0, records=records,
    )
    assert set(out) >= {"thput@1", "thput@3", "trial_thput@3", "per_kind@3", "converged_frac"}
    assert out["n_trials"] == 2
    assert set(out["per_kind@3"]) == {k.name for k in kinds}
    assert out["thput@3"] >= out["thput@1"]
    assert [(r["seed"], r["kind"]) for r in records] == [(s, k.name) for s, k, _ in factories]
