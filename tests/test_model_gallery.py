"""End-to-end tests for the static model-gallery generator (scripts/model_gallery.py).

Uses a freshly-initialised (untrained) AgentCNN saved to a temp .pt so the tests
exercise the real checkpoint-load + rollout + render path without needing W&B or a
trained model. We assert on page *structure*, not on what the random model builds.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import torch

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

import model_gallery  # noqa: E402
from model_gallery import Args, SchemaMismatch, generate_gallery, load_agent  # noqa: E402
from ppo import AgentCNN, make_env  # noqa: E402

SIZE = 8


def _make_agent(layers=(16, 16, 16), kernel_size=3, cat_embed_dim=8) -> AgentCNN:
    env_id = "factorion/FactorioEnv-v0-galtest"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, SIZE, "galtest")])
    try:
        return AgentCNN(
            envs, layers=layers, kernel_size=kernel_size, cat_embed_dim=cat_embed_dim
        )
    finally:
        envs.close()


def _random_checkpoint(tmp_path) -> str:
    """Save a random-init AgentCNN sized for SIZE and return its .pt path."""
    path = str(tmp_path / "rand.pt")
    torch.save(_make_agent().state_dict(), path)
    return path


def test_generate_gallery_multipage(tmp_path):
    ckpt = _random_checkpoint(tmp_path)
    args = Args(
        checkpoint=ckpt,
        wandb_run=None,
        size=SIZE,
        seeds_per_lesson=2,
        lessons=["MOVE_ONE_ITEM", "SPLITTER_SPLIT"],
        output=str(tmp_path / "gallery"),
    )
    pages = generate_gallery(args)

    # One index page plus one page per requested lesson.
    assert set(pages) == {"index.html", "MOVE_ONE_ITEM.html", "SPLITTER_SPLIT.html"}

    index = pages["index.html"]
    # Index links to each lesson page and carries the per-lesson summary table.
    assert 'href="MOVE_ONE_ITEM.html"' in index
    assert 'href="SPLITTER_SPLIT.html"' in index
    assert "mean thput" in index
    # The index is the lightweight landing page — no factory grids on it.
    assert "grid-label" not in index.replace(model_gallery._PAGE_CSS, "")

    lesson = pages["MOVE_ONE_ITEM.html"]
    # Two factories, each rendered as problem / model build / ground truth.
    assert lesson.count("PROBLEM".lower()) or 'class="grid-label">problem<' in lesson
    assert lesson.count('class="grid-label">model build<') == 2
    assert lesson.count('class="grid-label">ground truth<') == 2
    assert '<a href="index.html">' in lesson  # back link


def test_icons_are_deduped(tmp_path):
    """The whole point of the dedup pass: no inline PNG survives in the output,
    every icon is a shared `content: url(...)` CSS rule instead."""
    ckpt = _random_checkpoint(tmp_path)
    args = Args(
        checkpoint=ckpt,
        wandb_run=None,
        size=SIZE,
        seeds_per_lesson=2,
        lessons=["MOVE_ONE_ITEM"],
        output=str(tmp_path / "gallery"),
    )
    lesson = generate_gallery(args)["MOVE_ONE_ITEM.html"]
    assert "src='data:image/png;base64," not in lesson  # no inline PNGs left
    assert "content:url(data:image/png;base64," in lesson  # icons hoisted to CSS


def test_pages_are_written_to_disk(tmp_path):
    ckpt = _random_checkpoint(tmp_path)
    out = tmp_path / "site"
    model_gallery.main(
        Args(
            checkpoint=ckpt,
            wandb_run=None,
            size=SIZE,
            seeds_per_lesson=1,
            lessons=["MOVE_ONE_ITEM"],
            output=str(out),
        )
    )
    assert (out / "index.html").exists()
    assert (out / "MOVE_ONE_ITEM.html").exists()


def test_nondefault_arch_is_inferred(tmp_path):
    """A checkpoint with non-default arch hyperparameters (depth, widths,
    kernel, embedding dim) loads without being told them — load_agent recovers
    the whole architecture from the tensor shapes."""
    agent = _make_agent(layers=(24, 32), kernel_size=5, cat_embed_dim=4)
    ckpt = str(tmp_path / "arch.pt")
    torch.save(agent.state_dict(), ckpt)

    arch = model_gallery._infer_arch(agent.state_dict())
    assert arch.layers == [24, 32]
    assert arch.kernel_size == 5
    assert arch.cat_embed_dim == 4

    # And it round-trips through the full generator (no arch args passed).
    pages = generate_gallery(
        Args(
            checkpoint=ckpt,
            wandb_run=None,
            size=SIZE,
            seeds_per_lesson=1,
            lessons=["MOVE_ONE_ITEM"],
            output=str(tmp_path / "g"),
        )
    )
    assert "MOVE_ONE_ITEM.html" in pages


def test_schema_mismatch_raises_clear_error(tmp_path):
    """A checkpoint whose catalog-sized tensors don't match the current env
    (simulated by shrinking the entity head) raises SchemaMismatch naming the
    diverging tensor — not a raw torch shape-error."""
    state = _make_agent().state_dict()
    # Chop the entity head down as if trained on a smaller entity catalog.
    state["ent_head.weight"] = state["ent_head.weight"][:-3].clone()
    state["ent_head.bias"] = state["ent_head.bias"][:-3].clone()

    with pytest.raises(SchemaMismatch) as excinfo:
        load_agent(state, SIZE, torch.device("cpu"))
    assert "ent_head" in str(excinfo.value)


def test_remap_vocab_rows_preserves_prefix_and_tail():
    """The append-only remap: old real-item rows land on the same indices, the
    trailing Source/Sink pair moves to the new tail, and grown rows keep init."""
    old = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)  # 3 real + Sink + Source
    template = torch.full((8, 2), -1.0)  # 6 real + Sink + Source
    out = model_gallery._remap_vocab_rows(old, template, has_sink_source=True)
    assert torch.equal(out[:3], old[:3])       # real items keep their indices
    assert torch.equal(out[-2], old[3])        # Sink → new tail
    assert torch.equal(out[-1], old[4])        # Source → new tail
    assert torch.equal(out[3:6], template[3:6])  # never-seen items keep init

    # No Source/Sink tail (the entity head): pure prefix copy.
    out2 = model_gallery._remap_vocab_rows(old, template, has_sink_source=False)
    assert torch.equal(out2[:5], old)
    assert torch.equal(out2[5:], template[5:])


def test_catalog_remap_gate(tmp_path):
    """A checkpoint with a smaller (append-only) catalog is rejected by default
    with a hint, and loads when allow_catalog_remap=True."""
    state = _make_agent().state_dict()
    # Simulate an older, smaller catalog by dropping interior real-item rows
    # from every vocab tensor (keeping the trailing Source/Sink pair where present).
    for key, has_ss in model_gallery._VOCAB_TENSORS.items():
        t = state[key]
        if has_ss:
            state[key] = torch.cat([t[:-4], t[-2:]]).clone()  # drop 2 real rows
        else:
            state[key] = t[:-2].clone()

    with pytest.raises(SchemaMismatch) as excinfo:
        load_agent(state, SIZE, torch.device("cpu"))
    assert "allow_catalog_remap" in str(excinfo.value)  # the actionable hint

    agent = load_agent(state, SIZE, torch.device("cpu"), allow_catalog_remap=True)
    assert agent is not None  # remap path loads without raising


def test_recipe_label_reads_assembler_recipe():
    """_recipe_label surfaces the assembler's tagged recipe (or the sink item)."""
    from factorion import Channel, build_factory

    factory = build_factory(
        size=11, kind=model_gallery.LessonKind.MEMORISE_1_INGREDIENT_RECIPES, seed=1
    )
    assert factory is not None
    world_WHC = factory.world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
    label = model_gallery._recipe_label(world_WHC)
    # This lesson always tags a real recipe on its assembler, so a non-empty,
    # known item name must come back.
    assert label
    assert isinstance(label, str)
    # Sanity: the returned name is a real item that appears in the ITEMS channel.
    from factorion import items

    present = {items[v].name for v in np.unique(world_WHC[:, :, Channel.ITEMS.value])}
    assert label in present
