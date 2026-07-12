"""Generate a static, self-contained HTML gallery of a model's factory completions.

The problem this solves: after training, a W&B metric like ``eval/MOVE_ONE_ITEM/thput
= 0.88`` is hard to interpret — you want to *see* what the model actually builds.
The live ``scripts/factory_builder.py`` UI answers that, but it needs a laptop, a
running server, and per-click inference. This script bakes the same information into
a single static page you can host and open from anywhere.

For each ``(lesson, seed)`` it:
  1. builds the ground-truth factory,
  2. blanks it to an *empty* grid and has the model greedily rebuild it (the same
     rollout SFT/PPO use for ``eval/thput``),
  3. renders **problem → model build → ground truth** as HTML tables plus the
     throughput the model achieved.

Inference runs once, here, at generation time. The output is one static ``.html``
file — every icon inlined as base64, no JS, no server. That's what makes hosting on
GitHub Pages work: an Action runs this (it has the compute), Pages just serves the
finished file.

Usage:
    uv run python scripts/model_gallery.py --wandb-run j0s5y2mc --output gallery/index.html
    uv run python scripts/model_gallery.py --checkpoint sft.pt --seeds-per-lesson 5
    uv run python scripts/model_gallery.py --lessons MOVE_ONE_ITEM SPLITTER_SPLIT
"""

from __future__ import annotations

import dataclasses
import datetime
import html as html_lib
import os
import re
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    LessonKind,
    build_factory,
    entities,
    items,
    world2html,
)
from ppo import (  # noqa: E402
    AgentCNN,
    FactorioEnv,
    _resolve_wandb_checkpoint,
    make_env,
)
from sft import _apply_legal_tile_mask  # noqa: E402


@dataclasses.dataclass
class Args:
    checkpoint: Optional[str] = None
    """Local path to a trained SFT/PPO checkpoint (.pt). Takes precedence
    over --wandb-run when set."""
    wandb_run: Optional[str] = "j0s5y2mc"
    """W&B run id (or full 'entity/project/run_id'). The run's most recent
    model-type artifact is downloaded and loaded. Ignored if --checkpoint is
    given. Defaults to the canonical SFT base j0s5y2mc."""
    wandb_project: str = "factorion"
    """W&B project to look in when --wandb-run is a bare id."""
    wandb_entity: Optional[str] = None
    """W&B entity (team or user). None = your default entity."""
    size: int = 11
    """Grid size (NxN). Should match the checkpoint's training size."""
    seeds_per_lesson: int = 5
    """How many distinct factories to show per lesson kind."""
    lessons: Optional[list[str]] = None
    """Subset of LessonKind names to include (e.g. MOVE_ONE_ITEM). None = all."""
    seed_start: int = 1_000
    """First seed searched per lesson. A fixed high base keeps the gallery's
    factories disjoint from the training seeds and stable across regenerations."""
    seed_budget: int = 5_000
    """How many consecutive seeds to try per lesson before giving up (some
    (size, kind, seed) combos fail build_factory's rejection sampler)."""
    output: str = "gallery"
    """Directory to write the static site into (index.html + one page per lesson)."""


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclasses.dataclass
class _Arch:
    """The AgentCNN architecture hyperparameters recovered from a checkpoint's
    tensor shapes — everything needed to rebuild the exact same network without
    a sidecar, so tuning widths/depth/kernel/embedding dim between runs stays
    transparent to this loader."""

    layers: list[int]        # per-conv-layer channel widths (depth = len)
    kernel_size: int         # conv kernel size (square)
    cat_embed_dim: int       # entity/item embedding width (0 = no embeddings)
    in_channels: int         # channels the first conv consumes (the encoder input)


def _infer_arch(state: dict) -> _Arch:
    """Recover the full AgentCNN architecture from a checkpoint's shapes.

    Conv layers are located by 4-D weight shape and sorted by index (robust to
    interleaved Dropout2d shifting encoder.0/2/4 → 0/3/6). The embedding width
    comes from `ent_embed.weight`; the first conv's in-channels tells us what
    input encoding it was trained for (raw channels vs the categorical
    expansion), which is how we later detect an incompatible obs schema."""
    conv_keys = sorted(
        (
            k
            for k, v in state.items()
            if k.startswith("encoder.") and k.endswith(".weight") and v.dim() == 4
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    if not conv_keys:
        raise ValueError("no encoder conv weights found in checkpoint")
    layers = [int(state[k].shape[0]) for k in conv_keys]
    kernel_size = int(state[conv_keys[0]].shape[-1])
    in_channels = int(state[conv_keys[0]].shape[1])
    embed = state.get("ent_embed.weight")
    cat_embed_dim = int(embed.shape[1]) if embed is not None else 0
    return _Arch(layers, kernel_size, cat_embed_dim, in_channels)


def resolve_checkpoint(args: Args) -> tuple[dict, dict]:
    """Return (state_dict, source) for the checkpoint named by args.

    ``source`` is provenance for the page header: {"kind": "local", "path"} or
    {"kind": "wandb", "run_id", "run_url", "run_name", "artifact"}. Checkpoints
    saved by a torch.compile'd PPO agent carry an "_orig_mod." prefix on every
    key; we strip it so SFT and PPO checkpoints load identically."""
    if args.checkpoint:
        path = args.checkpoint
        source = {"kind": "local", "path": path}
    elif args.wandb_run:
        path, source = _resolve_wandb_checkpoint(
            args.wandb_run, args.wandb_project, args.wandb_entity
        )
    else:
        raise ValueError("provide either --checkpoint or --wandb-run")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    return state, source


class SchemaMismatch(RuntimeError):
    """A checkpoint that can't be faithfully loaded because it was trained
    against a different *environment* schema (item/entity catalog or input
    encoding), not merely different architecture hyperparameters. The message
    names exactly what diverged so the fix ('retrain on current code') is
    obvious — as opposed to a raw torch shape-mismatch traceback."""


def load_agent(state: dict, size: int, device: torch.device) -> AgentCNN:
    """Build an AgentCNN matching the checkpoint's architecture and load it.

    Architecture hyperparameters (conv widths, depth, kernel size, embedding
    dim) are inferred from the checkpoint via `_infer_arch`, so tuning them
    between runs needs no changes here — hand it a run id and it self-configures.

    What inference *can't* fix is an environment-schema change: the item/entity
    catalog growing (embedding/head vocab sizes) or the observation encoding
    changing (first-conv in-channels). Those shift what each learned index
    *means*, so loading anyway would render a faithful-looking but wrong gallery.
    We detect that case and raise SchemaMismatch instead.

    The critic head is never used at inference and its flat dim depends on W*H,
    so it's dropped; the eot head is kept only when its saved shape matches this
    grid size (we rely on it to report where the model signals 'done')."""
    arch = _infer_arch(state)
    env_id = "factorion/FactorioEnv-v0-gallery"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, size, "gallery")])
    try:
        agent = AgentCNN(
            envs,
            layers=arch.layers,
            kernel_size=arch.kernel_size,
            cat_embed_dim=arch.cat_embed_dim or 8,
        )
    finally:
        envs.close()

    # critic_head / eot_head flat dims depend on W*H, so they legitimately
    # mismatch when the gallery grid size differs from training — drop those
    # (critic always: unused at inference; eot only on a genuine size mismatch).
    expected_flat = arch.layers[-1] * size * size
    saved_eot_w = state.get("eot_head.1.weight")
    keep_eot = saved_eot_w is not None and saved_eot_w.shape[1] == expected_flat
    drop_prefixes: tuple[str, ...] = ("critic_head.",)
    if not keep_eot:
        drop_prefixes = drop_prefixes + ("eot_head.",)
    filtered = {k: v for k, v in state.items() if not k.startswith(drop_prefixes)}

    # Any *remaining* shape mismatch is a schema divergence inference can't
    # bridge (catalog vocab or obs encoding), so surface it clearly rather than
    # letting load_state_dict throw a wall of per-tensor size errors.
    model_sd = agent.state_dict()
    mismatches = [
        (k, tuple(v.shape), tuple(model_sd[k].shape))
        for k, v in filtered.items()
        if k in model_sd and tuple(model_sd[k].shape) != tuple(v.shape)
    ]
    if mismatches:
        details = "\n".join(
            f"  {k}: checkpoint {ckpt} vs current model {cur}"
            for k, ckpt, cur in mismatches
        )
        raise SchemaMismatch(
            "checkpoint was trained against a different environment schema than "
            "the current code — architecture hyperparameters are inferred "
            "automatically, but catalog/observation-encoding changes can't be "
            "reconciled (loading anyway would misrender). Retrain on the current "
            f"code to visualise it. Diverging tensors:\n{details}"
        )

    agent.load_state_dict(filtered, strict=False)
    return agent.to(device).eval()


def pick_seeds(kind: LessonKind, size: int, n: int, seed_start: int, budget: int) -> list[int]:
    """First `n` seeds in [seed_start, seed_start+budget) where build_factory
    succeeds for this (size, kind). Deterministic, so regenerations show the
    same factories."""
    found: list[int] = []
    s = seed_start
    while len(found) < n and s < seed_start + budget:
        if build_factory(size=size, kind=kind, seed=s) is not None:
            found.append(s)
        s += 1
    return found


@dataclasses.dataclass
class Completion:
    """One (lesson, seed) result: the three grids plus rollout metrics."""

    kind: LessonKind
    seed: int
    problem_WHC: np.ndarray
    model_WHC: np.ndarray
    truth_WHC: np.ndarray
    thput: float          # normalized throughput of the built-to-done grid
    thput_eot: float      # throughput at the step the model signalled 'done'
    thput_raw: float      # raw items/s of the built-to-done grid
    max_raw: float        # the ground-truth factory's raw items/s (the ceiling)
    steps: int            # placements taken
    eot_step: Optional[int]  # step the eot head first crossed 0.5, or None


def _to_WHC(world_CWH: torch.Tensor) -> np.ndarray:
    return world_CWH.permute(1, 2, 0).to(torch.int64).cpu().numpy()


def rollout(
    agent: AgentCNN,
    env: FactorioEnv,
    seed: int,
    kind: LessonKind,
    size: int,
    device: torch.device,
) -> Completion:
    """Greedily rebuild the (seed, kind) factory from an empty grid and capture
    the result. Mirrors sft.run_rollout_eval's single-slot logic: argmax every
    head over legal tiles, ignore the eot head for stepping (so we see the full
    build), but snapshot the throughput the first time it crosses 0.5."""
    obs, info = env.reset(
        seed=seed,
        options={"num_missing_entities": size * size, "kind": kind},
    )
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    thput = float(info.get("thput_normed", 0.0))
    thput_raw = float(info.get("thput_raw", 0.0))
    eot_step: Optional[int] = None
    eot_thput = 0.0
    steps = 0
    # Env truncates at its own max_steps, but cap defensively so a model that
    # never places a legal tile can't spin here forever.
    step_cap = size * size * 4

    with torch.no_grad():
        while steps < step_cap:
            encoded = agent.encoder(agent._encode_input(obs_t))
            eot_prob = float(torch.sigmoid(agent.eot_head(encoded).squeeze(-1)).item())
            tile_logits = _apply_legal_tile_mask(
                agent.tile_logits(encoded).reshape(1, -1), obs_t
            )
            if not torch.isfinite(tile_logits).any():
                break  # no legal tile left to place on
            tile_idx = int(tile_logits.argmax(dim=1).item())
            x, y = tile_idx // size, tile_idx % size
            feats = encoded[0, :, x, y].unsqueeze(0)
            ent = int(agent.ent_head(feats).argmax(dim=1).item())
            direction = int(agent.dir_head(feats).argmax(dim=1).item())
            item = int(agent.item_head(feats).argmax(dim=1).item())
            misc = int(agent.misc_head(feats).argmax(dim=1).item())

            if eot_step is None and eot_prob > 0.5:
                eot_step = steps
                eot_thput = thput

            action = {
                "xy": np.array([x, y], dtype=int),
                "entity": ent,
                "direction": direction,
                "item": item,
                "misc": misc,
            }
            next_obs, _r, terminated, truncated, info = env.step(action)
            thput = float(info.get("thput_normed", thput))
            thput_raw = float(info.get("thput_raw", thput_raw))
            steps += 1
            if terminated or truncated:
                break
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

    return Completion(
        kind=kind,
        seed=seed,
        problem_WHC=_to_WHC(env._original_world_CWH),
        model_WHC=_to_WHC(env._world_CWH),
        truth_WHC=_to_WHC(env._solved_world_CWH),
        thput=thput,
        # eot never fired → the model would have built to done, so its
        # eot-respecting throughput equals the final throughput.
        thput_eot=eot_thput if eot_step is not None else thput,
        thput_raw=thput_raw,
        max_raw=float(getattr(env, "_max_throughput", 0.0)),
        steps=steps,
        eot_step=eot_step,
    )


def _recipe_label(world_WHC: np.ndarray) -> str:
    """Best-effort recipe/item name carried by the factory — the recipe tagged
    on an assembler, else the item the sink carries. Empty string if neither."""
    ent_ch = Channel.ENTITIES.value
    item_ch = Channel.ITEMS.value
    W, H, _ = world_WHC.shape
    sink_item = ""
    for x in range(W):
        for y in range(H):
            ent = entities[world_WHC[x, y, ent_ch]].name
            item = items[world_WHC[x, y, item_ch]].name
            if ent == "assembling_machine_1" and item != "empty":
                return item
            if ent == "bulk_inserter" and item != "empty":  # sink
                sink_item = item
    return sink_item


def _color_for_thput(t: float) -> str:
    """Green→amber→red band for a throughput in [0, 1], for the card header."""
    if t >= 0.99:
        return "#1b7f2e"
    if t >= 0.5:
        return "#b8860b"
    return "#b02a2a"


def _card_html(c: Completion) -> str:
    recipe = _recipe_label(c.truth_WHC)
    recipe_html = f" · recipe <code>{html_lib.escape(recipe)}</code>" if recipe else ""
    if c.eot_step is not None:
        eot_html = f"stopped at step {c.eot_step} (thput_eot {c.thput_eot:.2f})"
    else:
        eot_html = "never signalled done"
    color = _color_for_thput(c.thput)
    problem = world2html(c.problem_WHC).text
    model = world2html(c.model_WHC).text
    truth = world2html(c.truth_WHC).text
    return f"""
<div class="card">
  <div class="card-head">
    seed <b>{c.seed}</b>{recipe_html} ·
    thput <b style="color:{color}">{c.thput:.2f}</b>
    ({c.thput_raw:.2f}/{c.max_raw:.2f} items/s) ·
    {c.steps} placements · {eot_html}
  </div>
  <div class="grids">
    <div class="grid-col"><div class="grid-label">problem</div>{problem}</div>
    <div class="grid-col"><div class="grid-label">model build</div>{model}</div>
    <div class="grid-col"><div class="grid-label">ground truth</div>{truth}</div>
  </div>
</div>"""


# world2html inlines a full base64 PNG into every cell's <img src>, so the same
# ~two dozen icons repeat thousands of times across a 70-factory page (tens of
# MiB). We hoist each *unique* data URI into one CSS rule (`content: url(...)`
# swaps the rendered image) and point every cell at it by class, collapsing the
# icon bytes to one copy each. Purely a size optimisation over world2html output.
_DATA_URI_RE = re.compile(r"src='(data:image/png;base64,[^']+)'")
# A 1x1 transparent GIF keeps the <img> valid before CSS `content` replaces it.
_PLACEHOLDER_SRC = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="


def _dedupe_icons(body: str) -> tuple[str, str]:
    """Replace every inline PNG data URI in `body` with a shared CSS class.
    Returns (rewritten_body, css) where css defines one `content: url(...)`
    rule per unique icon."""
    uri_to_class: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        uri = match.group(1)
        cls = uri_to_class.get(uri)
        if cls is None:
            cls = f"ic{len(uri_to_class)}"
            uri_to_class[uri] = cls
        return f"src='{_PLACEHOLDER_SRC}' class='{cls}'"

    rewritten = _DATA_URI_RE.sub(repl, body)
    css = "".join(f".{cls}{{content:url({uri})}}" for uri, cls in uri_to_class.items())
    return rewritten, css


def _source_html(source: dict) -> str:
    if source.get("kind") == "wandb":
        url = source.get("run_url", "")
        rid = source.get("run_id", "")
        name = source.get("run_name", "")
        label = html_lib.escape(f"{name} ({rid})" if name else rid)
        return f'W&B run <a href="{html_lib.escape(url)}">{label}</a>'
    return f'local checkpoint <code>{html_lib.escape(source.get("path", "?"))}</code>'


# Shared page CSS. `{icon_css}` (the deduped per-page icon rules) is appended
# at render time; kept out of the constant so the icon block can vary per page.
_PAGE_CSS = """
  body { font-family: system-ui, sans-serif; margin: 0; color: #1c1c1c; background: #fafafa; }
  header { background: #1c2c3a; color: #fff; padding: 1em 1.2em; }
  header h1 { margin: 0 0 0.2em; font-size: 1.3em; }
  header .meta { font-size: 0.85em; opacity: 0.85; }
  header a { color: #9ecbff; }
  main { max-width: 1400px; margin: 0 auto; padding: 1em 1.2em 4em; }
  table.summary { border-collapse: collapse; margin: 1em 0; font-size: 0.95em; }
  table.summary td, table.summary th { border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: left; }
  table.summary th { background: #eee; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 0.7em; margin: 0.8em 0; }
  .card-head { font-size: 0.9em; margin-bottom: 0.5em; }
  .grids { display: flex; flex-wrap: wrap; gap: 1.2em; align-items: flex-start; }
  .grid-col { display: flex; flex-direction: column; gap: 0.3em; }
  .grid-label { font-size: 0.75em; text-transform: uppercase; color: #666; letter-spacing: 0.05em; }
  code { background: #eee; padding: 0 0.3em; border-radius: 3px; }
  a { color: #1560b0; }
  .back { font-size: 0.85em; }
"""


def _page_shell(title: str, source: dict, generated: str, subtitle: str, body: str, icon_css: str) -> str:
    """Wrap page `body` in the common <html> shell with header + styles."""
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html_lib.escape(title)}</title>
<style>{_PAGE_CSS}{icon_css}</style></head><body>
<header>
  <h1>{html_lib.escape(title)}</h1>
  <div class="meta">{_source_html(source)} · {subtitle} · generated {generated}</div>
</header>
<main>{body}</main>
</body></html>"""


def build_index_page(by_kind: dict[str, list[Completion]], source: dict, generated: str, args: Args) -> str:
    """The landing page: a per-lesson summary table linking to each lesson page.
    No grids here, so it stays tiny and loads instantly on a phone."""
    total = sum(len(cs) for cs in by_kind.values())
    rows = []
    for name in sorted(by_kind):
        cs = by_kind[name]
        mean_thput = sum(c.thput for c in cs) / len(cs)
        color = _color_for_thput(mean_thput)
        rows.append(
            f'<tr><td><a href="{name}.html">{name}</a></td>'
            f'<td style="color:{color};font-weight:bold">{mean_thput:.2f}</td>'
            f"<td>{len(cs)}</td></tr>"
        )
    body = f"""
  <p>Each lesson page shows held-out factories blanked to an empty grid, what the
  model greedily rebuilt, and the ground-truth solution. <b>thput</b> is
  normalized items/s (1.00 = a perfect rebuild). Tap a lesson to see its builds.</p>
  <table class="summary">
    <tr><th>lesson</th><th>mean thput</th><th>factories</th></tr>
    {''.join(rows)}
  </table>"""
    subtitle = f"grid {args.size}×{args.size} · {len(by_kind)} lessons · {total} factories"
    return _page_shell("Factorion model gallery", source, generated, subtitle, body, "")


def build_lesson_page(name: str, completions: list[Completion], source: dict, generated: str, args: Args) -> str:
    """One lesson's page: every factory rendered as problem/model/truth cards.
    Icons are deduped per page (one CSS rule each), so the page weight is
    dominated by cell markup, not repeated base64."""
    mean_thput = sum(c.thput for c in completions) / len(completions)
    cards = "".join(_card_html(c) for c in completions)
    body_inner = f"""
  <p class="back"><a href="index.html">← all lessons</a></p>
  <p>mean thput <b>{mean_thput:.2f}</b> over {len(completions)} factories</p>
  {cards}"""
    body, icon_css = _dedupe_icons(body_inner)
    subtitle = f"grid {args.size}×{args.size} · {len(completions)} factories"
    return _page_shell(f"{name} — model gallery", source, generated, subtitle, body, icon_css)


def generate_gallery(args: Args) -> dict[str, str]:
    """Load the checkpoint, roll the model out over all requested (lesson, seed)
    factories, and return a {filename: html} map for the static site (an
    index.html plus one page per lesson)."""
    device = _resolve_device()
    state, source = resolve_checkpoint(args)
    agent = load_agent(state, args.size, device)

    if args.lessons:
        kinds = [LessonKind[name] for name in args.lessons]
    else:
        kinds = list(LessonKind)

    env = FactorioEnv(size=args.size, idx=0)
    by_kind: dict[str, list[Completion]] = {}
    for kind in kinds:
        seeds = pick_seeds(
            kind, args.size, args.seeds_per_lesson, args.seed_start, args.seed_budget
        )
        if not seeds:
            print(f"[warn] no valid seeds for {kind.name} at size {args.size} — skipping")
            continue
        by_kind[kind.name] = [
            rollout(agent, env, seed, kind, args.size, device) for seed in seeds
        ]
        print(f"{kind.name}: {len(seeds)} factories")

    generated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pages = {"index.html": build_index_page(by_kind, source, generated, args)}
    for name, completions in by_kind.items():
        pages[f"{name}.html"] = build_lesson_page(name, completions, source, generated, args)
    return pages


def main(args: Args) -> None:
    try:
        pages = generate_gallery(args)
    except SchemaMismatch as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for filename, html in pages.items():
        (out_dir / filename).write_text(html)
        total += len(html)
    print(f"Wrote {len(pages)} pages to {out_dir}/ ({total // 1024} KiB total)")


if __name__ == "__main__":
    main(tyro.cli(Args))
