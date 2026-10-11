"""Whole-factory denoiser: a fully-convolutional UNet that reads an entire
(possibly corrupted) factory and predicts the clean factory for every tile at
once.

Each tile's state is three tokens: a *type* (entity × direction × underground
mode, see `TYPES`), a *recipe* (assemblers only) and a *part* — the tile's
offset inside its multi-tile entity, which is what lets two touching
assemblers with the same recipe be told apart. Sources, sinks and walls are
the conditioning: they are never corrupted, never predicted.

Training corrupts a known-correct factory by resampling each free tile from
`noise_prior` with probability t ~ U[0, 1] (t = 0: nothing to fix, t = 1:
pure noise) and always regresses the *clean* factory — x0-prediction, as in
SUNDAE (Savinov et al., "Step-unrolled Denoising Autoencoders for Text
Generation", ICLR 2022), whose unrolled second step `sft.py` also uses. The
net is not told t: like the noise-unconditional denoisers of Sun et al. ("Is
Noise Conditioning Necessary for Denoising Generative Models?", 2025) it has
to judge for itself how broken its input is, so a clean factory maps to
itself and `generate` can stop at the fixed point.
"""

from __future__ import annotations

import functools
import math
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import factorion_rs
from factorion import Direction, Factory, LESSON_IS_TRIAL, LessonKind, Misc, count_dangling_inserters, entities, items
from ppo import (
    _ASM_MACHINE_ENT_ID,
    _CH_DIR,
    _CH_ENT,
    _CH_FOOTPRINT,
    _CH_ITEMS,
    _CH_MISC,
    _EMPTY_ENT_ID,
    _ENTITY_FOOTPRINT_AREAS,
    _ENTITY_UNIT_COSTS,
    _FOOTPRINT_UNAVAILABLE,
    _SINK_ENT_ID,
    _SOURCE_ENT_ID,
    _UG_BELT_ENT_ID,
    _VALID_ENTITY_ACTION_IDS,
    _ASM_RECIPE_ITEM_IDS,
)

# The model's state is the world's channels plus the part map.
CH_PART = _CH_FOOTPRINT + 1
STATE_CHANNELS = CH_PART + 1

TYPES: tuple[tuple[int, int, int], ...] = (
    (_EMPTY_ENT_ID, Direction.NONE.value, Misc.NONE.value),
    *(
        (e, d, m)
        for e in _VALID_ENTITY_ACTION_IDS
        if e != _EMPTY_ENT_ID
        # Generators store a facing on assemblers; NONE is what a placement
        # action writes. Both are legal worlds.
        for d in (range(len(Direction)) if e == _ASM_MACHINE_ENT_ID else range(1, len(Direction)))
        for m in ((Misc.UNDERGROUND_DOWN.value, Misc.UNDERGROUND_UP.value) if e == _UG_BELT_ENT_ID else (Misc.NONE.value,))
    ),
)
RECIPES: tuple[int, ...] = (0, *_ASM_RECIPE_ITEM_IDS)
N_TYPES, N_RECIPES = len(TYPES), len(RECIPES)
# part = dx * 3 + dy, the tile's offset from its entity's min corner (the
# anchor `py_entity_tiles` and the engine's `unit_anchors` both use).
N_PARTS = 9

_TYPES_NP = np.array(TYPES, dtype=np.int64)
_RECIPES_NP = np.array(RECIPES, dtype=np.int64)
_TYPE_LUT_NP = np.full((len(entities), len(Direction), len(Misc)), -1, dtype=np.int64)
for _i, (_e, _d, _m) in enumerate(TYPES):
    _TYPE_LUT_NP[_e, _d, _m] = _i
_RECIPE_LUT_NP = np.zeros(len(items), dtype=np.int64)
_RECIPE_LUT_NP[_RECIPES_NP] = np.arange(N_RECIPES)
_IS_ASM_NP = _TYPES_NP[:, 0] == _ASM_MACHINE_ENT_ID
# Footprint offsets per type, in part order; one (0, 0) row for 1x1 types.
_OFFSETS: list[list[tuple[int, int]]] = []
for _e, _d, _m in TYPES:
    _p = entities[_e]
    _t = factorion_rs.py_entity_tiles(0, 0, _d, _p.width, _p.height) if _p.width * _p.height > 1 else None
    _OFFSETS.append(sorted(tuple(t) for t in _t) if _t else [(0, 0)])
_IS_MULTI_NP = np.array([len(o) > 1 for o in _OFFSETS])
_N_PARTS_OF_NP = np.array([len(o) for o in _OFFSETS], dtype=np.int64)
_PART_TABLE_NP = np.zeros((N_TYPES, N_PARTS), dtype=np.int64)
_PART_VALID_NP = np.zeros((N_TYPES, N_PARTS), dtype=bool)
for _i, _o in enumerate(_OFFSETS):
    for _k, (_dx, _dy) in enumerate(_o):
        _PART_TABLE_NP[_i, _k] = _dx * 3 + _dy
        _PART_VALID_NP[_i, _dx * 3 + _dy] = True
# Per-tile share of a type's entity cost, for the differentiable cost loss.
_TYPE_TILE_COST_NP = _ENTITY_UNIT_COSTS[_TYPES_NP[:, 0]] / _ENTITY_FOOTPRINT_AREAS[_TYPES_NP[:, 0]]


@functools.cache
def _luts(device: torch.device) -> SimpleNamespace:
    """Device copies of the token tables."""
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return SimpleNamespace(
        types=t(_TYPES_NP), recipes=t(_RECIPES_NP), type_lut=t(_TYPE_LUT_NP), recipe_lut=t(_RECIPE_LUT_NP),
        is_asm=t(_IS_ASM_NP), n_parts=t(_N_PARTS_OF_NP), part_table=t(_PART_TABLE_NP),
        part_valid=t(_PART_VALID_NP), tile_cost=t(_TYPE_TILE_COST_NP / _TYPE_TILE_COST_NP.max()).float(),
    )


def part_map(world_CWH) -> np.ndarray:
    """(W, H) part index of every tile of a valid world (0 for 1x1 / empty)."""
    w = np.asarray(world_CWH, dtype=np.int64)
    ent, dirs = w[_CH_ENT], w[_CH_DIR]
    out = np.zeros(ent.shape, dtype=np.int64)
    claimed = np.zeros(ent.shape, dtype=bool)
    # Raster order reaches each multi-tile entity at its min corner first,
    # exactly as the engine's `unit_anchors` groups them.
    for x, y in zip(*np.nonzero(_ENTITY_FOOTPRINT_AREAS[ent] > 1)):
        if claimed[x, y]:
            continue
        p = entities[int(ent[x, y])]
        for tx, ty in factorion_rs.py_entity_tiles(int(x), int(y), int(dirs[x, y]), p.width, p.height) or []:
            if 0 <= tx < ent.shape[0] and 0 <= ty < ent.shape[1]:
                claimed[tx, ty] = True
                out[tx, ty] = (tx - x) * 3 + (ty - y)
    return out


def world_to_state(world_CWH) -> np.ndarray:
    """(STATE_CHANNELS, W, H) uint8 model state of a valid world."""
    w = np.asarray(world_CWH, dtype=np.int64)
    return np.concatenate([w, part_map(w)[None]]).astype(np.uint8)


def free_mask(state):
    """Tiles the model owns: buildable and not a source/sink."""
    ent = state[:, _CH_ENT]
    return (state[:, _CH_FOOTPRINT] != _FOOTPRINT_UNAVAILABLE) & (ent != _SOURCE_ENT_ID) & (ent != _SINK_ENT_ID)


def state_to_tokens(state):
    """(type, recipe, part) token grids of a (B, STATE_CHANNELS, W, H) state.
    Off-vocabulary tiles (only ever fixed ones) map to type 0."""
    L = _luts(state.device)
    s = state.long()
    typ = L.type_lut[s[:, _CH_ENT], s[:, _CH_DIR], s[:, _CH_MISC]].clamp(min=0)
    return typ, L.recipe_lut[s[:, _CH_ITEMS]], s[:, CH_PART]


def tokens_to_state(cond, free, typ, recipe, part):
    """`cond` with every free tile replaced by its tokens."""
    L = _luts(cond.device)
    edm = L.types[typ]
    tok = torch.stack(
        [edm[..., 0], edm[..., 1], L.recipes[recipe], edm[..., 2], cond[:, _CH_FOOTPRINT].long(), part], dim=1
    )
    return torch.where(free.unsqueeze(1), tok, cond.long())


def noise_tokens(prior, shape, generator=None):
    """Tokens drawn from the noise distribution: a type from `prior`, then a
    uniform recipe (assemblers) and a uniform part of that type's footprint."""
    L = _luts(prior.device)
    n = math.prod(shape)
    typ = torch.multinomial(prior, n, replacement=True, generator=generator).view(shape)
    u = torch.rand((2, *shape), device=prior.device, generator=generator)
    recipe = torch.where(L.is_asm[typ], 1 + (u[0] * (N_RECIPES - 1)).long(), 0)
    part = L.part_table[typ, (u[1] * L.n_parts[typ]).long()]
    return typ, recipe, part


def corrupt(tokens, free, t, prior, generator=None):
    """Resample each free tile from the noise distribution with probability
    `t` (per sample, shape (B,))."""
    typ = tokens[0]
    hit = free & (torch.rand(typ.shape, device=typ.device, generator=generator) < t.view(-1, 1, 1))
    noise = noise_tokens(prior, typ.shape, generator)
    return tuple(torch.where(hit, n, x) for n, x in zip(noise, tokens))


def _pick(logits, temperature, generator, dim=1):
    if temperature == 0:
        return logits.argmax(dim)
    u = torch.rand(logits.shape, device=logits.device, generator=generator).clamp_(1e-20, 1.0)
    return (logits / temperature - (-u.log()).log()).argmax(dim)


def split_logits(logits):
    """The (type, recipe, part) head logits."""
    return logits[:, :N_TYPES], logits[:, N_TYPES : N_TYPES + N_RECIPES], logits[:, N_TYPES + N_RECIPES :]


def pick_tokens(logits, temperature=0.0, generator=None):
    """Sample (temperature > 0) or argmax tokens from the head logits: the
    type first, then a recipe and a part consistent with it."""
    L = _luts(logits.device)
    tl, rl, pl = split_logits(logits)
    typ = _pick(tl, temperature, generator)
    is_asm = L.is_asm[typ]
    recipe = torch.where(is_asm, 1 + _pick(rl[:, 1:], temperature, generator), 0)
    pl = pl.masked_fill(~L.part_valid[typ].permute(0, 3, 1, 2), float("-inf"))
    part = _pick(pl, temperature, generator)
    return typ, recipe, part


class _ResBlock(nn.Module):
    def __init__(self, cin, cout, dropout):
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(32, cin), cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(math.gcd(32, cout), cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        # Each block starts as the identity, so depth doesn't destabilise init.
        for p in self.conv2.parameters():
            nn.init.zeros_(p)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        return self.skip(x) + self.conv2(self.drop(F.silu(self.norm2(h))))


def _rope(t, cos, sin):
    t1, t2 = t[..., 0::2], t[..., 1::2]
    return torch.stack([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1).flatten(-2).to(t.dtype)


class _AttnBlock(nn.Module):
    """Pre-norm transformer block over grid cells with 2-D rotary position
    embeddings (half of each head rotates with x, half with y): attention
    sees relative offsets only, so it is as size-agnostic as the convs."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_out = nn.Linear(4 * dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), mlp_out)
        self.dropout = dropout
        for p in (*self.proj.parameters(), *mlp_out.parameters()):
            nn.init.zeros_(p)

    def forward(self, x, cos, sin):
        B, N, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, N, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(
            _rope(q, cos, sin), _rope(k, cos, sin), v, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.norm2(x))


class DenoiserUNet(nn.Module):
    """UNet over the factory grid with self-attention at the bottleneck.

    `levels` resolutions (channels doubling, stride-2 downsamples), `blocks`
    residual blocks per level and side, and `attn_layers` rotary transformer
    blocks over the bottleneck cells. Grids are padded with wall tiles to a
    multiple of the total stride and cropped back, so any size runs.
    """

    def __init__(self, width=128, levels=2, blocks=2, attn_layers=4, attn_heads=8, dropout=0.0, embed_dim=16):
        super().__init__()
        self.arch = dict(
            width=width, levels=levels, blocks=blocks, attn_layers=attn_layers, attn_heads=attn_heads,
            dropout=dropout, embed_dim=embed_dim,
        )
        self.amp = False
        self.stride = 2 ** (levels - 1)
        self.ent_embed = nn.Embedding(len(entities), embed_dim)
        self.item_embed = nn.Embedding(len(items), embed_dim)
        self.part_embed = nn.Embedding(N_PARTS, embed_dim)
        in_ch = 3 * embed_dim + len(Direction) + len(Misc) + 2
        self.stem = nn.Conv2d(in_ch, width, 3, padding=1)
        chans = [width * 2**i for i in range(levels)]
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        c = width
        for i, ch in enumerate(chans):
            self.enc.append(nn.Sequential(*(_ResBlock(c if j == 0 else ch, ch, dropout) for j in range(blocks))))
            c = ch
            if i < levels - 1:
                self.down.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        if c // attn_heads % 4:
            raise ValueError(f"bottleneck head dim {c}/{attn_heads} must be a multiple of 4 for 2-D RoPE")
        self.mid_in = _ResBlock(c, c, dropout)
        self.attn = nn.ModuleList(_AttnBlock(c, attn_heads, dropout) for _ in range(attn_layers))
        self.mid_out = _ResBlock(c, c, dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(levels)):
            ch = chans[i]
            self.up.append(nn.Conv2d(c, ch, 3, padding=1) if i < levels - 1 else nn.Identity())
            self.dec.append(nn.Sequential(*(_ResBlock(2 * ch if j == 0 else ch, ch, dropout) for j in range(blocks))))
            c = ch
        self.head = nn.Sequential(
            nn.GroupNorm(math.gcd(32, c), c), nn.SiLU(), nn.Conv2d(c, N_TYPES + N_RECIPES + N_PARTS, 1)
        )
        # The noise distribution is part of the model: generation must start
        # from the same distribution training corrupted with.
        self.noise_prior: torch.Tensor
        self.register_buffer("noise_prior", torch.full((N_TYPES,), 1.0 / N_TYPES))

    def _embed(self, s):
        fixed = ~free_mask(s)
        return torch.cat(
            [
                self.ent_embed(s[:, _CH_ENT]).permute(0, 3, 1, 2),
                self.item_embed(s[:, _CH_ITEMS]).permute(0, 3, 1, 2),
                self.part_embed(s[:, CH_PART].clamp(0, N_PARTS - 1)).permute(0, 3, 1, 2),
                F.one_hot(s[:, _CH_DIR], len(Direction)).permute(0, 3, 1, 2).float(),
                F.one_hot(s[:, _CH_MISC], len(Misc)).permute(0, 3, 1, 2).float(),
                s[:, _CH_FOOTPRINT : _CH_FOOTPRINT + 1].float(),
                fixed.unsqueeze(1).float(),
            ],
            dim=1,
        )

    def forward(self, state):
        """(B, STATE_CHANNELS, W, H) state -> (B, N_TYPES + N_RECIPES + N_PARTS, W, H)
        fp32 logits; split them with `split_logits`."""
        s = state.long()
        W, H = s.shape[2:]
        pw, ph = -W % self.stride, -H % self.stride
        # Zero padding is a wall tile: the off-grid world is unbuildable.
        s = F.pad(s, (0, ph, 0, pw))
        with torch.autocast(s.device.type, dtype=torch.bfloat16, enabled=self.amp):
            h = self.stem(self._embed(s))
            skips = []
            for i, blocks in enumerate(self.enc):
                h = blocks(h)
                skips.append(h)
                if i < len(self.down):
                    h = self.down[i](h)
            h = self.mid_in(h)
            B, C, w, hh = h.shape
            cos, sin = _rope_tables(w, hh, C // self.arch["attn_heads"], h.device)
            t = h.flatten(2).transpose(1, 2)
            for blk in self.attn:
                t = blk(t, cos, sin)
            h = t.transpose(1, 2).reshape(B, C, w, hh)
            h = self.mid_out(h)
            for up, blocks, skip in zip(self.up, self.dec, reversed(skips)):
                if not isinstance(up, nn.Identity):
                    h = up(F.interpolate(h, scale_factor=2, mode="nearest"))
                h = blocks(torch.cat([h, skip], dim=1))
            out = self.head(h)
        return out[:, :, :W, :H].float()


def _rope_tables(w, h, head_dim, device):
    q = head_dim // 4
    inv = 1.0 / (100.0 ** (torch.arange(q, device=device, dtype=torch.float32) / q))
    xs = torch.arange(w, device=device, dtype=torch.float32).repeat_interleave(h)
    ys = torch.arange(h, device=device, dtype=torch.float32).repeat(w)
    ang = torch.cat([xs[:, None] * inv, ys[:, None] * inv], dim=1)
    return ang.cos(), ang.sin()


def load_denoiser(path, device="cpu") -> DenoiserUNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = DenoiserUNet(**ckpt["arch"])
    model.load_state_dict(ckpt["model"])
    return model.to(device)


# ── Turning tokens into a real factory ─────────────────────────────────


def legalize(cond_CWH: np.ndarray, typ: np.ndarray, recipe: np.ndarray, part: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """The valid (C, W, H) world closest to a token grid.

    Every multi-tile tile votes for the anchor its part implies; an entity is
    placed where a strict majority of its footprint agrees and every footprint
    tile is free and unclaimed, strongest vote first (its recipe/facing by
    vote). 1x1 tokens fill the remaining free tiles; multi-tile tokens no
    entity claimed become empty. The simulator would otherwise score a stray
    assembler tile as a whole machine."""
    world = cond_CWH.copy()
    ent = world[_CH_ENT]
    W, H = ent.shape
    free = (world[_CH_FOOTPRINT] != _FOOTPRINT_UNAVAILABLE) & (ent != _SOURCE_ENT_ID) & (ent != _SINK_ENT_ID)
    claimed = np.zeros((W, H), dtype=bool)
    votes: dict[tuple, list] = {}
    for x, y in zip(*np.nonzero(free & _IS_MULTI_NP[typ])):
        ty, p = int(typ[x, y]), int(part[x, y])
        e = int(_TYPES_NP[ty, 0])
        ax, ay = int(x) - p // 3, int(y) - p % 3
        # Splitters' facing decides their footprint, an assembler's doesn't.
        key = (ax, ay, e, -1 if e == _ASM_MACHINE_ENT_ID else ty)
        c = float(conf[x, y])
        vote = votes.setdefault(key, [0, 0.0, {}, {}])
        vote[0] += 1
        vote[1] += c
        vote[2][ty] = vote[2].get(ty, 0.0) + c
        r = int(recipe[x, y])
        vote[3][r] = vote[3].get(r, 0.0) + c
    for (ax, ay, e, _), (count, score, tys, rs) in sorted(votes.items(), key=lambda kv: -kv[1][1]):
        ty = max(tys, key=tys.__getitem__)
        offs = _OFFSETS[ty]
        if 2 * count <= len(offs):
            continue
        tiles = [(ax + dx, ay + dy) for dx, dy in offs]
        if not all(0 <= u < W and 0 <= v < H and free[u, v] and not claimed[u, v] for u, v in tiles):
            continue
        r = max(rs, key=rs.__getitem__) if e == _ASM_MACHINE_ENT_ID else 0
        for u, v in tiles:
            claimed[u, v] = True
            world[_CH_ENT, u, v], world[_CH_DIR, u, v], world[_CH_MISC, u, v] = _TYPES_NP[ty]
            world[_CH_ITEMS, u, v] = RECIPES[r]
    single = free & ~claimed
    flat = np.where(_IS_MULTI_NP[typ], 0, typ)
    for c, col in ((_CH_ENT, 0), (_CH_DIR, 1), (_CH_MISC, 2)):
        world[c][single] = _TYPES_NP[flat, col][single]
    world[_CH_ITEMS][single] = 0
    return world


def entity_cost(world_CWH: np.ndarray, cost_scale: float) -> tuple[float, float]:
    """(entity_cost, cost_efficiency), as `FactorioEnv._world_cost` scores it."""
    counts = np.bincount(world_CWH[_CH_ENT].ravel(), minlength=len(entities))[: len(entities)]
    cost = float(np.dot(counts / _ENTITY_FOOTPRINT_AREAS, _ENTITY_UNIT_COSTS))
    return cost, 1.0 / (1.0 + cost_scale * cost)


def score_world(world_CWH: np.ndarray, max_throughput: float, cost_scale: float) -> dict:
    raw, _ = factorion_rs.simulate_throughput(np.ascontiguousarray(world_CWH.transpose(1, 2, 0)))
    cost, eff = entity_cost(world_CWH, cost_scale)
    return {
        "thput_raw": raw,
        "thput": min(1.0, raw / max_throughput) if max_throughput > 0 else 0.0,
        # PPO's reward: uncapped, discounted by entity cost.
        "score": raw * eff / max_throughput if max_throughput > 0 else 0.0,
        "entity_cost": cost,
    }


# ── Generation ─────────────────────────────────────────────────────────


@torch.no_grad()
def generate(model, cond, steps=8, refine=8, temperature=1.0, generator=None):
    """Denoise pure noise into factories, one chain per row of `cond`.

    `steps` ancestral steps sample the clean-factory prediction and re-corrupt
    it to a noise level falling linearly 1 → 0; up to `refine` greedy steps
    then iterate the prediction until it reproduces its own input (the chain
    has *converged*: the model calls the factory finished). Returns the
    tokens, the confidence of each tile's type, and per-chain converged flag
    and refine steps taken."""
    free = free_mask(cond)
    prior = model.noise_prior
    toks = noise_tokens(prior, free.shape, generator)
    B = cond.shape[0]
    for k in range(steps):
        logits = model(tokens_to_state(cond, free, *toks))
        toks = pick_tokens(logits, temperature, generator)
        t_next = 1.0 - (k + 1) / steps
        if t_next > 0:
            toks = corrupt(toks, free, torch.full((B,), t_next, device=cond.device), prior, generator)
    converged = torch.zeros(B, dtype=torch.bool, device=cond.device)
    n_refine = torch.zeros(B, dtype=torch.long, device=cond.device)
    logits = model(tokens_to_state(cond, free, *toks))
    for _ in range(refine):
        new = pick_tokens(logits)
        same = torch.stack([((a == b) | ~free).flatten(1).all(1) for a, b in zip(new, toks)]).all(0)
        converged |= same
        if bool(converged.all()):
            break
        n_refine += ~converged
        toks = new
        logits = model(tokens_to_state(cond, free, *toks))
    conf = F.softmax(logits[:, :N_TYPES], dim=1).gather(1, toks[0].unsqueeze(1)).squeeze(1)
    return toks, conf, converged, n_refine


def run_denoise_eval(
    model,
    factories: list[tuple[int, LessonKind, Factory]],
    device,
    *,
    best_of: int,
    steps: int,
    refine: int,
    temperature: float,
    cost_scale: float,
    seed: int,
    batch_chains: int = 2048,
    records: Optional[list[dict]] = None,
) -> dict:
    """Best-of-N generation eval on held-out factories.

    Each `(seed, kind, Factory)` is fully blanked to its markers and walls,
    `best_of` chains denoise it from pure noise, every chain's factory is
    legalized and simulated, and the factory scores its best chain (ties: the
    cheaper build, then the lower chain index). Returns `"thput@n"` /
    `"trial_thput@n"` / `"per_kind@n"` for n in 1, 8 (when ≤ best_of) and
    best_of — best-of-n reads the first n chains — plus chain-level
    convergence and the selected factories' dangling-inserter counts.
    `records` collects one dict (seed, kind, thput, render, ...) per factory.
    """
    from factorion import blank_entities, render_factory

    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    conds = []
    for _, _, factory in factories:
        blank, _ = blank_entities(factory)
        conds.append(world_to_state(blank.numpy()))
    cond_all = torch.as_tensor(np.stack(conds))
    chains = [(f, j) for f in range(len(factories)) for j in range(best_of)]
    results: list[list[dict]] = [[] for _ in factories]
    n_conv = n_refine_total = 0
    for lo in range(0, len(chains), batch_chains):
        chunk = chains[lo : lo + batch_chains]
        cond = cond_all[[f for f, _ in chunk]].to(device)
        toks, conf, conv, nref = generate(model, cond, steps, refine, temperature, gen)
        n_conv += int(conv.sum())
        n_refine_total += int(nref.sum())
        typ, rec, part = (t.cpu().numpy() for t in toks)
        conf_np, cond_np = conf.float().cpu().numpy(), cond.cpu().numpy().astype(np.int64)
        for i, (f, j) in enumerate(chunk):
            world = legalize(cond_np[i, :CH_PART], typ[i], rec[i], part[i], conf_np[i])
            r = score_world(world, factories[f][2].max_throughput, cost_scale)
            r.update(chain=j, world=world)
            results[f].append(r)

    ns = sorted({1, min(8, best_of), best_of})
    out: dict = {"converged_frac": n_conv / len(chains), "refine_steps": n_refine_total / len(chains)}
    for n in ns:
        lesson, trial, per_kind = [], [], {}
        dangling = [0, 0, 0, 0]  # lesson in/out, trial in/out
        for (s, kind, _), rs in zip(factories, results):
            best = max(rs[:n], key=lambda r: (r["thput"], -r["entity_cost"], -r["chain"]))
            is_trial = LESSON_IS_TRIAL[kind]
            (trial if is_trial else lesson).append(best["thput"])
            per_kind.setdefault(kind.name, []).append(best["thput"])
            if n == best_of:
                d_in, d_out = count_dangling_inserters(torch.from_numpy(best["world"]))
                dangling[2 * is_trial] += d_in
                dangling[2 * is_trial + 1] += d_out
                if records is not None:
                    records.append(
                        {
                            "seed": s,
                            "kind": kind.name,
                            "thput": best["thput"],
                            "entity_cost": best["entity_cost"],
                            "inserters_no_input": d_in,
                            "inserters_no_output": d_out,
                            "render": render_factory(best["world"]),
                        }
                    )
        out[f"thput@{n}"] = float(np.mean(lesson)) if lesson else 0.0
        out[f"trial_thput@{n}"] = float(np.mean(trial)) if trial else 0.0
        out[f"per_kind@{n}"] = {k: float(np.mean(v)) for k, v in per_kind.items()}
        if n == best_of:
            out["dangling"] = dangling
            out["n_trials"] = len(trial)
    if was_training:
        model.train()
    return out
