"""Qualitative diff of two policies: on the factories where they disagree,
what does each one actually build?

A compare report says whether a metric moved. This says *where* it moved and
*what changed on the grid*: every checkpoint greedy-rolls the same held-out
factories, and each factory the two sides consistently disagree on is rendered
side by side, biggest gap first — the view that tells "PPO learned X" apart
from "PPO forgot Y".

    uv run python -m ci compare-renders <pr_ckpts> <main_ckpts>

where a checkpoint is a local .pt path (assumed to be the `training_config.py`
architecture) or a W&B run id (whose own config is used).
"""

from __future__ import annotations

import os
from dataclasses import fields
from statistics import mean
from types import SimpleNamespace

from training_config import PpoArgs

PER_KIND_DEFAULT = 50
# Nobody reads the 40th-largest gap, and a wall of grids buries the ones that
# matter. The per-lesson tally still covers every factory.
MAX_RENDERS_DEFAULT = 25
# Throughputs come from the same simulator, so equal factories give bitwise
# equal rates; anything above this is a real disagreement.
THPUT_TOL = 1e-6

_ENV_ID = "factorion/FactorioEnv-v0-diff"


def load_agent(spec: str, device):
    """Load a checkpoint into an `AgentCNN` built to its run's architecture."""
    import gymnasium as gym
    import torch

    from ppo import AgentCNN, _resolve_wandb_checkpoint, make_env

    config: dict = {}
    if os.path.exists(spec):
        path = spec
    else:
        path, source = _resolve_wandb_checkpoint(
            spec, PpoArgs.wandb_project_name, PpoArgs.wandb_entity
        )
        config = source.get("config", {})
    args = PpoArgs(**{f.name: config[f.name] for f in fields(PpoArgs) if f.name in config})

    if _ENV_ID not in gym.registry:
        gym.register(id=_ENV_ID, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(_ENV_ID, 0, False, args.size, "diff")])
    agent = AgentCNN(
        envs,
        conv_channels=args.conv_channels,
        kernel_size=args.kernel_size,
        attn_dim=args.attn_dim,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        attn_pos_embed=args.attn_pos_embed,
        global_feat_dim=args.global_feat_dim,
    )
    envs.close()
    # PPO saves after torch.compile, which prefixes every parameter name.
    state = torch.load(path, map_location="cpu", weights_only=True)
    agent.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    agent.to(device).eval()
    return agent, args


def collect(spec: str, *, seed: int, per_kind: int = PER_KIND_DEFAULT) -> list[dict]:
    """Greedy-rollout `spec` over `per_kind` held-out factories per LessonKind.

    The factory set is PPO's own eval set at this seed, so two checkpoints
    collected at the same seed and grid size see exactly the same factories.
    """
    import torch

    from ppo import _build_eval_set
    from sft import run_rollout_eval

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, args = load_agent(spec, device)
    eval_args = SimpleNamespace(
        size=args.size, seed=seed, max_level=0, eval_seeds_per_kind=per_kind
    )
    seeds_to_kind = _build_eval_set(eval_args)
    records: list[dict] = []
    run_rollout_eval(
        agent,
        eval_args,  # ty: ignore[invalid-argument-type]
        seeds_to_kind,
        device,
        max_seeds=len(seeds_to_kind),
        records=records,
    )
    return records


def _side_by_side(left: str, right: str, left_head: str, right_head: str) -> str:
    ls = [left_head, *left.splitlines()]
    rs = [right_head, *right.splitlines()]
    width = max(len(x) for x in ls)
    ls += [""] * (len(rs) - len(ls))
    rs += [""] * (len(ls) - len(rs))
    return "\n".join(f"{a:<{width}}  |  {b}".rstrip() for a, b in zip(ls, rs))


def _by_key(runs: list[list[dict]]) -> dict[tuple[str, int], list[dict]]:
    """{(lesson, factory seed): one record per run}, dropping any factory a run
    is missing — a verdict must never rest on a partially-covered factory."""
    out: dict[tuple[str, int], list[dict]] = {}
    for records in runs:
        for r in records:
            out.setdefault((r["kind"], r["seed"]), []).append(r)
    return {k: v for k, v in out.items() if len(v) == len(runs)}


def _thputs(recs: list[dict]) -> list[float]:
    return [r["thput"] for r in recs]


def _verdict(pr: list[dict], main: list[dict]) -> int:
    """+1 when EVERY pr run beat EVERY main run on this factory, -1 when every
    one lost it, 0 when the two ranges overlap.

    A compare runs several training seeds per side, and one seed moving is
    noise from that seed, not a property of the branch. Requiring the ranges
    to be disjoint is the cheap version of "this reproduces": with one seed a
    side it degenerates to "the throughput differs".
    """
    if min(_thputs(pr)) - max(_thputs(main)) > THPUT_TOL:
        return 1
    if min(_thputs(main)) - max(_thputs(pr)) > THPUT_TOL:
        return -1
    return 0


def _side_cell(recs: list[dict], label: str) -> tuple[str, str]:
    """(header, render) for one side: its mean throughput (the number Δ and the
    summary table are computed from), every seed's value, and the median run's
    grid — the side's typical build, not its luckiest or unluckiest one."""
    median = sorted(recs, key=lambda r: r["thput"])[len(recs) // 2]
    seeds = (
        "" if len(recs) == 1
        else f" [{' '.join(f'{t:.2f}' for t in sorted(_thputs(recs)))}]"
    )
    head = (
        f"{label}  thput {mean(_thputs(recs)):.3f}{seeds}  "
        f"cost {median['entity_cost']:.1f}"
    )
    return head, median["render"]


def _summary_table(
    keys: list[tuple[str, int]],
    pr_by_key: dict[tuple[str, int], list[dict]],
    main_by_key: dict[tuple[str, int], list[dict]],
) -> list[str]:
    """Per-lesson tally of where the two sides ended up (movers first). The
    throughput columns are means over every factory AND every seed."""
    by_kind: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        by_kind.setdefault(key[0], []).append(key)
    rows = []
    for kind, kind_keys in by_kind.items():
        pr_t = mean(mean(_thputs(pr_by_key[k])) for k in kind_keys)
        main_t = mean(mean(_thputs(main_by_key[k])) for k in kind_keys)
        verdicts = [_verdict(pr_by_key[k], main_by_key[k]) for k in kind_keys]
        rows.append((pr_t - main_t, kind, len(kind_keys), verdicts, main_t, pr_t))
    rows.sort(key=lambda r: -abs(r[0]))
    return [
        "| Lesson | factories | PR better | PR worse | μ main thput | μ PR thput | μ Δ |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| `{kind}` | {n} | {sum(1 for v in verdicts if v > 0)} "
            f"| {sum(1 for v in verdicts if v < 0)} "
            f"| {main_t:.3f} | {pr_t:.3f} | {delta:+.3f} |"
            for delta, kind, n, verdicts, main_t, pr_t in rows
        ),
    ]


def diff_markdown(
    pr_runs: list[list[dict]],
    main_runs: list[list[dict]],
    *,
    pr_label: str = "PR",
    main_label: str = "main",
    note: str = "",
    max_renders: int = MAX_RENDERS_DEFAULT,
) -> str:
    """Markdown report over one or more runs per side: a per-lesson tally, then
    every consistently-different factory rendered side by side, largest mean
    throughput gap first."""
    pr_by_key = _by_key(pr_runs)
    main_by_key = _by_key(main_runs)
    keys = sorted(set(pr_by_key) & set(main_by_key))
    if not keys:
        return ""

    def delta(key) -> float:
        return mean(_thputs(pr_by_key[key])) - mean(_thputs(main_by_key[key]))

    differing = sorted(
        (k for k in keys if _verdict(pr_by_key[k], main_by_key[k])),
        key=lambda k: -abs(delta(k)),
    )
    lines = [
        f"## Greedy factory diff: {main_label} vs {pr_label}",
        "",
        f"Every checkpoint rebuilt the same {len(keys)} held-out factories from "
        f"a blank grid, stopping where its own EOT head fired. "
        f"{len(differing)} came out consistently different. {len(pr_runs)} run(s) "
        "per side; a factory counts as different only when every run on one side "
        "beat every run on the other, so a single seed wandering is not reported.",
        "",
        # Provenance (which runs, which groups) belongs where a reader lands,
        # not in a footer under 25 grids.
        *([note, ""] if note else []),
        *_summary_table(keys, pr_by_key, main_by_key),
        "",
    ]
    if not differing:
        return "\n".join(lines + ["No factory moved consistently across seeds."])

    shown = differing[:max_renders]
    lines += [
        f"<details><summary>{len(differing)} consistently different factories, "
        "largest throughput gap first</summary>",
        "",
        "",  # markdown inside a <details> only renders after a blank line
    ]
    for i, key in enumerate(shown, start=1):
        main_head, main_render = _side_cell(main_by_key[key], main_label)
        pr_head, pr_render = _side_cell(pr_by_key[key], pr_label)
        lines += [
            f"**{i}. `{key[0]}` factory {key[1]} — Δthput {delta(key):+.3f}**",
            "",
            "```text",
            _side_by_side(main_render, pr_render, main_head, pr_head),
            "```",
            "",
        ]
    if len(shown) < len(differing):
        lines += [
            f"_{len(differing) - len(shown)} further factories omitted; "
            f"the {len(shown)} largest gaps are shown._",
            "",
        ]
    return "\n".join(lines + ["</details>"])


def compare_checkpoints(
    pr_specs: list[str],
    main_specs: list[str],
    *,
    seed: int = 1,
    per_kind: int = PER_KIND_DEFAULT,
    note: str = "",
) -> str:
    """Roll out every checkpoint over ONE shared factory set and diff the sides.

    `seed` picks the factory set, not the training seed of any checkpoint: all
    of them must rebuild the same factories for the per-factory comparison to
    mean anything.
    """
    return diff_markdown(
        [collect(s, seed=seed, per_kind=per_kind) for s in pr_specs],
        [collect(s, seed=seed, per_kind=per_kind) for s in main_specs],
        note=note,
    )
