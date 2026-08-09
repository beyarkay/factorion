"""Qualitative diff of two policies: on the factories where they disagree,
what does each one actually build?

A compare report says whether a metric moved. This says *where* it moved and
*what changed on the grid*: both checkpoints greedy-roll the same (lesson,
seed) held-out factories, and every pair whose throughput differs is rendered
side by side, biggest gap first — the view that tells "PPO learned X" apart
from "PPO forgot Y".

    uv run python -m ci compare-renders <pr_ckpt> <main_ckpt>

where each checkpoint is a local .pt path or a W&B run id.
"""

from __future__ import annotations

import os
from dataclasses import fields
from types import SimpleNamespace
from typing import Optional

from training_config import PpoArgs

PER_KIND_DEFAULT = 50
# GitHub caps a comment at 65536 chars; leave room for the report's own prose.
MAX_CHARS_DEFAULT = 58_000
# Throughputs come from the same simulator, so equal factories give bitwise
# equal rates; anything above this is a real disagreement.
THPUT_TOL = 1e-6

_ENV_ID = "factorion/FactorioEnv-v0-diff"


def _resolve(spec: str, project: str, entity: Optional[str]) -> tuple[str, dict]:
    """(local .pt path, training config) for a checkpoint spec. A path on disk
    is used as-is (no config, so the `training_config.py` defaults apply);
    anything else is a W&B run id whose model artifact is downloaded."""
    from ppo import _resolve_wandb_checkpoint

    if os.path.exists(spec):
        return spec, {}
    path, source = _resolve_wandb_checkpoint(spec, project, entity)
    return path, source.get("config", {})


def load_agent(spec: str, device, project: str, entity: Optional[str] = None):
    """Load a checkpoint into an `AgentCNN` built to the run's own architecture."""
    import gymnasium as gym
    import torch

    from ppo import AgentCNN, layers_from_args, make_env

    path, config = _resolve(spec, project, entity)
    args = PpoArgs(**{f.name: config[f.name] for f in fields(PpoArgs) if f.name in config})

    if _ENV_ID not in gym.registry:
        gym.register(id=_ENV_ID, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(_ENV_ID, 0, False, args.size, "diff")])
    agent = AgentCNN(
        envs,
        layers=layers_from_args(args),
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


def collect(
    spec: str,
    *,
    seed: int,
    per_kind: int = PER_KIND_DEFAULT,
    project: str = PpoArgs.wandb_project_name,
    entity: Optional[str] = None,
    device=None,
    num_envs: int = 8,
) -> list[dict]:
    """Greedy-rollout `spec` over `per_kind` held-out factories per LessonKind.

    The factory set is PPO's own eval set at this seed, so two checkpoints
    collected at the same seed and grid size see exactly the same factories.
    """
    import torch

    from ppo import _build_eval_set
    from sft import run_rollout_eval

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent, args = load_agent(spec, device, project, entity)
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
        num_envs=num_envs,
        records=records,
    )
    return records


def _side_by_side(left: str, right: str, left_head: str, right_head: str) -> str:
    ls = [left_head, *left.splitlines()]
    rs = [right_head, *right.splitlines()]
    width = max(len(x) for x in ls)
    height = max(len(ls), len(rs))
    ls += [""] * (height - len(ls))
    rs += [""] * (height - len(rs))
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


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _representative(recs: list[dict]) -> dict:
    """The median-throughput run's factory — the side's typical build, not its
    luckiest or unluckiest one."""
    return sorted(recs, key=lambda r: r["thput"])[len(recs) // 2]


def _thput_label(recs: list[dict]) -> str:
    """`0.700` for one run, `0.700 [0.60 0.70 0.80]` for several."""
    shown = _representative(recs)["thput"]
    if len(recs) == 1:
        return f"{shown:.3f}"
    return f"{shown:.3f} [{' '.join(f'{t:.2f}' for t in sorted(_thputs(recs)))}]"


def _summary_table(
    keys: list[tuple[str, int]],
    pr_by_key: dict[tuple[str, int], list[dict]],
    main_by_key: dict[tuple[str, int], list[dict]],
) -> list[str]:
    """Per-lesson tally of where the two sides ended up (movers first)."""
    by_kind: dict[str, list[tuple[str, int]]] = {}
    for key in keys:
        by_kind.setdefault(key[0], []).append(key)
    rows = []
    for kind, kind_keys in by_kind.items():
        pr_t = [_mean(_thputs(pr_by_key[k])) for k in kind_keys]
        main_t = [_mean(_thputs(main_by_key[k])) for k in kind_keys]
        verdicts = [_verdict(pr_by_key[k], main_by_key[k]) for k in kind_keys]
        rows.append(
            (
                _mean(pr_t) - _mean(main_t),
                kind,
                len(kind_keys),
                sum(1 for v in verdicts if v > 0),
                sum(1 for v in verdicts if v < 0),
                _mean(pr_t),
                _mean(main_t),
            )
        )
    rows.sort(key=lambda r: -abs(r[0]))
    lines = [
        "| Lesson | factories | PR better | PR worse | PR thput | MAIN thput | mean Δ |",
        "|---|---|---|---|---|---|---|",
    ]
    for mean_d, kind, n, better, worse, pr_mean, main_mean in rows:
        lines.append(
            f"| `{kind}` | {n} | {better} | {worse} "
            f"| {pr_mean:.3f} | {main_mean:.3f} | {mean_d:+.3f} |"
        )
    return lines


def diff_markdown(
    pr_runs: list[list[dict]],
    main_runs: list[list[dict]],
    *,
    pr_label: str = "PR",
    main_label: str = "MAIN",
    max_chars: int = MAX_CHARS_DEFAULT,
) -> str:
    """Markdown report over one or more runs per side: a per-lesson tally, then
    every consistently-different factory rendered side by side, largest mean
    throughput gap first."""
    pr_by_key = _by_key(pr_runs)
    main_by_key = _by_key(main_runs)
    keys = sorted(set(pr_by_key) & set(main_by_key))
    if not keys:
        return ""
    differing = sorted(
        (k for k in keys if _verdict(pr_by_key[k], main_by_key[k])),
        key=lambda k: -abs(
            _mean(_thputs(pr_by_key[k])) - _mean(_thputs(main_by_key[k]))
        ),
    )

    seeds_note = (
        f"{len(pr_runs)} run(s) per side; a factory counts as different only "
        "when every run on one side beat every run on the other, so a single "
        "seed wandering is not reported."
    )
    lines = [
        f"## Greedy factory diff: {pr_label} vs {main_label}",
        "",
        f"Every checkpoint rebuilt the same {len(keys)} held-out factories from "
        f"a blank grid, stopping where its own EOT head fired. "
        f"{len(differing)} came out consistently different. {seeds_note}",
        "",
        *_summary_table(keys, pr_by_key, main_by_key),
        "",
    ]
    if not differing:
        return "\n".join(lines + ["No factory moved consistently across seeds."])

    lines += [
        f"<details><summary>{len(differing)} consistently different factories, "
        "largest throughput gap first</summary>",
        "",
        "",  # markdown inside a <details> only renders after a blank line
    ]
    head = "\n".join(lines)
    blocks, shown = [], 0
    budget = max_chars - len(head)
    for i, key in enumerate(differing, start=1):
        pr, main = pr_by_key[key], main_by_key[key]
        delta = _mean(_thputs(pr)) - _mean(_thputs(main))
        pr_rep, main_rep = _representative(pr), _representative(main)
        block = "\n".join(
            [
                f"**{i}. `{key[0]}` factory {key[1]} — Δthput {delta:+.3f}**",
                "",
                "```text",
                _side_by_side(
                    pr_rep["render"],
                    main_rep["render"],
                    f"{pr_label}  thput {_thput_label(pr)}  "
                    f"cost {pr_rep['entity_cost']:.1f}",
                    f"{main_label}  thput {_thput_label(main)}  "
                    f"cost {main_rep['entity_cost']:.1f}",
                ),
                "```",
                "",
            ]
        )
        if budget - len(block) < 200:
            break
        budget -= len(block)
        blocks.append(block)
        shown += 1
    tail = ["</details>"]
    if shown < len(differing):
        tail = [
            f"_{len(differing) - shown} further factories omitted "
            "(GitHub comment size limit)._",
            "",
        ] + tail
    return head + "\n".join(blocks) + "\n".join(tail)


def compare_checkpoints(
    pr_specs: list[str],
    main_specs: list[str],
    *,
    seed: int = 1,
    per_kind: int = PER_KIND_DEFAULT,
    pr_label: str = "PR",
    main_label: str = "MAIN",
    project: str = PpoArgs.wandb_project_name,
    entity: Optional[str] = None,
    max_chars: int = MAX_CHARS_DEFAULT,
) -> str:
    """Roll out every checkpoint over ONE shared factory set and diff the sides.

    `seed` picks the factory set, not the training seed of any checkpoint: all
    of them must rebuild the same factories for the per-factory comparison to
    mean anything.
    """
    kw = dict(seed=seed, per_kind=per_kind, project=project, entity=entity)
    return diff_markdown(
        [collect(s, **kw) for s in pr_specs],  # ty: ignore[invalid-argument-type]
        [collect(s, **kw) for s in main_specs],  # ty: ignore[invalid-argument-type]
        pr_label=pr_label,
        main_label=main_label,
        max_chars=max_chars,
    )
