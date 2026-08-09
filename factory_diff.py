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


def _summary_table(pairs: list[tuple[dict, dict]]) -> list[str]:
    """Per-lesson tally of where the two sides ended up (movers first)."""
    by_kind: dict[str, list[tuple[dict, dict]]] = {}
    for a, b in pairs:
        by_kind.setdefault(a["kind"], []).append((a, b))
    rows = []
    for kind, items in by_kind.items():
        deltas = [a["thput"] - b["thput"] for a, b in items]
        rows.append(
            (
                sum(deltas) / len(deltas),
                kind,
                len(items),
                sum(1 for d in deltas if abs(d) > THPUT_TOL),
                sum(1 for d in deltas if d > THPUT_TOL),
                sum(1 for d in deltas if d < -THPUT_TOL),
                sum(a["thput"] for a, _ in items) / len(items),
                sum(b["thput"] for _, b in items) / len(items),
            )
        )
    rows.sort(key=lambda r: -abs(r[0]))
    lines = [
        "| Lesson | factories | differ | PR better | PR worse | PR thput | MAIN thput | mean Δ |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mean_d, kind, n, differ, better, worse, pr_t, main_t in rows:
        lines.append(
            f"| `{kind}` | {n} | {differ} | {better} | {worse} "
            f"| {pr_t:.3f} | {main_t:.3f} | {mean_d:+.3f} |"
        )
    return lines


def diff_markdown(
    pr_records: list[dict],
    main_records: list[dict],
    *,
    pr_label: str = "PR",
    main_label: str = "MAIN",
    max_chars: int = MAX_CHARS_DEFAULT,
) -> str:
    """Markdown report: per-lesson tally, then every differing factory rendered
    side by side, largest throughput gap first."""
    pr_by_key = {(r["kind"], r["seed"]): r for r in pr_records}
    main_by_key = {(r["kind"], r["seed"]): r for r in main_records}
    keys = sorted(set(pr_by_key) & set(main_by_key))
    if not keys:
        return ""
    pairs = [(pr_by_key[k], main_by_key[k]) for k in keys]
    differing = sorted(
        (p for p in pairs if abs(p[0]["thput"] - p[1]["thput"]) > THPUT_TOL),
        key=lambda p: -abs(p[0]["thput"] - p[1]["thput"]),
    )

    lines = [
        f"## Greedy factory diff: {pr_label} vs {main_label}",
        "",
        f"Both checkpoints rebuilt the same {len(pairs)} held-out factories from "
        f"a blank grid, stopping where their own EOT head fired. "
        f"{len(differing)} ended at a different throughput.",
        "",
        *_summary_table(pairs),
        "",
    ]
    if not differing:
        return "\n".join(lines + ["Both sides built identically-performing factories."])

    lines += [
        f"<details><summary>{len(differing)} differing factories, "
        "largest throughput gap first</summary>",
        "",
        "",  # markdown inside a <details> only renders after a blank line
    ]
    head = "\n".join(lines)
    blocks, shown = [], 0
    budget = max_chars - len(head)
    for i, (pr, main) in enumerate(differing, start=1):
        delta = pr["thput"] - main["thput"]
        block = "\n".join(
            [
                f"**{i}. `{pr['kind']}` seed {pr['seed']} — Δthput {delta:+.3f}**",
                "",
                "```text",
                _side_by_side(
                    pr["render"],
                    main["render"],
                    f"{pr_label}  thput {pr['thput']:.3f}  cost {pr['entity_cost']:.1f}",
                    f"{main_label}  thput {main['thput']:.3f}  cost {main['entity_cost']:.1f}",
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
    pr_spec: str,
    main_spec: str,
    *,
    seed: int = 1,
    per_kind: int = PER_KIND_DEFAULT,
    pr_label: str = "PR",
    main_label: str = "MAIN",
    project: str = PpoArgs.wandb_project_name,
    entity: Optional[str] = None,
    max_chars: int = MAX_CHARS_DEFAULT,
) -> str:
    """Roll out both checkpoints over one shared factory set and diff them."""
    kw = dict(seed=seed, per_kind=per_kind, project=project, entity=entity)
    pr_records = collect(pr_spec, **kw)  # ty: ignore[invalid-argument-type]
    main_records = collect(main_spec, **kw)  # ty: ignore[invalid-argument-type]
    return diff_markdown(
        pr_records,
        main_records,
        pr_label=pr_label,
        main_label=main_label,
        max_chars=max_chars,
    )
