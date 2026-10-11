"""Score checkpoints of either architecture on one held-out factory set.

Each spec — a local .pt or a W&B run id, holding either an `AgentCNN` (PPO,
pre-denoiser SFT) or a denoiser (`sft.py`) — builds every factory of PPO's
eval set at `--seed` from its markers, best of N for each `--best-of`, scored
by the same simulator. Alongside lesson / trial / per-lesson throughput it
reports inference cost: forward passes and GFLOP per generated factory, since
the two architectures spend very different compute per sample.

    uv run python scripts/head_to_head.py mvbipbr0 gyh6chx8 --best-of 1 8 64
"""

import json
import os
import sys
from dataclasses import dataclass, field, fields
from types import SimpleNamespace
from typing import Optional

import torch
import tyro
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training_config import PpoArgs, SftArgs  # noqa: E402


@dataclass
class Args:
    specs: tyro.conf.Positional[list[str]]
    best_of: list[int] = field(default_factory=lambda: [1, 8, 64])
    seed: int = 1
    per_kind: int = 16
    """held-out factories per lesson / trial kind"""
    out: Optional[str] = None
    """write the results as JSON here"""


def _gflop(model, x) -> float:
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        # Train mode keeps nn.TransformerEncoder off its fused fast path,
        # which the counter cannot see into.
        model.train()
        model(x) if not hasattr(model, "sample_action") else model.sample_action(x)
        model.eval()
    return counter.get_total_flops() / 1e9


def eval_agent_cnn(spec, args: Args, device) -> dict:
    from factory_diff import load_agent
    from ppo import _build_eval_set, run_rollout_eval

    agent, a = load_agent(spec, device)
    eval_args = SimpleNamespace(size=a.size, seed=args.seed, max_level=0, eval_seeds_per_kind=args.per_kind)
    seeds_to_kind = _build_eval_set(eval_args)
    x = torch.zeros(1, 5, a.size, a.size, device=device)
    x[:, 4] = 1
    out = {"params_M": sum(p.numel() for p in agent.parameters()) / 1e6, "gflop_per_forward": _gflop(agent, x)}
    for n in args.best_of:
        roll = run_rollout_eval(agent, eval_args, seeds_to_kind, device, max_seeds=len(seeds_to_kind), best_of=n)
        out[f"thput@{n}"] = roll["overall"]
        out[f"trial_thput@{n}"] = roll["trial_overall"]
        out[f"per_kind@{n}"] = {k: v for k, v in roll["per_kind"].items() if roll["per_kind_n"][k]}
        # One forward per step; the selected rollouts' step counts.
        out[f"forwards@{n}"] = sum(roll["per_kind_eot_step_n"].values()) / len(seeds_to_kind)
    return out


def eval_denoiser(path, config, args: Args, device) -> dict:
    from denoise import STATE_CHANNELS, load_denoiser, run_denoise_eval
    from sft import build_eval_factories

    a = SftArgs(**{f.name: config[f.name] for f in fields(SftArgs) if f.name in config})
    a.seed, a.eval_seeds_per_kind = args.seed, args.per_kind
    model = load_denoiser(path, device)
    model.amp = device.type == "cuda"
    x = torch.zeros(1, STATE_CHANNELS, a.size, a.size, dtype=torch.long, device=device)
    x[:, 4] = 1
    out = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6, "gflop_per_forward": _gflop(model, x)}
    roll = run_denoise_eval(
        model, build_eval_factories(a), device, best_of=max(args.best_of), steps=a.gen_steps,
        refine=a.gen_refine, temperature=a.gen_temperature, cost_scale=PpoArgs.entity_cost_scale, seed=args.seed,
        batch_chains=a.eval_batch_chains,
    )
    for n in args.best_of:
        # best-of-n reads the first n of the same chains; run_denoise_eval
        # reports 1, 8 and its maximum.
        if f"thput@{n}" in roll:
            out[f"thput@{n}"], out[f"trial_thput@{n}"], out[f"per_kind@{n}"] = (
                roll[f"thput@{n}"], roll[f"trial_thput@{n}"], roll[f"per_kind@{n}"]
            )
    out["forwards_per_chain"] = a.gen_steps + 1 + roll["refine_steps"]
    out["converged_frac"] = roll["converged_frac"]
    return out


def main(args: Args) -> dict:
    from factory_diff import _resolve

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for spec in args.specs:
        path, config = _resolve(spec)
        is_denoiser = "arch" in torch.load(path, map_location="cpu", weights_only=True)
        results[spec] = eval_denoiser(path, config, args, device) if is_denoiser else eval_agent_cnn(spec, args, device)
        r = results[spec]
        fwd = r.get("forwards_per_chain") or r.get(f"forwards@{max(args.best_of)}", 0.0)
        r["gflop_per_factory_sample"] = fwd * r["gflop_per_forward"]
        print(spec, json.dumps({k: v for k, v in r.items() if not k.startswith("per_kind")}, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    return results


if __name__ == "__main__":
    main(tyro.cli(Args))
