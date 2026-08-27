"""PPO's per-iteration WSD LR envelopes (`ppo._iteration_lrs`).

The SFT side of the shared shape function (`training_config.wsd_multiplier`)
is covered by tests/test_sft.py::TestLRSchedule through `build_lr_schedule`;
here we cover the shape function directly plus the PPO wiring: independent
actor/critic envelopes over the post-critic-warmup window.
"""

import os
import sys

import pytest
import tyro

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import _iteration_lrs  # noqa: E402
from training_config import PpoArgs, wsd_multiplier  # noqa: E402


def make_args(num_iterations: int = 100, **overrides) -> PpoArgs:
    args = PpoArgs(**overrides)
    args.num_iterations = num_iterations  # filled at runtime by ppo's __main__
    return args


def peaks(args) -> tuple[float, float]:
    return args.learning_rate, args.critic_lr


def test_wsd_multiplier_shape():
    # total 100, warmup 10, cooldown 20 (frac 0.2), floor 0.01.
    assert wsd_multiplier(0, 100, 10, 0.2, 0.01) == pytest.approx(0.1)
    assert wsd_multiplier(9, 100, 10, 0.2, 0.01) == pytest.approx(1.0)
    assert wsd_multiplier(50, 100, 10, 0.2, 0.01) == 1.0
    # Halfway into the cooldown the (1 - sqrt) shape has already shed ~71%.
    assert wsd_multiplier(90, 100, 10, 0.2, 0.01) == pytest.approx(
        0.01 + 0.99 * (1 - 0.5**0.5)
    )
    assert wsd_multiplier(100, 100, 10, 0.2, 0.01) == pytest.approx(0.01)


def test_freeze_window_holds_both_peaks():
    args = make_args(critic_warmup=9)
    for iteration in range(1, args.critic_warmup + 1):
        assert _iteration_lrs(args, iteration) == peaks(args)


def test_freeze_window_is_continuous_with_the_unfreeze():
    """With a warmup ramp, the frozen (inert) actor holds the ramp's first
    value and the critic holds its peak — the logged schedule never plunges
    at the unfreeze."""
    args = make_args(critic_warmup=9, lr_warmup_iters=5)
    for iteration in range(1, args.critic_warmup + 2):
        lr, critic_lr = _iteration_lrs(args, iteration)
        assert lr == pytest.approx(args.learning_rate / 5)
        assert critic_lr == pytest.approx(args.critic_lr)


def test_stable_phase_holds_peaks_and_is_budget_independent():
    short = make_args(num_iterations=100)
    long = make_args(num_iterations=1000)
    # Iterations 10..73 are inside the short run's stable phase (window 91,
    # cooldown 27); the long run must sit on the same values step for step.
    for iteration in range(10, 74):
        assert _iteration_lrs(short, iteration) == peaks(short)
        assert _iteration_lrs(long, iteration) == peaks(long)


def test_cooldown_lands_near_min_ratios():
    args = make_args(num_iterations=100)
    lr, critic_lr = _iteration_lrs(args, args.num_iterations)
    peak_lr, peak_critic = peaks(args)
    # The last iteration sits one step short of the cooldown's endpoint, so it
    # is near — not exactly at — the floor.
    assert peak_lr * args.lr_min_ratio <= lr < peak_lr * 0.1
    assert peak_critic * args.critic_lr_min_ratio <= critic_lr < peak_critic * 0.1


def test_envelopes_are_independent():
    args = make_args(lr_cooldown_frac=0.5, critic_lr_cooldown_frac=0.1)
    # Iteration 69 (step 59 of window 91): inside the actor's cooldown
    # (stable_end 45) but still in the critic's stable phase (stable_end 82).
    lr, critic_lr = _iteration_lrs(args, 69)
    assert lr < args.learning_rate
    assert critic_lr == pytest.approx(args.critic_lr)


def test_actor_warmup_ramps_only_the_actor():
    args = make_args(lr_warmup_iters=10)
    lr, critic_lr = _iteration_lrs(args, args.critic_warmup + 1)
    assert lr == pytest.approx(args.learning_rate / 10)
    assert critic_lr == pytest.approx(args.critic_lr)
    assert _iteration_lrs(args, args.critic_warmup + 10)[0] == pytest.approx(
        args.learning_rate
    )


def test_run_shorter_than_critic_warmup_stays_at_peak():
    args = make_args(num_iterations=5, critic_warmup=9)
    for iteration in range(1, args.num_iterations + 1):
        assert _iteration_lrs(args, iteration) == peaks(args)


def test_flags_round_trip_through_tyro():
    """`ppo.py` parses with `tyro.cli(PpoArgs)`, so the knobs must round-trip
    for CLI/sweep overrides to work."""
    args = tyro.cli(
        PpoArgs,
        args=[
            "--lr-warmup-iters", "3",
            "--lr-cooldown-frac", "0.25",
            "--lr-min-ratio", "0.02",
            "--critic-lr", "1e-4",
            "--critic-lr-cooldown-frac", "0.4",
            "--critic-lr-min-ratio", "0.1",
        ],
    )
    assert args.lr_warmup_iters == 3
    assert args.lr_cooldown_frac == 0.25
    assert args.lr_min_ratio == 0.02
    assert args.critic_lr == 1e-4
    assert args.critic_lr_cooldown_frac == 0.4
    assert args.critic_lr_min_ratio == 0.1
