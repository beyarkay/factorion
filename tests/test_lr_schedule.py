"""Tests for the LR/entropy anneal shapes (`PpoArgs.anneal_shape`).

The point of `wsd` is that the stable phase does not depend on the total
budget, so a short tuning run and the prefix of a long production run are the
same trajectory and can be compared at matched `global_step`. `linear` (the
default) keeps the pre-existing whole-run decay, where every step's value
depends on `total_timesteps`.
"""

import os
import sys

import pytest
import tyro

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import (  # noqa: E402
    PpoArgs,
    _decay_multiplier,
    _iteration_schedule,
    _lr_warmup_multiplier,
    _run_signature,
)

BATCH = 4096


def make_args(total_timesteps: int, **overrides) -> PpoArgs:
    """PpoArgs with the derived fields `__main__` computes before the loop."""
    args = PpoArgs(total_timesteps=total_timesteps, **overrides)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def lrs(args) -> list[float]:
    return [_iteration_schedule(args, i)[0] for i in range(1, args.num_iterations + 1)]


def ents(args) -> list[float]:
    return [_iteration_schedule(args, i)[1] for i in range(1, args.num_iterations + 1)]


class TestDefaultsUnchanged:
    """`linear` must reproduce the pre-WSD formula exactly, so existing
    baselines stay reproducible."""

    def test_default_shape_is_linear(self):
        assert PpoArgs().anneal_shape == "linear"

    def test_matches_the_original_inline_formula(self):
        args = make_args(2_000_000)
        anneal_total = max(1, args.num_iterations - args.critic_warmup)
        for iteration in range(1, args.num_iterations + 1):
            in_warmup = iteration <= args.critic_warmup
            anneal_iter = max(0, iteration - args.critic_warmup)
            frac = 1.0 if in_warmup else 1.0 - (anneal_iter - 1.0) / anneal_total
            want_lr = frac * args.learning_rate
            want_ent = args.ent_coef_end + frac * (
                args.ent_coef_start - args.ent_coef_end
            )
            got_lr, got_ent = _iteration_schedule(args, iteration)
            assert got_lr == pytest.approx(want_lr, rel=1e-12)
            assert got_ent == pytest.approx(want_ent, rel=1e-12)

    def test_linear_is_budget_dependent(self):
        """The behaviour WSD exists to fix: same step, different value."""
        short, long = make_args(2_000_000), make_args(10_000_000)
        i = 100  # past critic_warmup, so both are annealing
        assert _iteration_schedule(short, i)[0] != pytest.approx(
            _iteration_schedule(long, i)[0], rel=1e-6
        )


class TestWsdIsBudgetIndependent:
    def test_stable_phase_identical_across_budgets(self):
        short = make_args(2_000_000, anneal_shape="wsd")
        long = make_args(10_000_000, anneal_shape="wsd")
        # Every iteration the *short* run still holds at peak must match the
        # long run step for step — that is the whole property.
        stable_iters = int(
            (short.num_iterations - short.critic_warmup) * (1 - short.cooldown_frac)
        )
        assert stable_iters > 100, "test needs a meaningful stable phase"
        for i in range(1, stable_iters + 1):
            assert _iteration_schedule(short, i) == _iteration_schedule(long, i)

    def test_stable_phase_holds_peak_values(self):
        args = make_args(2_000_000, anneal_shape="wsd")
        lr, ent = _iteration_schedule(args, args.critic_warmup + 5)
        assert lr == pytest.approx(args.learning_rate)
        assert ent == pytest.approx(args.ent_coef_start)

    def test_cooldown_decays_to_the_end_values(self):
        args = make_args(2_000_000, anneal_shape="wsd")
        lr, ent = _iteration_schedule(args, args.num_iterations)
        assert lr < 0.02 * args.learning_rate
        assert ent == pytest.approx(args.ent_coef_end, abs=0.05 * args.ent_coef_start)

    def test_schedule_is_monotonically_decreasing(self):
        args = make_args(2_000_000, anneal_shape="wsd")
        for series in (lrs(args), ents(args)):
            for prev, cur in zip(series, series[1:]):
                assert cur <= prev + 1e-15

    def test_only_the_cooldown_tail_moves(self):
        args = make_args(2_000_000, anneal_shape="wsd")
        moved = [i for i, lr in enumerate(lrs(args), 1) if lr < args.learning_rate]
        expected_start = args.critic_warmup + int(
            (args.num_iterations - args.critic_warmup) * (1 - args.cooldown_frac)
        )
        assert moved[0] == pytest.approx(expected_start + 1, abs=1)

    def test_zero_cooldown_is_a_constant_schedule(self):
        args = make_args(2_000_000, anneal_shape="wsd", cooldown_frac=0.0)
        assert lrs(args) == [args.learning_rate] * args.num_iterations
        assert ents(args) == [args.ent_coef_start] * args.num_iterations


class TestLrWarmup:
    def test_off_by_default(self):
        args = make_args(2_000_000, anneal_shape="wsd")
        assert args.lr_warmup_steps == 0
        assert _iteration_schedule(args, args.critic_warmup + 1)[0] == pytest.approx(
            args.learning_rate
        )

    def test_ramps_from_near_zero_to_peak(self):
        warmup = 20 * BATCH
        args = make_args(2_000_000, anneal_shape="wsd", lr_warmup_steps=warmup)
        first = _iteration_schedule(args, args.critic_warmup + 1)[0]
        assert first == pytest.approx(args.learning_rate / 20)
        at_peak = _iteration_schedule(args, args.critic_warmup + 20)[0]
        assert at_peak == pytest.approx(args.learning_rate)

    def test_ramp_is_invariant_to_batch_size(self):
        """Warmup is in env steps, so halving the batch just doubles the number
        of iterations covering the same ramp."""
        big = make_args(
            2_000_000,
            anneal_shape="wsd",
            lr_warmup_steps=50_000,
            critic_warmup=0,
            num_steps=256,
        )
        small = make_args(
            2_000_000,
            anneal_shape="wsd",
            lr_warmup_steps=50_000,
            critic_warmup=0,
            num_steps=128,
        )
        assert big.batch_size == 2 * small.batch_size
        for global_step in (big.batch_size, 5 * big.batch_size, 12 * big.batch_size):
            lr_big = _iteration_schedule(big, global_step // big.batch_size)[0]
            lr_small = _iteration_schedule(small, global_step // small.batch_size)[0]
            assert lr_big == pytest.approx(lr_small)

    def test_entropy_is_not_warmed_up(self):
        """The entropy bonus should be at full strength immediately — ramping it
        would mean the least-trained policy also explores least."""
        args = make_args(2_000_000, anneal_shape="wsd", lr_warmup_steps=20 * BATCH)
        assert _iteration_schedule(args, args.critic_warmup + 1)[1] == pytest.approx(
            args.ent_coef_start
        )


class TestCriticWarmupWindow:
    def test_peak_is_held_while_the_actor_is_frozen(self):
        for shape in ("linear", "wsd"):
            args = make_args(2_000_000, anneal_shape=shape, lr_warmup_steps=20 * BATCH)
            for i in range(1, args.critic_warmup + 1):
                lr, ent = _iteration_schedule(args, i)
                assert lr == pytest.approx(args.learning_rate)
                assert ent == pytest.approx(args.ent_coef_start)


class TestDecayMultiplier:
    def test_linear_spans_one_to_zero(self):
        assert _decay_multiplier("linear", 0, 100, 0.2) == pytest.approx(1.0)
        assert _decay_multiplier("linear", 50, 100, 0.2) == pytest.approx(0.5)
        assert _decay_multiplier("linear", 100, 100, 0.2) == pytest.approx(0.0)

    def test_wsd_cooldown_is_one_minus_sqrt(self):
        # 20% cooldown: halfway through it, 1 - sqrt(0.5).
        assert _decay_multiplier("wsd", 90, 100, 0.2) == pytest.approx(1 - 0.5**0.5)
        assert _decay_multiplier("wsd", 80, 100, 0.2) == pytest.approx(1.0)
        assert _decay_multiplier("wsd", 100, 100, 0.2) == pytest.approx(0.0)

    def test_degenerate_windows_hold_the_start_value(self):
        assert _decay_multiplier("wsd", 0, 0, 0.2) == 1.0
        assert _decay_multiplier("linear", 0, 0, 0.2) == 1.0
        assert _decay_multiplier("wsd", 99, 100, 0.0) == 1.0

    def test_warmup_multiplier_clamps_at_one(self):
        assert _lr_warmup_multiplier(0, 10, 0) == 1.0
        assert _lr_warmup_multiplier(0, 10, 100) == pytest.approx(0.1)
        assert _lr_warmup_multiplier(990, 10, 100) == 1.0


class TestRunSignature:
    def test_linear_signature_is_unchanged(self):
        assert "wsd" not in _run_signature(make_args(2_000_000))

    def test_wsd_is_named_so_runs_are_distinguishable(self):
        sig = _run_signature(make_args(2_000_000, anneal_shape="wsd"))
        assert "-wsd0.2" in sig


class TestCli:
    """`ppo.py` builds its args with `tyro.cli(PpoArgs)`, so the flags have to
    round-trip through it for `/ci ppo` overrides to work."""

    def test_flags_round_trip(self):
        args = tyro.cli(
            PpoArgs,
            args=[
                "--anneal-shape", "wsd",
                "--lr-warmup-steps", "50000",
                "--cooldown-frac", "0.15",
            ],
        )
        assert args.anneal_shape == "wsd"
        assert args.lr_warmup_steps == 50_000
        assert args.cooldown_frac == 0.15

    def test_shape_is_constrained_to_known_values(self):
        with pytest.raises(SystemExit):
            tyro.cli(PpoArgs, args=["--anneal-shape", "cosine"])
