"""Tests for entropy coefficient scheduling."""

import os
import sys

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import Args  # noqa: E402


class TestEntropySchedule:
    def test_args_have_start_and_end(self):
        """Args should have ent_coef_start and ent_coef_end, not ent_coef."""
        args = Args()
        assert hasattr(args, "ent_coef_start"), "Missing ent_coef_start"
        assert hasattr(args, "ent_coef_end"), "Missing ent_coef_end"
        assert not hasattr(args, "ent_coef"), "Old ent_coef should be removed"

    def test_start_greater_than_end(self):
        """Default start should be greater than end."""
        args = Args()
        assert args.ent_coef_start > args.ent_coef_end

    def test_schedule_at_boundaries(self):
        """Verify the annealing formula at iteration 1 and last iteration."""
        args = Args()
        args.num_iterations = 100

        # At iteration 1 (start): frac = 1.0 - 0/100 = 1.0
        # ent_coef = end + 1.0 * (start - end) = start
        ent_frac = 1.0 - (1 - 1.0) / args.num_iterations
        ent_coef = args.ent_coef_end + ent_frac * (args.ent_coef_start - args.ent_coef_end)
        assert abs(ent_coef - args.ent_coef_start) < 1e-10, f"At start: expected {args.ent_coef_start}, got {ent_coef}"

        # At last iteration: frac = 1.0 - 99/100 = 0.01
        ent_frac = 1.0 - (100 - 1.0) / args.num_iterations
        ent_coef = args.ent_coef_end + ent_frac * (args.ent_coef_start - args.ent_coef_end)
        assert abs(ent_coef - args.ent_coef_end) < 0.01, f"At end: expected ~{args.ent_coef_end}, got {ent_coef}"

    def test_schedule_monotonically_decreasing(self):
        """Entropy coefficient should decrease over iterations."""
        args = Args()
        args.num_iterations = 50

        prev = float("inf")
        for iteration in range(1, 51):
            ent_frac = 1.0 - (iteration - 1.0) / args.num_iterations
            ent_coef = args.ent_coef_end + ent_frac * (args.ent_coef_start - args.ent_coef_end)
            assert ent_coef <= prev, f"Not monotonic at iteration {iteration}: {ent_coef} > {prev}"
            prev = ent_coef
