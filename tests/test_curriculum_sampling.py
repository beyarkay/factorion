"""Tests for prioritized curriculum sampling (P4).

sample_difficulty(max_missing_entities, geometric_p) should:
- Return values in [0, max_missing_entities]
- With geometric_p > 0, bias toward the frontier (max_missing_entities)
- With geometric_p == 0, fall back to uniform sampling
- Preserve some difficulty-0 episodes (essential scaffolding per PR #13)
"""

import os
import sys

import numpy as np
import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import sample_difficulty  # noqa: E402


class TestSampleDifficultyBounds:
    """All returned values must be in [0, max_missing_entities]."""

    @pytest.mark.parametrize("max_missing", [0, 1, 2, 5, 10])
    @pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_within_bounds(self, max_missing, p):
        np.random.seed(42)
        for _ in range(500):
            d = sample_difficulty(max_missing, p)
            assert 0 <= d <= max_missing, (
                f"sample_difficulty({max_missing}, {p}) returned {d}"
            )

    def test_max_missing_zero_always_returns_zero(self):
        np.random.seed(42)
        for p in [0.0, 0.5, 1.0]:
            for _ in range(100):
                assert sample_difficulty(0, p) == 0


class TestGeometricBias:
    """With geometric_p > 0, the frontier should get the most samples."""

    def test_frontier_gets_majority(self):
        """At p=0.5, ~50% of samples should be at the frontier."""
        np.random.seed(42)
        max_missing = 5
        n = 10000
        samples = [sample_difficulty(max_missing, 0.5) for _ in range(n)]
        frontier_frac = sum(1 for s in samples if s == max_missing) / n
        # With geometric(0.5), P(frontier) = 0.5.  Allow some tolerance.
        assert 0.45 <= frontier_frac <= 0.55, (
            f"Expected ~50% at frontier, got {frontier_frac:.3f}"
        )

    def test_frontier_minus_one_gets_quarter(self):
        """At p=0.5, ~25% of samples should be at frontier-1."""
        np.random.seed(42)
        max_missing = 5
        n = 10000
        samples = [sample_difficulty(max_missing, 0.5) for _ in range(n)]
        frac = sum(1 for s in samples if s == max_missing - 1) / n
        assert 0.20 <= frac <= 0.30, (
            f"Expected ~25% at frontier-1, got {frac:.3f}"
        )

    def test_difficulty_zero_still_sampled(self):
        """Difficulty-0 must still appear (essential scaffolding)."""
        np.random.seed(42)
        max_missing = 5
        n = 10000
        samples = [sample_difficulty(max_missing, 0.5) for _ in range(n)]
        zero_count = sum(1 for s in samples if s == 0)
        assert zero_count > 0, "Difficulty-0 should still be sampled"

    def test_higher_p_more_frontier(self):
        """Higher p concentrates more mass at the frontier."""
        np.random.seed(42)
        max_missing = 5
        n = 10000
        frac_low_p = sum(
            1 for _ in range(n) if sample_difficulty(max_missing, 0.3) == max_missing
        ) / n
        np.random.seed(42)
        frac_high_p = sum(
            1 for _ in range(n) if sample_difficulty(max_missing, 0.8) == max_missing
        ) / n
        assert frac_high_p > frac_low_p, (
            f"p=0.8 should give more frontier than p=0.3: "
            f"{frac_high_p:.3f} vs {frac_low_p:.3f}"
        )


class TestPClamping:
    """p=1.0 is clamped to 0.95 to preserve difficulty-0 scaffolding."""

    def test_p_one_still_samples_non_frontier(self):
        """With p=1.0 (clamped to 0.95), non-frontier levels must still appear."""
        np.random.seed(42)
        max_missing = 5
        n = 10000
        samples = [sample_difficulty(max_missing, 1.0) for _ in range(n)]
        non_frontier = sum(1 for s in samples if s < max_missing)
        assert non_frontier > 0, (
            "p=1.0 should be clamped so non-frontier difficulties still appear"
        )

    def test_p_one_frontier_fraction_matches_clamped(self):
        """p=1.0 should behave identically to p=0.95."""
        np.random.seed(42)
        n = 10000
        samples_1 = [sample_difficulty(5, 1.0) for _ in range(n)]
        np.random.seed(42)
        samples_95 = [sample_difficulty(5, 0.95) for _ in range(n)]
        assert samples_1 == samples_95


class TestUniformFallback:
    """With geometric_p == 0, sampling should be uniform."""

    def test_uniform_distribution(self):
        np.random.seed(42)
        max_missing = 4
        n = 20000
        samples = [sample_difficulty(max_missing, 0.0) for _ in range(n)]
        for d in range(max_missing + 1):
            frac = sum(1 for s in samples if s == d) / n
            expected = 1.0 / (max_missing + 1)
            assert abs(frac - expected) < 0.03, (
                f"Difficulty {d}: expected ~{expected:.3f}, got {frac:.3f}"
            )

    def test_negative_p_treated_as_uniform(self):
        """Negative p should also fall back to uniform."""
        np.random.seed(42)
        max_missing = 3
        n = 10000
        samples = [sample_difficulty(max_missing, -0.1) for _ in range(n)]
        for d in range(max_missing + 1):
            frac = sum(1 for s in samples if s == d) / n
            expected = 1.0 / (max_missing + 1)
            assert abs(frac - expected) < 0.03, (
                f"Difficulty {d}: expected ~{expected:.3f}, got {frac:.3f}"
            )


class TestMaxMissingOne:
    """Edge case: max_missing_entities=1 (the starting curriculum level)."""

    def test_both_levels_sampled(self):
        np.random.seed(42)
        n = 5000
        samples = [sample_difficulty(1, 0.5) for _ in range(n)]
        zero_count = sum(1 for s in samples if s == 0)
        one_count = sum(1 for s in samples if s == 1)
        assert zero_count > 0, "Difficulty 0 must appear"
        assert one_count > 0, "Difficulty 1 must appear"
        # With geometric(0.5), P(frontier=1) = 0.5, P(0) = 0.5
        frac_one = one_count / n
        assert 0.45 <= frac_one <= 0.55, (
            f"Expected ~50% at frontier, got {frac_one:.3f}"
        )


class TestReturnType:
    """sample_difficulty should return a plain Python int."""

    def test_returns_int(self):
        result = sample_difficulty(5, 0.5)
        assert isinstance(result, int), f"Expected int, got {type(result)}"
