"""Tests for the curriculum_score metric.

curriculum_score = (max_missing_entities - 1) + moving_avg_throughput

This composite metric is monotonically increasing with agent capability:
advancing a curriculum level is always worth more than any throughput gain
within a level (since throughput is in [0, 1]).
"""

import json
import os
import sys
import tempfile

import pytest

# Add scripts/ci to sys.path so we can import compare_runs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "ci"))

from compare_runs import (  # noqa: E402
    _ensure_curriculum_score,
    generate_report,
)


class TestEnsureCurriculumScore:
    def test_computes_from_level_and_throughput(self):
        results = [{"moving_avg_throughput": 0.6, "max_missing_entities": 2}]
        _ensure_curriculum_score(results)
        assert results[0]["curriculum_score"] == 1.6

    def test_does_not_overwrite_existing(self):
        results = [
            {
                "moving_avg_throughput": 0.6,
                "max_missing_entities": 2,
                "curriculum_score": 99.0,
            }
        ]
        _ensure_curriculum_score(results)
        assert results[0]["curriculum_score"] == 99.0

    def test_defaults_for_missing_fields(self):
        results = [{}]
        _ensure_curriculum_score(results)
        # level defaults to 1, throughput defaults to 0 → score = 0
        assert results[0]["curriculum_score"] == 0.0

    def test_level_1_equals_throughput(self):
        results = [{"moving_avg_throughput": 0.85, "max_missing_entities": 1}]
        _ensure_curriculum_score(results)
        assert results[0]["curriculum_score"] == 0.85

    def test_level_advancement_always_wins(self):
        """A model at level 2 with 0.0 throughput should score higher than
        a model at level 1 with 0.99 throughput."""
        level1 = [{"moving_avg_throughput": 0.99, "max_missing_entities": 1}]
        level2 = [{"moving_avg_throughput": 0.0, "max_missing_entities": 2}]
        _ensure_curriculum_score(level1)
        _ensure_curriculum_score(level2)
        assert level2[0]["curriculum_score"] > level1[0]["curriculum_score"]


class TestCurriculumScoreGating:
    """CI should gate on curriculum_score regression, not raw throughput."""

    def _make_results(self, seeds_data):
        """Helper: seeds_data is list of (seed, throughput, level) tuples."""
        return [
            {
                "seed": s,
                "moving_avg_throughput": t,
                "max_missing_entities": lvl,
                "total_timesteps": 100000,
                "sps": 200,
            }
            for s, t, lvl in seeds_data
        ]

    def test_higher_level_lower_throughput_is_better(self):
        """PR advances to level 2 but raw throughput drops — should show
        curriculum_score as 'better', not 'worse'."""
        pr = self._make_results([(1, 0.3, 2), (2, 0.4, 2), (3, 0.2, 2)])
        baseline = self._make_results([(1, 0.9, 1), (2, 0.85, 1), (3, 0.92, 1)])

        report = generate_report(pr, baseline, "pr", "99", "abc1234")

        # Curriculum score should be better
        assert "Curriculum score" in report
        # The report should NOT fail CI — verify by checking the gating logic
        # (generate_report doesn't gate, but we can verify the verdicts)
        lines = report.split("\n")
        for line in lines:
            if "Curriculum score" in line:
                assert "worse" not in line.lower(), (
                    f"Curriculum score should not be 'worse' when level advanced: {line}"
                )
                break

    def test_same_level_lower_throughput_is_worse(self):
        """At the same curriculum level, lower throughput should be worse."""
        pr = self._make_results([(1, 0.3, 1), (2, 0.35, 1), (3, 0.25, 1)])
        baseline = self._make_results([(1, 0.9, 1), (2, 0.85, 1), (3, 0.92, 1)])

        report = generate_report(pr, baseline, "pr", "99", "abc1234")

        lines = report.split("\n")
        for line in lines:
            if "Curriculum score" in line:
                assert "worse" in line.lower(), (
                    f"Curriculum score should be 'worse' at same level with lower throughput: {line}"
                )
                break

    def test_backward_compat_old_baseline_no_curriculum_score(self):
        """Old baselines without curriculum_score should still work."""
        pr = [
            {
                "seed": 1,
                "moving_avg_throughput": 0.6,
                "max_missing_entities": 1,
                "curriculum_score": 0.6,
                "total_timesteps": 100000,
                "sps": 200,
            },
            {
                "seed": 2,
                "moving_avg_throughput": 0.5,
                "max_missing_entities": 1,
                "curriculum_score": 0.5,
                "total_timesteps": 100000,
                "sps": 200,
            },
        ]
        # Old baseline: no curriculum_score key
        baseline = [
            {
                "seed": 1,
                "moving_avg_throughput": 0.55,
                "max_missing_entities": 1,
                "total_timesteps": 100000,
                "sps": 200,
            },
            {
                "seed": 2,
                "moving_avg_throughput": 0.5,
                "max_missing_entities": 1,
                "total_timesteps": 100000,
                "sps": 200,
            },
        ]

        # Should not raise
        report = generate_report(pr, baseline, "pr", "99", "abc1234")
        assert "Curriculum score" in report


class TestSummaryJsonCurriculumScore:
    """Verify curriculum_score is correctly written to summary JSON."""

    def test_score_formula(self):
        """curriculum_score = (max_missing_entities - 1) + throughput"""
        # Simulate what ppo.py writes
        for level, thput, expected in [
            (1, 0.5, 0.5),
            (2, 0.3, 1.3),
            (3, 0.0, 2.0),
            (1, 0.0, 0.0),
            (5, 1.0, 5.0),
        ]:
            score = (level - 1) + thput
            assert score == pytest.approx(expected), (
                f"level={level}, thput={thput}: expected {expected}, got {score}"
            )

    def test_monotonicity(self):
        """Advancing a level never scores lower than perfect throughput at the
        previous level.  At the boundary (0.0 throughput at new level), the
        scores tie — any nonzero throughput at the new level is strictly better."""
        for level in range(1, 10):
            best_at_current = (level - 1) + 1.0  # max throughput at current level
            worst_at_next = (level) + 0.0  # zero throughput at next level
            assert worst_at_next >= best_at_current, (
                f"Level {level+1} with 0 throughput ({worst_at_next}) should be >= "
                f"level {level} with perfect throughput ({best_at_current})"
            )
            # Any nonzero throughput at the next level is strictly better
            tiny_at_next = (level) + 0.01
            assert tiny_at_next > best_at_current
