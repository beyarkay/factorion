"""Tests for read_metric in scripts/ci/wandb_metric.py.

Regression coverage for the W&B sweep-report crash where a metric resolved
to a nested ``SummarySubDict`` and ``runs.sort`` raised
``TypeError: '<' not supported between instances of 'SummarySubDict'``.
"""

import math
import os
import sys

# Add scripts/ci to sys.path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "ci"))

from wandb_metric import read_metric  # noqa: E402


class SummarySubDict(dict):
    """Stand-in for W&B's nested summary node.

    Like the real thing, it is dict-like (supports ``.get``) and instances
    are unorderable, so feeding one to ``sorted``/``list.sort`` raises.
    ``.get`` does not traverse slash-separated keys — nested metrics live
    under nested ``SummarySubDict`` values, exactly like W&B.
    """


class TestReadMetric:
    def test_flat_scalar(self):
        summary = SummarySubDict({"val/acc": 0.9})
        assert read_metric(summary, "val/acc", float("-inf")) == 0.9

    def test_int_coerced_to_float(self):
        summary = SummarySubDict({"steps": 5})
        result = read_metric(summary, "steps", float("-inf"))
        assert result == 5.0
        assert isinstance(result, float)

    def test_nested_slash_key_resolves_leaf(self):
        # Full "a/b" key misses; the value lives under nested dicts.
        summary = SummarySubDict(
            {"curriculum": SummarySubDict({"throughput_avg": 0.5})}
        )
        assert read_metric(summary, "curriculum/throughput_avg", float("-inf")) == 0.5

    def test_metric_pointing_at_namespace_is_missing(self):
        # Metric name resolves to a namespace node, not a scalar leaf.
        summary = SummarySubDict(
            {"throughput": SummarySubDict({"greedy": 0.5, "random": 0.1})}
        )
        assert read_metric(summary, "throughput", float("-inf")) == float("-inf")

    def test_absent_metric_returns_missing(self):
        summary = SummarySubDict({"other": 1.0})
        assert read_metric(summary, "val/acc", float("-inf")) == float("-inf")
        assert read_metric(summary, "val/acc", float("inf")) == float("inf")

    def test_partial_nested_path_is_missing(self):
        # First hop exists, second does not.
        summary = SummarySubDict({"curriculum": SummarySubDict({"score": 1.0})})
        assert read_metric(summary, "curriculum/throughput_avg", float("inf")) == float(
            "inf"
        )

    def test_sorting_runs_with_nonscalar_metric_does_not_raise(self):
        # The original bug: two runs whose metric resolves to SummarySubDicts
        # made list.sort compare unorderable nodes and raise TypeError.
        runs = [
            SummarySubDict({"throughput": SummarySubDict({"greedy": 0.5})}),
            SummarySubDict({"throughput": SummarySubDict({"greedy": 0.9})}),
        ]
        missing = float("-inf")
        # Should not raise, and missing values sort consistently.
        runs.sort(key=lambda s: read_metric(s, "throughput", missing), reverse=True)
        assert all(read_metric(s, "throughput", missing) == missing for s in runs)

    def test_mixed_present_and_missing_sort_order(self):
        runs = [
            SummarySubDict({"val/acc": 0.3}),
            SummarySubDict({"other": 1.0}),  # missing -> -inf
            SummarySubDict({"val/acc": 0.8}),
        ]
        missing = float("-inf")
        runs.sort(key=lambda s: read_metric(s, "val/acc", missing), reverse=True)
        vals = [read_metric(s, "val/acc", missing) for s in runs]
        assert vals[0] == 0.8
        assert vals[1] == 0.3
        assert math.isinf(vals[2]) and vals[2] < 0
