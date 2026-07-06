"""Tests for read_metric in ci/wandb_metric.py.

Regression coverage for the W&B sweep-report crash where a metric resolved
to a nested ``SummarySubDict`` and ``runs.sort`` raised
``TypeError: '<' not supported between instances of 'SummarySubDict'``.
"""

import math

from ci.wandb_metric import read_metric


class SummarySubDict:
    """Stand-in for W&B's nested summary node.

    Mirrors the real ``wandb.apis.public.summary.SummarySubDict``: dict-LIKE
    (``get``/``keys``/``[]``/``in``) but deliberately **not** a ``dict``
    subclass — the real class isn't either, so ``isinstance(node, dict)`` is
    ``False`` for it. That is the exact footgun ``read_metric`` must survive;
    subclassing ``dict`` here would hide it. Instances are unorderable, so
    feeding one to ``sorted``/``list.sort`` raises. ``.get`` does not traverse
    slash-separated keys — nested metrics live under nested ``SummarySubDict``
    values, exactly like W&B.
    """

    def __init__(self, data):
        self._d = dict(data)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d


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

    def test_define_metric_max_dict_unwrapped(self):
        # define_metric(summary="max") stores {"max": value}, not a bare
        # scalar — the SFT throughput sweep's objective. Must unwrap to 0.04.
        summary = SummarySubDict({"val/throughput": SummarySubDict({"max": 0.04})})
        assert read_metric(summary, "val/throughput", float("-inf")) == 0.04

    def test_define_metric_min_dict_unwrapped(self):
        # A minimize objective records {"min": value}; with no "max" present
        # the unwrap falls through to "min".
        summary = SummarySubDict({"val/loss": SummarySubDict({"min": 1.2})})
        assert read_metric(summary, "val/loss", float("inf")) == 1.2

    def test_nested_slash_key_with_summary_stat_dict(self):
        # Slash key misses AND the leaf is a define_metric dict: resolve the
        # nested path, then unwrap the stat.
        summary = SummarySubDict(
            {"a": SummarySubDict({"b": SummarySubDict({"max": 0.5})})}
        )
        assert read_metric(summary, "a/b", float("-inf")) == 0.5

    def test_multikey_namespace_without_stats_still_missing(self):
        # A namespace dict whose keys are NOT summary stats must stay missing
        # (we can't choose a metric for the caller) — guards the unwrap from
        # over-firing.
        summary = SummarySubDict(
            {"val/throughput": SummarySubDict({"greedy": 0.5, "random": 0.1})}
        )
        assert read_metric(summary, "val/throughput", float("-inf")) == float("-inf")

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
