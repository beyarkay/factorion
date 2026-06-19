"""Read a sweep metric out of a W&B run summary as a sortable scalar.

Two W&B summary representations defeat a naive ``float(summary.get("a/b"))``:

1. **Slash-namespaced metrics** (e.g. ``curriculum/throughput_avg``) can be
   stored as *nested* dicts, so the full ``"a/b"`` key misses and the value
   lives at ``summary["a"]["b"]``. A metric name pointing at a namespace
   rather than a leaf yields a nested ``SummarySubDict`` node.

2. **``define_metric(summary="max"|"min"|...)`` metrics** are stored as a
   single-stat dict — ``val/throughput`` becomes ``{"max": 0.04}`` rather
   than a bare ``0.04``. ``define_metric`` is exactly how the SFT greedy
   throughput sweep records its objective, so this is the common case, not
   an edge one.

Both yield a non-scalar (a dict / ``SummarySubDict``) where the caller wants
a number; ``float(dict)`` raises ``TypeError`` and unorderable nodes break
``sorted()``/``list.sort()``. ``read_metric`` resolves the slash path,
unwraps a summary-stat dict, and only ever returns a float (or the caller's
``missing`` sentinel), so runs can be sorted safely.
"""

# define_metric(summary=...) stores the chosen statistic under one of these
# keys. Ordered by preference so a maximize objective unwraps to its "max"
# and a minimize objective (no "max" present) falls through to "min".
_SUMMARY_STAT_KEYS = ("max", "min", "last", "mean", "value")


def read_metric(summary, metric_name, missing):
    """Return ``metric_name`` from a W&B run summary as a float.

    Falls back to ``missing`` when the metric is absent, non-scalar, or
    resolves to a nested namespace rather than a leaf value.

    Args:
        summary: A run's ``summary`` (dict-like, supports ``.get``).
        metric_name: Metric key, possibly slash-namespaced ("a/b").
        missing: Value to return when no scalar can be resolved (e.g.
            ``float("-inf")`` so missing runs sort last when maximizing).
    """
    val = summary.get(metric_name)
    # The full slash key missed but the value may be nested under it.
    if val is None and "/" in metric_name:
        node = summary
        for part in metric_name.split("/"):
            try:
                node = node.get(part)
            except AttributeError:
                node = None
            if node is None:
                break
        val = node
    # A define_metric(summary="max"|...) value is a single-stat mapping
    # ({"max": 0.04}); unwrap it to the underlying scalar. W&B's SummarySubDict
    # is dict-LIKE but does NOT subclass dict, so duck-type on `.keys()` rather
    # than isinstance(dict) (which silently misses the real object). A namespace
    # mapping with no stat keys (e.g. {"greedy": .., "random": ..}) is left
    # alone and falls through to `missing` below — we can't pick one for the
    # caller.
    if val is not None and hasattr(val, "keys"):
        keys = set(val.keys())
        for stat in _SUMMARY_STAT_KEYS:
            if stat in keys:
                val = val[stat]
                break
    try:
        return float(val)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return missing
