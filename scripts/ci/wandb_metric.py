"""Read a sweep metric out of a W&B run summary as a sortable scalar.

W&B stores slash-namespaced metrics (e.g. ``curriculum/throughput_avg``)
as *nested* dicts in a run's summary. A plain ``summary.get("a/b")`` can
therefore return a nested ``SummarySubDict`` node — either because the
full slash key misses (the value lives at ``summary["a"]["b"]``) or
because the metric name points at a namespace rather than a leaf. Those
nodes are unorderable, so feeding them to ``sorted()``/``list.sort()``
raises ``TypeError: '<' not supported between instances of
'SummarySubDict'``.

``read_metric`` resolves the slash path and only ever returns a float (or
the caller's ``missing`` sentinel), so runs can be sorted safely.
"""


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
    try:
        return float(val)
    except (TypeError, ValueError):
        return missing
