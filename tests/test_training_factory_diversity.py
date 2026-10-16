"""Guard against the "train on the same N factories forever" bug: SFT's
factory stream marches its seed forward, so training covers about one
distinct factory per generated lesson, scaling with num_samples."""

import os
import sys

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── SFT: the factory stream marches its seed ─────────────────────────────────


class TestSFTFactoryDiversity:
    @staticmethod
    def _states(worker_id, num_workers, n):
        from itertools import islice

        from sft import _iter_factories

        return [s.numpy().tobytes() for s in islice(_iter_factories(9, 1, worker_id, num_workers), n)]

    def test_stream_covers_many_distinct_factories(self):
        assert len(set(self._states(0, 1, 40))) >= 30

    def test_workers_draw_disjoint_factories(self):
        """Each worker walks its own seed residue, so two workers' streams
        don't hand the trainer the same factories."""
        a, b = self._states(0, 2, 20), self._states(1, 2, 20)
        assert len(set(a) & set(b)) <= 2
