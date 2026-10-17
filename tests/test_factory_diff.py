"""Tests for the side-by-side checkpoint diff (factory_diff.py)."""

import os
import sys

import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import TINY  # noqa: E402

from factory_diff import collect, diff_markdown  # noqa: E402


def _record(kind, seed, thput, cost=1.0, render="b>"):
    return {
        "seed": seed,
        "kind": kind,
        "thput": thput,
        "entity_cost": cost,
        "render": render,
    }


def test_collect_records_every_held_out_factory(tmp_path):
    from denoise import DenoiserUNet

    model = DenoiserUNet(**TINY)
    path = tmp_path / "dn.pt"
    torch.save({"arch": model.arch, "model": model.state_dict()}, path)
    records = collect(str(path), seed=1, per_kind=1)
    assert records and {"seed", "kind", "thput", "entity_cost", "render"} <= set(records[0])


class TestDiffMarkdown:
    def test_only_differing_factories_rendered_biggest_gap_first(self):
        pr = [
            [
                _record("MOVE_ONE_ITEM", 1, 0.9, render="PRSMALL"),
                _record("MOVE_ONE_ITEM", 2, 0.5),
                _record("SPLITTER_SPLIT", 3, 0.1, render="PRBIG"),
            ]
        ]
        main = [
            [
                _record("MOVE_ONE_ITEM", 1, 0.8, render="MAINSMALL"),
                _record("MOVE_ONE_ITEM", 2, 0.5),
                _record("SPLITTER_SPLIT", 3, 0.9, render="MAINBIG"),
            ]
        ]
        md = diff_markdown(pr, main)

        assert "3 held-out factories" in md and "2 came out consistently" in md
        # The unchanged factory is tallied but never rendered.
        assert "factory 2" not in md
        assert md.index("factory 3") < md.index("factory 1"), "biggest gap first"
        assert "Δthput -0.800" in md and "Δthput +0.100" in md
        for token in ("PRBIG", "MAINBIG", "PRSMALL", "MAINSMALL"):
            assert token in md
        # Per-lesson tally: one better, one worse, one lesson at each.
        assert "| `SPLITTER_SPLIT` | 1 | 0 | 1 | 0.900 | 0.100 | -0.800 |" in md
        assert "μ main thput" in md and "MAIN" not in md.split("```")[0]
        assert "| `MOVE_ONE_ITEM` | 2 | 1 | 0 | 0.650 | 0.700 | +0.050 |" in md

    def test_identical_sides_say_so(self):
        md = diff_markdown([[_record("MOVE_ONE_ITEM", 1, 0.4)]], [[_record("MOVE_ONE_ITEM", 1, 0.4)]])
        assert "No factory moved consistently" in md
        assert "<details>" not in md

    def test_unpaired_factories_ignored(self):
        md = diff_markdown(
            [[_record("MOVE_ONE_ITEM", 1, 0.4)]], [[_record("MOVE_ONE_ITEM", 9, 0.9)]]
        )
        assert md == ""

    def test_caps_the_number_of_renders(self):
        """Even a tiny-grid report has to stay readable."""
        pr = [[_record("MOVE_ONE_ITEM", i, 1.0) for i in range(40)]]
        main = [[_record("MOVE_ONE_ITEM", i, 0.0) for i in range(40)]]
        md = diff_markdown(pr, main, max_renders=3)
        assert md.count("Δthput") == 3
        assert "37 further factories omitted; the 3 largest gaps are shown." in md
        assert md.rstrip().endswith("</details>")

    def test_note_lands_under_the_intro(self):
        """Provenance links belong where a reader lands, above the grids."""
        md = diff_markdown(
            [[_record("MOVE_ONE_ITEM", 1, 0.4)]],
            [[_record("MOVE_ONE_ITEM", 1, 0.9)]],
            note="seed 1: PR [`abc`](u1) vs main [`def`](u2)",
        )
        assert md.index("[`abc`](u1)") < md.index("| Lesson |")

    def test_side_labels_carry_through(self):
        md = diff_markdown(
            [[_record("MOVE_ONE_ITEM", 1, 0.4)]],
            [[_record("MOVE_ONE_ITEM", 1, 0.9)]],
            pr_label="cmp-abc-pr",
            main_label="cmp-abc-main",
        )
        assert "cmp-abc-main vs cmp-abc-pr" in md
        assert "cmp-abc-pr  thput 0.400" in md
        assert "cmp-abc-main  thput 0.900" in md


class TestMultiSeedSides:
    """With several training seeds a side, only a factory that moved for EVERY
    seed counts — one seed wandering is noise, not a property of the branch."""

    def _sides(self, pr_thputs, main_thputs):
        pr = [[_record("MOVE_ONE_ITEM", 1, t, render="PR")] for t in pr_thputs]
        main = [[_record("MOVE_ONE_ITEM", 1, t, render="MAIN")] for t in main_thputs]
        return diff_markdown(pr, main)

    def test_separated_ranges_are_reported(self):
        md = self._sides([0.7, 0.8, 0.6], [0.5, 0.5, 0.5])
        assert "1 came out consistently different" in md
        assert "Δthput +0.200" in md  # mean 0.7 vs mean 0.5
        # Every seed's value is shown next to the representative factory.
        assert "thput 0.700 [0.60 0.70 0.80]" in md
        assert "thput 0.500 [0.50 0.50 0.50]" in md

    def test_one_seed_wandering_is_not_reported(self):
        md = self._sides([0.5, 0.5, 0.6], [0.5, 0.5, 0.5])
        assert "No factory moved consistently" in md

    def test_a_side_that_is_worse_everywhere_counts(self):
        md = self._sides([0.1, 0.2, 0.3], [0.4, 0.5, 0.9])
        assert "Δthput -0.400" in md
        assert "| `MOVE_ONE_ITEM` | 1 | 0 | 1 | 0.600 | 0.200 | -0.400 |" in md

    def test_factories_missing_from_a_run_are_dropped(self):
        """A factory only some runs scored can't support a verdict."""
        pr = [
            [_record("MOVE_ONE_ITEM", 1, 0.9), _record("MOVE_ONE_ITEM", 2, 0.9)],
            [_record("MOVE_ONE_ITEM", 1, 0.9)],
        ]
        main = [[_record("MOVE_ONE_ITEM", 1, 0.1), _record("MOVE_ONE_ITEM", 2, 0.1)]] * 2
        md = diff_markdown(pr, main)
        assert "1 held-out factories" in md
        assert "factory 2" not in md
