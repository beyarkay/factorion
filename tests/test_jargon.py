"""Words the codebase reserves for one meaning (see CLAUDE.md's Jargon)."""

import re
import subprocess
from pathlib import Path

# Spelled in halves so this file doesn't trip its own check.
_WORD = "wa" + "ll"


def test_only_the_stone_wall_entity_has_that_name():
    """An unbuildable tile is "unbuildable"; the word is the stone_wall's alone."""
    root = Path(__file__).resolve().parent.parent
    hits = subprocess.run(["git", "grep", "-n", "-i", "-I", _WORD], cwd=root, capture_output=True, text=True).stdout
    strays = [h for h in hits.splitlines() if re.search(_WORD, re.sub(f"stone[-_ ]?{_WORD}", "", h, flags=re.I), re.I)]
    assert not strays, "\n".join(strays)
