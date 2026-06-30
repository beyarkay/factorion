"""Pytest conftest: ensure the tests directory is on sys.path so helpers.py can be imported."""

import os
import sys

# Add tests/ directory to sys.path so `from helpers import ...` works
sys.path.insert(0, os.path.dirname(__file__))

# The CPU-fallback guard (ppo.assert_device_ok) aborts on CPU everywhere except
# CI and under pytest. The test suite trains on CPU (sft/ppo end-to-end tests);
# the guard recognises the pytest-set PYTEST_CURRENT_TEST env var, so local
# `uv run pytest` runs CPU-only smoke tests without any CI=true env hack.
