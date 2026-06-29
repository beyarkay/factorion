"""Pytest conftest: ensure the tests directory is on sys.path so helpers.py can be imported."""

import os
import sys

# Add tests/ directory to sys.path so `from helpers import ...` works
sys.path.insert(0, os.path.dirname(__file__))

# The CPU-fallback guard (ppo.assert_device_ok) aborts on CPU everywhere except
# CI. The test suite trains on CPU (sft end-to-end tests), so mark the session
# as CI even when run locally on a CPU-only box. setdefault preserves a real CI
# value. GitHub Actions already sets CI=true.
os.environ.setdefault("CI", "true")
