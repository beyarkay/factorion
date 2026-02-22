"""Pytest conftest: ensure the tests directory is on sys.path so helpers.py can be imported."""

import os
import sys

# Add tests/ directory to sys.path so `from helpers import ...` works
sys.path.insert(0, os.path.dirname(__file__))
