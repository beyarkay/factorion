"""Pytest conftest: ensure the tests directory is on sys.path so helpers.py can be imported."""

import os
import sys

import torch

# Without the device, torch.manual_seed queues its seed for a device init that
# never comes, capturing a full stack trace each time. build_factory seeds
# torch on every call, so under pytest's deep stacks this dominates any test
# that scans seeds.
for _device in (torch.cuda, torch.xpu, torch.mtia):
    if not _device.is_available():
        setattr(_device, "manual_seed_all", lambda seed: None)

# Add tests/ directory to sys.path so `from helpers import ...` works
sys.path.insert(0, os.path.dirname(__file__))

# The CPU-fallback guard (ppo.assert_device_ok) aborts on CPU everywhere except
# CI and under pytest. The test suite trains on CPU (sft/ppo end-to-end tests);
# the guard recognises the pytest-set PYTEST_CURRENT_TEST env var, so local
# `uv run pytest` runs CPU-only smoke tests without any CI=true env hack.

os.environ["FACTORION_TRAIN_ALL_KINDS"] = "1"
