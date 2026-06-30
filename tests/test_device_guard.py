"""End-to-end tests for the always-on CPU-fallback guard
(``ppo.assert_device_ok``).

A GPU host whose NVIDIA driver is too old for the installed torch CUDA build
makes torch silently select ``cpu`` — training then "works" but crawls. The
guard turns that silent fallback into a hard error. CUDA (pods) and Apple MPS
(local Mac dev) are accepted; CPU aborts — except under CI or pytest, which run
CPU-only smoke tests.
"""

import pytest
import torch

from ppo import assert_device_ok

CPU = torch.device("cpu")
MPS = torch.device("mps")
CUDA = torch.device("cuda")


def test_cuda_accepted():
    """A real CUDA device (GPU pod) always passes."""
    assert_device_ok(CUDA)  # must not raise


def test_mps_accepted():
    """Local Mac dev (Apple MPS) keeps working."""
    assert_device_ok(MPS)  # must not raise


def test_cpu_rejected(monkeypatch):
    """Off CI and outside pytest, CPU aborts — the silent-fallback case we hit
    on a pod. Both signals must be cleared: pytest sets PYTEST_CURRENT_TEST for
    every test, which the guard would otherwise honour."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with pytest.raises(RuntimeError, match="Refusing to train on 'cpu'"):
        assert_device_ok(CPU)


def test_cpu_allowed_in_ci(monkeypatch):
    """CI (CI=true, set by GitHub Actions) runs CPU-only smoke tests."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("CI", "true")
    assert_device_ok(CPU)  # must not raise


def test_cpu_allowed_under_pytest(monkeypatch):
    """The local test suite (pytest sets PYTEST_CURRENT_TEST) runs CPU-only
    smoke tests without needing CI=true in the environment."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_device_guard.py::x")
    assert_device_ok(CPU)  # must not raise
