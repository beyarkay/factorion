"""End-to-end tests for the always-on CPU-fallback guard
(``ppo.assert_device_ok``).

A GPU host whose NVIDIA driver is too old for the installed torch CUDA build
makes torch silently select ``cpu`` — training then "works" but crawls. The
guard turns that silent fallback into a hard error. CUDA (pods) and Apple MPS
(local Mac dev) are accepted; CPU aborts — except under CI, which runs CPU-only
smoke tests.
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
    """Off CI, CPU aborts — the silent-fallback case we hit on a pod."""
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(RuntimeError, match="Refusing to train on 'cpu'"):
        assert_device_ok(CPU)


def test_cpu_allowed_in_ci(monkeypatch):
    """CI (CI=true, set by GitHub Actions) runs CPU-only smoke tests."""
    monkeypatch.setenv("CI", "true")
    assert_device_ok(CPU)  # must not raise
