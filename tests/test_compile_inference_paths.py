"""End-to-end tests for ``ppo.maybe_compile``.

`reduce-overhead` exists to get CUDA-graph replay, so it is CUDA-only. Off CUDA
it bought nothing but startup latency, and worse: inductor's CPU vectoriser
miscodegens the `_legal_tile_mask` select (mixing `VecMask<int,1>` and
`VecMask<float,1>` in one ternary), so every CPU run — including the CPU-only
smoke test — died in the C++ backend before the first rollout step.
"""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, make_env, maybe_compile  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-maybe-compile-test"
SIZE = 5


@pytest.fixture(scope="module")
def envs():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, SIZE, "test") for i in range(2)]
    )


@pytest.fixture(scope="module")
def agent(envs):
    return AgentCNN(envs, layers=(16, 16, 16))


def test_cpu_returns_fn_untouched(agent):
    """On CPU the callable comes back as-is, so inductor is never invoked."""
    fn = maybe_compile(agent.get_action_and_value, torch.device("cpu"))

    assert fn == agent.get_action_and_value
    # A dynamo wrapper would carry this.
    assert not hasattr(fn, "_torchdynamo_orig_callable")


@pytest.mark.parametrize("device_type", ["cuda", "mps"])
def test_accelerators_still_compile(agent, device_type):
    """Only CPU is excluded — the bug is in inductor's C++ backend, which MPS
    (Metal) and CUDA never reach. torch.compile is lazy, so this asserts the
    wrapper was requested without needing the hardware present."""
    fn = maybe_compile(agent.get_action_and_value, torch.device(device_type))

    assert fn != agent.get_action_and_value
    assert fn._torchdynamo_orig_callable == agent.get_action_and_value


def test_uncompiled_paths_still_act(agent, envs):
    """The passed-through callables really run the policy, so the rollout gets
    past the step that used to die in the C++ backend."""
    cpu = torch.device("cpu")
    rollout_act = maybe_compile(agent.get_action_and_value, cpu)
    rollout_value = maybe_compile(agent.get_value, cpu)
    obs = torch.randn(2, envs.single_observation_space.shape[0], SIZE, SIZE)

    with torch.no_grad():
        action_out, logp_B, entropy_B, value_B = rollout_act(obs)
        value_only_B = rollout_value(obs)

    assert action_out["xy"].shape == (2, 2)
    assert logp_B.shape == (2,)
    assert entropy_B.shape == (2,)
    assert value_B.shape == (2,)
    assert value_only_B.shape == (2,)
