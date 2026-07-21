"""Tests for spatial per-tile action prediction in AgentCNN."""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, PpoArgs, FactorioEnv, make_env  # noqa: E402
from training_config import SharedArgs  # noqa: E402
from helpers import Channel  # noqa: E402

NUM_CHANNELS = len(Channel)


ENV_ID = "factorion/FactorioEnv-v0-spatial-test"


@pytest.fixture(scope="module")
def registered_env():
    """Register the env once for all tests in this module."""
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


@pytest.fixture()
def envs(registered_env):
    """Create a small SyncVectorEnv for testing."""
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, 5, "test") for i in range(2)]
    )


@pytest.fixture()
def agent(envs):
    """Create an AgentCNN with default params."""
    return AgentCNN(envs, layers=(32, 64, 64))


class TestForwardPass:
    def test_output_shapes(self, agent):
        """Verify all output tensor shapes from get_action_and_value."""
        obs = torch.randn(2, NUM_CHANNELS, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)

        assert action_out["xy"].shape == (2, 2)
        assert action_out["entity"].shape == (2,)
        assert action_out["direction"].shape == (2,)
        assert action_out["item"].shape == (2,)
        assert action_out["misc"].shape == (2,)
        assert logp_B.shape == (2,)
        assert entropy_B.shape == (2,)
        assert value_B.shape == (2,)

    def test_single_batch(self, agent):
        """Verify forward pass works with batch size 1."""
        obs = torch.randn(1, NUM_CHANNELS, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)

        assert action_out["xy"].shape == (1, 2)
        assert logp_B.shape == (1,)
        assert entropy_B.shape == (1,)
        assert value_B.shape == (1,)

    def test_xy_within_bounds(self, agent):
        """Verify sampled x, y are within grid bounds."""
        obs = torch.randn(16, NUM_CHANNELS, 5, 5)
        for _ in range(10):
            action_out, _, _, _ = agent.get_action_and_value(obs)
            x = action_out["xy"][:, 0]
            y = action_out["xy"][:, 1]
            assert (x >= 0).all() and (x < 5).all(), f"x out of bounds: {x}"
            assert (y >= 0).all() and (y < 5).all(), f"y out of bounds: {y}"

    def test_get_value_shape(self, agent):
        """Verify get_value output shape."""
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        value = agent.get_value(obs)
        assert value.shape == (4,)

    def test_item_and_misc_within_head_bounds(self, agent):
        """Item and misc are now sampled from learned heads, so they must
        be valid indices into items/Misc — not the old hardcoded 0."""
        obs = torch.randn(8, NUM_CHANNELS, 5, 5)
        action_out, _, _, _ = agent.get_action_and_value(obs)
        assert (action_out["item"] >= 0).all()
        assert (action_out["item"] < agent.num_items).all()
        assert (action_out["misc"] >= 0).all()
        assert (action_out["misc"] < agent.num_misc).all()


class TestLogProbConsistency:
    def test_log_prob_matches_replay(self, agent):
        """Sample actions, then replay them and verify log_prob matches."""
        obs = torch.randn(8, NUM_CHANNELS, 5, 5)
        action_out, logp_B, _, _ = agent.get_action_and_value(obs)

        # Reconstruct action tensor as the training loop does
        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        item_B = action_out["item"]
        misc_B = action_out["misc"]
        eot_B = action_out["eot"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, item_B, misc_B, eot_B], dim=1
        )

        # Replay: pass the same obs and action tensor
        _, logp_replay, _, _ = agent.get_action_and_value(
            obs, action_tensor.long()
        )
        torch.testing.assert_close(logp_B, logp_replay)

    def test_log_prob_is_negative(self, agent):
        """Log probabilities should be negative (prob < 1)."""
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        _, logp_B, _, _ = agent.get_action_and_value(obs)
        assert (logp_B < 0).all()


class TestEntropy:
    def test_entropy_is_positive(self, agent):
        """Entropy should be non-negative."""
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        _, _, entropy_B, _ = agent.get_action_and_value(obs)
        assert (entropy_B >= 0).all()


class TestGradientFlow:
    def test_gradients_flow_through_all_params(self, agent):
        """Verify gradients flow to encoder, tile_logits, and all four
        per-tile heads (entity, direction, item, misc)."""
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        # Check tile_logits conv has gradients
        assert agent.tile_logits.weight.grad is not None
        assert agent.tile_logits.weight.grad.abs().sum() > 0

        # Check entity/direction/item/misc heads have gradients
        for head_name in ("ent_head", "dir_head", "item_head", "misc_head"):
            head = getattr(agent, head_name)
            assert head.weight.grad is not None, f"No grad for {head_name}.weight"
            assert head.weight.grad.abs().sum() > 0, f"Zero grad for {head_name}.weight"

        # Check encoder has gradients
        for name, param in agent.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_during_update(self, agent):
        """Simulate the PPO update path (action not None) and check grads
        propagate through every head."""
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        with torch.no_grad():
            action_out, _, _, _ = agent.get_action_and_value(obs)
        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        item_B = action_out["item"]
        misc_B = action_out["misc"]
        eot_B = action_out["eot"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, item_B, misc_B, eot_B], dim=1,
        )

        _, logp_B, entropy_B, value_B = agent.get_action_and_value(
            obs, action_tensor.long()
        )
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        # eot_head is part of the joint action distribution, so its grad must
        # flow too (this is what lets PPO train the stop decision).
        for head_name in ("tile_logits", "ent_head", "dir_head", "item_head", "misc_head", "eot_head"):
            head = getattr(agent, head_name)
            weight = head.weight if hasattr(head, "weight") else head[-1].weight
            assert weight.grad is not None, f"No grad for {head_name}"


class TestBatchConsistency:
    def test_single_vs_batch(self, agent):
        """Processing items one-at-a-time gives same results as batched."""
        obs = torch.randn(3, NUM_CHANNELS, 5, 5)
        # Create a fixed action to replay (within bounds for 5x5 grid)
        # 7 columns: xy(2), entity, direction, item, misc, eot
        action_tensor = torch.tensor(
            [
                [2, 3, 1, 2, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [4, 4, 1, 4, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        # Batched
        _, logp_batch, entropy_batch, value_batch = agent.get_action_and_value(
            obs, action_tensor
        )

        # One at a time
        for i in range(3):
            _, logp_single, entropy_single, value_single = (
                agent.get_action_and_value(obs[i : i + 1], action_tensor[i : i + 1])
            )
            torch.testing.assert_close(logp_batch[i : i + 1], logp_single)
            torch.testing.assert_close(entropy_batch[i : i + 1], entropy_single)
            torch.testing.assert_close(value_batch[i : i + 1], value_single)


class TestRegularisation:
    """Dropout + weight-decay knobs. Both CLI args default to a no-op so
    existing runs are unchanged; these tests pin that contract and verify
    a non-zero dropout actually regularises."""

    def test_arg_defaults_are_noop(self):
        """The CLI defaults must leave training behaviour unchanged."""
        assert PpoArgs().dropout == 0.0
        assert PpoArgs().weight_decay == 0.0

    def test_default_agent_carries_inert_dropout(self, agent):
        """Default AgentCNN has Dropout2d layers (so the knob exists) but
        at p=0 they are the identity."""
        drops = [m for m in agent.encoder if isinstance(m, torch.nn.Dropout2d)]
        assert len(drops) == 3
        assert all(d.p == 0.0 for d in drops)

    def test_dropout_zero_is_deterministic_in_train(self, agent):
        """p=0 in train() mode is a true no-op: repeated encodes match."""
        agent.train()
        # These probe the conv stack directly, downstream of the categorical
        # input encoding, so feed it the encoder's expanded input channels.
        obs = torch.randn(2, agent.input_channels, 5, 5)
        torch.testing.assert_close(agent.encoder(obs), agent.encoder(obs))

    def test_dropout_active_varies_in_train(self, envs):
        """p>0 in train() mode resamples the mask each pass, so the only
        source of variation — dropout — makes two encodes differ."""
        agent = AgentCNN(envs, layers=(32, 64, 64), dropout=0.5)
        agent.train()
        obs = torch.randn(2, agent.input_channels, 5, 5)
        assert not torch.allclose(agent.encoder(obs), agent.encoder(obs))

    def test_dropout_inert_in_eval(self, envs):
        """Dropout is disabled in eval() regardless of p, so inference is
        deterministic even with a high drop probability."""
        agent = AgentCNN(envs, layers=(32, 64, 64), dropout=0.5)
        agent.eval()
        obs = torch.randn(2, agent.input_channels, 5, 5)
        torch.testing.assert_close(agent.encoder(obs), agent.encoder(obs))

    def test_weight_decay_default_adds_no_penalty(self, agent):
        """Mirror ppo.py's optimiser wiring: the default weight_decay puts
        no L2 penalty in the Adam param group."""
        import torch.optim as optim

        opt = optim.Adam(agent.parameters(), weight_decay=PpoArgs().weight_decay)
        assert opt.param_groups[0]["weight_decay"] == 0.0
        tuned = optim.Adam(agent.parameters(), weight_decay=0.01)
        assert tuned.param_groups[0]["weight_decay"] == 0.01


class TestCategoricalInputEncoding:
    """The nominal observation channels (entity/item/direction/misc) are
    encoded categorically before the conv — embeddings for the two wide
    vocabularies, one-hots for the two narrow ones, footprint left scalar —
    rather than fed as raw ordinal id floats."""

    def test_encoded_shape(self, agent):
        """_encode_input expands len(Channel) into the conv's input channels:
        two embeddings + two one-hots + footprint + two coordinate planes."""
        d = agent.cat_embed_dim
        assert agent.input_channels == 2 * d + agent.num_directions + agent.num_misc + 1 + 2
        enc = agent._encode_input(torch.zeros(2, NUM_CHANNELS, 5, 5))
        assert enc.shape == (2, agent.input_channels, 5, 5)

    def test_direction_channel_is_one_hot(self, agent):
        """A direction id lands as a one-hot in the direction slice, not a
        scalar magnitude."""
        obs = torch.zeros(1, NUM_CHANNELS, 5, 5)
        obs[0, Channel.DIRECTION.value, 1, 1] = 2.0
        d = agent.cat_embed_dim
        dir_slice = agent._encode_input(obs)[0, 2 * d : 2 * d + agent.num_directions, 1, 1]
        assert dir_slice.argmax().item() == 2
        assert dir_slice.sum().item() == pytest.approx(1.0)

    def test_distinct_entities_get_independent_encodings(self, agent):
        """The whole point: two different entity ids map to independent
        embeddings, not scalar multiples as the old float channel implied."""
        obs1 = torch.zeros(1, NUM_CHANNELS, 5, 5)
        obs2 = torch.zeros(1, NUM_CHANNELS, 5, 5)
        obs1[0, Channel.ENTITIES.value, 1, 1] = 1.0
        obs2[0, Channel.ENTITIES.value, 1, 1] = 3.0
        d = agent.cat_embed_dim
        e1 = agent._encode_input(obs1)[0, :d, 1, 1]
        e2 = agent._encode_input(obs2)[0, :d, 1, 1]
        assert not torch.allclose(e1, e2)

    def test_embedding_gradients_flow(self, agent):
        """Gradients reach the new entity/item embedding tables through a
        full forward + backward."""
        obs = torch.zeros(2, NUM_CHANNELS, 5, 5)
        obs[:, Channel.ENTITIES.value] = 1.0
        obs[:, Channel.ITEMS.value] = 2.0
        _, logp_B, _, value_B = agent.get_action_and_value(obs)
        (logp_B.mean() + value_B.mean()).backward()
        assert agent.ent_embed.weight.grad is not None
        assert agent.item_embed.weight.grad is not None


# `attn_dim` is the swept dial (0 = conv-only ablation); the rest are knobs
# fixed at their winning defaults but still overridable. Each entry must
# construct, run the full PPO forward (sample + stored-action recompute), and
# round-trip through a state dict into a fresh same-config model.
ARCH_VARIANTS = [
    {},              # default: attention on at the ctor default dim
    {"attn_dim": 0},   # conv-only ablation baseline
    {"attn_dim": 16},
    {"attn_dim": 32},
    {"global_feat_dim": 0},   # no pooled global vector
    {"attn_pos_embed": 0},    # attention without positional embedding
    {"attn_dim": 24, "attn_heads": 4, "attn_layers": 1},
    {"attn_dim": 0, "global_feat_dim": 0},  # pure conv, window-only heads
]


class TestArchVariants:
    @pytest.mark.parametrize("kwargs", ARCH_VARIANTS)
    def test_forward_and_stored_action_recompute(self, envs, kwargs):
        agent = AgentCNN(envs, layers=(16, 16, 16), **kwargs)
        obs = torch.zeros(4, NUM_CHANNELS, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)
        assert logp_B.shape == entropy_B.shape == value_B.shape == (4,)
        assert torch.isfinite(logp_B).all() and torch.isfinite(value_B).all()
        # PPO-update path: recompute log-probs for the sampled action.
        stored = torch.cat(
            [
                action_out["xy"],
                action_out["entity"][:, None],
                action_out["direction"][:, None],
                action_out["item"][:, None],
                action_out["misc"][:, None],
                action_out["eot"][:, None].long(),
            ],
            dim=1,
        )
        _, logp2_B, _, _ = agent.get_action_and_value(obs, action=stored)
        torch.testing.assert_close(logp_B, logp2_B)

    @pytest.mark.parametrize("kwargs", ARCH_VARIANTS)
    def test_state_dict_roundtrip(self, envs, kwargs):
        agent = AgentCNN(envs, layers=(16, 16, 16), **kwargs)
        clone = AgentCNN(envs, layers=(16, 16, 16), **kwargs)
        clone.load_state_dict(agent.state_dict())
        obs = torch.zeros(2, NUM_CHANNELS, 5, 5)
        torch.manual_seed(0)
        _, logp_a, _, val_a = agent.get_action_and_value(obs)
        torch.manual_seed(0)
        _, logp_b, _, val_b = clone.get_action_and_value(obs)
        torch.testing.assert_close(logp_a, logp_b)
        torch.testing.assert_close(val_a, val_b)

    def test_coord_channels_always_present(self, envs):
        """CoordConv is fixed on: two extra normalized x/y planes are always
        appended to the encoder input."""
        agent = AgentCNN(envs, layers=(16, 16, 16))
        enc = agent._encode_input(torch.zeros(1, NUM_CHANNELS, 5, 5))
        assert enc.shape == (1, agent.input_channels, 5, 5)
        x_plane, y_plane = enc[0, -2], enc[0, -1]
        assert x_plane[0, 0].item() == pytest.approx(-1.0)
        assert x_plane[4, 0].item() == pytest.approx(1.0)
        assert y_plane[0, 0].item() == pytest.approx(-1.0)
        assert y_plane[0, 4].item() == pytest.approx(1.0)
        # x varies along W only, y along H only.
        torch.testing.assert_close(x_plane[:, 0], x_plane[:, 4])
        torch.testing.assert_close(y_plane[0, :], y_plane[4, :])

    def test_critic_and_eot_heads_flatten_the_map(self, envs):
        """Both value/eot heads are Flatten -> Linear, and `head[-1]` reaches
        the Linear (ppo.py's --start-from critic re-init depends on it)."""
        agent = AgentCNN(envs, layers=(16, 16, 16), attn_dim=0)
        assert isinstance(agent.critic_head[-1], torch.nn.Linear)
        assert isinstance(agent.eot_head[-1], torch.nn.Linear)
        assert agent.critic_head[-1].in_features == 16 * 5 * 5
        _, _, _, value_B = agent.get_action_and_value(torch.zeros(2, NUM_CHANNELS, 5, 5))
        assert value_B.shape == (2,)

    def test_attn_on_by_default_off_when_zero(self, envs):
        on = AgentCNN(envs, layers=(16, 16, 16))
        assert on.attn_dim > 0 and hasattr(on, "attn")
        # Head count and depth default to the SharedArgs (swept-winning) values.
        assert len(on.attn.transformer.layers) == SharedArgs.attn_layers
        assert on.attn.transformer.layers[0].self_attn.num_heads == SharedArgs.attn_heads
        # Positional embedding defaults on (one learned vector per grid cell).
        assert on.attn.pos_embed is not None
        assert on.attn.pos_embed.shape == (1, 25, on.attn_dim)

        off = AgentCNN(envs, layers=(16, 16, 16), attn_dim=0)
        assert off.attn_dim == 0 and not hasattr(off, "attn")

    def test_global_feat_dim_zero_disables_global_vector(self, envs):
        agent = AgentCNN(envs, layers=(16, 16, 16), global_feat_dim=0)
        assert not hasattr(agent, "global_proj")
        enc, g = agent.encode(torch.zeros(2, NUM_CHANNELS, 5, 5))
        assert g is None
        # Per-tile heads then read just the last_chan-wide column.
        assert agent.ent_head.in_features == 16
        _, logp_B, _, value_B = agent.get_action_and_value(torch.zeros(2, NUM_CHANNELS, 5, 5))
        assert torch.isfinite(logp_B).all() and torch.isfinite(value_B).all()

    def test_attn_is_identity_at_init(self, envs):
        """The out projection is zero-initialised, so the residual attention
        stage is exactly identity at init — the trunk starts as the plain
        conv encoder and only leans on attention as training moves it."""
        agent = AgentCNN(envs, layers=(16, 16, 16), attn_dim=16)
        x = torch.randn(3, 16, 5, 5)
        torch.testing.assert_close(agent.attn(x), x)

    def test_attn_preserves_channels_and_grid_size(self, envs):
        agent = AgentCNN(envs, layers=(16, 16, 16), attn_dim=32)
        enc, _ = agent.encode(torch.zeros(2, NUM_CHANNELS, 5, 5))
        # Attention is shape-preserving, so the per-tile heads still index a
        # last_chan-wide column at every one of the 5x5 cells.
        assert enc.shape == (2, 16, 5, 5)
        assert agent.tile_logits.in_channels == 16

    def test_attn_heads_snap_to_divisor(self, envs):
        """A small attn_dim not divisible by the fixed head count must still
        construct — heads snap down to the largest divisor."""
        agent = AgentCNN(envs, layers=(16, 16, 16), attn_dim=4)
        num_heads = agent.attn.transformer.layers[0].self_attn.num_heads
        assert 4 % num_heads == 0 and num_heads == 4
