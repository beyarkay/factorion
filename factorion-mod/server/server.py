"""Factorion model server.

Polls the Factorio mod for requests, runs the trained AgentCNN policy, and
streams each predicted entity back into the running game over RCON.

Run from the repo root so the `factorion` and `factorion_rs` modules are
importable (e.g. via `uv run python factorion-mod/server/server.py ...`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np
import torch

# Make the repo root importable when this script is run via uv from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from factorion import Channel, Misc, entities, items, str2ent  # noqa: E402
from ppo import AgentCNN, _resolve_wandb_checkpoint  # noqa: E402

import factorion_rs  # noqa: E402

from blueprint import _DIR_MODEL_TO_BP, _hyphenate  # noqa: E402

log = logging.getLogger("factorion-server")
MOD_GRID_SIZE = 11


# --------------------------------------------------------------------------- #
# RCON client: shared with parity.py, see rcon.py.
# --------------------------------------------------------------------------- #

from rcon import RconClient, RconError  # noqa: E402, F401


# --------------------------------------------------------------------------- #
# Model loading & inference.
# --------------------------------------------------------------------------- #

def _duck_envs(size: int):
    """Minimal vector-env shape surface needed to construct AgentCNN."""
    return SimpleNamespace(
        single_observation_space=SimpleNamespace(shape=(len(Channel), size, size)),
    )


@dataclass
class Hyperparams:
    grid_size: int = 11
    layers: tuple[int, ...] = (93, 69, 96)
    kernel_size: int = 3
    attn_dim: int = 0
    attn_heads: int = 12
    attn_layers: int = 4
    attn_pos_embed: int = 1
    global_feat_dim: int = 0

    @classmethod
    def from_mapping(cls, values: dict) -> "Hyperparams":
        """Read either current layer1..8 config or the legacy chan1..3 form."""
        size = int(values.get("size", values.get("grid_size", cls.grid_size)))
        if "layers" in values:
            layers = tuple(int(v) for v in values["layers"] if int(v) > 0)
        elif "layer1" in values:
            layers = tuple(
                int(values.get(f"layer{i}", 0))
                for i in range(1, 9)
                if int(values.get(f"layer{i}", 0)) > 0
            )
        else:
            layers = tuple(
                int(values.get(f"chan{i}", default))
                for i, default in enumerate(cls.layers, start=1)
            )
        if not layers:
            raise ValueError("checkpoint config has no positive-width encoder layers")
        return cls(
            grid_size=size,
            layers=layers,
            kernel_size=int(values.get("kernel_size", 3)),
            attn_dim=int(values.get("attn_dim", 0)),
            attn_heads=int(values.get("attn_heads", 12)),
            attn_layers=int(values.get("attn_layers", 4)),
            attn_pos_embed=int(values.get("attn_pos_embed", 1)),
            global_feat_dim=int(values.get("global_feat_dim", 0)),
        )

    @classmethod
    def from_json_sibling(cls, ckpt_path: Path) -> "Hyperparams":
        sidecar = ckpt_path.with_suffix(".hp.json")
        if sidecar.exists():
            with sidecar.open() as f:
                return cls.from_mapping(json.load(f))
        return cls()


def _wandb_run_path(spec: str) -> str:
    """Accept a bare run id, entity/project/id, or a normal W&B run URL."""
    if spec.startswith(("https://", "http://")):
        parts = [p for p in urlparse(spec).path.split("/") if p]
        try:
            runs_i = parts.index("runs")
            return "/".join((parts[runs_i - 2], parts[runs_i - 1], parts[runs_i + 1]))
        except (ValueError, IndexError):
            raise ValueError(f"not a W&B run URL: {spec}") from None
    return spec


def resolve_checkpoint(
    spec: str, project: str = "factorion", entity: Optional[str] = None,
) -> tuple[Path, Optional[Hyperparams], Optional[dict]]:
    """Resolve a local checkpoint or W&B run and recover its architecture."""
    local = Path(spec).expanduser()
    if local.exists():
        return local.resolve(), None, None
    if spec.endswith(".pt"):
        raise FileNotFoundError(f"checkpoint does not exist: {local}")

    path, source = _resolve_wandb_checkpoint(_wandb_run_path(spec), project, entity)
    config = source.get("config") or {}
    hp = Hyperparams.from_mapping(config)
    return Path(path), hp, source


def load_agent(ckpt_path: Path, hp: Hyperparams, device: torch.device) -> AgentCNN:
    log.info(
        "Loading checkpoint %s (grid=%d, layers=%s, kernel=%d, attention=%d, global=%d)",
        ckpt_path, hp.grid_size, "/".join(map(str, hp.layers)), hp.kernel_size,
        hp.attn_dim, hp.global_feat_dim,
    )
    agent = AgentCNN(
        _duck_envs(hp.grid_size),
        layers=hp.layers,
        kernel_size=hp.kernel_size,
        attn_dim=hp.attn_dim,
        attn_heads=hp.attn_heads,
        attn_layers=hp.attn_layers,
        attn_pos_embed=hp.attn_pos_embed,
        global_feat_dim=hp.global_feat_dim,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    current = agent.state_dict()
    expandable = {
        "ent_embed.weight",
        "item_embed.weight",
        "ent_head.weight",
        "ent_head.bias",
        "item_head.weight",
        "item_head.bias",
    }
    expanded = []
    for key, saved in state.items():
        target = current.get(key)
        if target is None or target.shape == saved.shape:
            continue
        can_expand_coord_channels = (
            key == "encoder.0.weight"
            and target.ndim == saved.ndim == 4
            and target.shape[0] == saved.shape[0]
            and target.shape[2:] == saved.shape[2:]
            and target.shape[1] == saved.shape[1] + 2
        )
        if can_expand_coord_channels:
            merged = target.new_zeros(target.shape)
            merged[:, :saved.shape[1]] = saved
            state[key] = merged
            expanded.append(
                f"{key} input channels:{saved.shape[1]}→{target.shape[1]}"
            )
            continue
        can_expand = (
            key in expandable
            and target.ndim == saved.ndim
            and target.shape[1:] == saved.shape[1:]
            and target.shape[0] > saved.shape[0]
        )
        if not can_expand:
            raise RuntimeError(
                f"checkpoint tensor {key} has shape {tuple(saved.shape)}, "
                f"current model expects {tuple(target.shape)}"
            )
        merged = target.clone()
        merged[:saved.shape[0]] = saved
        # New catalog entries were not present during training. Keep random
        # embeddings for input compatibility, but make new output rows lose
        # argmax so loading old checkpoints cannot emit unseen recipes.
        if key in {"ent_head.weight", "item_head.weight"}:
            merged[saved.shape[0]:].zero_()
        elif key in {"ent_head.bias", "item_head.bias"}:
            merged[saved.shape[0]:].fill_(-1e9)
        state[key] = merged
        expanded.append(f"{key}:{saved.shape[0]}→{target.shape[0]}")
    if expanded:
        log.info("Expanded append-only catalog tensors: %s", ", ".join(expanded))
    if "coord_grid" not in state:
        state["coord_grid"] = current["coord_grid"]
    agent.load_state_dict(state)
    agent.eval()
    return agent


# --------------------------------------------------------------------------- #
# Request → obs tensor.
# --------------------------------------------------------------------------- #

def _source_id() -> int:
    e = str2ent("source")
    assert e is not None, "factorion is missing the 'source' entity"
    return e.value


def _sink_id() -> int:
    e = str2ent("sink")
    assert e is not None, "factorion is missing the 'sink' entity"
    return e.value


def request_to_obs(req: dict) -> np.ndarray:
    """Build the (C, W, H) tensor the policy expects from a request dict."""
    size = req["grid_size"]
    C = len(Channel)
    obs = np.zeros((C, size, size), dtype=np.float32)

    # Footprint mask
    for x, y in req["footprint"]:
        obs[Channel.FOOTPRINT.value, x, y] = 1.0

    src_id, snk_id = _source_id(), _sink_id()

    for s in req.get("sources", []):
        x, y = s["x"], s["y"]
        obs[Channel.ENTITIES.value, x, y] = src_id
        obs[Channel.DIRECTION.value, x, y] = s["direction"]
        item = str2ent(s["item"])
        if item is not None:
            obs[Channel.ITEMS.value, x, y] = item.value

    for s in req.get("sinks", []):
        x, y = s["x"], s["y"]
        obs[Channel.ENTITIES.value, x, y] = snk_id
        obs[Channel.DIRECTION.value, x, y] = s["direction"]
        item = str2ent(s["item"])
        if item is not None:
            obs[Channel.ITEMS.value, x, y] = item.value

    return obs


# --------------------------------------------------------------------------- #
# Iterative inference: place one entity at a time, greedy (argmax).
# --------------------------------------------------------------------------- #

def _argmax_action(agent: AgentCNN, obs_CWH: np.ndarray, device) -> dict:
    x = torch.from_numpy(obs_CWH).unsqueeze(0).to(device)
    with torch.no_grad():
        # temperature=0 = the shared sampler's greedy (argmax) mode.
        act = agent.sample_action(
            x, temperature=0.0, legal_mask=True, compute_value=False,
        )["action"]
    return {
        "xy": (int(act["xy"][0, 0]), int(act["xy"][0, 1])),
        "entity": int(act["entity"].item()),
        "direction": int(act["direction"].item()),
        "item": int(act["item"].item()),
        "misc": int(act["misc"].item()),
    }


def _apply_placement(obs_CWH: np.ndarray, action: dict) -> bool:
    """Update obs in-place with the predicted entity. Returns True if the
    placement was non-empty (so we should keep iterating)."""
    ent_id = action["entity"]
    if ent_id == 0:  # no-op / empty
        return False
    x, y = action["xy"]
    ent_meta = entities.get(ent_id)
    if ent_meta is None or not ent_meta.is_placeable:
        # Head can technically emit recipe IDs (8+) since it's sized
        # len(entities)-2 to only exclude source/sink. Treat those as no-ops.
        log.debug("Skipping non-placeable predicted entity id %d", ent_id)
        return True

    direction = action["direction"]
    width, height = ent_meta.width, ent_meta.height

    try:
        tiles = factorion_rs.py_entity_tiles(x, y, direction, width, height)
    except Exception:
        tiles = None
    if tiles is None:
        # Fall back to the anchor tile; better to under-mark than crash.
        tiles = [(x, y)]

    _, W, H = obs_CWH.shape
    for tx, ty in tiles:
        if 0 <= tx < W and 0 <= ty < H:
            obs_CWH[Channel.ENTITIES.value, tx, ty] = ent_id
            obs_CWH[Channel.DIRECTION.value, tx, ty] = direction
    obs_CWH[Channel.ITEMS.value, x, y] = action["item"]
    obs_CWH[Channel.MISC.value, x, y] = action["misc"]
    return True


def action_to_placement(action: dict) -> dict:
    """Convert one model action into a Factorio create_entity specification."""
    ent_id = int(action["entity"])
    ent_meta = entities.get(ent_id)
    if ent_meta is None or not ent_meta.is_placeable:
        raise ValueError(f"entity id {ent_id} is not placeable")

    x, y = (int(v) for v in action["xy"])
    direction_model = int(action["direction"])
    direction = _DIR_MODEL_TO_BP.get(direction_model)
    if direction is None:
        direction = 0

    name = _hyphenate(ent_meta.name)
    if "inserter" in name:
        direction = (direction + 8) % 16

    width, height = ent_meta.width, ent_meta.height
    if direction_model in (2, 4):
        width, height = height, width

    placement = {
        "name": name,
        "tile_x": x,
        "tile_y": y,
        "width": width,
        "height": height,
        "x": x + width / 2.0,
        "y": y + height / 2.0,
        "direction": direction,
    }

    item_id = int(action["item"])
    item_meta = items.get(item_id)
    if name == "assembling-machine-1" and item_meta is not None:
        if item_meta.name != "empty":
            placement["recipe"] = _hyphenate(item_meta.name)

    if name == "underground-belt":
        misc = int(action["misc"])
        if misc == Misc.UNDERGROUND_DOWN.value:
            placement["type"] = "input"
        elif misc == Misc.UNDERGROUND_UP.value:
            placement["type"] = "output"

    return placement


def run_inference(
    agent: AgentCNN,
    req: dict,
    max_steps: int,
    device,
    eot_threshold: float = 0.5,
    on_placement: Optional[Callable[[dict], bool]] = None,
) -> tuple[np.ndarray, dict]:
    """Iteratively place entities until eot_head signals "done", the model
    emits a no-op, or we hit the safety budget."""
    obs = request_to_obs(req)

    # Dump the initial obs summary so we can see what the model is starting from
    src_ids = [(int(s["x"]), int(s["y"]), s.get("direction"), s.get("item"))
               for s in req.get("sources", [])]
    snk_ids = [(int(s["x"]), int(s["y"]), s.get("direction"), s.get("item"))
               for s in req.get("sinks", [])]
    log.info("  initial sources (x,y,dir,item): %s", src_ids)
    log.info("  initial sinks   (x,y,dir,item): %s", snk_ids)
    fp_count = int(obs[Channel.FOOTPRINT.value].sum())
    log.info("  footprint tiles: %d", fp_count)

    stats: dict = {
        "steps_taken": 0,
        "stop_reason": "max_steps",
        "first_eot_prob": None,
        "final_eot_prob": None,
        "placements": [],  # list of dicts, one per step
    }

    for step in range(max_steps):
        # Ask the model first: do you think we're done?
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(device)
            eot_p = float(agent.eot_prob(x).item())
        if step == 0:
            stats["first_eot_prob"] = eot_p
        stats["final_eot_prob"] = eot_p
        if eot_p > eot_threshold:
            log.info("  step %d: eot_prob=%.3f > %.2f → STOP", step, eot_p, eot_threshold)
            stats["stop_reason"] = "eot"
            stats["steps_taken"] = step
            break
        action = _argmax_action(agent, obs, device)
        ent_id = action["entity"]
        ent_name = entities[ent_id].name if ent_id in entities else "?"
        item_id = action["item"]
        item_name = items[item_id].name if item_id in items else "?"
        stats["placements"].append({
            "step": step,
            "eot": eot_p,
            "entity_id": ent_id,
            "entity_name": ent_name,
            "x": int(action["xy"][0]),
            "y": int(action["xy"][1]),
            "direction": int(action["direction"]),
            "item_id": item_id,
            "item_name": item_name,
            "misc": int(action["misc"]),
        })
        log.info(
            "  step %d: eot=%.3f place=%s(id=%d) at (%d,%d) dir=%d item=%s(id=%d) misc=%d",
            step, eot_p, ent_name, ent_id,
            action["xy"][0], action["xy"][1],
            action["direction"], item_name, item_id, action["misc"],
        )
        if not _apply_placement(obs, action):
            log.info("  → empty/no-op placement, stopping")
            stats["stop_reason"] = "empty"
            stats["steps_taken"] = step + 1
            break
        stats["steps_taken"] = step + 1
        if on_placement is not None and not on_placement(action):
            log.info("  → Factorio rejected the placement, stopping")
            stats["stop_reason"] = "placement_error"
            break
    else:
        log.info("Reached max_steps=%d without eot/empty.", max_steps)

    # Summarise final state
    ent_ch = obs[Channel.ENTITIES.value]
    placed = int((ent_ch != 0).sum())
    log.info("Final: %d non-empty tiles in ENTITIES channel", placed)
    stats["nonzero_entities_tiles"] = placed
    return obs, stats


# --------------------------------------------------------------------------- #
# RCON poll loop: single channel both ways.
# --------------------------------------------------------------------------- #

POLL_CMD = "/silent-command rcon.print(remote.call('factorion','poll_request'))"
MODEL_POLL_CMD = "/silent-command rcon.print(remote.call('factorion','poll_model'))"
PROTOCOL_CMD = (
    "/silent-command rcon.print(remote.call('factorion','protocol_version'))"
)


def _lua_string(value: str) -> str:
    """Quote a Python string for the small Lua command strings sent over RCON."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


def _stream_placement(
    rcon: RconClient,
    request_id: str,
    action: dict,
    *,
    placement_delay_s: float = 0.0,
) -> bool:
    placement = action_to_placement(action)
    payload = json.dumps(placement, separators=(",", ":"))
    response = rcon.exec(
        "/silent-command rcon.print(remote.call('factorion','place_prediction',"
        f"{_lua_string(request_id)},{_lua_string(payload)}))"
    ).strip()
    if response == "ok":
        if placement_delay_s > 0:
            time.sleep(placement_delay_s)
        return True
    log.error("Factorio rejected %s: %s", placement, response or "(empty response)")
    return False


def _finish_prediction(rcon: RconClient, request_id: str, stats: dict) -> None:
    summary = json.dumps({
        "steps_taken": stats["steps_taken"],
        "stop_reason": stats["stop_reason"],
    }, separators=(",", ":"))
    response = rcon.exec(
        "/silent-command rcon.print(remote.call('factorion','finish_prediction',"
        f"{_lua_string(request_id)},{_lua_string(summary)}))"
    ).strip()
    if response != "ok":
        raise RuntimeError(
            f"Factorio could not finish request {request_id}: "
            f"{response or '(empty response)'}"
        )


def _send_model_status(
    rcon: RconClient,
    player_index: int,
    ok: bool,
    message: str,
    model_name: Optional[str] = None,
    model_url: Optional[str] = None,
) -> None:
    model_arg = ",nil" if model_name is None else f",{_lua_string(model_name)}"
    url_arg = ",nil" if model_url is None else f",{_lua_string(model_url)}"
    rcon.exec(
        "/silent-command remote.call('factorion','model_status',"
        f"{player_index},{str(ok).lower()},{_lua_string(message)}{model_arg}{url_arg})"
    )


def _maybe_switch_model(
    agent: AgentCNN,
    rcon: RconClient,
    device: torch.device,
    project: str,
    entity: Optional[str],
    model_state: dict,
) -> AgentCNN:
    """Apply one queued in-game /model request, keeping the old model on error."""
    raw = rcon.exec(MODEL_POLL_CMD).strip()
    if not raw:
        return agent
    # A save hosted before this mod update has the older remote interface.
    # Keep serving predictions quietly; a newly hosted game will expose the
    # method and hot-swapping starts working automatically.
    if (
        "No such function: factorion.poll_model" in raw
        or "Unknown interface: factorion" in raw
    ):
        return agent
    try:
        request = json.loads(raw)
        spec = str(request["spec"])
        player_index = int(request["player_index"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        log.warning("Invalid model-switch request: %r", raw[:500])
        return agent

    log.info("In-game model switch requested: %s", spec)
    try:
        checkpoint, wandb_hp, source = resolve_checkpoint(spec, project, entity)
        hp = wandb_hp or Hyperparams.from_json_sibling(checkpoint)
        if hp.grid_size != MOD_GRID_SIZE:
            raise ValueError(
                f"checkpoint uses {hp.grid_size}x{hp.grid_size}; "
                f"the in-game region brush is fixed at {MOD_GRID_SIZE}x{MOD_GRID_SIZE}"
            )
        replacement = load_agent(checkpoint, hp, device)
        provenance = source["run_id"] if source else str(checkpoint)
        message = (
            f"Loaded {provenance}: {hp.grid_size}x{hp.grid_size}, "
            f"layers {'/'.join(map(str, hp.layers))}."
        )
        model_url = source["run_url"] if source else None
        if model_url:
            message += f" {model_url}"
        _send_model_status(
            rcon, player_index, True, message, provenance, model_url,
        )
        model_state["name"] = provenance
        model_state["url"] = model_url
        log.info(message)
        return replacement
    except Exception as exc:
        message = f"Could not load {spec}: {exc}"
        log.exception("Model switch failed")
        _send_model_status(rcon, player_index, False, message)
        return agent


def poll_loop(
    agent: AgentCNN,
    rcon: RconClient,
    *,
    poll_interval: float = 0.25,
    max_steps: int = 64,
    placement_delay_s: float = 0.02,
    device: torch.device,
    wandb_project: str = "factorion",
    wandb_entity: Optional[str] = None,
    model_state: Optional[dict] = None,
):
    """Poll Factorio over RCON for queued requests; handle each as it arrives.

    The transport is symmetric: both legs go through the same RCON
    connection. Reconnect on transient failures so the server survives a
    Factorio restart without needing to be restarted itself.
    """
    log.info("Polling factorion.poll_request every %.0f ms", poll_interval * 1000)
    log.info("Placement pacing delay: %.0f ms", placement_delay_s * 1000)
    model_state = model_state or {"name": "unknown", "url": None}
    last_model_publish = 0.0

    while True:
        try:
            agent = _maybe_switch_model(
                agent, rcon, device, wandb_project, wandb_entity, model_state,
            )
            # Re-publish periodically so a Factorio save/mod reload learns the
            # active model even though the long-running Python process stayed up.
            now = time.monotonic()
            if now - last_model_publish >= 5.0:
                name = model_state["name"]
                _send_model_status(
                    rcon, 0, True, f"Active model: {name}.",
                    name, model_state.get("url"),
                )
                last_model_publish = now
            raw = rcon.exec(POLL_CMD).strip()
        except (RconError, OSError) as e:
            log.warning("RCON poll failed (%s); reconnecting in 2s…", e)
            try:
                rcon.close()
            except Exception:
                pass
            time.sleep(2.0)
            try:
                rcon.connect()
                log.info("RCON reconnected.")
            except Exception as e2:
                log.warning("RCON reconnect failed: %s", e2)
            continue

        if not raw:
            time.sleep(poll_interval)
            continue

        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            if "Unknown interface: factorion" in raw:
                log.info("Factorion mod is not active yet; waiting for a loaded game…")
                time.sleep(2.0)
            else:
                log.warning("Non-JSON RCON response (first 200 chars): %r", raw[:200])
                time.sleep(poll_interval)
            continue

        try:
            handle_request(
                req,
                agent,
                rcon,
                max_steps=max_steps,
                placement_delay_s=placement_delay_s,
                device=device,
            )
        except Exception:
            log.exception("Failed to handle request %s", req.get("request_id"))


def handle_request(
    req: dict,
    agent: AgentCNN,
    rcon: RconClient,
    *,
    max_steps: int,
    placement_delay_s: float = 0.02,
    device,
):
    if req["grid_size"] != agent.width:
        raise ValueError(
            f"game requested a {req['grid_size']}x{req['grid_size']} grid, but "
            f"the checkpoint expects {agent.width}x{agent.height}; load an 11x11 model"
        )
    log.info("Request %s: grid=%dx%d, %d sources, %d sinks",
             req["request_id"], req["grid_size"], req["grid_size"],
             len(req.get("sources", [])), len(req.get("sinks", [])))

    t0 = time.time()
    _, stats = run_inference(
        agent,
        req,
        max_steps=max_steps,
        device=device,
        on_placement=lambda action: _stream_placement(
            rcon,
            req["request_id"],
            action,
            placement_delay_s=placement_delay_s,
        ),
    )
    _finish_prediction(rcon, req["request_id"], stats)
    log.info(
        "Inference %.2fs; streamed %d placement step(s), stop=%s",
        time.time() - t0, stats["steps_taken"], stats["stop_reason"],
    )


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Local .pt path, W&B run id, entity/project/id, or run URL.")
    ap.add_argument("--wandb-project", default="factorion")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=27015)
    ap.add_argument("--rcon-password", default="factorion")
    ap.add_argument("--grid-size", type=int, default=None,
                    help="Override checkpoint metadata (normally unnecessary).")
    ap.add_argument("--layers", default=None,
                    help="Comma-separated encoder widths; overrides checkpoint metadata.")
    ap.add_argument("--kernel-size", type=int, default=None,
                    help="Override checkpoint metadata (normally unnecessary).")
    ap.add_argument("--max-steps", type=int, default=64,
                    help="Iterative inference budget per request.")
    ap.add_argument(
        "--placement-delay-ms",
        type=float,
        default=20.0,
        help="Delay after each accepted world placement (default: 20 ms).",
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    if args.placement_delay_ms < 0:
        ap.error("--placement-delay-ms must be non-negative")

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device)
    checkpoint, wandb_hp, source = resolve_checkpoint(
        args.checkpoint, args.wandb_project, args.wandb_entity,
    )
    sidecar = Hyperparams.from_json_sibling(checkpoint)
    inferred = wandb_hp or sidecar
    cli_layers = tuple(int(v) for v in args.layers.split(",")) if args.layers else None
    hp = Hyperparams(
        grid_size=args.grid_size if args.grid_size is not None else inferred.grid_size,
        layers=cli_layers or inferred.layers,
        kernel_size=args.kernel_size if args.kernel_size is not None else inferred.kernel_size,
        attn_dim=inferred.attn_dim,
        attn_heads=inferred.attn_heads,
        attn_layers=inferred.attn_layers,
        attn_pos_embed=inferred.attn_pos_embed,
        global_feat_dim=inferred.global_feat_dim,
    )
    if hp.grid_size != MOD_GRID_SIZE:
        ap.error(
            f"checkpoint uses {hp.grid_size}x{hp.grid_size}; "
            f"the in-game region brush is fixed at {MOD_GRID_SIZE}x{MOD_GRID_SIZE}"
        )
    if source:
        log.info("W&B run: %s (%s)", source["run_id"], source["run_url"])
    agent = load_agent(checkpoint, hp, device)

    rcon = RconClient(args.rcon_host, args.rcon_port, args.rcon_password)
    while True:
        try:
            rcon.connect()
            break
        except (RconError, OSError) as exc:
            log.info("Waiting for Factorio RCON at %s:%d (%s)",
                     args.rcon_host, args.rcon_port, exc)
            time.sleep(2)
    try:
        log.info("RCON connected to %s:%d", args.rcon_host, args.rcon_port)
        while True:
            try:
                protocol = rcon.exec(PROTOCOL_CMD).strip()
                if protocol == "2":
                    break
                log.info(
                    "Waiting for streaming protocol v2; restart Factorio and "
                    "host a game with factorion 0.6.1 (%s)",
                    protocol or "no mod response",
                )
            except (RconError, OSError) as exc:
                log.info("Waiting for Factorio to reload factorion 0.6.1 (%s)", exc)
                rcon.close()
                time.sleep(2)
                try:
                    rcon.connect()
                except (RconError, OSError):
                    continue
            time.sleep(2)
        log.info("Factorion streaming protocol v2 ready")

        model_name = source["run_id"] if source else str(checkpoint)
        model_url = source["run_url"] if source else None
        try:
            _send_model_status(
                rcon, 0, True,
                f"Active model: {model_name} ({hp.grid_size}x{hp.grid_size}, "
                f"layers {'/'.join(map(str, hp.layers))}).",
                model_name,
                model_url,
            )
        except Exception:
            log.warning("Could not publish initial model identity to the mod")

        poll_loop(
            agent, rcon,
            poll_interval=0.25,
            max_steps=args.max_steps,
            placement_delay_s=args.placement_delay_ms / 1000,
            device=device,
            wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
            model_state={"name": model_name, "url": model_url},
        )
    finally:
        rcon.close()


if __name__ == "__main__":
    main()
