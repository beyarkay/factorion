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
MOD_PROTOCOL_VERSION = "4"


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
        """Read W&B's layer1..8 config or a sidecar's layers list."""
        size = int(values.get("size", values.get("grid_size", cls.grid_size)))
        if "layers" in values:
            layers = tuple(int(v) for v in values["layers"] if int(v) > 0)
        else:
            layers = tuple(
                int(values.get(f"layer{i}", 0))
                for i in range(1, 9)
                if int(values.get(f"layer{i}", 0)) > 0
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


def resolve_checkpoint(
    spec: str, project: str = "factorion", entity: Optional[str] = None,
) -> tuple[Path, Optional[Hyperparams], Optional[dict]]:
    """Resolve a local checkpoint or W&B run and recover its architecture."""
    local = Path(spec).expanduser()
    if local.exists():
        return local.resolve(), None, None
    if spec.endswith(".pt"):
        raise FileNotFoundError(f"checkpoint does not exist: {local}")

    path, source = _resolve_wandb_checkpoint(spec, project, entity)
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

    # The in-game region is a fixed square, so every tile is buildable.
    obs[Channel.FOOTPRINT.value] = 1.0

    src_id, snk_id = _source_id(), _sink_id()

    for existing in req.get("entities", []):
        entity = str2ent(existing.get("name"))
        if entity is None or not entity.is_placeable:
            log.warning(
                "Ignoring unsupported existing entity %r",
                existing.get("name"),
            )
            continue
        if entity.value in {src_id, snk_id}:
            # Endpoints have their own request lists because they also carry
            # the configured source/sink item.
            continue
        item = str2ent(existing.get("item", existing.get("recipe", "empty")))
        _apply_placement(obs, {
            "xy": (int(existing["x"]), int(existing["y"])),
            "entity": entity.value,
            "direction": int(existing.get("direction", 0)),
            "item": item.value if item is not None else 0,
            "misc": int(existing.get("misc", Misc.NONE.value)),
        })

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
        # A legal mask should prevent this, but stop safely if a mismatched
        # checkpoint still emits an unsupported catalog entry.
        log.debug("Skipping non-placeable predicted entity id %d", ent_id)
        return False

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
            obs_CWH[Channel.ITEMS.value, tx, ty] = action["item"]
            obs_CWH[Channel.MISC.value, tx, ty] = action["misc"]
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

    stats: dict = {
        "steps_taken": 0,
        "stop_reason": "max_steps",
    }

    for step in range(max_steps):
        # Ask the model first: do you think we're done?
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(device)
            eot_p = float(agent.eot_prob(x).item())
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

    return obs, stats


# --------------------------------------------------------------------------- #
# RCON poll loop: single channel both ways.
# --------------------------------------------------------------------------- #

POLL_CMD = "/silent-command rcon.print(remote.call('factorion','poll_request'))"
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


def poll_loop(
    agent: AgentCNN,
    rcon: RconClient,
    *,
    poll_interval: float = 0.25,
    max_steps: int = 64,
    placement_delay_s: float = 0.01,
    device: torch.device,
):
    """Poll Factorio over RCON for queued requests; handle each as it arrives.

    The transport is symmetric: both legs go through the same RCON
    connection. Reconnect on transient failures so the server survives a
    Factorio restart without needing to be restarted itself.
    """
    log.info("Polling factorion.poll_request every %.0f ms", poll_interval * 1000)
    log.info("Placement pacing delay: %.0f ms", placement_delay_s * 1000)

    while True:
        try:
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
    placement_delay_s: float = 0.01,
    device,
):
    t0 = time.time()
    stats = {"steps_taken": 0, "stop_reason": "server_error"}
    try:
        if req["grid_size"] != agent.width:
            raise ValueError(
                f"game requested a {req['grid_size']}x{req['grid_size']} grid, "
                f"but the checkpoint expects {agent.width}x{agent.height}"
            )
        log.info("Request %s: grid=%dx%d, %d sources, %d sinks",
                 req["request_id"], req["grid_size"], req["grid_size"],
                 len(req.get("sources", [])), len(req.get("sinks", [])))
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
    finally:
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
                    help="Local .pt path, W&B run id, or entity/project/id.")
    ap.add_argument("--wandb-project", default="factorion")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=27015)
    ap.add_argument("--rcon-password", default="factorion")
    ap.add_argument("--max-steps", type=int, default=64,
                    help="Iterative inference budget per request.")
    ap.add_argument(
        "--placement-delay-ms",
        type=float,
        default=10.0,
        help="Delay after each accepted world placement (default: 10 ms).",
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
    hp = wandb_hp or Hyperparams.from_json_sibling(checkpoint)
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
            rcon.close()
            log.info("Waiting for Factorio RCON at %s:%d (%s)",
                     args.rcon_host, args.rcon_port, exc)
            time.sleep(2)
    try:
        log.info("RCON connected to %s:%d", args.rcon_host, args.rcon_port)
        while True:
            try:
                protocol = rcon.exec(PROTOCOL_CMD).strip()
                if protocol == MOD_PROTOCOL_VERSION:
                    break
                log.info(
                    "Waiting for streaming protocol v%s; restart Factorio and "
                    "host a game with the current mod (%s)",
                    MOD_PROTOCOL_VERSION,
                    protocol or "no mod response",
                )
            except (RconError, OSError) as exc:
                log.info(
                    "Waiting for Factorio to reload the mod (%s)",
                    exc,
                )
                rcon.close()
                time.sleep(2)
                try:
                    rcon.connect()
                except (RconError, OSError):
                    continue
            time.sleep(2)
        log.info(
            "Factorion streaming protocol v%s ready",
            MOD_PROTOCOL_VERSION,
        )

        poll_loop(
            agent, rcon,
            poll_interval=0.25,
            max_steps=args.max_steps,
            placement_delay_s=args.placement_delay_ms / 1000,
            device=device,
        )
    finally:
        rcon.close()


if __name__ == "__main__":
    main()
