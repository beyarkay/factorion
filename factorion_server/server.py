"""Factorion model server.

Polls the Factorio mod for requests, generates a factory with the trained
denoiser, and streams each of its entities back into the running game over
RCON.

Installed as `factorion-server`; from a checkout, `uv run python -m factorion_server`.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
import torch

from denoise import DenoiserUNet, generate, legalize, load_denoiser, part_map, score_world, world_to_state
from factorion import Channel, Footprint, Misc, apply_placement_action, entities, items, str2ent
from sft import resolve_checkpoint
from training_config import RlArgs, SftArgs

from .blueprint import _DIR_MODEL_TO_BP, _hyphenate

log = logging.getLogger("factorion-server")
MOD_PROTOCOL_VERSION = "9"
RELEASES_URL = "https://api.github.com/repos/beyarkay/factorion/releases"
CHECKPOINT_CACHE = Path.home() / ".cache" / "factorion"
ModelLoader = Callable[[str], tuple[DenoiserUNet, str]]
"""Checkpoint spec (the mod's setting) → (model, display name)."""


# --------------------------------------------------------------------------- #
# RCON client: shared with parity.py, see rcon.py.
# --------------------------------------------------------------------------- #

from .rcon import RconClient, RconError  # noqa: E402, F401


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
    """The (C, W, H) world of a request: its markers, plus its existing
    entities as walls the model builds around, as a lesson protects its own."""
    size = req["grid_size"]
    C = len(Channel)
    obs = np.zeros((C, size, size), dtype=np.float32)

    # The in-game region is a fixed square, so every tile is buildable.
    obs[Channel.FOOTPRINT.value] = 1.0

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

    for existing in req.get("entities", []):
        entity = str2ent(existing.get("name"))
        if entity is None or not entity.is_placeable:
            log.warning(
                "Ignoring unsupported existing entity %r",
                existing.get("name"),
            )
            continue
        item = str2ent(existing.get("item", existing.get("recipe", "empty")))
        reason = apply_placement_action(
            torch.from_numpy(obs),
            {
                "xy": (int(existing["x"]), int(existing["y"])),
                "entity": entity.value,
                "direction": int(existing.get("direction", 0)),
                "item": item.value if item is not None else 0,
                "misc": int(existing.get("misc", Misc.NONE.value)),
            },
        )
        if reason is not None:
            log.warning("Ignoring existing %s: %s", entity.name, reason)

    ent = obs[Channel.ENTITIES.value]
    existing_mask = (ent != str2ent("empty").value) & (ent != src_id) & (ent != snk_id)
    obs[Channel.FOOTPRINT.value][existing_mask] = Footprint.UNAVAILABLE.value
    return obs


# --------------------------------------------------------------------------- #
# Inference: generate best-of-N as the eval does, stream the best build.
# --------------------------------------------------------------------------- #
def action_to_placement(action: dict) -> dict:
    """Convert one placement action into a Factorio create_entity specification.

    The mod first removes every entity the placement's tiles overlap, so an
    `empty` action becomes a name-less placement that only deletes."""
    ent_id = int(action["entity"])
    x, y = (int(v) for v in action["xy"])
    if ent_id == 0:
        return {"tile_x": x, "tile_y": y, "width": 1, "height": 1}
    ent_meta = entities.get(ent_id)
    if ent_meta is None or not ent_meta.is_placeable:
        raise ValueError(f"entity id {ent_id} is not placeable")

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


def best_build(model: DenoiserUNet, obs: np.ndarray, temperature: float, seed: int, device) -> np.ndarray:
    """The highest-throughput (then cheapest) of `eval_best_of` factories
    generated around `obs`'s markers."""
    cond = torch.as_tensor(world_to_state(obs)).unsqueeze(0).repeat(SftArgs.eval_best_of, 1, 1, 1).to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    (typ, rec, part), conf, _, _ = generate(model, cond, SftArgs.gen_steps, SftArgs.gen_refine, temperature, gen)
    typ, rec, part, conf = (t.cpu().numpy() for t in (typ, rec, part, conf.float()))
    worlds = [legalize(obs.astype(np.int64), typ[i], rec[i], part[i], conf[i]) for i in range(len(cond))]
    scores = [score_world(w, 1.0, RlArgs.entity_cost_scale) for w in worlds]
    return worlds[max(range(len(worlds)), key=lambda i: (scores[i]["thput_raw"], -scores[i]["entity_cost"]))]


def build_actions(obs: np.ndarray, world: np.ndarray) -> list[dict]:
    """One placement action per entity `world` adds to `obs`, in raster order
    of their anchors."""
    ent = world[Channel.ENTITIES.value]
    added = (ent != obs[Channel.ENTITIES.value]) & (part_map(world) == 0)
    return [
        {
            "xy": (int(x), int(y)),
            "entity": int(ent[x, y]),
            "direction": int(world[Channel.DIRECTION.value, x, y]),
            "item": int(world[Channel.ITEMS.value, x, y]),
            "misc": int(world[Channel.MISC.value, x, y]),
        }
        for x, y in zip(*np.nonzero(added))
    ]


def run_inference(
    model: DenoiserUNet,
    req: dict,
    device,
    on_placement: Optional[Callable[[dict], bool]] = None,
) -> tuple[np.ndarray, dict]:
    """Generate a factory for `req` and stream its entities until done or
    Factorio refuses one."""
    obs = request_to_obs(req)
    seed = int(req.get("seed") or secrets.randbits(31))
    temperature = float(req.get("temperature", SftArgs.gen_temperature))
    log.info("  seed=%d temperature=%g", seed, temperature)
    world = best_build(model, obs, temperature, seed, device)
    stats: dict = {"steps_taken": 0, "invalid_steps": 0, "stop_reason": "eot", "seed": seed}
    for action in build_actions(obs, world):
        log.info("  place %s at %s", entities[action["entity"]].name, action["xy"])
        if on_placement is not None and not on_placement(action):
            log.info("  → Factorio rejected the placement, stopping")
            stats["stop_reason"] = "placement_error"
            break
        stats["steps_taken"] += 1
    return world, stats


# --------------------------------------------------------------------------- #
# RCON poll loop: single channel both ways.
# --------------------------------------------------------------------------- #

POLL_CMD = (
    "/silent-command rcon.print(remote.call('factorion','poll_request',"
    f"'{MOD_PROTOCOL_VERSION}'))"
)
PROTOCOL_CMD = (
    "/silent-command rcon.print(remote.call('factorion','protocol_version'))"
)


def _lua_string(value: str) -> str:
    """Quote a Python string for the small Lua command strings sent over RCON."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ") + "'"


class PredictionCancelled(Exception):
    """The player cancelled the request in game (CTRL+R or a region clear)."""


def _stream_placement(
    rcon: RconClient,
    request_id: str,
    action: dict,
    *,
    placement_delay_s: float = 0.0,
) -> bool:
    placement = action_to_placement(action)
    payload = json.dumps(placement, separators=(",", ":"))
    response = _call_mod(rcon, "place_prediction", request_id, payload)
    if response == "ok":
        if placement_delay_s > 0:
            time.sleep(placement_delay_s)
        return True
    if response == "cancelled":
        raise PredictionCancelled
    log.error("Factorio rejected %s: %s", placement, response or "(empty response)")
    return False


def factorio_config() -> Path:
    base = "Library/Application Support/factorio" if sys.platform == "darwin" else ".factorio"
    return Path.home() / base / "config" / "config.ini"


def gui_rcon(config: Path) -> tuple[str, int, str]:
    """(host, port, password) of the RCON a GUI-hosted game binds, enabling
    it in Factorio's config.ini first if it isn't: that file is the only way
    to turn RCON on for a game hosted from the menu."""
    if not config.exists():
        sys.exit(f"No Factorio config at {config}: launch Factorio once, then run this again.")
    text = config.read_text()
    found = dict(re.findall(r"(?m)^\s*local-rcon-(socket|password)\s*=\s*(\S+)", text))
    if len(found) < 2:
        # Factorio writes its config back out, which would drop our edit.
        if subprocess.run(["pgrep", "-x", "factorio"], capture_output=True).returncode == 0:
            sys.exit("Quit Factorio and run this again: it needs to enable RCON in Factorio's config.")
        found = {"socket": "127.0.0.1:64502", "password": secrets.token_urlsafe(12)}
        lines = [line for line in text.splitlines() if not re.match(r"\s*local-rcon-(socket|password)\s*=", line)]
        if "[other]" not in lines:
            lines.append("[other]")
        lines[lines.index("[other]") + 1:lines.index("[other]") + 1] = [f"local-rcon-{k}={v}" for k, v in found.items()]
        config.write_text("\n".join(lines) + "\n")
        log.info("Enabled RCON in %s", config)
    host, port = found["socket"].rsplit(":", 1)
    return "127.0.0.1" if host == "0.0.0.0" else host, int(port), found["password"]


def released_checkpoint(run_id: str) -> Optional[Path]:
    """The `<run_id>.pt` a GitHub release of this repo carries, downloaded
    once, so players need no W&B account. None when no release has it."""
    dest = CHECKPOINT_CACHE / f"{run_id}.pt"
    if dest.exists():
        return dest
    resp = requests.get(RELEASES_URL, params={"per_page": 100}, timeout=30)
    resp.raise_for_status()
    urls = [a["browser_download_url"] for r in resp.json() for a in r["assets"] if a["name"] == dest.name]
    if not urls:
        return None
    log.info("Downloading checkpoint %s from %s", run_id, urls[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    with requests.get(urls[0], stream=True, timeout=30) as download:
        download.raise_for_status()
        with part.open("wb") as f:
            for chunk in download.iter_content(1 << 20):
                f.write(chunk)
    part.rename(dest)
    return dest


def _call_mod(rcon: RconClient, method: str, *args: str) -> str:
    quoted = ",".join(_lua_string(a) for a in args)
    return rcon.exec(
        f"/silent-command rcon.print(remote.call('factorion','{method}',{quoted}))"
    ).strip()


def _report_model(rcon: RconClient, checkpoint: str, get_model: ModelLoader, grid_size: int) -> None:
    try:
        _, name = get_model(checkpoint)
        method = "model_loaded"
        payload = {"checkpoint": checkpoint, "name": name, "grid_size": grid_size}
    except Exception as exc:
        log.exception("Could not load checkpoint %s", checkpoint)
        method = "model_failed"
        payload = {"checkpoint": checkpoint, "error": str(exc)}
    _call_mod(rcon, method, json.dumps(payload, separators=(",", ":")))


def _finish_prediction(rcon: RconClient, request_id: str, stats: dict) -> None:
    summary = json.dumps({
        "steps_taken": stats["steps_taken"],
        "invalid_steps": stats["invalid_steps"],
        "stop_reason": stats["stop_reason"],
        "seed": stats.get("seed"),
        "error": stats.get("error"),
    }, separators=(",", ":"))
    response = _call_mod(rcon, "finish_prediction", request_id, summary)
    if response not in ("ok", "cancelled"):
        raise RuntimeError(
            f"Factorio could not finish request {request_id}: "
            f"{response or '(empty response)'}"
        )


def poll_loop(
    get_model: ModelLoader,
    rcon: RconClient,
    *,
    poll_interval: float = 0.25,
    device: torch.device,
    grid_size: int,
):
    """Poll Factorio over RCON for queued requests; handle each as it arrives.

    The transport is symmetric: both legs go through the same RCON
    connection. Reconnect on transient failures so the server survives a
    Factorio restart without needing to be restarted itself.
    """
    log.info("Polling factorion.poll_request every %.0f ms", poll_interval * 1000)

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

        if req.get("type") == "protocol_mismatch":
            log.error(
                "The mod speaks protocol %s but this server speaks %s; restart "
                "Factorio or this server from the same checkout.",
                req.get("mod_protocol"), MOD_PROTOCOL_VERSION,
            )
            time.sleep(5.0)
            continue

        try:
            if req.get("type") == "load_model":
                _report_model(rcon, req["checkpoint"], get_model, grid_size)
                continue
            handle_request(
                req,
                get_model,
                rcon,
                device=device,
            )
        except Exception:
            log.exception("Failed to handle request %s", req.get("request_id"))


def handle_request(
    req: dict,
    get_model: ModelLoader,
    rcon: RconClient,
    *,
    device,
):
    t0 = time.time()
    stats = {"steps_taken": 0, "invalid_steps": 0, "stop_reason": "server_error"}
    try:
        model, _ = get_model(req["checkpoint"])
        log.info(
            "Requesting a factory converting [%s] to [%s] using checkpoint %s "
            "(%dx%d, %d existing entities)",
            ", ".join(e["item"] for e in req.get("sources", [])),
            ", ".join(e["item"] for e in req.get("sinks", [])),
            req["checkpoint"], req["grid_size"], req["grid_size"],
            len(req.get("entities", [])),
        )
        _, stats = run_inference(
            model,
            req,
            device=device,
            on_placement=lambda action: _stream_placement(
                rcon,
                req["request_id"],
                action,
                placement_delay_s=float(req.get("placement_delay_s", 0.0)),
            ),
        )
    except PredictionCancelled:
        stats["stop_reason"] = "cancelled"
    except Exception as exc:
        stats["error"] = str(exc)
        raise
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
    ap = argparse.ArgumentParser(
        description="The checkpoint is chosen in-game via the mod's "
        "'Factorion checkpoint' setting.",
    )
    ap.add_argument("--wandb-project", default="factorion")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, help="default: from Factorio's config.ini, enabling RCON there if needed")
    ap.add_argument("--rcon-password")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--grid-size", type=int, default=SftArgs.size, help="side of the region the mod plans")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.rcon_port is None or args.rcon_password is None:
        args.rcon_host, args.rcon_port, args.rcon_password = gui_rcon(factorio_config())

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    log.info("Running the model on %s", device)

    @functools.cache
    def get_model(spec: str) -> tuple[DenoiserUNet, str]:
        # A bare run id is looked up in the releases first; anything else, or a
        # run no release carries, goes to sft.resolve_checkpoint (local or W&B).
        released = None if "/" in spec or spec.endswith(".pt") else released_checkpoint(spec)
        if released is not None:
            return load_denoiser(released, device).eval(), spec
        path, source = resolve_checkpoint(spec, args.wandb_project, args.wandb_entity)
        return load_denoiser(path, device).eval(), source.get("run_name", Path(path).name)

    rcon = RconClient(args.rcon_host, args.rcon_port, args.rcon_password)
    waiting = log.info
    while True:
        try:
            rcon.connect()
            break
        except (RconError, OSError) as exc:
            rcon.close()
            waiting("Waiting for Factorio RCON at %s:%d (%s). Start Factorio and host a game: "
                     "Play > Multiplayer > Host new game (or Host saved game).",
                     args.rcon_host, args.rcon_port, exc)
            waiting = log.debug
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
            get_model, rcon,
            poll_interval=0.25,
            device=device,
            grid_size=args.grid_size,
        )
    finally:
        rcon.close()


if __name__ == "__main__":
    main()
