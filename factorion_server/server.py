"""Factorion model server.

Polls the Factorio mod for requests, denoises a factory for the requested
region with a trained `denoise.DenoiserUNet`, and streams every denoising
step into the running game over RCON as the entities that step changed.

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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests
import torch

from denoise import DenoiserUNet, generate, legalize, load_denoiser, part_map, score_world, world_to_state
from factorion import Channel, Footprint, Misc, apply_placement_action, entities, items, str2ent
from sft import resolve_checkpoint
from training_config import SftArgs

from .blueprint import _DIR_MODEL_TO_BP, _hyphenate

log = logging.getLogger("factorion-server")
MOD_PROTOCOL_VERSION = "10"
RELEASES_URL = "https://api.github.com/repos/beyarkay/factorion/releases"
CHECKPOINT_CACHE = Path.home() / ".cache" / "factorion"


# --------------------------------------------------------------------------- #
# RCON client: shared with parity.py, see rcon.py.
# --------------------------------------------------------------------------- #

from .rcon import RconClient, RconError  # noqa: E402, F401


# --------------------------------------------------------------------------- #
# Model loading.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    denoiser: DenoiserUNet
    name: str
    config: SftArgs
    """The run's grid size and generation schedule; defaults for a local file."""


def load_model(spec: str, project: str, entity: Optional[str], device: torch.device) -> Model:
    """A checkpoint spec's model. A bare run id is looked up in the releases
    first; anything else, or a run no release carries, goes to
    sft.resolve_checkpoint (local or W&B)."""
    released = None if "/" in spec or spec.endswith(".pt") else released_checkpoint(spec)
    source: dict = {"run_name": spec}
    if released is not None:
        path = str(released)
    else:
        try:
            path, source = resolve_checkpoint(str(Path(spec).expanduser()), project, entity)
        except ModuleNotFoundError as exc:
            # Players' installs leave out wandb: only developers have W&B keys.
            if exc.name != "wandb":
                raise
            raise RuntimeError(f"No release carries checkpoint {spec}. For a W&B run, restart the "
                               "server as `uvx --with wandb factorion-server` with a W&B API key.") from exc
    config = {k: v for k, v in (source.get("config") or {}).items() if k in ("size", "gen_steps", "gen_refine")}
    denoiser = load_denoiser(path, device).eval()
    log.info("Loaded checkpoint %s (%s)", path, denoiser.arch)
    return Model(denoiser, source.get("run_name", Path(path).name), SftArgs(**config))


ModelLoader = Callable[[str], Model]
"""Checkpoint spec (the mod's setting) → loaded model."""


# --------------------------------------------------------------------------- #
# Request → world.
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
    """The region's (C, W, H) world. Entities already in it become walls the
    model builds around, as a lesson's protected tiles are, and so does every
    tile outside the painted build mask, when there is one."""
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
    obs[Channel.FOOTPRINT.value][(ent != 0) & (ent != src_id) & (ent != snk_id)] = Footprint.UNAVAILABLE.value
    if req.get("allowed"):
        unpainted = np.ones((size, size), dtype=bool)
        for t in req["allowed"]:
            unpainted[t["x"], t["y"]] = False
        obs[Channel.FOOTPRINT.value][unpainted] = Footprint.UNAVAILABLE.value
    return obs.astype(np.int64)


# --------------------------------------------------------------------------- #
# Denoising: every step of the best chain becomes one batch of placements.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Generation:
    """Per-request generation knobs, set in the GPU server window."""

    temperature: float = 1.0
    steps: int = 0
    """Ancestral denoising steps; 0 = the checkpoint's `gen_steps`."""
    refine: int = 0
    """Most greedy refinement steps; 0 = the checkpoint's `gen_refine`."""
    best_of: int = 8

    def __str__(self) -> str:
        changed = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self) if getattr(self, f.name) != f.default
        ]
        return ", ".join(changed) or "default generation"

    @classmethod
    def from_request(cls, req: dict) -> "Generation":
        d = cls()
        return cls(
            temperature=float(req.get("temperature", d.temperature)),
            steps=int(req.get("steps", d.steps)),
            refine=int(req.get("refine", d.refine)),
            best_of=max(1, int(req.get("best_of", d.best_of))),
        )


@torch.no_grad()
def denoise_frames(
    model: Model, world_CWH: np.ndarray, gen: Generation, seed: int, device,
) -> tuple[list[np.ndarray], dict]:
    """Denoise `best_of` chains from noise around the region's markers and
    walls, and return the best chain's legalized prediction at every step,
    its final factory last. Best is the highest simulated throughput, then
    the cheapest build, as `val/thput@N` picks."""
    net = model.denoiser
    steps = gen.steps or model.config.gen_steps
    refine = gen.refine or model.config.gen_refine
    cond = torch.as_tensor(world_to_state(world_CWH), device=device)[None].repeat(gen.best_of, 1, 1, 1)
    trace: list = []
    toks, conf, converged, n_refine = generate(
        net, cond, steps, refine, gen.temperature,
        torch.Generator(device=device).manual_seed(seed), trace,
    )

    def legal(toks, conf, i: int) -> np.ndarray:
        typ, rec, part = (t[i].cpu().numpy() for t in toks)
        return legalize(world_CWH, typ, rec, part, conf[i].float().cpu().numpy())

    finals = [legal(toks, conf, i) for i in range(gen.best_of)]
    scores = [score_world(w, 1.0, 0.0) for w in finals]
    best = max(range(gen.best_of), key=lambda i: (scores[i]["thput_raw"], -scores[i]["entity_cost"], -i))
    taken = steps + int(n_refine[best])
    frames = [legal(t, c, best) for t, c in trace[:taken]] + [finals[best]]
    return frames, {
        "steps_taken": taken,
        "stop_reason": "converged" if converged[best] else "max_steps",
        "best_of": gen.best_of,
        "thput": scores[best]["thput_raw"],
    }


def action_to_placement(action: dict) -> dict:
    """Convert one entity into a Factorio create_entity specification.

    The mod first removes every entity the placement's tiles overlap, so an
    `empty` entity becomes a name-less placement that only deletes."""
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


def _units(world_CWH: np.ndarray) -> dict[tuple[int, int], tuple[int, ...]]:
    """Every entity of a valid world, keyed by the min-corner tile it is
    placed from."""
    anchors = (world_CWH[Channel.ENTITIES.value] != 0) & (part_map(world_CWH) == 0)
    return {(int(x), int(y)): tuple(world_CWH[:, x, y].tolist()) for x, y in zip(*np.nonzero(anchors))}


def frame_placements(prev_CWH: np.ndarray, next_CWH: np.ndarray) -> list[dict]:
    """The placements turning world `prev` into `next`: a deletion of every
    entity that changed, then every new entity."""
    before, after = _units(prev_CWH), _units(next_CWH)

    def placement(xy, unit):
        ent, d, item, misc = unit[:4]
        return action_to_placement({"xy": xy, "entity": ent, "direction": d, "item": item, "misc": misc})

    gone = [placement(xy, u) for xy, u in before.items() if after.get(xy) != u]
    built = [placement(xy, u) for xy, u in after.items() if before.get(xy) != u]
    return [{k: p[k] for k in ("tile_x", "tile_y", "width", "height")} for p in gone] + built


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


def _stream_step(rcon: RconClient, request_id: str, placements: list[dict]) -> bool:
    """Apply one denoising step's placements in game; False if Factorio
    refused any of them."""
    payload = json.dumps(placements, separators=(",", ":"))
    response = _call_mod(rcon, "place_predictions", request_id, payload)
    if response == "ok":
        return True
    if response == "cancelled":
        raise PredictionCancelled
    log.error("Factorio refused part of a step: %s", response or "(empty response)")
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
        model = get_model(checkpoint)
        method = "model_loaded"
        payload = {"checkpoint": checkpoint, "name": model.name, "grid_size": model.config.size}
    except Exception as exc:
        log.exception("Could not load checkpoint %s", checkpoint)
        method = "model_failed"
        payload = {"checkpoint": checkpoint, "error": str(exc)}
    _call_mod(rcon, method, json.dumps(payload, separators=(",", ":")))


def _finish_prediction(rcon: RconClient, request_id: str, stats: dict) -> None:
    summary = json.dumps(stats, separators=(",", ":"))
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
    seed = int(req.get("seed") or secrets.randbits(31))
    stats: dict = {"steps_taken": 0, "refused_steps": 0, "stop_reason": "server_error", "seed": seed}
    try:
        model = get_model(req["checkpoint"])
        gen = Generation.from_request(req)
        log.info(
            "Requesting a factory converting [%s] to [%s] using checkpoint %s "
            "(%dx%d, seed=%d, %s, %d existing entities)",
            ", ".join(e["item"] for e in req.get("sources", [])),
            ", ".join(e["item"] for e in req.get("sinks", [])),
            req["checkpoint"], req["grid_size"], req["grid_size"], seed, gen,
            len(req.get("entities", [])),
        )
        world = request_to_obs(req)
        frames, result = denoise_frames(model, world, gen, seed, device)
        log.info(
            "Denoised %d chain(s) in %.2fs; the best takes %d step(s) to %.3f items/s (%s)",
            gen.best_of, time.time() - t0, result["steps_taken"], result["thput"], result["stop_reason"],
        )
        for frame in frames:
            placements = frame_placements(world, frame)
            world = frame
            if not placements:
                continue
            if not _stream_step(rcon, req["request_id"], placements):
                stats["refused_steps"] += 1
            time.sleep(float(req.get("step_delay_s", 0.0)))
        stats.update(result)
    except PredictionCancelled:
        stats["stop_reason"] = "cancelled"
    except Exception as exc:
        stats["error"] = str(exc)
        raise
    finally:
        _finish_prediction(rcon, req["request_id"], stats)
    log.info("Request done in %.2fs, stop=%s", time.time() - t0, stats["stop_reason"])


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
    def get_model(spec: str) -> Model:
        return load_model(spec, args.wandb_project, args.wandb_entity, device)

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
