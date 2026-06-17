"""Factorion model server.

Watches a directory for `req-*.json` files written by the Factorio mod,
runs the trained AgentCNN policy on each, and pushes the resulting
blueprint back into the running game over RCON.

Run from the repo root so the `factorion` and `factorion_rs` modules are
importable (e.g. via `uv run python factorion-mod/server/server.py ...`).
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

# Make the repo root importable when this script is run via uv from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from factorion import Channel, entities, str2ent  # noqa: E402
from ppo import AgentCNN  # noqa: E402

import factorion_rs  # noqa: E402

from blueprint import world_tensor_to_blueprint_string  # noqa: E402

log = logging.getLogger("factorion-server")


# --------------------------------------------------------------------------- #
# RCON client (inlined; no extra deps).
# --------------------------------------------------------------------------- #

class RconError(RuntimeError):
    pass


class RconClient:
    """Minimal Source-RCON client. Factorio implements the same protocol."""

    AUTH = 3
    EXEC = 2
    RESP = 0

    def __init__(self, host: str, port: int, password: str, timeout: float = 5.0):
        self.host, self.port, self.password, self.timeout = host, port, password, timeout
        self._sock: Optional[socket.socket] = None
        self._counter = 1

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *a):
        self.close()

    def connect(self):
        self._sock = socket.create_connection((self.host, self.port), self.timeout)
        rid = self._send(self.AUTH, self.password)
        # Factorio sends an empty RESP first, then an AUTH response.
        resp_id, _, _ = self._recv()
        if resp_id == -1:
            raise RconError("RCON auth failed (bad password)")
        if resp_id != rid:
            # Drain the empty packet and read again.
            resp_id, _, _ = self._recv()
            if resp_id == -1:
                raise RconError("RCON auth failed (bad password)")

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def exec(self, command: str) -> str:
        if self._sock is None:
            self.connect()
        self._send(self.EXEC, command)
        _, _, body = self._recv()
        return body

    def _send(self, ptype: int, body: str) -> int:
        rid = self._counter
        self._counter += 1
        payload = body.encode("utf-8")
        # length = id(4) + type(4) + body(N) + 2 null bytes
        packet = struct.pack("<ii", rid, ptype) + payload + b"\x00\x00"
        self._sock.sendall(struct.pack("<i", len(packet)) + packet)
        return rid

    def _recv(self):
        assert self._sock is not None
        raw_len = self._read_exact(4)
        (length,) = struct.unpack("<i", raw_len)
        data = self._read_exact(length)
        rid, ptype = struct.unpack("<ii", data[:8])
        body = data[8:-2].decode("utf-8", errors="replace")
        return rid, ptype, body

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise RconError("RCON socket closed")
            buf += chunk
        return buf


# --------------------------------------------------------------------------- #
# Model loading & inference.
# --------------------------------------------------------------------------- #

def _duck_envs(size: int):
    """AgentCNN reads grid size off `envs.envs[0].unwrapped.size`."""
    return SimpleNamespace(envs=[SimpleNamespace(unwrapped=SimpleNamespace(size=size))])


@dataclass
class Hyperparams:
    grid_size: int = 8
    chan1: int = 32
    chan2: int = 64
    chan3: int = 64

    @classmethod
    def from_json_sibling(cls, ckpt_path: Path) -> "Hyperparams":
        sidecar = ckpt_path.with_suffix(".hp.json")
        if sidecar.exists():
            with sidecar.open() as f:
                return cls(**json.load(f))
        return cls()


def load_agent(ckpt_path: Path, hp: Hyperparams, device: torch.device) -> AgentCNN:
    log.info("Loading checkpoint %s (grid=%d, chans=%d/%d/%d)",
             ckpt_path, hp.grid_size, hp.chan1, hp.chan2, hp.chan3)
    agent = AgentCNN(_duck_envs(hp.grid_size),
                     layers=(hp.chan1, hp.chan2, hp.chan3)).to(device)
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
        encoded = agent.encoder(x)
        B = encoded.shape[0]
        tile_logits = agent.tile_logits(encoded).reshape(B, -1)
        tile_idx = tile_logits.argmax(dim=-1)
        x_B = tile_idx // agent.height
        y_B = tile_idx % agent.height
        feat = encoded[torch.arange(B), :, x_B, y_B]
        ent = agent.ent_head(feat).argmax(dim=-1)
        dir_ = agent.dir_head(feat).argmax(dim=-1)
        item = agent.item_head(feat).argmax(dim=-1)
        misc = agent.misc_head(feat).argmax(dim=-1)
    return {
        "xy": (int(x_B.item()), int(y_B.item())),
        "entity": int(ent.item()),
        "direction": int(dir_.item()),
        "item": int(item.item()),
        "misc": int(misc.item()),
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


def run_inference(
    agent: AgentCNN, req: dict, max_steps: int, device, eot_threshold: float = 0.5,
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
        item_name = entities[item_id].name if item_id in entities else "?"
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


def poll_loop(
    agent: AgentCNN,
    rcon: RconClient,
    *,
    poll_interval: float = 0.25,
    max_steps: int = 64,
    device: torch.device,
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
            log.warning("Non-JSON RCON response (first 200 chars): %r", raw[:200])
            continue

        try:
            handle_request(req, agent, rcon, max_steps=max_steps, device=device)
        except Exception:
            log.exception("Failed to handle request %s", req.get("request_id"))


def handle_request(
    req: dict, agent: AgentCNN, rcon: RconClient, *, max_steps: int, device
):
    log.info("Request %s: grid=%dx%d, %d sources, %d sinks",
             req["request_id"], req["grid_size"], req["grid_size"],
             len(req.get("sources", [])), len(req.get("sinks", [])))

    t0 = time.time()
    obs_CWH, stats = run_inference(agent, req, max_steps=max_steps, device=device)

    # Embed prediction metadata in the blueprint itself so it's visible in
    # the player's cursor / inventory — particularly useful for empty
    # results, where the player would otherwise see a blank blueprint with
    # no clue why.
    src_id, snk_id = _source_id(), _sink_id()
    placed_count = 0   # model-placed entities (belts, inserters, etc.)
    marker_count = 0   # source/sink markers (rendered as chests)
    for xx in range(obs_CWH.shape[1]):
        for yy in range(obs_CWH.shape[2]):
            eid = int(obs_CWH[Channel.ENTITIES.value, xx, yy])
            if eid == 0:
                continue
            if eid in (src_id, snk_id):
                marker_count += 1
            else:
                placed_count += 1
    total_entities = placed_count + marker_count  # matches blueprint contents

    # Direction enum (1=N, 2=E, 3=S, 4=W) → short label.
    _DIR_LABEL = {0: "-", 1: "N", 2: "E", 3: "S", 4: "W"}
    _MISC_LABEL = {0: "", 1: " UG_DOWN", 2: " UG_UP"}

    def _fmt_step(p: dict) -> str:
        # Factorio rich-text icon, then position + direction. No entity
        # name (the icon shows it) and no eot prob per step (keeps the
        # trace short — Factorio's blueprint description has a tight
        # character limit, ~500 chars in 2.0).
        ent_tag = "[item=" + p["entity_name"].replace("_", "-") + "]"
        dir_label = _DIR_LABEL.get(p["direction"], str(p["direction"]))
        item_tag = ""
        if p["item_name"] not in ("empty", "?", ""):
            item_tag = " [item=" + p["item_name"].replace("_", "-") + "]"
        return (f"{p['step']}: {ent_tag} ({p['x']},{p['y']}) "
                f"{dir_label}{_MISC_LABEL.get(p['misc'], '')}{item_tag}")

    placements = stats.get("placements", [])
    # Description has a hard ~500 char limit in Factorio 2.0; truncation
    # leaves a half-written rich-text tag at the end. Budget conservatively
    # and stop as soon as we'd exceed it.
    MAX_DESCRIPTION_CHARS = 480
    header_lines = [
        f"sources={len(req.get('sources', []))} "
        f"sinks={len(req.get('sinks', []))} "
        f"steps={stats['steps_taken']} stop={stats['stop_reason']}",
        "",
    ]
    used = sum(len(s) + 1 for s in header_lines)  # +1 for the newline
    trace_lines = []
    for p in placements:
        line = _fmt_step(p)
        if used + len(line) + 1 + len("...N more") > MAX_DESCRIPTION_CHARS:
            trace_lines.append(f"...{len(placements) - len(trace_lines)} more")
            break
        trace_lines.append(line)
        used += len(line) + 1

    label = (f"Factorion: {total_entities} entities "
             f"({placed_count} placed + {marker_count} markers)")
    description_parts = [
        f"sources={len(req.get('sources', []))} "
        f"sinks={len(req.get('sinks', []))} "
        f"steps={stats['steps_taken']} stop={stats['stop_reason']}",
        "",
    ] + trace_lines
    description = "\n".join(description_parts)
    bp_str = world_tensor_to_blueprint_string(
        obs_CWH, label=label, description=description,
    )
    log.info("Inference %.2fs; blueprint %d chars  label=%r",
             time.time() - t0, len(bp_str), label)

    # Decode the blueprint so we can log what's actually inside (entity
    # counts, names, positions) — easier than eyeballing the b64.
    try:
        from factorion import b64_to_dict
        decoded = b64_to_dict(bp_str)
        entities_list = decoded.get("blueprint", {}).get("entities", [])
        log.info("  blueprint contains %d entities:", len(entities_list))
        for e in entities_list[:40]:  # cap to 40 to avoid log spam
            log.info("    %s @ (%s,%s) dir=%s%s%s",
                     e.get("name"), e["position"]["x"], e["position"]["y"],
                     e.get("direction", "-"),
                     " recipe=" + e["recipe"] if "recipe" in e else "",
                     " type=" + e["type"] if "type" in e else "")
        if len(entities_list) > 40:
            log.info("    ... and %d more", len(entities_list) - 40)
        log.info("  blueprint string: %s", bp_str)
    except Exception:
        log.exception("Could not decode blueprint for logging")

    # Single-quoted Lua string: blueprint b64 alphabet [A-Za-z0-9+/=] plus
    # our "0" version prefix contains no single quotes, so this is safe.
    cmd = "/silent-command remote.call('factorion','deliver_blueprint','{}','{}')".format(
        req["request_id"], bp_str
    )
    resp = rcon.exec(cmd)
    if resp:
        log.info("RCON reply: %s", resp.strip())
    else:
        log.info("RCON reply: (empty — success)")


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to a torch.save'd AgentCNN state_dict.")
    ap.add_argument("--rcon-host", default="127.0.0.1")
    ap.add_argument("--rcon-port", type=int, default=27015)
    ap.add_argument("--rcon-password", default="factorion")
    ap.add_argument("--grid-size", type=int, default=8)
    ap.add_argument("--chan1", type=int, default=32)
    ap.add_argument("--chan2", type=int, default=64)
    ap.add_argument("--chan3", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=64,
                    help="Iterative inference budget per request.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device)
    # Sidecar JSON next to the checkpoint is the source of truth; CLI flags
    # override it if explicitly different from the defaults.
    sidecar = Hyperparams.from_json_sibling(args.checkpoint)
    defaults = Hyperparams()
    hp = Hyperparams(
        grid_size=args.grid_size if args.grid_size != defaults.grid_size else sidecar.grid_size,
        chan1=args.chan1 if args.chan1 != defaults.chan1 else sidecar.chan1,
        chan2=args.chan2 if args.chan2 != defaults.chan2 else sidecar.chan2,
        chan3=args.chan3 if args.chan3 != defaults.chan3 else sidecar.chan3,
    )
    agent = load_agent(args.checkpoint, hp, device)

    with RconClient(args.rcon_host, args.rcon_port, args.rcon_password) as rcon:
        log.info("RCON connected to %s:%d", args.rcon_host, args.rcon_port)
        # Sanity check: ping the mod. If a save isn't loaded yet, this
        # call will succeed at the RCON layer but the remote interface
        # won't exist; treat that as "wait, the user hasn't joined a
        # game yet" rather than a fatal error.
        try:
            r = rcon.exec("/silent-command rcon.print(remote.call('factorion','ping'))")
            log.info("Mod ping: %s", (r or "(no response — load a save with factorion enabled)").strip())
        except Exception:
            log.warning("Could not ping factorion mod — is the save loaded with the mod enabled?")

        poll_loop(
            agent, rcon,
            poll_interval=0.25, max_steps=args.max_steps, device=device,
        )


if __name__ == "__main__":
    main()
