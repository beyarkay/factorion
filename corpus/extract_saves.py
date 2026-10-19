"""Factorio saves -> scraped records of blueprints, for `corpus.build`.

    uv run python -m corpus.extract_saves saves/*.zip --index saves/index.jsonl --out raw/saves.jsonl.gz

Loads each save in the headless server (`factorion-mod/scripts/install_headless.sh`)
and blueprints the player's buildings on every surface, one blueprint per 256 ×
256 tile region that holds any, with the game's own `create_blueprint`, so a
world comes out exactly as a blueprint of it would. Each record's
`unbuildable` lists the terrain no belt can go on (water, lava, space, cliffs)
as `[x, y, length]` row runs in the blueprint's frame, which `corpus.build`
carries into the design and `corpus.convert` masks. `--index` (one JSON object
per line: file, url, title, mods_if_known) supplies each record's metadata and
skips saves with mods beyond base and the DLC.
"""

import argparse
import gzip
import json
import subprocess
import tempfile
import time
from pathlib import Path

from factorion_server.rcon import RconClient

# Stable loads 1.0..2.0 saves; 2.1's only load in the experimental build
# (`FACTORIO_HEADLESS_DIR=~/.cache/factorio-headless-experimental FACTORIO_VERSION=latest install_headless.sh`).
FACTORIO = Path.home() / ".cache/factorio-headless/factorio"
FACTORIO_EXPERIMENTAL = Path.home() / ".cache/factorio-headless-experimental/factorio"
OFFICIAL = {"base", "space-age", "quality", "elevated-rails", "recycler"}
REGION = 256

LUA = """/sc local inv = game.create_inventory(1)
local bp = inv[1]
bp.set_stack{name = "blueprint"}
-- Terrain a belt can't be built on (water, lava, oil ocean, space, out of map).
local belt = prototypes.entity["transport-belt"].collision_mask.layers
local blocking = {}
for name, t in pairs(prototypes.tile) do
  for layer in pairs(t.collision_mask.layers) do
    if belt[layer] then blocking[#blocking + 1] = name break end
  end
end
local n = 0
for _, surface in pairs(game.surfaces) do
  local regions = {}
  for chunk in surface.get_chunks() do
    local rx, ry = math.floor(chunk.x * 32 / REGION), math.floor(chunk.y * 32 / REGION)
    local key = rx .. "," .. ry
    if not regions[key] and surface.count_entities_filtered{area = chunk.area, force = "player", limit = 1} > 0 then
      regions[key] = {rx, ry}
    end
  end
  for _, r in pairs(regions) do
    local area = {{r[1] * REGION, r[2] * REGION}, {(r[1] + 1) * REGION, (r[2] + 1) * REGION}}
    bp.clear_blueprint()
    local made = bp.create_blueprint{surface = surface, force = "player", include_trains = false, area = area}
    local k, world = next(made)
    if k and bp.get_blueprint_entities() then
      -- The blueprint recentres its entities; shift terrain by the same offset.
      local ox, oy
      for _, e in pairs(bp.get_blueprint_entities()) do
        if e.entity_number == k then ox, oy = e.position.x - world.position.x, e.position.y - world.position.y end
      end
      local rows = {}
      local function add(x, y)
        rows[y] = rows[y] or {}
        rows[y][#rows[y] + 1] = x
      end
      for _, t in pairs(surface.find_tiles_filtered{area = area, name = blocking}) do add(t.position.x, t.position.y) end
      for _, c in pairs(surface.find_entities_filtered{area = area, type = "cliff"}) do
        local b = c.bounding_box
        for x = math.floor(b.left_top.x), math.ceil(b.right_bottom.x) - 1 do
          for y = math.floor(b.left_top.y), math.ceil(b.right_bottom.y) - 1 do add(x, y) end
        end
      end
      local runs = {}
      for y, xs in pairs(rows) do
        table.sort(xs)
        local i = 1
        while i <= #xs do
          local j = i
          while j < #xs and xs[j + 1] <= xs[j] + 1 do j = j + 1 end
          runs[#runs + 1] = {math.floor(xs[i] + ox + 0.5), math.floor(y + oy + 0.5), xs[j] - xs[i] + 1}
          i = j + 1
        end
      end
      helpers.write_file("OUT", helpers.table_to_json{surface = surface.name, x = r[1], y = r[2],
        string = bp.export_stack(), unbuildable = runs} .. "\\n", true)
      n = n + 1
    end
  end
end
inv.destroy()
rcon.print(n)""".replace("REGION", str(REGION))


def extract(save: Path, mods: set[str], factorio: Path, port: int = 27215) -> list[dict]:
    """(surface, x, y, string, unbuildable) of every region of `save` with
    buildings, loaded with the official `mods` it uses."""
    with tempfile.TemporaryDirectory() as tmp:
        listed = [{"name": m.name, "enabled": m.name in mods} for m in (factorio / "data").iterdir() if m.name in OFFICIAL]
        (Path(tmp) / "mod-list.json").write_text(json.dumps({"mods": listed}))
        settings = Path(tmp) / "settings.json"
        settings.write_text(json.dumps({"name": "extract", "description": "", "visibility": {"public": False, "lan": False}, "auto_pause": False}))
        out = factorio / "script-output" / f"extract-{save.stem}.jsonl"
        out.unlink(missing_ok=True)
        log = open(Path(tmp) / "server.log", "w+")
        server = subprocess.Popen(
            [str(factorio / "bin/x64/factorio"), "--start-server", str(save), "--mod-directory", tmp,
             "--server-settings", str(settings), "--rcon-bind", f"127.0.0.1:{port}", "--rcon-password", "pw",
             "--port", str(port + 1)],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        try:
            deadline = time.time() + 600
            while "Starting RCON interface" not in Path(log.name).read_text():
                if server.poll() is not None or time.time() > deadline:
                    raise RuntimeError(f"{save.name} did not load:\n{Path(log.name).read_text()[-2000:]}")
                time.sleep(1)
            with RconClient("127.0.0.1", port, "pw", timeout=1800) as rcon:
                rcon.exec("/sc rcon.print(1)")  # the first command only acknowledges disabling achievements
                reply = rcon.exec(LUA.replace('"OUT"', json.dumps(out.name)))
                if not reply.strip().isdigit():
                    raise RuntimeError(f"{save.name}: {reply}")
            return [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
        finally:
            server.terminate()
            server.wait()
            out.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("saves", nargs="+", type=Path)
    ap.add_argument("--index", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    index = {}
    if args.index:
        index = {(m := json.loads(line))["file"]: m for line in args.index.read_text().splitlines()}
    with gzip.open(args.out, "at") as f:
        for save in args.saves:
            meta = index.get(save.name, {})
            mods = set(meta.get("mods_if_known") or {"base": None})
            version = tuple(int(v) for v in (meta.get("factorio_version_if_known") or "2.0").split(".")[:2])
            if mods - OFFICIAL or version < (1, 0):
                print(f"skip {save.name}: version {version}, mods {sorted(mods)}")
                continue
            t = time.time()
            try:
                regions = extract(save, mods, FACTORIO_EXPERIMENTAL if version >= (2, 1) else FACTORIO)
            except RuntimeError as e:
                print(f"skip {e}")
                continue
            for r in regions:
                rec = {"source": f"save:{meta.get('source', 'unknown')}", "id": f"{save.name}#{r['surface']}#{r['x']},{r['y']}"}
                rec |= {"url": meta.get("url", ""), "title": meta.get("title", save.stem), "string": r["string"]}
                rec["unbuildable"] = r.get("unbuildable") or []
                f.write(json.dumps(rec) + "\n")
            print(f"{save.name}: {len(regions)} regions in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
