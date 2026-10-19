"""Factorio saves -> scraped records of blueprints, for `corpus.build`.

    uv run python -m corpus.extract_saves saves/*.zip --index saves/index.jsonl --out raw/saves.jsonl.gz

Loads each save in the headless server (`factorion-mod/scripts/install_headless.sh`)
and blueprints the player's buildings on every surface, one blueprint per 256 ×
256 tile region that holds any, with the game's own `create_blueprint`, so a
world comes out exactly as a blueprint of it would. Trees, rocks, cliffs and
water aren't blueprintable and so don't come out. `--index` (one JSON object
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

FACTORIO = Path.home() / ".cache/factorio-headless/factorio"
DLC = ("space-age", "quality", "elevated-rails")
REGION = 256

LUA = """/sc local inv = game.create_inventory(1)
local bp = inv[1]
bp.set_stack{name = "blueprint"}
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
    bp.clear_blueprint()
    bp.create_blueprint{surface = surface, force = "player", include_trains = false,
      area = {{r[1] * REGION, r[2] * REGION}, {(r[1] + 1) * REGION, (r[2] + 1) * REGION}}}
    if bp.is_blueprint_setup() then
      helpers.write_file("OUT", helpers.table_to_json{surface = surface.name, x = r[1], y = r[2],
        string = bp.export_stack()} .. "\\n", true)
      n = n + 1
    end
  end
end
inv.destroy()
rcon.print(n)""".replace("REGION", str(REGION))


def extract(save: Path, dlc: bool, port: int = 27215) -> list[dict]:
    """(surface, x, y, string) of every region of `save` with buildings."""
    with tempfile.TemporaryDirectory() as tmp:
        mods = [{"name": "base", "enabled": True}, *({"name": m, "enabled": dlc} for m in DLC)]
        (Path(tmp) / "mod-list.json").write_text(json.dumps({"mods": mods}))
        settings = Path(tmp) / "settings.json"
        settings.write_text(json.dumps({"name": "extract", "description": "", "visibility": {"public": False, "lan": False}, "auto_pause": False}))
        out = FACTORIO / "script-output" / f"extract-{save.stem}.jsonl"
        out.unlink(missing_ok=True)
        log = open(Path(tmp) / "server.log", "w+")
        server = subprocess.Popen(
            [str(FACTORIO / "bin/x64/factorio"), "--start-server", str(save), "--mod-directory", tmp,
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
            if mods - {"base", *DLC}:
                print(f"skip {save.name}: mods {sorted(mods)}")
                continue
            t = time.time()
            try:
                regions = extract(save, dlc=bool(mods & set(DLC)))
            except RuntimeError as e:
                print(f"skip {e}")
                continue
            for r in regions:
                rec = {"source": f"save:{meta.get('source', 'unknown')}", "id": f"{save.name}#{r['surface']}#{r['x']},{r['y']}"}
                rec |= {"url": meta.get("url", ""), "title": meta.get("title", save.stem), "string": r["string"]}
                f.write(json.dumps(rec) + "\n")
            print(f"{save.name}: {len(regions)} regions in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
