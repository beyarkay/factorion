"""Validate the mod in isolated Factorio data; capture real day/night grids.

uv run --no-sync python factorion-mod/art/review_cable_fold.py --factorio PATH

Only a temporary copy of the mod receives the review scene. The normal
Factorio profile and saves are not used. Requires the graphical game binary.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

SCENE = """
-- Temporary art review scene, appended only to the isolated mod copy.
script.on_nth_tick(1, function(event)
  if not storage.art_review then
    local surface = game.surfaces[1]
    surface.request_to_generate_chunks({0, 0}, 2)
    surface.force_generate_chunk_requests()
    for _, entity in pairs(surface.find_entities_filtered({area={{-20,-16},{20,16}}})) do
      if entity.type ~= "character" then entity.destroy() end
    end
    local tiles = {}
    for x=-18,18 do for y=-13,12 do
      tiles[#tiles+1] = {name="refined-concrete",position={x,y}}
    end end
    surface.set_tiles(tiles)
    storage.art_review = {}
    for row=0,3 do for col=0,2 do
      local entity=surface.create_entity{name=inference.GPU_SERVER, position={-10+4*col,-7+2*row},
        direction=defines.direction.north, force="player"}
      storage.art_review[#storage.art_review+1] = {entity=entity,level=(row*3+col)%6}
    end end
    for row=0,2 do for col=0,3 do
      local entity=surface.create_entity{name=inference.GPU_SERVER, position={5+2*col,-6+4*row},
        direction=defines.direction.east,force="player"}
      storage.art_review[#storage.art_review+1] = {entity=entity,level=(row*4+col)%6}
    end end
    for i,name in ipairs({"lab","assembling-machine-2","accumulator","steel-chest"}) do
      surface.create_entity{name=name, position={-10+4*(i-1),5},force="player"}
    end
    for _, pair in ipairs({{10,11},{21,22}}) do
      local a,b=storage.art_review[pair[1]].entity,storage.art_review[pair[2]].entity
      for _, id in ipairs({defines.wire_connector_id.circuit_red,defines.wire_connector_id.circuit_green}) do
        assert(a.get_wire_connector(id,true).connect_to(b.get_wire_connector(id,true)))
      end
      draw_cable(surface,a,b)
    end
    if #game.players==0 then game.create_player{name="CableFoldReview"} end
    for _, player in pairs(game.players) do
      local character = player.character
      player.set_controller{type=defines.controllers.god}
      if character and character.valid then character.destroy() end
    end
    game.players[1].teleport({0,0},surface)
    game.forces.player.chart(surface,{{-20,-16},{20,16}})
    surface.freeze_daytime = true
    surface.daytime = 0
  end
  for _, item in ipairs(storage.art_review) do
    server_power(item.entity).energy = 150
    local state=server_state(item.entity)
    state.utilisation=item.level/5
    gpu_visuals.update(state,true,state.utilisation,false)
  end
  -- The graphics benchmark can attach its local player after tick one.
  -- Remove that character too, so no player flashlight contaminates the test.
  for _, player in pairs(game.players) do
    local character = player.character
    if character then
      player.set_controller{type=defines.controllers.god}
      if character.valid then character.destroy() end
    end
  end
  -- Freeplay can leave an unowned character after changing controllers.
  -- Keep it for scale, but exclude its headlamp from the lighting comparison.
  for _, character in pairs(game.surfaces[1].find_entities_filtered{type="character"}) do
    character.disable_flashlight()
  end
  if event.tick == 31 then game.surfaces[1].daytime = 0.5 end
  if event.tick == 30 or event.tick == 60 then
    local shot = event.tick == 30 and {"engine-day.png",0} or {"engine-night.png",0.5}
      game.take_screenshot{player=1,surface=game.surfaces[1],position={0,-1},
        resolution={1600,1050},zoom=1.6,show_gui=false,show_entity_info=false,
        hide_clouds=true,hide_fog=true,anti_alias=true,force_render=true,
        daytime=shot[2],path="cable-fold/"..shot[1]}
    -- Exercise the real renderer's writable sprite and light properties.
    local state=server_state(storage.art_review[1].entity)
    local original=state.visuals.glow
    gpu_visuals.update(state,true,1,true)
    assert(state.visuals.glow==original)
    assert(rendering.get_object_by_id(original).valid)
    gpu_visuals.update(state,false,0,false)
    assert(state.visuals==nil)
    gpu_visuals.update(state,true,0,false)
    if event.tick == 60 then
      log("CABLE_FOLD_REVIEW_OK: both footprints, glow levels, power cycle, render reuse, circuit/data ports")
    end
  end
end)
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorio", type=Path, required=True)
    args = parser.parse_args()
    art = Path(__file__).resolve().parent
    work = Path(tempfile.mkdtemp(prefix="factorion-cable-fold-", dir="/private/tmp"))
    mods = work / "mods"
    mod = mods / "factorion_0.0.0"
    shutil.copytree(art.parent / "mod", mod)
    with (mod / "control.lua").open("a") as stream:
        stream.write("\n" + SCENE)
    (mods / "mod-list.json").write_text(
        '{"mods":[{"name":"base","enabled":true},{"name":"factorion","enabled":true}]}\n'
    )
    data = args.factorio.parent.parent / "data"
    config = work / "config.ini"
    config.write_text(
        f"[path]\nread-data={data}\nwrite-data={work / 'user'}\n[other]\ncheck-updates=false\n"
    )
    common = [
        str(args.factorio),
        "--config",
        str(config),
        "--mod-directory",
        str(mods),
        "--disable-audio",
    ]
    save = work / "review.zip"
    print("Isolated Factorio review:", work, flush=True)
    subprocess.run(
        common
        + [
            "--create",
            str(save),
            "--map-gen-seed",
            "24",
            "--check-unused-prototype-data",
        ],
        check=True,
    )
    # Steam's desktop build otherwise restarts through the launcher and drops
    # this subprocess's isolated configuration. Keep the app ID in the test cwd.
    (work / "steam_appid.txt").write_text("427520\n")
    env = dict(os.environ, SteamAppId="427520", SteamGameId="427520")
    subprocess.run(
        common
        + [
            "--benchmark-graphics",
            str(save),
            "--benchmark-ticks",
            "90",
            "--window-size",
            "1600x1050",
            "--force-graphics-preset",
            "high",
        ],
        check=True,
        cwd=work,
        env=env,
    )
    logs = (work / "user/factorio-current.log").read_text()
    if "CABLE_FOLD_REVIEW_OK" not in logs:
        raise RuntimeError("Engine review did not finish; inspect " + str(work))
    for image in ("engine-day.png", "engine-night.png"):
        shutil.copy2(
            work / "user/script-output/cable-fold" / image, art / "cable-fold" / image
        )
    useful_lines = [
        line
        for line in logs.splitlines()
        if any(
            token in line
            for token in (
                "Factorio 2.",
                "Loading mod factorion",
                "Checksum of factorion",
                "CABLE_FOLD_REVIEW_OK",
                "Error",
                "Warning",
                "Quitting:",
                "Goodbye",
            )
        )
    ]
    (art / "cable-fold/engine-review.log").write_text("\n".join(useful_lines) + "\n")
    print("Engine screenshots and log saved in", art / "cable-fold", flush=True)


if __name__ == "__main__":
    main()
