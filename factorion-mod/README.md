# Factorion Mod

A Factorio mod + local Python server that lets you ask a trained
Factorion policy to design a factory layout from inside the game and
place the predicted entities directly into the world as the model runs.

## Round trip

A single RCON connection carries both directions. The server polls a
`poll_request` remote interface every ~250 ms; the mod returns either an
empty string (nothing pending) or the next queued request JSON.

```
   ┌──────────────┐  poll: remote.call("factorion","poll_request")   ┌────────┐
   │              │ ◄─────────────────────────────────────────────── │        │
   │   Factorio   │                                                  │ Python │
   │   (mod)      │ ──────────────────────────────────────────────►  │ server │
   │              │  reply: request JSON  (or "" when empty)         │        │
   │              │                                                  │        │
   │              │ ◄─────────────────────────────────────────────── │        │
   └──────────────┘  each step: place_prediction(req_id, entity)     └────────┘
                     final: finish_prediction(req_id, summary)
```

Why is this asymmetric on the wire even though it's one channel? Because
**Factorio's modding Lua has no file-read API and no socket access** —
the sandbox is intentional for multiplayer determinism. The only inbound
channel for an external process is RCON.

## Source / sink representation

The normal workflow uses a green **source belt** and orange **sink belt**. They
are real 1×1 transport belts: place, mine, rotate, copy, paste, and blueprint
them like ordinary belts. Click a placed endpoint to choose the item it
provides or receives. Alt mode displays that configured item over the belt.
Each sink also displays the items per second it consumed over a rolling
5-second window, refreshed twice per second.

Green sources keep both lanes supplied with the chosen item; orange sinks
consume that item when it reaches their tile. The endpoint's current rotation
is sent to the model. Rerunning a prediction removes only the entities created
by the previous prediction, so endpoint belts stay in place until you mine
them.

## RCON setup

RCON only binds when Factorio is **hosting multiplayer**. Two paths:

- **Headless server** (no GUI): `factorio --start-server <save> --rcon-bind 127.0.0.1:PORT --rcon-password PW`. RCON binds at launch. CLI flags work directly.
- **GUI multiplayer host** (singleplayer-feeling): `factorio --host <save>` opens the GUI in host mode. For RCON to bind, add to `config.ini` under `[other]`:

  ```ini
  local-rcon-socket=127.0.0.1:64502
  local-rcon-password=<some password>
  ```

  CLI `--rcon-bind` is **ignored** for GUI mode — config.ini is the only path.

`scripts/launch.sh` automates the headless path with auto-generated port+password.

## Layout

```
factorion-mod/
├── README.md                 ← this file
├── mod/                      ← the actual Factorio mod (publishable)
│   ├── info.json
│   ├── .luarc.json           ← lua-language-server config (Factorio globals)
│   ├── control.lua           ← event handlers, endpoint belts, RCON interface
│   ├── parity.lua            ← engine-parity runner (build spec, measure throughput)
│   ├── data.lua / settings.lua
│   ├── locale/en/factorion.cfg
│   └── prototypes/           ← endpoint belts, selection tools, hotkey definitions
├── server/                   ← local inference daemon + parity harness
│   ├── server.py             ← RCON poll loop → model (with eot_head stop) → RCON push
│   ├── parity.py             ← engine ↔ Factorio throughput comparison (issue #261)
│   ├── rcon.py               ← shared minimal Source-RCON client
│   ├── blueprint.py          ← tensor → blueprint utility used by tests/tooling
│   └── README.md
└── scripts/
    ├── install_mod.sh        ← symlink mod/ into Factorio's mods dir
    ├── serve.sh              ← one-command GUI setup + W&B/local model server
    ├── launch.sh             ← spawn Factorio with auto-RCON + start server (headless)
    └── parity_launch.sh      ← headless Factorio + parity harness, one command
```

## Quick start (GUI)

1. Configure GUI-host RCON once in `config.ini` as described above. Then start
   the mod from the repo root:

   ```bash
   ./start-mod.sh
   ```

   This installs dependencies and builds the Rust extension when needed,
   installs and enables the mod, downloads the run's latest model artifact,
   reads the exact grid/encoder architecture from W&B, and waits for Factorio.
   It defaults to checkpoint `h76h80yb`; pass a local checkpoint or another W&B
   run as the first argument, or set `FACTORION_CHECKPOINT`.

2. Restart Factorio and choose **Play → Multiplayer → Host new game**. The
   region brush and run `h76h80yb` both use a fixed 11×11 grid; checkpoints
   trained at another size are rejected with a clear error.

For a manual or headless setup, start the server directly. `--checkpoint`
accepts a local `.pt`, bare W&B run id, `entity/project/id`, or run URL:

```bash
uv run python factorion-mod/server/server.py \
  --checkpoint h76h80yb \
  --rcon-port 64502 --rcon-password <pw>
```

For local files, an `agent.hp.json` sidecar with
`{"grid_size": 11, "layers": [93, 69, 96], "kernel_size": 3}` is read
automatically.

3. In Factorio:
   - Press `Ctrl+T` to receive the blue region tool plus ten source belts and
     ten sink belts.
   - Click once with the blue **region tool**. It stamps an 11×11 region
     centered on that tile—there is no size-sensitive drag.
   - Place the green **source belt** and orange **sink belt** inside it as
     ordinary belts. Click each endpoint to choose its item, and hover it and
     press `R` to rotate it. Alt mode shows the configured item.
   - Press `Ctrl+P` to request a prediction.
   - Each predicted entity is placed directly into the marked region as soon
     as the model emits it. When the model stops, chat reports how many
     entities were placed and why inference ended.

   Hotkeys (rebindable in Controls → Factorion):
   - `Ctrl+P` — request prediction
   - `Ctrl+R` — clear the region and model-placed entities
   - `Ctrl+T` — re-grant the region tool and endpoint belts

   Change checkpoints without restarting the game or Python server:

   ```text
   /model h76h80yb
   /model /absolute/path/to/agent.pt
   ```

   The server reports success or the load error in chat. Models not trained on
   an 11×11 grid are rejected because the in-game brush is intentionally fixed.

## Engine parity harness (issue #261)

The mod doubles as a measurement rig for checking that the Factorion
engine's throughput simulation matches real Factorio. The harness builds
known-good factories with `factorion.build_factory`, asks the engine what
each sink should receive (`factorion_rs.py_sink_deliveries`), replays the
same factory inside the game, and compares measured per-sink items/s.

```
   Python (parity.py)                      Factorio (parity.lua)
   ──────────────────                      ─────────────────────
   build_factory(lesson, seed)
   world → spec JSON        ──ᴿᶜᴼᴺ──►      parity_start(spec):
   py_sink_deliveries(world)                 lab-tiles surface, own force,
                                             EEI+substation power, entities
   poll parity_poll()       ◄──ᴿᶜᴼᴺ──       warmup → measure at game.speed≫1
   compare per-sink rates                    per-sink counts + diagnostics
```

Sources/sinks are placed as real transport-belts scripted every tick
(source lanes kept full, sink lanes counted then cleared), so the grid
stays 1:1 with the engine's tile model and lane semantics — side-loading,
curves, inserter drop lanes — come from the real game.

One command (headless, tears itself down when done):

```bash
bash factorion-mod/scripts/parity_launch.sh --lessons all --seeds 3 --size 11
```

Or against an already-running instance (headless or GUI host, same RCON
setup as above):

```bash
uv run python factorion-mod/server/parity.py \
  --rcon-port 64502 --rcon-password <pw> \
  --lessons MOVE_ONE_ITEM,SPLITTER_SPLIT --seeds 5 --size 11
```

Useful flags: `--dry-run` (print specs + engine expectations, no Factorio
needed), `--json-out results.json` (full dump for offline analysis),
`--rel-tol/--abs-tol` (pass thresholds on per-sink items/s),
`--warmup-ticks/--measure-ticks/--game-speed` (run shape; defaults
1800/3600/32 — a 30 s settle plus a 60 s counting window, sped up 32×).

Each factory prints one line per sink (`engine 15.000/s, factorio
14.870/s (err 0.9%) ok`); a mismatch additionally prints the rendered
factory and Factorio-side per-entity diagnostics — belt lane occupancy,
machine status counts (`item_ingredient_shortage`, `output_full`, …),
`products_finished` deltas, inserter held fractions — to localise where
flow stalls, per the diagnosis idea in #261. The exit code is 0 iff every
sink of every factory matched. This is a local, on-demand tool — it needs
a licensed Factorio install, so it deliberately does not run in CI.

Runs narrate themselves in-game: chat messages (`game.print`, which also
reach the headless server console) announce build errors, warmup→measure
transitions, periodic progress with live per-sink rates, and the final
rates; map overlays draw the grid outline, label every source/sink (sink
labels tick up with the measured rate), and hang a red status tag over
any machine/inserter that isn't `working` at sample time — so a
spectator literally watches where flow stalls. The Python side mirrors
the heartbeat, printing phase + percent lines while it polls.

To watch a run from the GUI: `/c game.player.teleport({5, 5},
"factorion-parity")` (the grid's top-left tile is at 0,0 on that
surface). Runs execute on their own surface and force, so they never
touch the hosting save's world; `game.speed` is restored to 1.0 when the
run ends or `parity_abort` is called.

### Parity status: what's verified vs. needs a live game

Verified without Factorio: Lua syntax (`luac -p`), the tensor→spec
conversion across every `LessonKind` (`tests/test_parity_spec.py` —
prototype names, direction conversion incl. the inserter flip, splitter
centers, UG types, recipes, RCON-safe JSON), and that
`py_sink_deliveries` aggregates back to `simulate_throughput`'s score.

To check on the first live run (in rough order):

1. `remote.call('factorion','parity_start', …)` round trip:
   `bash factorion-mod/scripts/parity_launch.sh --lessons MOVE_ONE_ITEM --seeds 1`
   — a pure belt line, engine says 15.0/s. Expect ~15/s measured.
2. Scripted source/sink belts actually saturate/drain (sources feed
   ~15/s in the result's `sources[].rate`).
3. `create_entity` direction semantics for inserters match blueprint
   import (the +8 flip) — MEMORISE lessons stall at 0/s if wrong.
4. Power: machines/inserters show `working`, not `no_power`, in
   `status_counts` (substation ring + electric-energy-interface).
5. AM1 accepts 3-5-ingredient recipes on the all-recipes force
   (MEMORISE_3.._5 lessons).
6. `game.speed` actually reached (wall-clock per run ≈
   (warmup+measure)/60/speed seconds, CPU permitting).

## Debug interfaces (RCON)

Server-callable remote methods exposed by the mod:

- `ping()` — round-trip check
- `place_prediction(request_id, placement_json)` /
  `finish_prediction(request_id, summary_json)` — streamed model placement
- `parity_start(spec_json)` / `parity_poll()` / `parity_abort()` — the
  engine-parity runner (see above; driven by `server/parity.py`)
- `introspect()` — outbox depth, pending requests, players known
- `dump_state(player_index?)` — full footprint + sources + sinks dump
- `inject_request(json, deliver_to_player_index)` — synthesise a request
  without using the hotkey (for headless / scripted tests). Pass
  `player_index=0` for the headless sentinel, which logs streamed placements.

## Status

### Verified end-to-end (Factorio 2.0.76)

- Mod loads cleanly, `lua-language-server --check` reports clean.
- Headless round trip via `--start-server-load-scenario base/freeplay` +
  `inject_request` (proves the wire).
- Source and sink items place configurable one-tile endpoint belts with normal
  mining, rotation, copy/paste, blueprint, and Alt-mode behavior.
- Model actions stream over RCON and create real entities in the player's
  world; rerun/reset cleanup is limited to entities created by the mod.
- `eot_head` is wired as the iterative stop signal (PPO PR #103 landed
  via the main-merge).

### What doesn't work / wasn't possible

- **Symmetric file-based transport** — tried; Factorio's modding Lua has
  no `game.read_file`, no `loadfile`, no socket access. Inbound side
  must be RCON.
- **Avoiding launch flags entirely** — RCON config can't be set
  in-game. For GUI hosting, `config.ini`'s `local-rcon-socket` /
  `local-rcon-password` is the only path (no env-vars, no in-game UI).
- **Poking mod `storage` directly from RCON** — `/silent-command` runs
  in *level scope*, not mod scope. The mod's `remote.add_interface`
  methods are the only way in.
- **`game.reload_script()` picking up `control.lua` edits from disk** —
  reloads from the **save's embedded mod scripts**, so changes only
  stick after save → exit → host-saved-game cycle.
- **World placement in headless mode** — no player exists to select a surface
  and force. `player_index=0` remains a sentinel that logs each placement.

### Not yet integrated

- **Stochastic / temperature sampling.** Inference is argmax-only; with
  weak checkpoints the policy can argmax-loop on the same tile. A
  `--temperature` flag with `Categorical(logits)` sampling is ~10 LoC.
- **Full multi-tile action masking.** The tile head masks occupied anchor
  tiles, and Factorio rejects a streamed entity whose rotated footprint lies
  outside the region or collides in the world. The model-side mask does not
  yet pre-mask every invalid multi-tile anchor.
- **Cross-platform `launch.sh` binary discovery**. macOS Steam install
  verified; Linux / Windows paths are heuristics. Override with
  `FACTORIO_BIN=…` if it can't find yours.

## Lua linting

The mod directory ships a `.luarc.json` so
[`lua-language-server`](https://github.com/LuaLS/lua-language-server)
recognises Factorio's runtime globals (`game`, `storage`, `script`,
`remote`, `helpers`, `defines`, `data`, `settings`, `rcon`, `log`,
`table_size`).

CLI check:

```bash
lua-language-server --check factorion-mod/mod --checklevel=Warning
```

For full API typing, install
[FMTK / vscode-factoriomod-debug](https://github.com/justarandomgeek/vscode-factoriomod-debug)
which ships full Factorio API definitions for LLS.
