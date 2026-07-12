# Factorion Mod

A Factorio mod + local Python server that lets you ask a trained
Factorion policy to design a factory layout from inside the game and
paste it back as a blueprint.

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
   └──────────────┘  push: remote.call("factorion",                  └────────┘
                          "deliver_blueprint", req_id, bp_b64)
```

Why is this asymmetric on the wire even though it's one channel? Because
**Factorio's modding Lua has no file-read API and no socket access** —
the sandbox is intentional for multiplayer determinism. The only inbound
channel for an external process is RCON.

## Source / sink representation

The normal workflow uses separate green **source** and orange **sink** tools.
Click a tile and a dialog opens; choose what that endpoint provides or receives
and the direction items flow. The tool places a real constant-combinator on the
tile with the item, role, and arrow signals, plus a floating colored label such
as `SOURCE for iron-plate →`. Mining the combinator removes the endpoint.

Configured **constant-combinators** remain available as an advanced/legacy
alternative. Place one in the region and configure its first section with
three filters:

| filter | purpose         | values                                                                  |
| ------ | --------------- | ----------------------------------------------------------------------- |
| item   | what flows      | `iron-plate`, `electronic-circuit`, … (items tab)                       |
| role   | source vs sink  | `signal-output` (source) or `signal-input` (sink) (virtual signals)     |
| arrow  | facing          | `up-arrow` / `right-arrow` / `down-arrow` / `left-arrow` (virtual)      |

Slot order doesn't matter — the mod parses by signal name. The server
re-renders these same combinators in the output blueprint so the round
trip is loss-free; you can paste-over your existing markers and only the
model's belts/inserters get added as ghosts.

There's also a fallback click-tool (drag = sources, Shift+drag = sinks)
if you'd rather mark by tile without dropping combinators.

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
│   ├── control.lua           ← event handlers, RCON interface, combinator auto-detect
│   ├── parity.lua            ← engine-parity runner (build spec, measure throughput)
│   ├── data.lua / settings.lua
│   ├── locale/en/factorion.cfg
│   └── prototypes/           ← footprint + marker selection tools, hotkey definitions
├── server/                   ← local inference daemon + parity harness
│   ├── server.py             ← RCON poll loop → model (with eot_head stop) → RCON push
│   ├── parity.py             ← engine ↔ Factorio throughput comparison (issue #261)
│   ├── rcon.py               ← shared minimal Source-RCON client
│   ├── blueprint.py          ← obs tensor → Factorio blueprint b64 (combinator markers)
│   └── README.md
└── scripts/
    ├── install_mod.sh        ← symlink mod/ into Factorio's mods dir
    ├── serve.sh              ← one-command GUI setup + W&B/local model server
    ├── launch.sh             ← spawn Factorio with auto-RCON + start server (headless)
    └── parity_launch.sh      ← headless Factorio + parity harness, one command
```

## Quick start (GUI)

1. Build the project's Rust extension and install deps (one-time, from
   the repo root):

   ```bash
   uv sync
   uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
   ```

2. Configure GUI-host RCON once in `config.ini` as described above. Then run
   the all-in-one command with either a local checkpoint or a W&B run id/URL:

   ```bash
   bash factorion-mod/scripts/serve.sh 38vza7tf
   ```

   This installs and enables the mod, downloads the run's latest model artifact,
   reads the exact grid/encoder architecture from W&B, and waits for Factorio.

3. Restart Factorio and choose **Play → Multiplayer → Host new game**. The
   region brush and run `38vza7tf` both use a fixed 11×11 grid; checkpoints
   trained at another size are rejected with a clear error.

For a manual or headless setup, start the server directly. `--checkpoint`
accepts a local `.pt`, bare W&B run id, `entity/project/id`, or run URL:

```bash
uv run python factorion-mod/server/server.py \
  --checkpoint 38vza7tf \
  --rcon-port 64502 --rcon-password <pw>
```

For local files, an `agent.hp.json` sidecar with
`{"grid_size": 11, "layers": [93, 69, 96], "kernel_size": 3}` is read
automatically.

4. In Factorio:
   - Press `Ctrl+T` to receive the three custom tools.
   - Click once with the blue **region tool**. It stamps an 11×11 region
     centered on that tile—there is no size-sensitive drag.
   - Click inside it with the green **source tool**, then choose the supplied
     item and flow direction. Do the same with the orange **sink tool**. Each
     endpoint becomes a labeled constant-combinator on the map.
   - Press `Ctrl+P` to request a prediction.
   - A blueprint titled `Factorion: N entities (M placed + K markers)`
     lands in your cursor. Description has a per-step trace with item
     icons. Paste to materialise.

   Hotkeys (rebindable in Controls → Factorion):
   - `Ctrl+P` — request prediction
   - `Ctrl+R` — clear footprint, sources and sinks
   - `Ctrl+T` — re-grant selection tools

   Change checkpoints without restarting the game or Python server:

   ```text
   /model 38vza7tf
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
- `parity_start(spec_json)` / `parity_poll()` / `parity_abort()` — the
  engine-parity runner (see above; driven by `server/parity.py`)
- `introspect()` — outbox depth, pending requests, players known
- `dump_state(player_index?)` — full footprint + sources + sinks dump
- `inject_request(json, deliver_to_player_index)` — synthesise a request
  without using the hotkey (for headless / scripted tests). Pass
  `player_index=0` for the headless sentinel where `deliver_blueprint`
  logs the result instead of cursor-injecting.

## Status

### Verified end-to-end (Factorio 2.0.76)

- Mod loads cleanly, `lua-language-server --check` reports clean.
- Headless round trip via `--start-server-load-scenario base/freeplay` +
  `inject_request` (proves the wire).
- Live GUI round trip via `--host <save>` with config.ini RCON +
  W&B SFT checkpoint `0g9hfjna`: place combinators → drag footprint →
  auto-predict → blueprint in cursor → paste materialises markers + model
  placements with no Factorio import warnings.
- Constant-combinator markers round-trip through blueprint import/export
  preserving all three filter signals.
- Per-step inference trace embedded in the blueprint description with
  rich-text item icons; description char-budget aware so it doesn't
  truncate mid-tag.
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
- **Cursor injection in headless mode** — no player exists, so
  `cursor_stack` ops would crash. Server uses `player_index=0` as the
  sentinel and the mod logs the blueprint via `log()` instead.

### Not yet integrated

- **Stochastic / temperature sampling.** Inference is argmax-only; with
  weak checkpoints the policy can argmax-loop on the same tile. A
  `--temperature` flag with `Categorical(logits)` sampling is ~10 LoC.
- **Tile-mask in the tile head**: prevent the model from re-picking a
  tile that's already non-empty. Would also break degenerate loops.
- **Multi-tile entity validation** during iterative inference. The
  obs-side state update treats splitters / undergrounds as 1×1; final
  blueprint emission handles them correctly. To enforce validity during
  the loop, drive `FactorioEnv.step()` directly.
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

For full API typing (typed `LuaPlayer.cursor_stack` etc.), install
[FMTK / vscode-factoriomod-debug](https://github.com/justarandomgeek/vscode-factoriomod-debug)
which ships full Factorio API definitions for LLS.
