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

The mod uses **constant-combinators** as source/sink markers. Place one
in your footprint and configure its first section with three filters:

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
│   ├── data.lua / settings.lua
│   ├── locale/en/factorion.cfg
│   └── prototypes/           ← footprint + marker selection tools, hotkey definitions
├── server/                   ← local inference daemon
│   ├── server.py             ← RCON poll loop → model (with eot_head stop) → RCON push
│   ├── blueprint.py          ← obs tensor → Factorio blueprint b64 (combinator markers)
│   └── README.md
└── scripts/
    ├── install_mod.sh        ← symlink mod/ into Factorio's mods dir
    └── launch.sh             ← spawn Factorio with auto-RCON + start server (headless)
```

## Quick start

1. Build the project's Rust extension and install deps (one-time, from
   the repo root):

   ```bash
   uv venv --python 3.11
   uv pip install -r requirements.txt
   uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
   ```

2. Symlink the mod into Factorio:

   ```bash
   bash factorion-mod/scripts/install_mod.sh
   ```

3. Set `mod settings → Factorion grid size` to match what your checkpoint
   was trained on (default 8; the included `0g9hfjna` checkpoint is 11).

4. Enable RCON. Either headless (use `scripts/launch.sh`) or GUI host
   mode (edit `config.ini` per above).

5. Start the server pointed at your checkpoint:

   ```bash
   uv run python factorion-mod/server/server.py \
     --checkpoint path/to/agent.pt \
     --rcon-port 64502 --rcon-password <pw>
   ```

   A sidecar `agent.hp.json` next to the `.pt` with `{"grid_size": N, "chan1": …}`
   is read automatically.

6. In Factorio:
   - Place one or more **constant-combinators** in the area you want
     the factory built. Configure each per the table above.
   - Take the **Factorion footprint tool** (granted on first join; or
     `Ctrl+T` to re-grant).
   - Drag-select the rectangular footprint over your combinators. The
     mod auto-fires a prediction as soon as the area is valid (≥1 source
     + ≥1 sink detected). If you add combinators later, press `Ctrl+P`.
   - A blueprint titled `Factorion: N entities (M placed + K markers)`
     lands in your cursor. Description has a per-step trace with item
     icons. Paste to materialise.

   Hotkeys (rebindable in Controls → Factorion):
   - `Ctrl+P` — request prediction
   - `Ctrl+R` — clear footprint, sources and sinks
   - `Ctrl+T` — re-grant selection tools

## Debug interfaces (RCON)

Server-callable remote methods exposed by the mod:

- `ping()` — round-trip check
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
