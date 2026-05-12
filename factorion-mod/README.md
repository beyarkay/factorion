# Factorion Mod

A Factorio mod + local Python server that lets you ask a trained Factorion
policy to design a factory layout from inside the game and paste it back as
a blueprint.

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
channel for an external process is RCON. So:

- The mod *enqueues* outbound requests into a Lua table; the server
  *pulls* them by RCON-polling `poll_request`.
- The server *pushes* blueprint responses by RCON-calling
  `deliver_blueprint`.

One TCP socket, two directions, no script-output files involved.

## Why does Factorio need a special launch?

RCON is configured only by command-line flags (`--rcon-bind`,
`--rcon-password`) — there is no `config.ini` or env-var equivalent. The
included `scripts/launch.sh` spawns Factorio with those flags
pre-baked (random free port + random password) and starts the server
pointed at the same endpoint, so you never type them yourself.

If you'd rather launch Factorio manually (e.g. through Steam), pass:

```bash
factorio --rcon-bind 127.0.0.1:27015 --rcon-password factorion
```

…and start the server with matching `--rcon-port` / `--rcon-password`.

## Layout

```
factorion-mod/
├── README.md                 ← this file
├── mod/                      ← the actual Factorio mod (publishable)
│   ├── info.json
│   ├── .luarc.json           ← lua-language-server config (Factorio globals)
│   ├── control.lua           ← event handlers, RCON interface (poll + deliver)
│   ├── data.lua              ← prototype loader
│   ├── settings.lua          ← mod settings (grid size, default item)
│   ├── locale/en/factorion.cfg
│   └── prototypes/
│       ├── selection-tools.lua  ← footprint / source / sink tools
│       └── custom-inputs.lua    ← hotkey: execute prediction
├── server/                   ← local inference daemon
│   ├── server.py             ← RCON poll loop → model → RCON push
│   ├── blueprint.py          ← tensor → Factorio blueprint JSON
│   └── README.md
└── scripts/
    ├── install_mod.sh        ← symlink mod/ into Factorio's mods dir
    └── launch.sh             ← spawn Factorio with auto-RCON + start server
```

## Quick start

1. Build the project's Rust extension and install deps (one-time, from the
   repo root):

   ```bash
   uv venv --python 3.11
   uv pip install -r requirements.txt
   uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
   ```

2. Symlink the mod into Factorio:

   ```bash
   bash factorion-mod/scripts/install_mod.sh
   ```

3. Launch Factorio + the model server with one command:

   ```bash
   bash factorion-mod/scripts/launch.sh runs/<your-run>/agent.pt
   ```

   This spawns Factorio with RCON pre-configured on a random free port
   and starts the server pointed at the same endpoint. Load a save with
   the `factorion` mod enabled — the server connects automatically.

4. In Factorio:
   - Take the **Factorion footprint tool** from your inventory (granted on
     first join). Drag-select an 8×8 region.
   - Take the **Factorion source/sink tool**. Left-click tiles on the
     footprint border to add a source; right-click to add a sink.
   - Press **Ctrl+Shift+P** (default) to request a prediction.
   - A blueprint with the predicted layout appears in your cursor. Paste.

## Status

### What works (verified end-to-end against Factorio 2.0.76, 2026-05-12)

- Mod loads cleanly in Factorio 2.0 (`factorio --start-server-load-scenario base/freeplay`).
- Server connects via RCON, pings the mod, starts the poll loop.
- Client injects a request via the `inject_request` remote interface.
- Server polls, deserialises, runs iterative inference, emits a
  valid Factorio blueprint b64 string, pushes it back via RCON.
- Headless mode reports the blueprint via `log()` since there's no
  player cursor to inject into.
- `lua-language-server --check` reports clean on the mod.

### What doesn't work / wasn't possible

- **Symmetric file-based transport** — tried; Factorio's modding Lua
  has no `game.read_file`, no `loadfile`, no socket access. The sandbox
  is intentional for multiplayer determinism. Inbound side must be RCON.
- **Avoiding the `--rcon-port` launch flag** — tried; RCON is only
  configurable via command-line flags. No `config.ini` or env-var
  alternative exists in Factorio. `scripts/launch.sh` hides the flag
  by spawning the binary itself, but the flag is still there.
- **Poking mod `storage` directly from RCON** — tried; `/silent-command`
  runs in *level (scenario) scope*, not any mod's scope. The `storage`
  visible to RCON commands is the scenario's, not the mod's. Required
  adding an `inject_request` remote interface as the only way in.
- **Cursor injection in headless mode** — tried; no player exists in
  `--start-server-load-scenario`, so `game.get_player(1)` is nil and
  `cursor_stack` ops would crash. Worked around with a
  `deliver_to_player_index=0` sentinel that logs the blueprint instead.

### Not yet tried / known partial

- **Live in-game UI flow** (footprint drag-select, source/sink marker,
  Ctrl+Shift+P hotkey, cursor injection into a real player) — needs a
  human in the seat with a non-headless game. The wire is proven; the
  GUI events are untested.
- **Source/sink item picker** — hardcoded to `iron-plate` until a per-
  source GUI is added.
- **Multi-tile entities (splitters, undergrounds) during iterative
  inference** — emitted in the final blueprint correctly, but the
  server's iterative state update treats them as 1×1. The policy may
  compose them in odd ways until the stepping loop calls into
  `factorion_rs` for proper validation.
- **Cross-platform `launch.sh` binary discovery** — macOS path verified
  (Steam install), Linux/Windows paths are heuristics. Override with
  `FACTORIO_BIN=/path/to/factorio` if it can't find yours.
- **Reconnect after `/c game.reload_mods()`** — server has reconnect
  logic but reloading the mod mid-session hasn't been exercised.

## Lua linting

The mod directory ships a `.luarc.json` so
[`lua-language-server`](https://github.com/LuaLS/lua-language-server)
recognises Factorio's runtime globals (`game`, `storage`, `script`,
`remote`, `helpers`, `defines`, `data`, `settings`, `rcon`, `log`).

CLI check:

```bash
lua-language-server --check factorion-mod/mod --checklevel=Warning
```

Exits clean when there are no diagnostics. For deeper API-level type
checking (e.g. typed `LuaPlayer.cursor_stack`), install
[FMTK / vscode-factoriomod-debug](https://github.com/justarandomgeek/vscode-factoriomod-debug)
which ships full Factorio API definitions for LLS.
