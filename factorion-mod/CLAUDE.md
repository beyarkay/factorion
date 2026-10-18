# CLAUDE.md — factorion-mod

Notes for working on the Factorio mod + parity harness. Read
`factorion-mod/README.md` for the full picture; this file is just the
gotchas that bite during development.

## Running a parity check (runbook)

The parity harness replays engine-generated factories (and the textual test
fixtures) inside a real Factorio and compares each sink's measured items/s
against the engine's prediction. Full sequence, from nothing to a report:

1. **One-time setup** (per machine):
   - Symlink + enable the mod: `bash factorion-mod/scripts/install_mod.sh`
     (the script updates `mod-list.json` idempotently). See *install_mod.sh is
     macOS-safe*.
   - Enable RCON for GUI hosting: add `local-rcon-socket` /
     `local-rcon-password` under `[other]` in Factorio's `config.ini` (see
     *RCON to a GUI-hosted game*). On this laptop it's already
     `127.0.0.1:64502`.
   - Build the engine + deps from the repo root: `uv sync` then
     `uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml`.

2. **Start Factorio hosting the game so RCON binds:** launch Factorio, then
   **Play → Multiplayer → Host new game** (a fresh freeplay is fine — the
   harness builds on its own private surface). RCON only binds while hosting.

3. **Confirm the mod is live** (the ping check in *RCON to a GUI-hosted
   game*). If it errors, the mod isn't loaded — re-check step 1 and re-host.

4. **Run a check.** No Factorio needed for `--dry-run` (prints specs +
   engine expectations only). Against the hosted game:

   ```bash
   # a quick sweep of every lesson, 3 seeds each
   uv run python -m factorion_server.parity \
     --rcon-port 64502 --rcon-password <pw> --lessons all --seeds 3 --game-speed 100

   # one lesson / focused run
   uv run python -m factorion_server.parity \
     --rcon-port 64502 --rcon-password <pw> --lessons MEMORISE_1_INGREDIENT_RECIPES --seeds 10

   # full regression (lessons + every YAML fixture) — see the section below
   SEEDS=40 factorion-mod/scripts/parity_regression.sh \
     --rcon-port 64502 --rcon-password <pw>
   ```

   `<pw>` is the `local-rcon-password` from `config.ini`. Add `--json-out
   results.json` to dump every spec/result for offline analysis.

5. **Read the report.** Each factory prints per-sink `engine X/s, factorio
   Y/s (err Z%)`; the run ends with a ranked list of every sink with >0
   error and a pass/fail count (non-zero exit if any factory mismatches
   beyond `--rel-tol`/`--abs-tol`). Known-expected divergences (so a real
   regression stands out): long-handed
   inserter (0.86 vs ~1.23), the flat-inserter ~8% under, uncraftable
   smelting recipes → 0. Transport (belts/splitters/undergrounds) should be
   exact. Intentional engine simplifications that also show up here:
   - **Slow jams.** A lane carrying an item nobody drains, at less than the
     lane's full rate, stays open in the engine; in-game it backs up and
     in time blocks anything sideloading onto it. (A lane saturated by its
     straight-on flow blocks sideloads in both, `sideload_priority.yaml`.)
   - **Starved assemblers.** An assembler short of one ingredient still
     draws the others at the rate its output inserters could drain, so a
     shared belt downstream of it is under-counted.
   - **Sideload refill.** An inserter picking from the tile a lane is
     sideloaded onto leaves a gap the game refills on the same tile; the
     engine charges the pickup to the lane (≈7.5 → 6.64 downstream).

6. **After editing the mod Lua**, re-host to reload it (see *Reloading the
   mod after editing*) before the next run.

## Lua syntax-checking

Factorio's runtime Lua is 5.2-ish, but any `luac` parses the syntax fine.
On this laptop (macOS, Homebrew) the compiler is:

```bash
/opt/homebrew/bin/luac -p factorion-mod/mod/parity.lua factorion-mod/mod/control.lua
```

`luac -p` only parses (no output on success) — a fast syntax gate before
reloading the mod in-game. `/opt/homebrew/bin/lua` (5.4) is also present.
There is no `luac5.2` here; don't assume the sandbox's Lua tooling.

The mod also ships `mod/.luarc.json` for lua-language-server (declares the
Factorio globals: `game`, `storage`, `rendering`, `remote`, …). If you add
a new engine global, add it there too.

## Reloading the mod after editing

Factorio reads `parity.lua`/`control.lua` from disk only when a **new**
game is hosted — `game.reload_script()` reloads the *save's* embedded copy,
not disk (see the root MEMORY notes). So after editing mod Lua:

1. In Factorio: quit to the main menu.
2. Play → Multiplayer → **Host new game** (a fresh freeplay is fine).

The mod is symlinked into the mods dir by `scripts/install_mod.sh`, so no
re-copy is needed — just the re-host.

### install_mod.sh is macOS-safe

`scripts/install_mod.sh` parses `info.json` with POSIX `[[:space:]]` (not
GNU `\s`, which BSD/macOS `sed` treats as a literal `s` and silently
yields a garbage mod-dir name). Keep it that way.

## RCON to a GUI-hosted game

RCON binds only while Factorio is hosting multiplayer. For the GUI host
path it comes from `config.ini` `[other]` (`local-rcon-socket` /
`local-rcon-password`), **not** `--rcon-bind`. On this laptop it's already
set to `127.0.0.1:64502`. Quick liveness check:

```bash
uv run python -c "from factorion_server.rcon import RconClient; \
print(RconClient('127.0.0.1',64502,'<pw>').__enter__().exec(\
\"/silent-command rcon.print(remote.call('factorion','ping'))\"))"
```

## Parity harness quick reference

```bash
# needs a hosted game with the mod; --dry-run needs neither
uv run python -m factorion_server.parity \
  --rcon-port 64502 --rcon-password <pw> --lessons all --seeds 3
```

Measurement is adaptive: auto-warmup until the delivery rate plateaus, then
measure until the cumulative rate converges (counts are zeroed at the
boundary so buffer-fill can't inflate the rate), with hard tick caps. Crank
`--game-speed` (100+) to compress wall-clock; slow assembler lessons still
need many game-seconds to resolve a fractional items/s rate.

## Regression check after an engine change

`scripts/parity_regression.sh` is a thin wrapper (no new logic): it rebuilds
the engine, dumps the textual fixtures (`dump_fixtures_for_parity`), and runs
`parity.py --lessons all --seeds N --fixtures …` so one command replays every
lesson (N seeds each) plus every `tests/factories/*.yaml` fixture through a
running Factorio and reports any sink that diverges (non-zero exit on
mismatch). Pass RCON args through:

```bash
SEEDS=40 factorion-mod/scripts/parity_regression.sh \
  --rcon-port 64502 --rcon-password <pw>
```

`parity.py --fixtures <json>` folds the dumped fixtures into a normal run;
they batch/compare/report exactly like the lesson factories.
