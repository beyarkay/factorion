---
name: factorio
description: Check something against real Factorio — engine parity (does factorion_rs's throughput match the game?), how an entity or mechanic actually behaves in-game, or running the factorion mod. Use whenever asked to "check this works in Factorio", verify against the real game, or run the parity harness. Downloads the free headless server on demand.
---

# Real Factorio, on demand

The headless server is a free download that needs no account or licence. It
is not installed by default; install it (idempotent, ~6 s, Linux only):

```bash
bash factorion-mod/scripts/install_headless.sh   # prints the binary path
```

You have free rein over it: start and stop servers, run any Lua over RCON,
build on any surface, edit and reload the mod, keep saves. It is a throwaway
server inside your sandbox. Stop every server you start when you're done.

## Parity checks

`parity_launch.sh` finds the headless binary, starts a server, runs
`server/parity.py`, and tears it down:

```bash
bash factorion-mod/scripts/parity_launch.sh --lessons MOVE_ONE_ITEM,SPLITTER_SPLIT --seeds 3 --game-speed 100
```

Flags, fixtures (`parity_regression.sh`), how to read the report, and the
known expected divergences are in `factorion-mod/CLAUDE.md`.

## Anything else: drive a server over RCON

```bash
B=~/.cache/factorio-headless/factorio
# Without auto_pause off, a server with no player connected runs at ~0.1 ticks/s.
echo '{"name": "factorion", "description": "", "visibility": {"public": false, "lan": false}, "auto_pause": false}' > $B/server-settings.json
$B/bin/x64/factorio --start-server-load-scenario base/freeplay --server-settings $B/server-settings.json \
  --rcon-bind 127.0.0.1:27015 --rcon-password pw > $B/server.log 2>&1 &
PID=$!   # kill $PID when done
```

```python
import sys; sys.path.insert(0, "factorion-mod/server")
from rcon import RconClient
with RconClient("127.0.0.1", 27015, "pw") as r:
    r.exec("/sc rcon.print(1)")  # a fresh server refuses its first Lua command (achievements warning)
    print(r.exec("/sc rcon.print(remote.call('factorion','ping'))"))
```

Any Lua runs through `/sc …`; return values via `rcon.print` (use
`helpers.table_to_json` for structures). The mod's remote interfaces are listed
under *Debug interfaces* in `factorion-mod/README.md`. `game.speed` is capped by
CPU at ~2,500 ticks/s on 4 cores. The server reads the mod's Lua only at
launch, so restart it after editing the mod.

## Limits

The headless build has no renderer: `game.take_screenshot` does nothing, so
there are no screenshots. Use `factorion.render_factory` for an ASCII view.
