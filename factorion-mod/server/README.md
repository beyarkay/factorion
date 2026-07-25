# Factorion server

Loads a trained Factorion `AgentCNN` checkpoint (local or directly from a W&B
run), polls a running
Factorio instance over RCON for prediction requests, runs iterative
greedy inference, and streams each predicted entity back via the same RCON
connection for immediate placement in the world.

## Run

Easiest GUI path: run the root-level launcher. It installs dependencies and
builds the Rust extension when needed, installs/enables the mod, reads
Factorio's GUI RCON config, downloads the default W&B model, and waits for a
hosted game:

```bash
./start-mod.sh
```

Pass a local checkpoint or another W&B run as the first argument when needed.
Predicted entities are paced with a 20 ms delay after each placement, making
the sequence easier to follow. Override it in milliseconds when needed:

```bash
./start-mod.sh --placement-delay-ms 30
```

Use `--placement-delay-ms 0` to restore unpaced placement.

Manual path:

```bash
# from the repo root, so `factorion` and `factorion_rs` import cleanly
uv run python factorion-mod/server/server.py \
  --checkpoint h76h80yb \
  --rcon-host 127.0.0.1 \
  --rcon-port 27015 \
  --rcon-password factorion
```

If you go manual, start Factorio with matching flags:

```bash
factorio --rcon-bind 127.0.0.1:27015 --rcon-password factorion
```

Required flags for the server:

- `--checkpoint` — a local `torch.save(agent.state_dict(), ...)` file, W&B run
  id, `entity/project/id`, or normal W&B run URL. W&B runs supply grid and
  architecture metadata automatically.
- `--rcon-port` / `--rcon-password` — must match what Factorio was
  launched with.

Architecture overrides (normally only needed for old local checkpoints with no
sidecar metadata):

- `--grid-size 11`
- `--layers 93,69,96 --kernel-size 3`

If you put a sidecar JSON next to the checkpoint named `agent.hp.json`
with `{"grid_size": 11, "layers": [93, 69, 96], "kernel_size": 3}`, the
server reads it automatically (explicit CLI flags override metadata).

While the server is running, `/model <path-or-wandb-id>` in Factorio hot-swaps
the in-memory checkpoint. A failed load leaves the previous model active. W&B
models report their direct run URL in chat, and every prediction queue message
names the active model.

## Protocol

The server runs a poll loop. Every 250 ms it sends, over RCON:

```
/silent-command rcon.print(remote.call('factorion','poll_request'))
```

The mod's `poll_request` returns either an empty string (queue empty)
or the next pending request JSON:

```json
{
  "request_id": "1234-5-7",
  "player_index": 1,
  "grid_size": 8,
  "footprint": [[0, 0], [0, 1], ...],
  "entities": [
    {"name": "transport-belt", "x": 2, "y": 3, "direction": 2},
    {"name": "assembling-machine-1", "x": 4, "y": 4, "direction": 0,
     "item": "iron-gear-wheel"}
  ],
  "sources": [{"x": 0, "y": 3, "direction": 2, "item": "iron-plate"}],
  "sinks":   [{"x": 7, "y": 3, "direction": 2, "item": "iron-plate"}],
  "default_item": "iron-plate"
}
```

`direction` uses Factorion's enum (1=N, 2=E, 3=S, 4=W), *not* Factorio's
16-step runtime convention; the server converts before emitting.
Ctrl-P snapshots supported entities already inside the region into `entities`,
including their complete rotated footprint, recipes, and underground-belt
input/output state. Inserter directions are converted from Factorio's drop
direction to the model's pickup direction.

After every accepted action, the server sends:

```
/silent-command rcon.print(remote.call(
  'factorion','place_prediction','<req_id>','<placement_json>'))
```

The placement contains the Factorio prototype, center position relative to the
region, rotated footprint, direction, and optional recipe or underground-belt
type. The mod looks up the requesting player and creates the entity on that
player's surface. The server then calls `finish_prediction` with the stop
reason; rerunning or resetting removes only entities created by this protocol.

## Reconnect

The poll loop survives Factorio restarts: on `RconError` / `OSError`,
it closes the socket, sleeps 2 s, and tries `connect()` again. So you
can leave the server running across save reloads or even full Factorio
relaunches.

## Debugging without Factorio

Spin up `nc` as a fake RCON endpoint to verify the wire format, or run
just the inference path:

```python
from server import action_to_placement, load_agent, run_inference
import torch
agent = load_agent(...)
obs, stats = run_inference(
    agent, fake_req, max_steps=64, device=torch.device("cpu"),
    on_placement=lambda action: print(action_to_placement(action)) or True,
)
print(stats)
```
