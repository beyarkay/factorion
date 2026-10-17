# Factorion server

Loads the trained Factorion denoiser checkpoint named by the mod's
"Factorion checkpoint" setting (local or directly from a W&B run), polls a
running Factorio instance over RCON for prediction requests, generates eight
factories around each request's markers (its existing entities are walls to
build around), and streams the highest-throughput one's entities back via
the same RCON connection for immediate placement in the world.

## Run

Easiest GUI path: run the root-level launcher. It installs dependencies and
builds the Rust extension when needed, installs/enables the mod, reads
Factorio's GUI RCON config, and waits for a hosted game:

```bash
./start-mod.sh
```

Placement pacing and the sampling temperature and seed are set in game;
each request carries their current values. Each prediction
draws a fresh seed (reported in chat) unless the seed setting fixes one.

Manual path:

```bash
# from the repo root, so `factorion` and `factorion_rs` import cleanly
uv run python factorion-mod/server/server.py \
  --rcon-host 127.0.0.1 \
  --rcon-port 27015 \
  --rcon-password factorion
```

If you go manual, start Factorio with matching flags:

```bash
factorio --rcon-bind 127.0.0.1:27015 --rcon-password factorion
```

Required flags for the server:

- `--rcon-port` / `--rcon-password` — must match what Factorio was
  launched with.

The checkpoint setting accepts a W&B run id, `entity/project/id`, or a local
`.pt` checkpoint saved by `sft.py` or `rl.py`. The denoiser runs at any grid
size; `--grid-size` (default 15) is the region the mod plans.

## Protocol

The server runs a poll loop. Every 250 ms it sends, over RCON:

```
/silent-command rcon.print(remote.call('factorion','poll_request'))
```

The mod's `poll_request` returns either an empty string (queue empty)
or the next queued message. A new game or a change to the checkpoint setting
queues

```json
{ "type": "load_model", "checkpoint": "dr17bmik" }
```

which the server answers with `model_loaded` (`checkpoint`, run `name`,
`grid_size`) or `model_failed` (`checkpoint`, `error`). `Ctrl+P` queues a
prediction request:

```json
{
  "type": "predict",
  "request_id": "1234-5-7",
  "checkpoint": "dr17bmik",
  "grid_size": 15,
  "entities": [
    { "name": "transport-belt", "x": 2, "y": 3, "direction": 2 },
    {
      "name": "assembling-machine-1",
      "x": 4,
      "y": 4,
      "direction": 0,
      "item": "iron-gear-wheel"
    }
  ],
  "sources": [{ "x": 0, "y": 3, "direction": 2, "item": "iron-plate" }],
  "sinks": [{ "x": 10, "y": 3, "direction": 2, "item": "iron-plate" }]
}
```

`direction` uses Factorion's enum (1=N, 2=E, 3=S, 4=W), _not_ Factorio's
16-step runtime convention; the server converts before emitting.
Ctrl-P snapshots supported entities already inside the region into `entities`,
including their complete rotated footprint, recipes, and underground-belt
input/output state. Inserter directions are converted from Factorio's drop
direction to the model's pickup direction.

For each entity of the chosen build, the server sends:

```
/silent-command rcon.print(remote.call(
  'factorion','place_prediction','<req_id>','<placement_json>'))
```

The placement contains the Factorio prototype, center position relative to the
region, rotated footprint, direction, and optional recipe or underground-belt
type. The mod looks up the requesting player and creates the entity on that
player's surface. The server then calls `finish_prediction` with the number
of entities placed, the stop reason, and any server error; rerunning or resetting removes only entities created by this protocol.

## Reconnect

The poll loop survives Factorio restarts: on `RconError` / `OSError`,
it closes the socket, sleeps 2 s, and tries `connect()` again. So you
can leave the server running across save reloads or even full Factorio
relaunches.

## Debugging without Factorio

Spin up `nc` as a fake RCON endpoint to verify the wire format, or run
just the inference path:

```python
from denoise import load_denoiser
from server import action_to_placement, run_inference
import torch
model = load_denoiser("sft_checkpoint.pt")
world, stats = run_inference(
    model, fake_req, device=torch.device("cpu"),
    on_placement=lambda action: print(action_to_placement(action)) or True,
)
print(stats)
```
