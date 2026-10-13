# Factorion server

Loads the trained Factorion denoiser (`denoise.DenoiserUNet`) checkpoint named
by the mod's "Factorion checkpoint" setting (local, from a GitHub release, or
from a W&B run), polls a running Factorio instance over RCON for prediction
requests, denoises best-of-N factories from noise around the region's markers,
and streams the best chain's steps back via the same RCON connection, one batch
of changed entities per step.

## Run

Players install it from PyPI: `uvx factorion-server@<version>` (see the root
README). From a checkout, the easiest GUI path is the root-level launcher. It
installs dependencies and builds the Rust extension when needed,
installs/enables the mod, and waits for a hosted game:

```bash
./start-mod.sh
```

The denoising step delay is a map setting under **Settings → Mod settings →
Map**; the generation knobs (temperature, denoising and refinement steps, best
of, seed) are in the GPU server window. Each request carries their current
values. Each prediction draws a fresh seed (reported in chat) unless the seed
setting fixes one.

Manual path:

```bash
# from the repo root, so `factorion` and `factorion_rs` import cleanly
uv run python -m factorion_server \
  --rcon-host 127.0.0.1 \
  --rcon-port 27015 \
  --rcon-password factorion
```

If you go manual, start Factorio with matching flags:

```bash
factorio --rcon-bind 127.0.0.1:27015 --rcon-password factorion
```

Without `--rcon-port` / `--rcon-password` the server reads Factorio's
`config.ini` (`local-rcon-socket` / `local-rcon-password` under `[other]`),
writing them there first if absent; pass both to reach a headless server.

The checkpoint setting accepts a W&B run id of a denoiser SFT run (fetched
from a GitHub release asset `<run_id>.pt` if one exists, else from W&B),
`entity/project/id`, or a local `.pt` checkpoint saved by `sft.py` or `rl.py`.
W&B runs supply the grid size and generation schedule (`size`, `gen_steps`,
`gen_refine`); a release asset or local file uses `SftArgs`' defaults.
`--device auto` (the default) uses CUDA, then MPS, then the CPU.

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
direction to the model's pickup direction. These entities are walls the
denoiser builds around and never replaces.

After each denoising step that changed anything, the server sends:

```
/silent-command rcon.print(remote.call(
  'factorion','place_predictions','<req_id>','<placements_json>'))
```

`placements_json` is a list: first a name-less deletion over each entity the
step changed, then each new entity's Factorio prototype, center position
relative to the region, rotated footprint, direction, and optional recipe or
underground-belt type. The mod looks up the requesting player and applies them
on that player's surface, answering `ok`, `cancelled`, or an error naming the
placements Factorio refused. The server then calls `finish_prediction` with
the step count, stop reason (`converged` or `max_steps`), best-of count,
simulated items/s, seed, steps with a refusal, and any server error; rerunning
or resetting removes only entities created by this protocol.

## Reconnect

The poll loop survives Factorio restarts: on `RconError` / `OSError`,
it closes the socket, sleeps 2 s, and tries `connect()` again. So you
can leave the server running across save reloads or even full Factorio
relaunches.

## Debugging without Factorio

Spin up `nc` as a fake RCON endpoint to verify the wire format, or run
just the denoising path:

```python
from server import Generation, denoise_frames, frame_placements, load_model, request_to_obs
from factorion import render_factory
import torch
model = load_model("dr17bmik", "factorion", None, torch.device("cpu"))
world = request_to_obs(fake_req)
frames, stats = denoise_frames(model, world, Generation(), seed=1, device=torch.device("cpu"))
for frame in frames:
    print(frame_placements(world, frame))
    world = frame
print(render_factory(world), stats)
```
