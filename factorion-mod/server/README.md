# Factorion server

Loads a trained Factorion `AgentCNN` checkpoint, polls a running
Factorio instance over RCON for prediction requests, runs iterative
greedy inference, and pushes the resulting blueprint back via the same
RCON connection.

## Run

Easiest path: use the launcher in `scripts/`, which spawns Factorio +
server together with auto-generated RCON port/password:

```bash
bash factorion-mod/scripts/launch.sh path/to/agent.pt
```

Manual path:

```bash
# from the repo root, so `factorion` and `factorion_rs` import cleanly
uv run python factorion-mod/server/server.py \
  --checkpoint path/to/agent.pt \
  --rcon-host 127.0.0.1 \
  --rcon-port 27015 \
  --rcon-password factorion
```

If you go manual, start Factorio with matching flags:

```bash
factorio --rcon-bind 127.0.0.1:27015 --rcon-password factorion
```

Required flags for the server:

- `--checkpoint` — the `torch.save(agent.state_dict(), ...)` file from
  PPO (`runs/<run>/agent.pt`) or SFT (`sft_runs/<run>/agent.pt`).
- `--rcon-port` / `--rcon-password` — must match what Factorio was
  launched with.

Architecture knobs (must match how the checkpoint was trained):

- `--grid-size 8` (default 8)
- `--chan1 32 --chan2 64 --chan3 64`

If you put a sidecar JSON next to the checkpoint named `agent.hp.json`
with `{"grid_size": 8, "chan1": 32, ...}`, the server reads it
automatically (CLI flags override only when explicitly non-default).

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
  "sources": [{"x": 0, "y": 3, "direction": 2, "item": "iron-plate"}],
  "sinks":   [{"x": 7, "y": 3, "direction": 2, "item": "iron-plate"}],
  "default_item": "iron-plate"
}
```

`direction` uses Factorion's enum (1=N, 2=E, 3=S, 4=W), *not* Factorio's
16-step blueprint convention; the server converts before emitting.

The server's response leg goes back over the same RCON connection:

```
/silent-command remote.call('factorion','deliver_blueprint','<req_id>','<bp_b64>')
```

The mod's `deliver_blueprint` handler looks up which player asked, sets
their cursor stack to a blueprint, and `import_stack`s the b64 string.

## Reconnect

The poll loop survives Factorio restarts: on `RconError` / `OSError`,
it closes the socket, sleeps 2 s, and tries `connect()` again. So you
can leave the server running across save reloads or even full Factorio
relaunches.

## Debugging without Factorio

Spin up `nc` as a fake RCON endpoint to verify the wire format, or run
just the inference path:

```python
from server import load_agent, request_to_obs, run_inference
from blueprint import world_tensor_to_blueprint_string
import torch
agent = load_agent(...)
obs = run_inference(agent, fake_req, max_steps=64, device=torch.device("cpu"))
print(world_tensor_to_blueprint_string(obs))
```
