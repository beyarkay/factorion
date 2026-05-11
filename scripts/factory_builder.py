"""Interactive factory builder.

Spins up a tiny local HTTP server that serves a drag-and-drop UI for
designing a factory and visualising the flow graph it produces.

Why a server (and not a single static HTML file like
``visualise_sft_data.py``): graph construction (``world2graph``) and
throughput calculation (``calc_throughput``) live in Python, so the
browser POSTs the grid to the server, which runs them and returns a
rendered graph image.

Usage:
    uv run python scripts/factory_builder.py
    # then open http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion  # noqa: E402

_, _objs = factorion.datatypes.run()
_, _fns = factorion.functions.run()

Channel = _objs["Channel"]
Direction = _objs["Direction"]
Misc = _objs["Misc"]
Footprint = _objs["Footprint"]
items = _objs["items"]
entities = _objs["entities"]
new_world = _fns["new_world"]
world2graph = _fns["world2graph"]
calc_throughput = _fns["calc_throughput"]
plot_flow_network = _fns["plot_flow_network"]
ent_str2b64img = _fns["ent_str2b64img"]


# Order shown in the palette and dropdowns.
PLACEABLE_ENTITIES = [
    "transport_belt",
    "underground_belt",
    "splitter",
    "inserter",
    "stack_inserter",
    "bulk_inserter",
    "assembling_machine_1",
]
NON_PLACEABLE_ITEMS = [
    "copper_cable",
    "copper_plate",
    "iron_plate",
    "electronic_circuit",
    "iron_gear_wheel",
]
# 10-slot hotbar mapped to keys 1..9,0 (key '0' is the last slot).
# `None` means the slot is unbound and pressing that key is a no-op.
# "empty" is the eraser.
HOTBAR = [
    "transport_belt",
    "underground_belt",
    "splitter",
    "inserter",
    "stack_inserter",
    "bulk_inserter",
    "assembling_machine_1",
    None,
    None,
    "empty",
]
# Display labels for slots whose canonical entity name differs from how
# they're shown to the user. `stack_inserter` / `bulk_inserter` are the
# Python-facing names for `Item::Source` / `Item::Sink` (see
# factorion_rs/src/types.rs); the UI says what they actually are.
DISPLAY_NAME = {
    "stack_inserter": "source",
    "bulk_inserter": "sink",
}
DIRECTIONS = ["NONE", "NORTH", "EAST", "SOUTH", "WEST"]
MISC_VALUES = ["NONE", "UNDERGROUND_DOWN", "UNDERGROUND_UP"]


@dataclass
class Args:
    port: int = 8765
    """port for the local HTTP server"""
    size: int = 8
    """default grid size"""


def build_world(grid: list[list[dict]]) -> torch.Tensor:
    """Convert a JSON grid (rows of {entity, direction, item, misc,
    footprint} dicts) into a world tensor in WHC layout."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if w != h:
        raise ValueError(f"grid must be square, got {w}x{h}")
    world = new_world(width=w, height=h)
    name_to_value = {it.name: it.value for it in items.values()}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            ent_name = cell.get("entity") or "empty"
            world[x, y, Channel.ENTITIES.value] = name_to_value.get(
                ent_name, name_to_value["empty"]
            )
            world[x, y, Channel.DIRECTION.value] = Direction[
                cell.get("direction", "NONE")
            ].value
            item_name = cell.get("item") or "empty"
            world[x, y, Channel.ITEMS.value] = name_to_value.get(
                item_name, name_to_value["empty"]
            )
            world[x, y, Channel.MISC.value] = Misc[cell.get("misc", "NONE")].value
            footprint = cell.get("footprint", "AVAILABLE")
            world[x, y, Channel.FOOTPRINT.value] = Footprint[footprint].value
    return torch.tensor(world)


def render_graph_png(grid: list[list[dict]]) -> dict:
    """Build the world, run world2graph, and return a base64 PNG plus
    text describing the nodes/edges/throughput."""
    world = build_world(grid)
    G = world2graph(world)
    if len(G.nodes) == 0:
        return {
            "png": "",
            "info": "No entities placed yet — drop something onto the grid.",
            "edges": [],
        }

    fig = plt.figure(figsize=(max(6, len(G.nodes) ** 0.5 * 3),
                              max(4, len(G.nodes) ** 0.5 * 3)))
    try:
        plot_flow_network(G)
        # plot_flow_network calls plt.show() which is a no-op under Agg;
        # the active figure is still the one we just created via
        # plot_flow_network's plt.figure(...) call. Use plt.gcf() so we
        # capture whichever figure is current.
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=110)
        png_b64 = base64.b64encode(buf.getvalue()).decode()
    finally:
        plt.close("all")

    try:
        throughput, num_unreachable = calc_throughput(G)
        tp_text = ", ".join(
            f"{k}: {v:.2f}" for k, v in throughput.items()
        ) or "(none)"
        info = f"throughput: {tp_text}  ·  unreachable nodes: {num_unreachable}"
    except Exception as e:
        info = f"throughput failed: {e}"

    # Node names contain literal '\n' (e.g. "transport_belt\n@0,0"), so use
    # repr() to make embedded newlines visible as \n in the edge list panel
    # instead of breaking the line mid-name.
    edges = [[repr(u), repr(v)] for u, v in G.edges]
    return {"png": png_b64, "info": info, "edges": edges}


# Cache palette icons so the page payload stays small per cell.
def _icon_b64(name: str) -> str:
    try:
        return ent_str2b64img(name)
    except Exception:
        return ""


PALETTE_ICONS = {n: _icon_b64(n) for n in PLACEABLE_ENTITIES + ["empty"]}
ITEM_ICONS = {n: _icon_b64(n) for n in NON_PLACEABLE_ITEMS}


def render_index(default_size: int) -> str:
    def _hotbar_slot(idx: int, name: str | None) -> str:
        key_label = str((idx + 1) % 10)
        if name is None:
            return (
                f'<div class="hb-slot empty-slot" data-slot="{idx}" '
                f'title="(unbound)"><div class="hb-key">{key_label}</div></div>'
            )
        if name == "empty":
            return (
                f'<div class="hb-slot eraser" data-slot="{idx}" '
                f'data-entity="empty" draggable="true" title="eraser">'
                f'<div class="hb-key">{key_label}</div>'
                f'<img draggable="false" src="{PALETTE_ICONS["empty"]}" alt="eraser">'
                f'<div class="hb-name">eraser</div></div>'
            )
        display = DISPLAY_NAME.get(name, name)
        return (
            f'<div class="hb-slot" data-slot="{idx}" data-entity="{name}" '
            f'draggable="true" title="{display} ({name})">'
            f'<div class="hb-key">{key_label}</div>'
            f'<img draggable="false" src="{PALETTE_ICONS[name]}" alt="{display}">'
            f'<div class="hb-name">{display}</div></div>'
        )

    hotbar_html = "".join(_hotbar_slot(i, n) for i, n in enumerate(HOTBAR))
    item_options = "".join(
        f'<option value="{n}">{n}</option>'
        for n in (["empty"] + PLACEABLE_ENTITIES + NON_PLACEABLE_ITEMS)
    )
    direction_options = "".join(
        f'<option value="{d}">{d}</option>' for d in DIRECTIONS
    )
    misc_options = "".join(
        f'<option value="{m}">{m}</option>' for m in MISC_VALUES
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Factory builder</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1em; color: #222; }}
  h1 {{ margin: 0 0 0.4em; }}
  .layout {{ display: grid; grid-template-columns: 1fr 280px; gap: 1em; }}
  .panel {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.6em; background: #fafafa; }}
  .panel h3 {{ margin: 0 0 0.4em; font-size: 0.9em; text-transform: uppercase; color: #555; }}
  .hotbar {{
    display: flex; gap: 0.3em; flex-wrap: wrap;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot {{
    position: relative; width: 64px; height: 78px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 2px;
    border: 2px solid #ddd; border-radius: 4px; background: white;
    cursor: grab; font-size: 0.7em; padding: 2px;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot img {{ width: 32px; height: 32px; pointer-events: none; }}
  .hb-slot .hb-key {{
    position: absolute; top: 1px; left: 3px;
    font-size: 0.7em; font-weight: bold; color: #888;
  }}
  .hb-slot .hb-name {{
    font-size: 0.65em; line-height: 1; text-align: center;
    word-break: break-all; pointer-events: none;
  }}
  .hb-slot.active {{ border-color: #28c850; background: #e8ffe8; }}
  .hb-slot.eraser {{ background: #fee; }}
  .hb-slot.empty-slot {{
    background: #f4f4f4; cursor: default; color: #bbb;
  }}
  .grid-wrap {{ display: flex; flex-direction: column; align-items: flex-start; gap: 0.6em; min-width: 0; }}
  .grid-graph-row {{
    display: flex; gap: 1em; align-items: flex-start;
    width: 100%; min-width: 0;
  }}
  .grid-graph-row > #grid-host {{ flex: 0 0 auto; }}
  .graph-view {{ flex: 1 1 0; min-width: 0; }}
  .graph-view h3 {{ margin: 0 0 0.4em; font-size: 0.9em; text-transform: uppercase; color: #555; }}
  .controls {{ display: flex; gap: 0.5em; flex-wrap: wrap; align-items: center; }}
  .controls input[type=number] {{ width: 4em; }}
  table.grid {{ border-collapse: collapse; }}
  table.grid td {{
    width: 56px; height: 56px; border: 1px solid #bbb; padding: 0;
    position: relative; background: #fff;
  }}
  table.grid td.selected {{ outline: 2px solid #28c850; outline-offset: -2px; }}
  table.grid td.unavailable {{
    background:
      repeating-linear-gradient(45deg,
        rgba(80,80,80,0.12), rgba(80,80,80,0.12) 2px,
        transparent 2px, transparent 8px),
      repeating-linear-gradient(-45deg,
        rgba(80,80,80,0.12), rgba(80,80,80,0.12) 2px,
        transparent 2px, transparent 8px);
  }}
  .cell-inner {{ position: relative; width: 100%; height: 100%; }}
  .cell-inner img.ent {{ position: absolute; top: 10%; left: 10%; width: 60%; height: 60%; }}
  .cell-inner img.itm {{ position: absolute; bottom: 4%; right: 4%; width: 30%; height: 30%; }}
  .cell-inner .arrow {{
    position: absolute; bottom: 0; left: 2px;
    font-size: 16px; line-height: 16px; color: #222;
  }}
  .cell-inner .misc {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-weight: bold; color: white; text-shadow: 0 0 2px black; font-size: 18px;
  }}
  .cell-inner .xy {{ position: absolute; top: 0; left: 1px; font-size: 8px; opacity: 0.5; }}
  .editor label {{ display: block; font-size: 0.8em; margin-top: 0.4em; }}
  .editor select, .editor button {{ width: 100%; padding: 0.2em; }}
  .out-img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .info {{ font-family: monospace; font-size: 0.85em; margin: 0.4em 0; }}
  .help {{ font-size: 0.85em; color: #555; }}
  pre.edges {{
    font-size: 0.75em; max-height: 240px; overflow: auto;
    background: #222; color: #cfc; padding: 0.5em; border-radius: 4px;
  }}
</style></head><body>

<h1>Factory builder</h1>
<p class="help">
  Press <b>1</b>–<b>9</b> or <b>0</b> to pick a hotbar slot, then click a
  tile to place it. Drag a slot onto a tile to do the same. Click a tile
  to select/edit it (ENTITY / DIRECTION / ITEM / MISC / FOOTPRINT in the
  right panel). With a tile selected: <b>r</b> rotates clockwise,
  <b>R</b> counter-clockwise; <b>Delete</b> / <b>Backspace</b> clears it;
  right-click also clears. The graph recomputes automatically on every
  change. <b>Esc</b> deselects the active hotbar slot.
</p>

<div class="layout">

  <div class="grid-wrap">
    <div class="hotbar" id="hotbar">{hotbar_html}</div>
    <div class="controls">
      <label>size <input id="size" type="number" min="2" max="20" value="{default_size}"></label>
      <button id="resize">resize / clear</button>
      <button id="compute">Compute graph</button>
      <button id="export">copy state JSON</button>
    </div>
    <div class="grid-graph-row">
      <div id="grid-host"></div>
      <div class="graph-view">
        <h3>Graph</h3>
        <div class="info" id="info">(graph will appear after Compute graph)</div>
        <img id="out-img" class="out-img" alt="" style="display:none">
        <pre class="edges" id="edges" style="display:none"></pre>
      </div>
    </div>
  </div>

  <div class="panel editor" id="editor">
    <h3>Selected cell</h3>
    <div id="sel-info" class="help">(click a cell to edit)</div>
    <label>entity
      <select id="ed-entity">{item_options}</select>
    </label>
    <label>direction
      <select id="ed-direction">{direction_options}</select>
    </label>
    <label>item (recipe / filter)
      <select id="ed-item">{item_options}</select>
    </label>
    <label>misc
      <select id="ed-misc">{misc_options}</select>
    </label>
    <label>footprint
      <select id="ed-footprint">
        <option value="AVAILABLE">AVAILABLE</option>
        <option value="UNAVAILABLE">UNAVAILABLE</option>
      </select>
    </label>
    <button id="clear-cell" style="margin-top:0.6em;">clear cell</button>
  </div>

</div>

<script>
const PALETTE_ICONS = {json.dumps(PALETTE_ICONS)};
const ITEM_ICONS = {json.dumps(ITEM_ICONS)};
const HOTBAR = {json.dumps(HOTBAR)};
const DIR_ARROW = {{ NONE: '', NORTH: '↑', EAST: '→', SOUTH: '↓', WEST: '←' }};
const MISC_GLYPH = {{ NONE: '', UNDERGROUND_DOWN: '⭳', UNDERGROUND_UP: '⭱' }};
const DIR_CYCLE = ['NORTH', 'EAST', 'SOUTH', 'WEST'];

let SIZE = {default_size};
let grid = [];           // grid[y][x] = cell dict
let selected = null;     // {{x, y}} or null
let activeHotbar = null; // 0..9 or null

function emptyCell() {{
  return {{
    entity: 'empty', direction: 'NONE', item: 'empty',
    misc: 'NONE', footprint: 'AVAILABLE',
  }};
}}

function newGrid(n) {{
  const g = [];
  for (let y = 0; y < n; y++) {{
    const row = [];
    for (let x = 0; x < n; x++) row.push(emptyCell());
    g.push(row);
  }}
  return g;
}}

function iconFor(name) {{
  return PALETTE_ICONS[name] || ITEM_ICONS[name] || '';
}}

function renderGrid() {{
  const host = document.getElementById('grid-host');
  const tbl = document.createElement('table');
  tbl.className = 'grid';
  for (let y = 0; y < SIZE; y++) {{
    const tr = document.createElement('tr');
    for (let x = 0; x < SIZE; x++) {{
      const td = document.createElement('td');
      td.dataset.x = x; td.dataset.y = y;
      const c = grid[y][x];
      if (c.footprint === 'UNAVAILABLE') td.classList.add('unavailable');
      if (selected && selected.x === x && selected.y === y) td.classList.add('selected');

      const inner = document.createElement('div');
      inner.className = 'cell-inner';
      let html = `<div class="xy">${{x}},${{y}}</div>`;
      if (c.entity && c.entity !== 'empty')
        html += `<img class="ent" src="${{iconFor(c.entity)}}">`;
      if (c.item && c.item !== 'empty')
        html += `<img class="itm" src="${{iconFor(c.item)}}">`;
      const arrow = DIR_ARROW[c.direction] || '';
      if (arrow) html += `<div class="arrow">${{arrow}}</div>`;
      const m = MISC_GLYPH[c.misc] || '';
      if (m) html += `<div class="misc">${{m}}</div>`;
      inner.innerHTML = html;
      td.appendChild(inner);

      td.addEventListener('click', () => {{
        selected = {{x, y}};
        if (activeHotbar !== null) {{
          const ent = HOTBAR[activeHotbar];
          if (ent !== null) {{
            placeEntity(x, y, ent);
            return;
          }}
        }}
        renderGrid(); syncEditor();
      }});
      td.addEventListener('contextmenu', (ev) => {{
        ev.preventDefault();
        grid[y][x] = emptyCell();
        renderGrid();
        if (selected && selected.x === x && selected.y === y) syncEditor();
        scheduleCompute();
      }});
      td.addEventListener('dragover', (ev) => ev.preventDefault());
      td.addEventListener('drop', (ev) => {{
        ev.preventDefault();
        const ent = ev.dataTransfer.getData('text/plain');
        if (!ent) return;
        placeEntity(x, y, ent);
      }});
      tr.appendChild(td);
    }}
    tbl.appendChild(tr);
  }}
  host.replaceChildren(tbl);
}}

function syncEditor() {{
  const info = document.getElementById('sel-info');
  if (!selected) {{ info.textContent = '(click a cell to edit)'; return; }}
  const c = grid[selected.y][selected.x];
  info.textContent = `(${{selected.x}}, ${{selected.y}})`;
  document.getElementById('ed-entity').value = c.entity;
  document.getElementById('ed-direction').value = c.direction;
  document.getElementById('ed-item').value = c.item;
  document.getElementById('ed-misc').value = c.misc;
  document.getElementById('ed-footprint').value = c.footprint;
}}

function bindEditor() {{
  const map = {{
    'ed-entity': 'entity', 'ed-direction': 'direction',
    'ed-item': 'item', 'ed-misc': 'misc', 'ed-footprint': 'footprint',
  }};
  for (const [id, field] of Object.entries(map)) {{
    document.getElementById(id).addEventListener('change', (ev) => {{
      if (!selected) return;
      grid[selected.y][selected.x][field] = ev.target.value;
      renderGrid();
      scheduleCompute();
    }});
  }}
  document.getElementById('clear-cell').addEventListener('click', () => {{
    if (!selected) return;
    grid[selected.y][selected.x] = emptyCell();
    renderGrid(); syncEditor();
    scheduleCompute();
  }});
}}

function renderHotbar() {{
  document.querySelectorAll('.hb-slot').forEach(el => {{
    const idx = parseInt(el.dataset.slot, 10);
    el.classList.toggle('active', idx === activeHotbar);
  }});
}}

function setActiveHotbar(idx) {{
  if (idx !== null && HOTBAR[idx] === null) return;
  activeHotbar = (activeHotbar === idx) ? null : idx;
  renderHotbar();
}}

function bindHotbar() {{
  document.querySelectorAll('.hb-slot').forEach(el => {{
    const idx = parseInt(el.dataset.slot, 10);
    if (HOTBAR[idx] === null) return;
    el.addEventListener('dragstart', (ev) => {{
      ev.dataTransfer.setData('text/plain', el.dataset.entity);
    }});
    el.addEventListener('click', () => setActiveHotbar(idx));
  }});
}}

function placeEntity(x, y, ent) {{
  if (ent === 'empty') {{
    grid[y][x] = emptyCell();
  }} else {{
    grid[y][x].entity = ent;
    if (grid[y][x].direction === 'NONE') grid[y][x].direction = 'EAST';
  }}
  selected = {{x, y}};
  renderGrid(); syncEditor();
  scheduleCompute();
}}

function rotateSelected(cw) {{
  if (!selected) return;
  const c = grid[selected.y][selected.x];
  let i = DIR_CYCLE.indexOf(c.direction);
  if (i < 0) {{
    c.direction = cw ? 'NORTH' : 'WEST';
  }} else {{
    c.direction = DIR_CYCLE[(i + (cw ? 1 : -1) + 4) % 4];
  }}
  renderGrid(); syncEditor();
  scheduleCompute();
}}

function clearSelected() {{
  if (!selected) return;
  grid[selected.y][selected.x] = emptyCell();
  renderGrid(); syncEditor();
  scheduleCompute();
}}

let _computeTimer = null;
function scheduleCompute() {{
  clearTimeout(_computeTimer);
  _computeTimer = setTimeout(computeGraph, 200);
}}

async function computeGraph() {{
  document.getElementById('info').textContent = 'computing…';
  const resp = await fetch('/graph', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ grid }}),
  }});
  const data = await resp.json();
  if (data.error) {{
    document.getElementById('info').textContent = 'error: ' + data.error;
    document.getElementById('out-img').style.display = 'none';
    document.getElementById('edges').style.display = 'none';
    return;
  }}
  document.getElementById('info').textContent = data.info || '';
  const img = document.getElementById('out-img');
  if (data.png) {{
    img.src = 'data:image/png;base64,' + data.png;
    img.style.display = 'block';
  }} else {{
    img.style.display = 'none';
  }}
  const edges = document.getElementById('edges');
  if (data.edges && data.edges.length) {{
    edges.textContent = data.edges.map(e => e[0] + '  →  ' + e[1]).join('\\n');
    edges.style.display = 'block';
  }} else {{
    edges.style.display = 'none';
  }}
}}

document.getElementById('compute').addEventListener('click', computeGraph);
document.getElementById('resize').addEventListener('click', () => {{
  const n = parseInt(document.getElementById('size').value, 10);
  if (!Number.isFinite(n) || n < 2 || n > 20) return;
  SIZE = n;
  grid = newGrid(SIZE);
  selected = null;
  renderGrid(); syncEditor();
  scheduleCompute();
}});
document.getElementById('export').addEventListener('click', async () => {{
  const text = JSON.stringify({{ size: SIZE, grid }}, null, 2);
  try {{ await navigator.clipboard.writeText(text); }} catch (_) {{}}
  console.log(text);
  alert('state copied to clipboard (also logged to console)');
}});

document.addEventListener('keydown', (ev) => {{
  const t = ev.target;
  const tag = t && t.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (/^[0-9]$/.test(ev.key)) {{
    const n = parseInt(ev.key, 10);
    const idx = (n === 0) ? 9 : n - 1;
    setActiveHotbar(idx);
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'r') {{ rotateSelected(true); ev.preventDefault(); return; }}
  if (ev.key === 'R') {{ rotateSelected(false); ev.preventDefault(); return; }}
  if (ev.key === 'Delete' || ev.key === 'Backspace') {{
    clearSelected(); ev.preventDefault(); return;
  }}
  if (ev.key === 'Escape') {{
    if (activeHotbar !== null) setActiveHotbar(activeHotbar);
    return;
  }}
}});

grid = newGrid(SIZE);
renderGrid();
bindHotbar();
bindEditor();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "FactoryBuilder/0.1"

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[%s] %s\n" % (self.address_string(), format % args))

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            body = render_index(self.server.default_size).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path != "/graph":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
            result = render_graph_png(payload["grid"])
        except Exception as e:
            traceback.print_exc()
            result = {"error": f"{type(e).__name__}: {e}"}
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(args: Args) -> None:
    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
    httpd.default_size = args.size
    print(f"Serving factory builder on http://127.0.0.1:{args.port}")
    print("Press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main(tyro.cli(Args))
