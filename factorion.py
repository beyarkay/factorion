import marimo

__generated_with = "0.8.22"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import traceback
    from dataclasses import dataclass
    from enum import Enum
    from torch.distributions import Categorical
    from tqdm import trange
    from tqdm.notebook import tqdm
    import base64
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import random
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.init as init
    import torch.optim as optim
    import wandb
    import zlib
    import math

    wandb.login()
    mo.md("Imports")
    return (
        Categorical,
        Enum,
        F,
        base64,
        dataclass,
        go,
        init,
        json,
        math,
        mo,
        nn,
        np,
        nx,
        optim,
        pd,
        plt,
        random,
        sys,
        torch,
        tqdm,
        traceback,
        trange,
        wandb,
        zlib,
    )


@app.cell(hide_code=True)
def datatypes(Enum, dataclass, mo):
    class Channel(Enum):
        # What entity occupies this tile?
        ENTITIES = 0
        # What recipe OR filter is set?
        # RECIPES = 1
        # what direction is the entity facing?
        DIRECTION = 1
        # Undergrounds and splitter mechanics, see class Misc(Enum)
    #     MISC = 3
        # 1 if you can build there, 0 if you can't
    #     FOOTPRINT = 4

    class Footprint(Enum):
        UNAVAILABLE = 0
        AVAILABLE = 1

    class Misc(Enum):
        NONE = 0
        UNDERGROUND_DOWN = 1
        UNDERGROUND_UP = 2
    #     SPLITTER_INPUT_LEFT_OUTPUT_LEFT = 3
    #     SPLITTER_INPUT_LEFT_OUTPUT_NONE = 4
    #     SPLITTER_INPUT_LEFT_OUTPUT_RIGHT = 5
    #     SPLITTER_INPUT_NONE_OUTPUT_LEFT = 6
    #     SPLITTER_INPUT_NONE_OUTPUT_NONE = 7
    #     SPLITTER_INPUT_NONE_OUTPUT_RIGHT = 8
    #     SPLITTER_INPUT_RIGHT_OUTPUT_LEFT = 9
    #     SPLITTER_INPUT_RIGHT_OUTPUT_NONE = 10
    #     SPLITTER_INPUT_RIGHT_OUTPUT_RIGHT = 11

    class Dim(Enum):
        X = 0
        Y = 1

    class Direction(Enum):
        """Directions the entity can be facing"""
        NONE = 0
        NORTH = 1
        EAST = 2
        SOUTH = 3
        WEST = 4
    #     NONE = -1
    #     NORTH = 0
    # #     NORTH_EAST = 2
    #     EAST = 4
    # #     SOUTH_EAST = 6
    #     SOUTH = 8
    # #     SOUTH_WEST = 10
    #     WEST = 12
    # #     NORTH_WEST = 14

    @dataclass
    class Prototype:
        name: str
        value: int
        flow: float
        width: int
        height: int

    prototypes = {
        0:  Prototype(name='empty',                 value=0,  width=1, height=1, flow=0.0),
        1: Prototype(name='transport_belt',        value=1, width=1, height=1, flow=15.0),
        # sink
        2: Prototype(name='bulk_inserter',         value=2, width=1, height=1, flow=float('inf')),
        # source
        3: Prototype(name='stack_inserter',        value=3, width=1, height=1, flow=float('inf')),
    #     1:  Prototype(name='sink',                  value=1,  width=1, height=1, flow=float('inf')),
    #     2:  Prototype(name='source',                value=2,  width=1, height=1, flow=float('inf')),
    #     3:  Prototype(name='assembling_machine_1',  value=3,  width=3, height=3, flow=0.5),
    #     4:  Prototype(name='copper_cable',          value=4,  width=1, height=1, flow=0.0),
    #     5:  Prototype(name='copper_ore',            value=5,  width=1, height=1, flow=0.0),
    #     6:  Prototype(name='copper_plate',          value=6,  width=1, height=1, flow=0.0),
    #     7:  Prototype(name='electric_mining_drill', value=7,  width=3, height=3, flow=0.5),
        # 1:  Prototype(name='electronic_circuit',    value=1,  width=1, height=1, flow=0.0),
    #     9:  Prototype(name='hazard_concrete',       value=9,  width=1, height=1, flow=0.0),
    #     10: Prototype(name='inserter',              value=10, width=1, height=1, flow=0.86),
    #     11: Prototype(name='iron_ore',              value=11, width=1, height=1, flow=0.0),
    #     12: Prototype(name='iron_plate',            value=12, width=1, height=1, flow=0.0),
    #     13: Prototype(name='splitter',              value=13, width=2, height=1, flow=15.0),
    #     14: Prototype(name='steel_chest',           value=14, width=1, height=1, flow=0.0),
        # TODO: this doesn't account for the speed of items
        # underground (which is identical to a transport belt)
    #     16: Prototype(name='underground_belt',      value=16, width=1, height=1, flow=15.0),
    }

    @dataclass
    class Recipe:
        consumes: dict[str, float]
        produces: dict[str, float]

    recipes = {
        'electronic_circuit': Recipe(
            consumes={ 'copper_cable': 6.0, 'iron_plate': 2.0 },
            produces={'electronic_circuit': 2.0},
        ),
    #     'copper_cable': Recipe(
    #         consumes={ 'copper_plate': 2.0 },
    #         produces={'copper_cable': 4.0},
    #     ),
    }

    mo.md("Datatypes")
    return (
        Channel,
        Dim,
        Direction,
        Footprint,
        Misc,
        Prototype,
        Recipe,
        prototypes,
        recipes,
    )


@app.cell(hide_code=True)
def functions(
    Categorical,
    Channel,
    Direction,
    Footprint,
    Misc,
    Prototype,
    base64,
    go,
    json,
    mo,
    np,
    nx,
    plt,
    prototypes,
    recipes,
    torch,
    traceback,
    zlib,
):
    def b64_to_dict(blueprint_string):
        decoded = base64.b64decode(blueprint_string.strip()[1:])  # Skip the version byte
        json_data = zlib.decompress(decoded).decode('utf-8')
        return json.loads(json_data)

    def dict_to_b64(dictionary):
        compressed = zlib.compress(json.dumps(dictionary).encode('utf-8'))
        b64_encoded = base64.b64encode(compressed).decode('utf-8')
        blueprint_string = '0' + b64_encoded  # Add version byte
        return blueprint_string

    def prototype_from_str(s):
        if s is None:
            print(f"WARN: entity is None")
            return None
        for v in prototypes.values():
            if v.name == s.replace('-', '_'):
                return v
        # TODO I'm almost certianly going to regret hardcoding this
        if s == 'electronic_circuit':
            return Prototype(name='electronic_circuit', value=len(prototypes), width=1, height=1, flow=0.0)
        print(f"WARN: unknown entity {s}")
        return None

    def b64img_from_str(s, base_path='factorio-icons'):
        proto = prototype_from_str(s)
        path = proto.name
        try:
            with open(f"{base_path}/{path}.png", "rb") as image_file:
                return 'data:image/png;base64,' + base64.b64encode(image_file.read()).decode("utf-8")
        except:
            return ''

    def new_world(width=8, height=8):
        channels = len(Channel)
    #     print(f"Making world w={width}, h={height}, c={channels}")
        world = np.zeros((width, height, channels), dtype=int)
        world[:, :, Channel.ENTITIES.value] = prototype_from_str('empty').value
        world[:, :, Channel.DIRECTION.value] = Direction.NONE.value
        if any(['Channel.FOOTPRINT' in i for i in map(str, list(Channel))]):
            world[:, :, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        return world

    def add_entity(world, proto_str, x, y, direction=Direction.NONE, recipe='empty', misc=Misc.NONE):
        proto = prototype_from_str(proto_str)
        EMPTY = prototype_from_str('empty')
        if proto is None:
            proto = EMPTY
        recipe_proto = prototype_from_str(recipe)
        if recipe_proto is None:
            recipe_proto = EMPTY
        assert (world[x, y, Channel.ENTITIES.value] == EMPTY.value), f"Can't place {proto_str} at {x},{y} because {prototypes[world[x, y, Channel.ENTITIES.value]]} is there"
        assert 0 <= x < len(world), f'{x=} is not in [0, {len(world)})'
        assert 0 <= y < len(world[0]), f'{y=} is not in [0, {len(world[0])})'

        world[x, y, Channel.ENTITIES.value] = proto.value
        world[x, y, Channel.DIRECTION.value] = direction.value
        # world[x, y, Channel.RECIPES.value] = recipe_proto.value
        # world[x, y, Channel.MISC.value] = misc.value

    def world_to_html(world):
        assert len(world.shape) == 3, f"Expected 3 dimensions got {world.shape}"
        assert world.shape[0] == world.shape[1], f"Expected square got {world.shape}"
        if type(world) is not np.ndarray:
            world = np.array(world)
        DIRECTION_ARROWS = {
            -1: "",
            0: "↑",
            2: "↗",
            4: "→",
            6: "↘",
            8: "↓",
            10: "↙",
            12: "←",
            14: "↖",
        }
        html = ["<table style='border-collapse: collapse;'>"]
        for y in range(len(world[0])):
            html.append("<tr>")
            for x in range(len(world)):
                proto = prototypes[world[x, y, Channel.ENTITIES.value]]
                # recipe = prototypes[world[x, y, Channel.RECIPES.value]]
                direction = world[x, y, Channel.DIRECTION.value]
    #             entity, direction, recipe = get_entity_info(world, x, y)
                entity_icon = b64img_from_str(proto.name)
                # recipe_icon = b64img_from_str(recipe.name)
                direction_arrow = DIRECTION_ARROWS.get(direction, "")
                if any(['Channel.MISC' in i for i in map(str, list(Channel))]):
                    misc = Misc(world[x, y, Channel.MISC.value])
                else:
                    misc = Misc.NONE
                underground_symbol = "⭳" if misc == Misc.UNDERGROUND_DOWN else "⭱" if misc == Misc.UNDERGROUND_UP else ""

    #             print(direction_arrow, direction)
                if any(['Channel.FOOTPRINT' in i for i in map(str, list(Channel))]):
                    available = world[x, y, Channel.FOOTPRINT.value] == Footprint.AVAILABLE.value
                else:
                    available = True
                bg_style = "background: rgba(255, 0, 0, 0.3);" if not available else ""
    #             tint_style = "filter: brightness(1.5) sepia(1) hue-rotate(30deg);" if available else ""
                cell_content = f"""
                    <div style='position: relative; width: 50px; height: 50px; {bg_style}'>
                        <img src='{entity_icon}' style='width: 60%; height: 60%;'>
                        <div style='position: absolute; bottom: 0; left: 0; font-size: 20px;'>{direction_arrow}</div>
                        <div style='position: absolute; top: 50%; left: 50%;
                        font-size: 20px; font-weight: bold; color: white;'>{underground_symbol}</div>
                    </div>
                """
                html.append(f"<td style='border: 1px solid black; padding: 0;'>{cell_content}</td>")
            html.append("</tr>")
        html.append("</table>")
        return mo.Html("".join(html))

    def world_from_blueprint(bp):
        obj = b64_to_dict(bp)

        min_x = float('inf')
        min_y = float('inf')
        max_y = -float('inf')
        max_x = -float('inf')

        for e in obj['blueprint']['entities']:
            # NOTE: might be OBOEs here
            e['position']['x'] = int(e['position']['x'] - 0.5)
            e['position']['y'] = int(e['position']['y'] - 0.5)

        for e in obj['blueprint'].get('entities', []) + obj['blueprint'].get('tiles', []):
            min_x = min(min_x, e['position']['x'])
            min_y = min(min_y, e['position']['y'])
            max_y = max(max_y, e['position']['y'])
            max_x = max(max_x, e['position']['x'])

        for e in obj['blueprint'].get('entities', []):
            e['position']['x'] -= min_x
            e['position']['y'] -= min_y
            if 'direction' not in e and e['name'] == 'transport-belt':
                # transport belts have an implicit direction
                e['direction'] = 0
            # inserter's direction is towards their source, which we don't want, so flip them around
            if 'inserter' in e['name']:
    #             print(e)
                e['direction'] = (e.get('direction', 0) + 8) % 16

        # Add one, because of the 0.5 alignment of entities vs tiles
        world = new_world(width=max_x - min_x + 1, height=max_y - min_y + 1)

        # Use Hazard concrete to indicate the footprint
        world[:, :, Channel.FOOTPRINT.value] = Footprint.UNAVAILABLE.value
        for t in obj['blueprint'].get('tiles', []):
            if t['name'] == 'refined-hazard-concrete-left':
                x = t['position']['x'] - min_x
                y = t['position']['y'] - min_y
                world[x, y, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
            else:
                print(f"Ignoring tile {t}")

        for e in obj['blueprint'].get('entities', []):
            entity = prototype_from_str(e['name'])
            if entity is None:
                entity = prototype_from_str('empty')

            if 'recipe' in e:
                recipe = prototype_from_str(e['recipe'])
            else:
                if len(e.get('filters', [])) == 1:
                    recipe = prototype_from_str(e['filters'][0]['name'])
                else:
                    recipe = prototype_from_str('empty')

            # underground belts. Output = emerging, input = descending
            if e.get('type', None) == 'output':
                misc = Misc.UNDERGROUND_UP
            elif e.get('type', None) == 'input':
                misc = Misc.UNDERGROUND_DOWN
            else:
                misc = Misc.NONE

            direction = Direction(e.get('direction', -1))

            add_entity(
                world,
                e['name'],
                e['position']['x'],
                e['position']['y'],
                recipe=recipe.name,
                direction=direction,
                misc=misc,
            )
        return world

    def graph_from_world(world, debug=False):
        assert torch.is_tensor(world), f"world is {type(world)}, not a tensor"
        assert len(world.shape) == 3, f"Expected world to have 3 dimensions, but is of shape {world.shape}"
        assert world.shape[0] == world.shape[1], f"Expected world to be square, but is of shape {world.shape}"
        world = world.numpy()
        G = nx.DiGraph()
        def dbg(s):
            if debug: print(s)
        for x in range(len(world)):
            for y in range(len(world[0])):
                e = prototypes[world[x, y, Channel.ENTITIES.value]]
                if e.name == 'empty':
                    continue

                # while we're mocking the recipe, just hardcode electronic_circuit
                r = prototype_from_str('electronic_circuit')
                d = Direction(world[x, y, Channel.DIRECTION.value])

                input_ = {}
                output = {}
                if e.name == 'stack_inserter':
                    output = {r.name: float('inf')}

                self_name = f"{e.name}\n@{x},{y}"
                G.add_node(
                    self_name,
                    input_=input_,
                    output=output,
                    recipe=r.name if 'assembling_machine' in e.name else None
                )
                dbg(f"Created node {repr(self_name)}: {G.nodes[self_name]}, direction is {d}, recipe is {r.name}")

                # Figure out coords for source and destination
                if d == Direction.EAST:
                    src = [x - 1, y]
                    dst = [x + 1, y]
                elif d == Direction.WEST:
                    src = [x + 1, y]
                    dst = [x - 1, y]
                elif d == Direction.NORTH:
                    src = [x, y + 1]
                    dst = [x, y - 1]
                elif d == Direction.SOUTH:
                    src = [x, y - 1]
                    dst = [x, y + 1]
                else:
                    assert False, f"Can't handle direction {d} for entity {e}"
                # Connect the inserters' & belts' nodes
                # TODO here we connect nodes twice, once on source->me and again on me->destination.
                x_src_valid = 0 <= src[0] < len(world)
                y_src_valid = 0 <= src[1] < len(world[0])
                x_dst_valid = 0 <= dst[0] < len(world)
                y_dst_valid = 0 <= dst[1] < len(world[0])
                if 'inserter' in e.name:
                    if x_src_valid and y_src_valid:
                        src_entity = prototypes[world[src[0], src[1], Channel.ENTITIES.value]]
                        src_direction = Direction(world[src[0], src[1], Channel.DIRECTION.value])
                        src_not_empty = src_entity.name != 'empty'
                        if src_not_empty:
                            G.add_edge(
                                f"{src_entity.name}\n@{src[0]},{src[1]}",
                                f"{e.name}\n@{x},{y}",
                            )
                            dbg(f"{src_entity.name}@{src[0]},{src[1]} -> {e.name}@{x},{y}")
                    if x_dst_valid and y_dst_valid:
                        dst_entity = prototypes[world[dst[0], dst[1], Channel.ENTITIES.value]]
                        # TODO: This doesn't allow for the case where
                        # an inserter can put things on the ground to
                        # be picked up by another inserter
                        dst_is_insertable = (
                            'belt' in dst_entity.name
                            or 'assembling_machine' in dst_entity.name
                        )
                        if dst_is_insertable:
                            G.add_edge(
                                f"{e.name}\n@{x},{y}",
                                f"{dst_entity.name}\n@{dst[0]},{dst[1]}",
                            )
                            dbg(f"{e.name}@{x},{y} -> {dst_entity.name}@{dst[0]},{dst[1]}")

                elif 'transport_belt' in e.name:
                    if x_src_valid and y_src_valid:
                        src_entity = prototypes[world[src[0], src[1], Channel.ENTITIES.value]]
                        src_direction = Direction(world[src[0], src[1], Channel.DIRECTION.value])
                        src_is_beltish = (
                            'belt' in src_entity.name
                            # Check the other belt is directly behind me and pointing the same direction
                            and src_direction == d
                        )
                        if src_is_beltish:
                            G.add_edge(
                                f"{src_entity.name}\n@{src[0]},{src[1]}",
                                f"{e.name}\n@{x},{y}",
                            )
                            dbg(f"{src_entity.name}@{src[0]},{src[1]} -> {e.name}@{x},{y}",)

                    if x_dst_valid and y_dst_valid:
                        dst_entity = prototypes[world[dst[0], dst[1], Channel.ENTITIES.value]]
                        dst_direction = Direction(world[dst[0], dst[1], Channel.DIRECTION.value])
                        dst_not_empty = dst_entity.name != 'empty'
                        dst_is_belt = 'belt' in dst_entity.name
                        dst_opposing_belt = (
                            dst_is_belt
                            and abs(dst_direction.value - d.value) == 8
                        )
                        if dst_is_belt and not dst_opposing_belt:
                            G.add_edge(
                                f"{e.name}\n@{x},{y}",
                                f"{dst_entity.name}\n@{dst[0]},{dst[1]}",
                            )
                            dbg(f"{e.name}@{x},{y} -> {dst_entity.name}@{dst[0]},{dst[1]}",)

                # connect up the assembling machines
                elif 'assembling_machine' in e.name:
                    # search the blocks around the 3x3 assembling machine for inputs
                    for dx in range(-2, 3):
                        if not (0 <= x + dx < len(world)):
                            continue
                        for dy in range(-2, 3):
                            if not (0 <= y + dy < len(world[0])):
                                continue
                            if abs(dx) == abs(dy):
                                continue
                            if abs(dx) != 2 and abs(dy) != 2:
                                continue
                            other_e = prototypes[world[x + dx, y + dy, Channel.ENTITIES.value]]
                            other_d = Direction(world[x + dx, y + dy, Channel.DIRECTION.value])
                            # Only inserters can insert into an assembling machine
                            if 'inserter' not in other_e.name:
                                continue
    #                         if f"{other_e.name}\n@{x + dx},{y + dy}" == 'inserter\n@2,0':
    #                             print(other_e, e, other_d, dy, dx)

                            other_str = f"{other_e.name}\n@{x + dx},{y + dy}"
                            self_str = f"{e.name}\n@{x},{y}"

                            # Direction is self -> other
                            if (
                                   (other_d == Direction.NORTH and dy < 0)
                                or (other_d == Direction.SOUTH and dy > 0)
                                or (other_d == Direction.WEST  and dx < 0)
                                or (other_d == Direction.EAST  and dx > 0)
                            ):
    #                             print(f'self -> other')
                                src = self_str
                                dst = other_str
                            else:
                            # Direction is other -> self
    #                             print(f'other -> self')
                                src = other_str
                                dst = self_str

                            G.add_edge(src, dst)
                            dbg(f'{repr(src)} -> {repr(dst)}')

                elif 'underground_belt' in e.name:
                    m = Misc(world[x, y, Channel.MISC.value])
                    # Only down-undergrounds look for their upgoing counterparts,
                    # not the other way aroud
                    assert e.name == 'underground_belt', "don't know how to handle other undergrounds yet"
                    if m == Misc.UNDERGROUND_DOWN:
                        max_delta = 6
                    elif m == Misc.UNDERGROUND_UP:
                        max_delta = 1
                    else:
                        assert False, f"Dont understand {m}"
                    for delta in range(1, max_delta):
                        if d == Direction.EAST:
                            src = [x - 1, y]
                            dst = [x + delta, y]
                        elif d == Direction.WEST:
                            src = [x + 1, y]
                            dst = [x - delta, y]
                        elif d == Direction.NORTH:
                            src = [x, y + 1]
                            dst = [x, y - delta]
                        elif d == Direction.SOUTH:
                            src = [x, y - 1]
                            dst = [x, y + delta]
                        x_valid = 0 <= dst[0] < len(world)
                        y_valid = 0 <= dst[1] < len(world[0])
                        if x_valid and y_valid:
                            dst_entity = prototypes[world[dst[0], dst[1], Channel.ENTITIES.value]]
                            going_underground = (
                                dst_entity.name == 'underground_belt'
                                and m == Misc.UNDERGROUND_DOWN
                            )
                            cxn_to_belt = (
                                'transport_belt' in dst_entity.name
                                and m == Misc.UNDERGROUND_UP
                            )
                            if going_underground or cxn_to_belt:
                                G.add_edge(
                                    f"{e.name}\n@{x},{y}",
                                    f"{dst_entity.name}\n@{dst[0]},{dst[1]}",
                                )
                else:
                    assert False, f"Don't know how to handle {e.name} at {x} {y}"

        return G

    def plot_flow_network(G):
        # Extract x, y coordinates from node names
        pos = {
            node: (int(x), -int(y))
            for node, (x, y)
            in ((n, n.split("@")[1].split(",")) for n in G.nodes)
        }
        plt.figure(figsize=(
            (len(G.nodes) ** 0.5) * 3,
            (len(G.nodes) ** 0.5) * 3,
        ))

        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12, font_weight='bold')

        # Add throughput labels
    #     labels = {node: G.nodes[node].get("throughput", {}) for node in G.nodes}
    #     nx.draw_networkx_labels(G, pos, labels=labels, font_color="red")

        plt.show()

    def calc_throughput(G, debug=False):
        foobar = 1
        def dbg(s):
            if debug: print(s)
        if len(list(nx.simple_cycles(G))) > 0:
            dbg(f'Returning 0 reward due to cycles: {list(nx.simple_cycles(G))}')
            return {'foobar': 0.0}, 0
        # Now go through the graph and propogate the ingredients from producers to consumers.
        # the flow rate should depend on the intermediate rates.
        stack_inserters = [node for node, data in G.nodes(data=True) if 'stack_inserter' in node]
        nodes = stack_inserters[:]
        reachable_from_stack_inserters = []
        for s in stack_inserters:
            reachable_from_stack_inserters.extend(list(nx.descendants(G, s)))
        reachable_from_stack_inserters = list(set(reachable_from_stack_inserters))
        dbg(f'{reachable_from_stack_inserters=}')
        already_processed = []
        count = len(G.nodes) * len(G.nodes) # a reasonable upper bound

        dbg(f"Pre-calcs:")
        for n in G.nodes:
            dbg(f'- {repr(n)}: {G.nodes[n]}')

        while nodes and count > 0:
            dbg(f"Nodes: {nodes[::-1]}, already processed: {already_processed}")
            count -= 1
            node = nodes.pop()
            true_dependencies = filter(
                lambda n: n in reachable_from_stack_inserters,
                G.predecessors(node)
            )
            if any([n not in already_processed for n in true_dependencies]):
                unprocessed = [n for n in G.predecessors(node) if n not in already_processed]
    #             dbg(f"These nodes still need to be processed: {unprocessed}")
                assert len(nodes) > 0
                # Move the node to the back (NOTE: doesn't do loop detection)
                # TODO: need some way to detect if the unprocessed
                # nodes are actually never going to do anything. Maybe trim?
    #             nodes = list(set(
    #                 unprocessed + nodes
    #             ))
                nodes.insert(0, node)
                dbg(f"  Moved {repr(nodes[0])} to front of queue, some dependants {unprocessed} haven't been processed")
                continue
            assert node not in already_processed
            dbg(f"\nChecking node {repr(node)}")

            curr = G.nodes[node]
            proto = prototype_from_str(node.split('\n@')[0])
            dbg(f"  {curr=}")
            # Don't bother checking the initial sources for input rates
            if len(curr['output']) == 0:
                # Given all the predecessors' output *rates*, calculate this node's input rate
                curr['input_'] = {}
                for prev in G.predecessors(node):
                    for item, flow_rate in G.nodes[prev]['output'].items():
                        if item not in curr['input_']:
                            curr['input_'][item] = 0
                        curr['input_'][item] += flow_rate
                dbg(f"  curr[input_] is now: {curr['input_']}")
                if 'assembling_machine' in node:
                    dbg(f'  asm machine: {curr}')
                    assert curr['recipe'] != 'empty'
                    min_ratio = 1
                    # TODO crafting speed???
                    for item, rate in recipes[curr['recipe']].consumes.items():
                        ratio = curr['input_'][item] / rate
    #                     print(f"    [{item}] input:{curr['input_'][item]} / max:{rate} = {ratio}")
                        min_ratio = min(min_ratio, ratio)
                    dbg(f"  Minimum ratio for {curr} is {min_ratio}")
                    dbg(f"    Recipe consumables: {recipes[curr['recipe']].consumes}")
                    dbg(f"    Recipe products: {recipes[curr['recipe']].produces}")
                    curr['output'] = {
                        k: v * min_ratio
                        for k, v in
                        recipes[curr['recipe']].produces.items()
                    }
                else:
                    # Given this node's total input, calculate it's total output
                    for k, v in curr['input_'].items():
                        curr['output'][k] = min(v, proto.flow)
                    dbg(f'  made input_ match output: {curr["input_"]=} {curr["output"]=}')
            dbg(f"  after: {curr=}")
            dbg(f"Calcs:")
            for n in G.nodes:
                dbg(f'- {repr(n)}: {G.nodes[n]}')

            nodes = list(set(
                [n for n in G.neighbors(node) if n not in already_processed]
                + nodes
            ))
            dbg(f"Nodes: {nodes}")
            already_processed.append(node)

        assert count > 0, '"Recursion" depth reached, halting'

        output = {}
        dbg("iterating G.nodes")
        for n in G.nodes:
            dbg(f'- {repr(n)}: {G.nodes[n]}')
            if 'bulk_inserter' not in n:
                continue
            dbg(f"{repr(n)} is bulk inserter, examining")
            for k, v in G.nodes[n]['output'].items():
                if k not in output:
                    output[k] = 0
                output[k] += v
                dbg(f'- Added {v} to output[{k}] to make {output[k]} from {repr(n)}')

        sources = [n for n in G if 'stack_inserter' in n]
        sinks = [n for n in G if 'bulk_inserter' in n]

        can_reach_sink = set().union(*(nx.ancestors(G, s) | {s} for s in sinks))
        reachable_from_source = set().union(*(nx.descendants(G, s) | {s} for s in sources))
        unreachable = set(G.nodes) - can_reach_sink - reachable_from_source

        return output, len(unreachable)

    def show_two_factories(one, two, title_one="First Factory", title_two="Second Factory"):
        return mo.Html(f"""<table>
        <th>{title_one}</th>
        <th>{title_two}</th>
        <tr>
        <td>{world_to_html(one).data}</td>
        <td>{world_to_html(two).data}</td>
        </tr></table>""")

    def plot_loss_history(loss_history):
        # Create a figure
        fig = go.Figure()
        # Add traces for each key in loss_history
        for k in loss_history[-1].keys():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(loss_history))),  # X-axis: range of iterations
                    y=[float(v[k]) if k in v else np.nan for v in loss_history],  # Y-axis: values for the current key
                    mode='lines',  # Plot as lines
                    name=k,  # Legend label
                    line=dict(width=0.5),  # Set line width
                )
            )
        # Update layout for better readability
        fig.update_layout(
            title="Loss History",  # Title of the plot
            xaxis_title="Iteration",  # X-axis label
            yaxis_title="Loss",  # Y-axis label
            width=800,  # Set figure width
            height=500,  # Set figure height
        )
        # Show the plot
        fig.show()

    def normalise_world(world_T, og_world):
        assert torch.is_tensor(world_T), f"world_T is {type(world_T)}, not a tensor"
        assert torch.is_tensor(og_world), f"og_world is {type(og_world)}, not a tensor"
        assert len(world_T.shape) == 3, f"Expected world_T to have 3 dimensions, but is of shape {world_T.shape}"
        assert len(og_world.shape) == 3, f"Expected og_world to have 3 dimensions, but is of shape {og_world.shape}"
        assert world_T.shape[0] == world_T.shape[1], f"Expected world_T to be square, but is of shape {world_T.shape}"
        assert og_world.shape[0] == og_world.shape[1], f"Expected og_world to be square, but is of shape {og_world.shape}"

        empty_entity_value = prototype_from_str('empty').value

        bulk_inserter_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == prototype_from_str('bulk_inserter').value
        )
        world_T[:, :, Channel.ENTITIES.value][bulk_inserter_mask] = empty_entity_value

        stack_inserter_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == prototype_from_str('stack_inserter').value
        )
        world_T[:, :, Channel.ENTITIES.value][stack_inserter_mask] = empty_entity_value

        green_circ_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == prototype_from_str('electronic_circuit').value
        )
        world_T[:, :, Channel.ENTITIES.value][green_circ_mask] = empty_entity_value

        # Remove all transport belts without direction
        belt_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == prototype_from_str('transport_belt').value
        )
        no_direction_mask = (
            world_T[:, :, Channel.DIRECTION.value]
            == Direction.NONE.value
        )
        world_T[:, :, Channel.ENTITIES.value][belt_mask & no_direction_mask] = empty_entity_value

        # # Ensure belts don't have recipes
        # belt_entity_value = prototype_from_str('transport_belt').value
        # belt_entities = (world_T[:, :, Channel.ENTITIES.value] == belt_entity_value)
        # world_T[:, :, Channel.RECIPES.value][belt_entities] = empty_entity_value

        # # Ensure all empty entities have no recipe, no direction
        # no_entity = (world_T[:, :, Channel.ENTITIES.value] == empty_entity_value)
        # world_T[:, :, Channel.RECIPES.value][no_entity] = empty_entity_value
        # world_T[:, :, Channel.DIRECTION.value][no_entity] = Direction.NONE.value

        # Ensure the model can't just overwrite existing factories with a simpler thing.
        tworld = og_world.clone().detach().to(torch.int64)
        # tworld = torch.tensor(og_world, dtype=torch.int64)
        original_had_something = (tworld[:, :, Channel.ENTITIES.value] != empty_entity_value)
        for ch in list(Channel):
            replacements = tworld[:, :, ch.value][original_had_something]
            world_T[:, :, ch.value][original_had_something] = replacements
        return world_T

    def get_min_belts(world_CWH):
        assert world_CWH.shape[1] == world_CWH.shape[2], "Wrong shape: {world_CWH.shape}"
        C, W, H = world_CWH.shape

        stack_inserter_id = prototype_from_str("stack_inserter").value
        bulk_inserter_id = prototype_from_str("bulk_inserter").value
        coords1 = torch.where(world_CWH[Channel.ENTITIES.value] == bulk_inserter_id)
        assert len(coords1[0]) == len(coords1[1]) == 1, f"Expected 1 bulk inserter, found {coords1} in world {world_CWH}"
        w1, h1 = coords1[0][0], coords1[1][0]


        coords2 = torch.where(world_CWH[Channel.ENTITIES.value] == stack_inserter_id)
        assert len(coords2[0]) == len(coords2[1]) == 1, f"Expected 1 stack inserter, found {coords2} in world {world_CWH}"
        w2, h2 = coords2[0][0], coords2[1][0]

        # we want an estimate for how many belts are required, so get the
        # coords of the transport belt tile closest to the source/sink
        w1 = torch.clamp(w1, 1, W-2)
        h1 = torch.clamp(h1, 1, H-2)
        w2 = torch.clamp(w2, 1, W-2)
        h2 = torch.clamp(h2, 1, H-2)

        manhat_dist = torch.abs(w1 - w2) + torch.abs(h1 - h2)
        min_belts = manhat_dist + 1
        return min_belts

    def get_new_world(seed, n=6, min_belts=None):
        if seed is not None:
            np.random.seed(seed)
        assert min_belts != [1], f"min_belts of [1] is sometimes unsatisfiable"
        if min_belts is None:
            min_belts = list(range(0, 64))
        w = new_world(width=n, height=n)
        boundary_tiles = []
        for i in range(n):
            for j in range(n):
                if i in (0, n-1) and j in (0, n-1):
                    continue
                if i in (0, n-1) or j in (0, n-1):
                    boundary_tiles.append((i, j))

        # Put a source and a sink on one of the boundaries
        source = boundary_tiles[np.random.choice(len(boundary_tiles))]
        w[source[0], source[1], Channel.ENTITIES.value] = prototype_from_str('stack_inserter').value
        # TODO not the most efficient, but it'll be okay for now
        while True:
            # Find random location for the sink
            sink = boundary_tiles[np.random.choice(len(boundary_tiles))]
            # Ensure the sink isn't on top of the source
            if source == sink:
                continue
            # Add the sink to the world
            w[sink[0], sink[1], Channel.ENTITIES.value] = prototype_from_str('bulk_inserter').value
            # Calculate the manhatten distance
            min_belt = get_min_belts(torch.tensor(w).permute(2, 0, 1))
            # If manhatten distance is acceptable and source != sink, we've got
            # our world
            if (source != sink) and (min_belt in min_belts):
                break
            # else, remove the sink from the world and try again
            w[sink[0], sink[1], Channel.ENTITIES.value] = prototype_from_str('empty').value

        # Add the source + sink to the world
        # w[source[0], source[1], Channel.RECIPES.value] = prototype_from_str('electronic_circuit').value
        # w[sink[0], sink[1], Channel.RECIPES.value] = prototype_from_str('electronic_circuit').value

        # Figure out the direction of the source + sink
        for x, y, is_source in [(*source, True), (*sink, False)]:
            if x == 0:
                w[x, y, Channel.DIRECTION.value] = (Direction.EAST if is_source else Direction.WEST).value
            if x == n-1:
                w[x, y, Channel.DIRECTION.value] = (Direction.WEST if is_source else Direction.EAST).value
            if y == 0:
                w[x, y, Channel.DIRECTION.value] = (Direction.SOUTH if is_source else Direction.NORTH).value
            if y == n-1:
                w[x, y, Channel.DIRECTION.value] = (Direction.NORTH if is_source else Direction.SOUTH).value

        return torch.tensor(w).to(torch.float)

    def sample_world(probabilities):
        assert torch.is_tensor(probabilities), f'probabilities is {type(probabilities)} not torch.Tensor'
        distribution = Categorical(probs=probabilities)
        samples = distribution.sample()
        # make the directions fit the expected values
        d_direction = samples[:, :, Channel.DIRECTION.value]
        mask = (d_direction > 0)
        d_direction[mask] = d_direction[mask] * 4 - 4
        d_direction[~mask] = -1
        return samples

    def eval_model(actor, critic, pars, num_evaluations=1_000, pbar=False):
        torch.manual_seed(42)
        evals = []
        iterator = torch.randint(0, 2**16-1, (num_evaluations,)).tolist()
        if pbar:
            iterator = mo.status.progress_bar(iterator)
        for seed in iterator:
            original_world = get_new_world(seed, n=4)
            probabilities = actor(original_world)
            normalised_world = normalise_world(sample_world(probabilities), original_world)
            # value = critic(normalised_world)
            value = critic(normalised_world.to(torch.float))
            # Maybe having throughput being calculated as a black box is the problem?
            throughput = torch.tensor(
                funge_throughput(normalised_world)[0] / 15.0,
                dtype=value.dtype,
            )
            num_entities = (normalised_world[:, :, Channel.ENTITIES.value] != prototype_from_str('empty').value).sum()
            evals.append({
                'seed': seed,
                'original_world': original_world,
                'normalised_world': normalised_world,
                'throughput': throughput,
                'num_entities': num_entities,
            })

        avg_throughput = sum([eval['throughput'] for eval in evals]) / len(evals)
        avg_num_entities = sum([eval['num_entities'] for eval in evals]) / len(evals)

        return evals, avg_throughput, float(avg_num_entities)

    def funge_throughput(world, debug=False):
        assert torch.is_tensor(world), f"world is {type(world)}, not a tensor"
        assert len(world.shape) == 3, f"Expected world to have 3 dimensions, but is of shape {world.shape}"
        assert world.shape[0] == world.shape[1], f"Expected world to be square, but is of shape {world.shape}"
        try:
            throughput, num_unreachable = calc_throughput(graph_from_world(world, debug=debug), debug=debug)
            if len(throughput) == 0:
                return 0, num_unreachable
            actual_throughput = list(throughput.values())[0]
            assert  actual_throughput < float('inf'), f"throughput is +inf, probably a bug, world is: {torch.tensor(world).permute(2, 0, 1)}"
            return actual_throughput, num_unreachable
        except AssertionError:
            breakpoint()
            traceback.print_exc()
            return 0, 0

    mo.md("Functions")
    return (
        add_entity,
        b64_to_dict,
        b64img_from_str,
        calc_throughput,
        dict_to_b64,
        eval_model,
        funge_throughput,
        get_min_belts,
        get_new_world,
        graph_from_world,
        new_world,
        normalise_world,
        plot_flow_network,
        plot_loss_history,
        prototype_from_str,
        sample_world,
        show_two_factories,
        world_from_blueprint,
        world_to_html,
    )


@app.cell
def _(torch):
    import gymnasium as gym
    from cleanrl.cleanrl.ppo import AgentCNN, FactorioEnv

    path = 'cleanrl/artifacts/agent-1.000000-factorion-FactorioEnv-v0__ppo__1__1745516456.pt'
    path = 'cleanrl/artifacts/agent-0.994240-factorion-FactorioEnv-v0__ppo__1__2025-04-24T22-15-15.pt'

    path = 'cleanrl/artifacts/agent-1.023271-factorion-FactorioEnv-v0__ppo__1__2025-04-25T04-00-29.pt'

    path = 'cleanrl/artifacts/agent-0.966447-factorion-FactorioEnv-v0__ppo__1__2025-04-25T11-17-30.pt'


    def make_env():
        def _thunk():
            return FactorioEnv()
        return _thunk

    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(4)])

    agent = AgentCNN(envs)
    agent.load_state_dict(torch.load(path, weights_only=False))
    agent.eval()
    return AgentCNN, FactorioEnv, agent, envs, gym, make_env, path


@app.cell
def __(
    Channel,
    agent,
    funge_throughput,
    get_new_world,
    normalise_world,
    torch,
    world_to_html,
):
    size = 5
    world_CWH = get_new_world(seed=None, n=size).permute(2, 0, 1)
    world_CWH


    example_input = torch.randn(1, agent.channels, agent.width, agent.height)
    with torch.no_grad():
        action_BCWH, logprob, entropy, value = agent.get_action_and_value(world_CWH.unsqueeze(0))

    action_CWH = action_BCWH[0, :, :, :]
    dir_CWH = action_CWH[Channel.DIRECTION.value, :, :]
    mask = (dir_CWH > 0)
    dir_CWH[mask] = dir_CWH[mask] * 4 - 4
    dir_CWH[~mask] = -1
    action_WHC = torch.tensor(action_CWH).permute(1, 2, 0)
    world_WHC = world_CWH.permute(1, 2, 0)
    normalised_world_WHC = normalise_world(action_WHC, world_WHC)
    throughput, num_unreachable = funge_throughput(normalised_world_WHC, debug=False)
    throughput /= 15.0
    frac_reachable = 1.0 - float(num_unreachable) / (size*size)
    normalised_world_CWH = normalised_world_WHC.permute(2, 0, 1)

    hallucination_rate = (normalised_world_WHC != action_CWH.permute(1, 2, 0)).sum() / action_CWH.numel()

    (

        "(white inserter is source, green inserter is sink)",
        "---Initial world---",
        world_to_html(world_CWH.permute(1, 2, 0)),
        "---Before normalisation---",
        world_to_html(action_CWH.permute(1, 2, 0)),
        "---Predicted world---",
        world_to_html(normalised_world_WHC),
        f"Throughput: {throughput*100:.0f}%, hallu: {hallucination_rate:.2f}, frac: {frac_reachable}"
    )
    return (
        action_BCWH,
        action_CWH,
        action_WHC,
        dir_CWH,
        entropy,
        example_input,
        frac_reachable,
        hallucination_rate,
        logprob,
        mask,
        normalised_world_CWH,
        normalised_world_WHC,
        num_unreachable,
        size,
        throughput,
        value,
        world_CWH,
        world_WHC,
    )


if __name__ == "__main__":
    app.run()
