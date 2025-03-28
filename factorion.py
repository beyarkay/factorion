import marimo

__generated_with = "0.11.24"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    from dataclasses import dataclass
    from enum import Enum
    from torch.distributions import Categorical
    from tqdm import trange
    from tqdm.notebook import tqdm
    import base64
    import numpy as np
    import json
    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import plotly.graph_objects as go
    import sys
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import zlib
    import wandb

    wandb.login()
    mo.md("Imports")
    return (
        Categorical,
        Enum,
        F,
        base64,
        dataclass,
        go,
        json,
        mo,
        nn,
        np,
        nx,
        optim,
        plt,
        sys,
        torch,
        tqdm,
        trange,
        wandb,
        zlib,
    )


@app.cell(hide_code=True)
def _(Enum, dataclass, mo):
    class Channel(Enum):
        # What entity occupies this tile?
        ENTITIES = 0
        # What recipe OR filter is set?
        RECIPES = 1
        # what direction is the entity facing?
        DIRECTION = 2
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
        NONE = -1
        NORTH = 0
    #     NORTH_EAST = 2
        EAST = 4
    #     SOUTH_EAST = 6
        SOUTH = 8
    #     SOUTH_WEST = 10
        WEST = 12
    #     NORTH_WEST = 14

    @dataclass
    class Prototype:
        name: str
        value: int
        flow: float
        width: int
        height: int

    prototypes = {
        0:  Prototype(name='empty',                 value=0,  width=1, height=1, flow=0.0),
    #     1:  Prototype(name='sink',                  value=1,  width=1, height=1, flow=float('inf')),
    #     2:  Prototype(name='source',                value=2,  width=1, height=1, flow=float('inf')),
    #     3:  Prototype(name='assembling_machine_1',  value=3,  width=3, height=3, flow=0.5),
    #     4:  Prototype(name='copper_cable',          value=4,  width=1, height=1, flow=0.0),
    #     5:  Prototype(name='copper_ore',            value=5,  width=1, height=1, flow=0.0),
    #     6:  Prototype(name='copper_plate',          value=6,  width=1, height=1, flow=0.0),
    #     7:  Prototype(name='electric_mining_drill', value=7,  width=3, height=3, flow=0.5),
        1:  Prototype(name='electronic_circuit',    value=1,  width=1, height=1, flow=0.0),
    #     9:  Prototype(name='hazard_concrete',       value=9,  width=1, height=1, flow=0.0),
    #     10: Prototype(name='inserter',              value=10, width=1, height=1, flow=0.86),
    #     11: Prototype(name='iron_ore',              value=11, width=1, height=1, flow=0.0),
    #     12: Prototype(name='iron_plate',            value=12, width=1, height=1, flow=0.0),
    #     13: Prototype(name='splitter',              value=13, width=2, height=1, flow=15.0),
    #     14: Prototype(name='steel_chest',           value=14, width=1, height=1, flow=0.0),
        2: Prototype(name='transport_belt',        value=2, width=1, height=1, flow=15.0),
        # TODO: this doesn't account for the speed of items
        # underground (which is identical to a transport belt)
    #     16: Prototype(name='underground_belt',      value=16, width=1, height=1, flow=15.0),
        3: Prototype(name='bulk_inserter',         value=3, width=1, height=1, flow=float('inf')),
        4: Prototype(name='stack_inserter',        value=4, width=1, height=1, flow=float('inf')),
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
def _(
    Categorical,
    Channel,
    Direction,
    Footprint,
    Misc,
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
        world[x, y, Channel.RECIPES.value] = recipe_proto.value
        world[x, y, Channel.MISC.value] = misc.value

    def world_to_html(world):
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
                recipe = prototypes[world[x, y, Channel.RECIPES.value]]
                direction = world[x, y, Channel.DIRECTION.value]
    #             entity, direction, recipe = get_entity_info(world, x, y)
                entity_icon = b64img_from_str(proto.name)
                recipe_icon = b64img_from_str(recipe.name)
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
                        <img src='{recipe_icon}' style='position: absolute; top: 0; right: 0; width: 20px; height: 20px;'>
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
        G = nx.DiGraph()
        def dbg(s):
            if debug: print(s)

        for x in range(len(world)):
            for y in range(len(world[0])):
                e = prototypes[world[x, y, Channel.ENTITIES.value]]
                if e.name == 'empty':
                    continue
                r = prototypes[world[x, y, Channel.RECIPES.value]]
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
            return {'foobar': 0.0}
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
    #                 print(f"  curr output is {curr['output']}")
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
        return output

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

        # Ensure belts don't have recipes
        belt_entity_value = prototype_from_str('transport_belt').value
        belt_entities = (world_T[:, :, Channel.ENTITIES.value] == belt_entity_value)
        world_T[:, :, Channel.RECIPES.value][belt_entities] = empty_entity_value

        # Ensure all empty entities have no recipe, no direction
        no_entity = (world_T[:, :, Channel.ENTITIES.value] == empty_entity_value)
        world_T[:, :, Channel.RECIPES.value][no_entity] = empty_entity_value
        world_T[:, :, Channel.DIRECTION.value][no_entity] = Direction.NONE.value

        # Ensure the model can't just overwrite existing factories with a simpler thing.
        tworld = og_world.clone().detach().to(torch.int64)
        # tworld = torch.tensor(og_world, dtype=torch.int64)
        original_had_something = (tworld[:, :, Channel.ENTITIES.value] != empty_entity_value)
        for ch in list(Channel):
            replacements = tworld[:, :, ch.value][original_had_something]
            world_T[:, :, ch.value][original_had_something] = replacements
        return world_T

    def get_new_world(seed, n=6):
        np.random.seed(seed)
        w = new_world(width=n, height=n)
        boundary_tiles = []
        for i in range(n):
            for j in range(n):
                if i in (0, n-1) and j in (0, n-1):
                    continue
                if i in (0, n-1) or j in (0, n-1):
                    boundary_tiles.append((i, j))
        source = boundary_tiles[np.random.choice(len(boundary_tiles))]
        sink = boundary_tiles[np.random.choice(len(boundary_tiles))]
        while source == sink:
            sink = boundary_tiles[np.random.choice(len(boundary_tiles))]

        w[source[0], source[1], Channel.ENTITIES.value] = prototype_from_str('stack_inserter').value
        w[sink[0], sink[1], Channel.ENTITIES.value] = prototype_from_str('bulk_inserter').value
        w[source[0], source[1], Channel.RECIPES.value] = prototype_from_str('electronic_circuit').value
        w[sink[0], sink[1], Channel.RECIPES.value] = prototype_from_str('electronic_circuit').value

        for x, y, flipper in [(*source, 0), (*sink, 8)]:
            if x == 0:
                d = Direction.EAST
            if x == n-1:
                d = Direction.WEST
            if y == 0:
                d = Direction.SOUTH
            if y == n-1:
                d = Direction.NORTH

            d = (d.value + flipper) % 16
            w[x, y, Channel.DIRECTION.value] = d

        return torch.tensor(w).to(torch.float)

    def sample_world(probabilities):
        distribution = Categorical(probs=probabilities)
        samples = distribution.sample()
        # make the directions fit the expected values
        d_direction = samples[:, :, Channel.DIRECTION.value]
        mask = (d_direction > 0)
        d_direction[mask] = d_direction[mask] * 4 - 4
        d_direction[~mask] = -1
        return samples

    def eval_model(model, pars, num_evaluations=1_000, pbar=False):
        torch.manual_seed(42)
        evals = []
        iterator = torch.randint(0, 2**16-1, (num_evaluations,)).tolist()
        if pbar:
            iterator = mo.status.progress_bar(iterator)
        for seed in iterator:
            original_world = get_new_world(seed, n=4)
            probabilities, value = model(original_world)
            normalised_world = normalise_world(sample_world(probabilities), original_world)
            throughput = funge_throughput(normalised_world)
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

    def funge_throughput(world):
         try:
             throughput = calc_throughput(graph_from_world(world.numpy()))
             if len(throughput) == 0:
                 return 0
             return list(throughput.values())[0]
         except AssertionError:
             return 0

    mo.md("Functions")
    return (
        add_entity,
        b64_to_dict,
        b64img_from_str,
        calc_throughput,
        dict_to_b64,
        eval_model,
        funge_throughput,
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
def _(mo, wandb):
    pars = {
        'policy_lr': 5e-4,
        'value_lr': 1e-3,
        'num_epochs': 2_000,
        'world_size': 4,
        'hidden_channels': [8, 16, 8],
        'entropy_loss_multiplier': 0.1,
    }
    run = wandb.init(
        project="factorion",  # Specify your project
        config=pars,
    )

    mo.md("Define hyperparameters")
    return pars, run


@app.cell(hide_code=True)
def _(mo, nn):
    class PPOModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_entities):
            super().__init__()
            # channels = [in_channels, *hidden_channels]

            # self.cnn = nn.Sequential(*[
            #     layer
            #     for c_in, c_out in zip(channels[:-1], channels[1:])
            #     for layer in [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.ReLU()]
            # ][:-1])  # Remove last ReLU

            self.policy_head = nn.Sequential(
               nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(8, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, out_channels * num_entities, kernel_size=1),
            )

            self.value_head = nn.Sequential(
               nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 1, kernel_size=1)
            )

            self.num_entities = num_entities
            self.out_channels = out_channels
            self.hidden_channels = hidden_channels

        def forward(self, x):
            # (W, H, C) -> (B=1, C, W, H)
            x = x.permute(2, 0, 1).unsqueeze(0)  

            # features = self.cnn(x)
            # Policy head
            # Shape: (1, out_channels * num_entities, W, H)
            policy = self.policy_head(x)
            policy = policy.view(1, self.out_channels, self.num_entities, *policy.shape[2:])
            # Shape: (W, H, out_channels, num_entities)
            policy = policy.permute(3, 4, 0, 1, 2).squeeze(2)
            probabilities = policy.softmax(dim=-1)

            # Value head
            value = self.value_head(x).mean(dim=(-1, -2)).squeeze()
            return probabilities, value

    class ActorModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_entities):
            super().__init__()
            self.policy_head = nn.Sequential(
               nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(8, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, out_channels * num_entities, kernel_size=1),
            )

            self.num_entities = num_entities
            self.out_channels = out_channels
            self.hidden_channels = hidden_channels

        def forward(self, x):
            x = x.permute(2, 0, 1).unsqueeze(0)  

            policy = self.policy_head(x)
            policy = policy.view(1, self.out_channels, self.num_entities, *policy.shape[2:])
            # Shape: (W, H, out_channels, num_entities)
            policy = policy.permute(3, 4, 0, 1, 2).squeeze(2)
            probabilities = policy.softmax(dim=-1)

            return probabilities

    class CriticModel(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, num_entities):
            super().__init__()
            self.value_head = nn.Sequential(
               nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(16, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Conv2d(64, 1, kernel_size=1)
            )

            self.num_entities = num_entities
            self.out_channels = out_channels
            self.hidden_channels = hidden_channels

        def forward(self, x):
            # (W, H, C) -> (B=1, C, W, H)
            x = x.permute(2, 0, 1).unsqueeze(0)  
            # Value head
            value = self.value_head(x).mean(dim=(-1, -2)).squeeze()
            return value


    mo.md("Define Model")
    return ActorModel, CriticModel, PPOModel


@app.cell(hide_code=True)
def calc_grad_norms():
    def calc_grad_norms(model):
        mean_grad_norm = 0
        n = 0
        value_grad_norm = 0
        policy_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                mean_grad_norm += param.grad.norm()
                # print(f"- {name}: {param.grad.norm():.4f}")
                if 'policy_head' in name:
                    policy_grad_norm += param.grad.norm() / 2
                if 'value_head' in name:
                    value_grad_norm += param.grad.norm() / 2
                n += 1
        mean_grad_norm /= n

        return {
            'mean_grad_norm': mean_grad_norm,
            'policy_grad_norm': policy_grad_norm,
            'value_grad_norm': value_grad_norm,
        }
    return (calc_grad_norms,)


@app.cell
def _(
    ActorModel,
    Categorical,
    Channel,
    CriticModel,
    F,
    calc_grad_norms,
    funge_throughput,
    get_new_world,
    mo,
    normalise_world,
    optim,
    pars,
    plot_history,
    prototypes,
    sample_world,
    torch,
):
    import random

    def _():
        epsilon = 0.2
        history = []

        actor = ActorModel(
            in_channels=len(Channel),
            hidden_channels=pars['hidden_channels'],
            out_channels=len(Channel),
            num_entities=len(prototypes),
        )
        critic = CriticModel(
            in_channels=len(Channel),
            hidden_channels=pars['hidden_channels'],
            out_channels=len(Channel),
            num_entities=len(prototypes),
        )
        print(f"Actor has {sum(p.numel() for p in actor.parameters())} params")
        print(f"Critic has {sum(p.numel() for p in critic.parameters())} params")

        actor_optimizer = optim.Adam(actor.parameters(), lr=pars['policy_lr'])
        critic_optimizer = optim.Adam(critic.parameters(), lr=pars['value_lr'])
    # 
        # # Initial probabilities for use in PPO improvement calcs
        probabilities = actor(get_new_world(42, n=pars['world_size']))
        sampled_world = Categorical(probs=probabilities).sample()
        old_probs = probabilities.gather(
            dim=-1, 
            index=sampled_world.unsqueeze(-1)
        ).squeeze(-1).detach()
    
        pbar = mo.status.progress_bar(range(pars['num_epochs']), title='Training Value+Policy model')
        for epoch in pbar:
            critic_optimizer.zero_grad()
            actor_optimizer.zero_grad()
    
            original_world = get_new_world(epoch, n=pars['world_size'])

            while True:
                probabilities = actor(original_world)
    
                if torch.isnan(probabilities).any():
                    print("!Probabilites are NaN!")
                    return history
        
                normalised_world = normalise_world(sample_world(probabilities), original_world)
                value = critic(normalised_world.to(torch.float))
                throughput = torch.tensor(
                    funge_throughput(normalised_world) / 15.0, 
                    dtype=value.dtype,
                )
                assert 0 <= throughput <= 1.0, f'Throughput is {throughput}, world is shape {normalised_world.shape}:\n{normalised_world}'
        
                advantage = throughput - value
        
                # Get the action probabilities for the taken sampled_world
                sampled_world = Categorical(probs=probabilities).sample()
                action_probs = probabilities.gather(
                    dim=-1, 
                    index=sampled_world.unsqueeze(-1)
                ).squeeze(-1)
                ratio = action_probs / old_probs
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        
                entropy_loss = -(probabilities * probabilities.log()).sum(dim=-1).mean()
    
                old_probs = action_probs.clone().detach()
            
                value_loss = F.mse_loss(value, throughput)
                value_loss.backward(retain_graph=True)
            
                policy_loss = (
                    - torch.min(ratio * advantage, clipped_ratio * advantage).mean()  
                    - entropy_loss * pars['entropy_loss_multiplier']
                )
                policy_loss.backward()
            
                actor_optimizer.step()
                critic_optimizer.step()
        
                history.append({
                    'policy_loss': policy_loss,
                    'value_loss': value_loss,
                    'entropy_loss': -pars['entropy_loss_multiplier'] * entropy_loss,
                    'throughput': throughput.item(),
                    'value': value.item(),
                    'clipped_ratio': clipped_ratio.mean().item(),
                    'advantage': advantage.item(),
                } | { k: v for k, v in calc_grad_norms(actor).items() if v != 0 } 
                  | { k: v for k, v in calc_grad_norms(critic).items() if v != 0 })
            
                if throughput == 0.0:
                    break
    
            # if epoch % 5_000 == 0:
            #     evaluations, avg_thrpt, avg_entities = eval_model(model, pars)
            #     print(f"[{epoch}] avg throughput: {avg_thrpt:.4f}, avg #entities: {avg_entities:.4f}, value_loss: {value_loss:.4f}, policy_loss: {policy_loss:.4f}")

        return history

    plot_history(_(), hide=['mean_grad_norm', 'value_grad_norm', 'throughput', 'clipped_ratio', 'advantage'])
    return (random,)


@app.cell
def _(go, np):
    def plot_history(history, hide=None):
        hide = [] if hide is None else hide
        # Create a figure
        fig = go.Figure()
        # Add traces for each key in loss_history

        all_keys = list(set([k for h in history for k in h.keys()]))
        print(all_keys)
        for key in all_keys:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(history))),
                    y=[float(v[key]) if key in v else np.nan for v in history], 
                    mode='lines',  # Plot as lines
                    name=key,  # Legend label
                    line=dict(width=0.75),  # Set line width
                )
            )
        # Update layout for better readability
        fig.update_layout(
            title="Training History",  # Title of the plot
            xaxis_title="Epoch",  # X-axis label
            yaxis_title="Values",  # Y-axis label
            template='plotly_dark',
        )
        fig.update_traces(
            visible="legendonly", 
            selector=lambda t: t.name in hide
        )
        # Show the plot
        fig.show()
    return (plot_history,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
