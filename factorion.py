import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import deque, defaultdict
    from dataclasses import dataclass
    from enum import Enum
    from torch.distributions import Categorical
    from tqdm import trange
    from tqdm.notebook import tqdm
    from typing import List, Tuple
    import base64
    import json
    import marimo as mo
    import math
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
    import traceback
    import wandb
    import zlib

    wandb.login()
    mo.md("Imports")
    return (
        Categorical,
        Enum,
        List,
        Tuple,
        base64,
        dataclass,
        defaultdict,
        deque,
        go,
        json,
        mo,
        np,
        nx,
        plt,
        random,
        torch,
        traceback,
        zlib,
    )


@app.cell(hide_code=True)
def datatypes(Enum, dataclass, mo):
    class Channel(Enum):
        # What entity occupies this tile?
        ENTITIES = 0
        # what direction is the entity facing?
        DIRECTION = 1
        # What recipe OR filter is set?
        ITEMS = 2
        # Undergrounds mechanics, see class Misc(Enum)
        MISC = 3
        # 1 if you can build there, 0 if you can't
        # FOOTPRINT = 4


    class Footprint(Enum):
        UNAVAILABLE = 0
        AVAILABLE = 1


    class Misc(Enum):
        NONE = 0
        UNDERGROUND_DOWN = 1
        UNDERGROUND_UP = 2


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


    @dataclass
    class Item:
        name: str
        value: int


    @dataclass
    class Entity:
        name: str
        value: int
        flow: float
        width: int
        height: int


    # NOTE: Don't forget to update the "value" as well as the order
    items = {
        0: Item(name="empty", value=0),
        1: Item(name="copper_cable", value=1),
        2: Item(name="copper_plate", value=2),
        3: Item(name="iron_plate", value=3),
        4: Item(name="electronic_circuit", value=4),
    }

    # Also update the pretty printer
    entities = {
        0: Entity(name="empty", value=0, width=1, height=1, flow=0.0),
        1: Entity(name="transport_belt", value=1, width=1, height=1, flow=15.0),
        2: Entity(name="inserter", value=2, width=1, height=1, flow=0.86),
        3: Entity(
            name="assembling_machine_1", value=3, width=3, height=3, flow=0.5
        ),
        # underground (which is identical to a transport belt)
        4: Entity(name="underground_belt", value=4, width=1, height=1, flow=15.0),
        # sink
        5: Entity(
            name="bulk_inserter", value=5, width=1, height=1, flow=float("inf")
        ),
        # source
        6: Entity(
            name="stack_inserter", value=6, width=1, height=1, flow=float("inf")
        ),
        #     4:  Entity(name='copper_cable',          value=4,  width=1, height=1, flow=0.0),
        #     6:  Entity(name='copper_plate',          value=6,  width=1, height=1, flow=0.0),
        #     5:  Entity(name='copper_ore',            value=5,  width=1, height=1, flow=0.0),
        #     7:  Entity(name='electric_mining_drill', value=7,  width=3, height=3, flow=0.5),
        # 1:  Entity(name='electronic_circuit',    value=1,  width=1, height=1, flow=0.0),
        #     9:  Entity(name='hazard_concrete',       value=9,  width=1, height=1, flow=0.0),
        #     11: Entity(name='iron_ore',              value=11, width=1, height=1, flow=0.0),
        #     12: Entity(name='iron_plate',            value=12, width=1, height=1, flow=0.0),
        #     13: Entity(name='splitter',              value=13, width=2, height=1, flow=15.0),
        #     14: Entity(name='steel_chest',           value=14, width=1, height=1, flow=0.0),
    }


    @dataclass
    class Recipe:
        consumes: dict[str, float]
        produces: dict[str, float]


    recipes = {
        "electronic_circuit": Recipe(
            consumes={"copper_cable": 6.0, "iron_plate": 2.0},
            produces={"electronic_circuit": 2.0},
        ),
        "copper_cable": Recipe(
            consumes={"copper_plate": 2.0},
            produces={"copper_cable": 4.0},
        ),
    }


    class LessonKind(Enum):
        MOVE_ONE_ITEM = 0
        MOVE_TWO_ITEMS_NO_UNDERGROUND = 1
        MOVE_TWO_ITEMS_WITH_UNDERGROUND = 2
        CREATE_COPPER_WIRE = 3
        CREATE_ELECTRONIC_CIRCUIT = 4


    # Map Enum <--> grid deltas
    DIR_TO_DELTA = {
        Direction.NORTH: (0, -1),
        Direction.EAST: (1, 0),
        Direction.SOUTH: (0, 1),
        Direction.WEST: (-1, 0),
    }

    mo.md("Datatypes")
    return (
        Channel,
        DIR_TO_DELTA,
        Direction,
        Entity,
        Footprint,
        LessonKind,
        Misc,
        entities,
        items,
        recipes,
    )


@app.cell
def functions(
    Categorical,
    Channel,
    DIR_TO_DELTA,
    Direction,
    Entity,
    Footprint,
    LessonKind,
    List,
    Misc,
    Tuple,
    base64,
    defaultdict,
    deque,
    entities,
    go,
    items,
    json,
    mo,
    np,
    nx,
    plt,
    random,
    recipes,
    torch,
    traceback,
    zlib,
):
    def b64_to_dict(blueprint_string):
        decoded = base64.b64decode(
            blueprint_string.strip()[1:]
        )  # Skip the version byte
        json_data = zlib.decompress(decoded).decode("utf-8")
        return json.loads(json_data)


    def dict2b64(dictionary):
        compressed = zlib.compress(json.dumps(dictionary).encode("utf-8"))
        b64_encoded = base64.b64encode(compressed).decode("utf-8")
        blueprint_string = "0" + b64_encoded  # Add version byte
        return blueprint_string


    def str2item(s):
        assert s is not None, "input cannot be None"
        return next(
            (v for k, v in items.items() if v.name == s.replace("-", "_")), None
        )


    def str2ent(s):
        if s is None:
            print(f"WARN: given string  is None")
            return None
        if s == "source":
            s = "stack_inserter"
        elif s == "sink":
            s = "bulk_inserter"

        for v in entities.values():
            if v.name == s.replace("-", "_"):
                return v
        # TODO I'm almost certianly going to regret hardcoding this
        if s == "electronic_circuit":
            return Entity(
                name="electronic_circuit",
                value=len(entities),
                width=1,
                height=1,
                flow=0.0,
            )
        print(f"WARN: unknown entity {s}")
        return None


    def _str2b64img(path, base_path="factorio-icons"):
        try:
            with open(f"{base_path}/{path}.png", "rb") as image_file:
                return "data:image/png;base64," + base64.b64encode(
                    image_file.read()
                ).decode("utf-8")
        except:
            return ""


    def ent_str2b64img(ent, base_path="factorio-icons"):
        return _str2b64img(str2ent(ent).name)


    def item_str2b64img(item, base_path="factorio-icons"):
        return _str2b64img(str2item(item).name)


    def new_world(width=8, height=8):
        channels = len(Channel)
        #     print(f"Making world w={width}, h={height}, c={channels}")
        world = np.zeros((width, height, channels), dtype=int)
        world[:, :, Channel.ENTITIES.value] = str2ent("empty").value
        world[:, :, Channel.DIRECTION.value] = Direction.NONE.value
        if any(["Channel.FOOTPRINT" in i for i in map(str, list(Channel))]):
            world[:, :, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        return world


    def add_entity(
        world,
        proto_str,
        x,
        y,
        direction=Direction.NONE,
        recipe="empty",
        misc=Misc.NONE,
    ):
        proto = str2ent(proto_str)
        EMPTY = str2ent("empty")
        if proto is None:
            proto = EMPTY
        recipe_proto = str2ent(recipe)
        if recipe_proto is None:
            recipe_proto = EMPTY
        assert world[x, y, Channel.ENTITIES.value] == EMPTY.value, (
            f"Can't place {proto_str} at {x},{y} because {entities[world[x, y, Channel.ENTITIES.value]]} is there"
        )
        assert 0 <= x < len(world), f"{x=} is not in [0, {len(world)})"
        assert 0 <= y < len(world[0]), f"{y=} is not in [0, {len(world[0])})"

        world[x, y, Channel.ENTITIES.value] = proto.value
        world[x, y, Channel.DIRECTION.value] = direction.value
        # world[x, y, Channel.ITEMS.value] = recipe_proto.value
        # world[x, y, Channel.MISC.value] = misc.value


    def world2html(world_WHC):
        assert len(world_WHC.shape) == 3, (
            f"Expected 3 dimensions got {world_WHC.shape}"
        )
        assert world_WHC.shape[0] == world_WHC.shape[1], (
            f"Expected square got {world_WHC.shape}"
        )
        if type(world_WHC) is not np.ndarray:
            world_WHC = np.array(world_WHC)
        DIRECTION_ARROWS = {
            Direction.NONE.value: "",
            Direction.NORTH.value: "↑",
            Direction.EAST.value: "→",
            Direction.SOUTH.value: "↓",
            Direction.WEST.value: "←",
            # 0: "↘",
            # 10: "↙",
            # 14: "↖",
        }
        html = ["<table style='border-collapse: collapse;'>"]
        W, H, C = world_WHC.shape
        display_asm_ghost = 0
        ghosts = []
        for y in range(H):
            html.append("<tr>")
            for x in range(W):
                proto = entities[world_WHC[x, y, Channel.ENTITIES.value]]
                if proto.width != 1 or proto.height != 1:
                    ghosts.append(
                        {
                            "x_lo": x,
                            "y_lo": y,
                            "x_hi": x + proto.width,
                            "y_hi": y + proto.height,
                            "name": proto.name,
                        }
                    )
                item = items[world_WHC[x, y, Channel.ITEMS.value]]
                direction = world_WHC[x, y, Channel.DIRECTION.value]
                #             entity, direction, recipe = get_entity_info(world_WHC, x, y)
                entity_icon = ent_str2b64img(proto.name)

                ghost_icons = []
                for ghost in ghosts:
                    if (
                        ghost["x_lo"] <= x < ghost["x_hi"]
                        and ghost["y_lo"] <= y < ghost["y_hi"]
                    ):
                        ghost_icons.append(ent_str2b64img(ghost["name"]))

                item_icon = item_str2b64img(item.name)
                direction_arrow = DIRECTION_ARROWS.get(direction, "")
                if any(["Channel.MISC" in i for i in map(str, list(Channel))]):
                    misc = Misc(world_WHC[x, y, Channel.MISC.value])
                else:
                    misc = Misc.NONE
                underground_symbol = (
                    "⭳"
                    if misc == Misc.UNDERGROUND_DOWN
                    else "⭱"
                    if misc == Misc.UNDERGROUND_UP
                    else ""
                )

                #             print(direction_arrow, direction)
                if any(
                    ["Channel.FOOTPRINT" in i for i in map(str, list(Channel))]
                ):
                    available = (
                        world_WHC[x, y, Channel.FOOTPRINT.value]
                        == Footprint.AVAILABLE.value
                    )
                else:
                    available = True
                bg_style = (
                    "background: rgba(255, 0, 0, 0.3);" if not available else ""
                )
                #             tint_style = "filter: brightness(1.5) sepia(1) hue-rotate(30deg);" if available else ""

                ghost_imgs = "\n".join(
                    [
                        f"<img src='{ghost_icon}' style=' position: absolute; top: 10%;  left: 10%;  width: 60%; height: 60%; opacity: 20%;'>"
                        for ghost_icon in ghost_icons
                    ]
                )

                xy_str = f"{x},{y}"
                cell_content = f"""
               <div style='position: relative; width: 50px; height: 50px; {bg_style}; border: 1px solid grey;'>
        <img src='{entity_icon}' style=' position: absolute; top: 10%;   left: 10%;  width: 60%; height: 60%; '>
        {ghost_imgs}
        <img src='{item_icon}' style=' position: absolute; bottom: 5%;  right: 5%;  width: 20%; height: 20%; '>
        <div style='position: absolute; top: 0; left: 0; font-size: 8px; opacity: 50%'>{xy_str}</div>
        <div style='position: absolute; bottom: 0; left: 0; font-size: 20px;'>{direction_arrow}</div>
        <div style=' position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);  font-size: 20px; font-weight: bold; color: white; '>{underground_symbol}</div>
    </div>
                """
                html.append(
                    f"<td style='border: 1px solid black; padding: 0;'>{cell_content}</td>"
                )
            html.append("</tr>")
        html.append("</table>")
        return mo.Html("".join(html))


    def blueprint2world(bp):
        obj = b64_to_dict(bp)

        min_x = float("inf")
        min_y = float("inf")
        max_y = -float("inf")
        max_x = -float("inf")

        for e in obj["blueprint"]["entities"]:
            # NOTE: might be OBOEs here
            e["position"]["x"] = int(e["position"]["x"] - 0.5)
            e["position"]["y"] = int(e["position"]["y"] - 0.5)

        for e in obj["blueprint"].get("entities", []) + obj["blueprint"].get(
            "tiles", []
        ):
            min_x = min(min_x, e["position"]["x"])
            min_y = min(min_y, e["position"]["y"])
            max_y = max(max_y, e["position"]["y"])
            max_x = max(max_x, e["position"]["x"])

        for e in obj["blueprint"].get("entities", []):
            e["position"]["x"] -= min_x
            e["position"]["y"] -= min_y
            if "direction" not in e and e["name"] == "transport-belt":
                # transport belts have an implicit direction
                e["direction"] = 0
            # inserter's direction is towards their source, which we don't want, so flip them around
            if "inserter" in e["name"]:
                #             print(e)
                e["direction"] = (e.get("direction", 0) + 8) % 16

        # Add one, because of the 0.5 alignment of entities vs tiles
        world = new_world(width=max_x - min_x + 1, height=max_y - min_y + 1)

        # Use Hazard concrete to indicate the footprint
        world[:, :, Channel.FOOTPRINT.value] = Footprint.UNAVAILABLE.value
        for t in obj["blueprint"].get("tiles", []):
            if t["name"] == "refined-hazard-concrete-left":
                x = t["position"]["x"] - min_x
                y = t["position"]["y"] - min_y
                world[x, y, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
            else:
                print(f"Ignoring tile {t}")

        for e in obj["blueprint"].get("entities", []):
            entity = str2ent(e["name"])
            if entity is None:
                entity = str2ent("empty")

            if "recipe" in e:
                recipe = str2ent(e["recipe"])
            else:
                if len(e.get("filters", [])) == 1:
                    recipe = str2ent(e["filters"][0]["name"])
                else:
                    recipe = str2ent("empty")

            # underground belts. Output = emerging, input = descending
            if e.get("type", None) == "output":
                misc = Misc.UNDERGROUND_UP
            elif e.get("type", None) == "input":
                misc = Misc.UNDERGROUND_DOWN
            else:
                misc = Misc.NONE

            direction = Direction(e.get("direction", -1))

            add_entity(
                world,
                e["name"],
                e["position"]["x"],
                e["position"]["y"],
                recipe=recipe.name,
                direction=direction,
                misc=misc,
            )
        return world


    def plot_flow_network(G):
        # Extract x, y coordinates from node names
        pos = {
            node: (int(x), -int(y))
            for node, (x, y) in ((n, n.split("@")[1].split(",")) for n in G.nodes)
        }
        plt.figure(
            figsize=(
                (len(G.nodes) ** 0.5) * 3,
                (len(G.nodes) ** 0.5) * 3,
            )
        )

        # Draw the graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=12,
            font_weight="bold",
        )

        # Add throughput labels
        #     labels = {node: G.nodes[node].get("throughput", {}) for node in G.nodes}
        #     nx.draw_networkx_labels(G, pos, labels=labels, font_color="red")

        plt.show()


    def calc_throughput(G, debug=False):
        foobar = 1

        def dbg(s):
            if debug:
                print(s)

        if len(list(nx.simple_cycles(G))) > 0:
            dbg(f"Returning 0 reward due to cycles: {list(nx.simple_cycles(G))}")
            return {"foobar": 0.0}, 0
        # Now go through the graph and propogate the ingredients from producers to consumers.
        # the flow rate should depend on the intermediate rates.
        stack_inserters = [
            node for node, data in G.nodes(data=True) if "stack_inserter" in node
        ]
        nodes = stack_inserters[:]
        reachable_from_stack_inserters = []
        for s in stack_inserters:
            reachable_from_stack_inserters.extend(list(nx.descendants(G, s)))
        reachable_from_stack_inserters = list(set(reachable_from_stack_inserters))
        dbg(f"{reachable_from_stack_inserters=}")
        already_processed = []
        count = len(G.nodes) * len(G.nodes)  # a reasonable upper bound

        dbg(f"Pre-calcs:")
        for n in G.nodes:
            dbg(f"- {repr(n)}: {G.nodes[n]}")

        while nodes and count > 0:
            dbg(f"Nodes: {nodes[::-1]}, already processed: {already_processed}")
            count -= 1
            node = nodes.pop()
            true_dependencies = filter(
                lambda n: n in reachable_from_stack_inserters, G.predecessors(node)
            )
            if any([n not in already_processed for n in true_dependencies]):
                unprocessed = [
                    n for n in G.predecessors(node) if n not in already_processed
                ]
                #             dbg(f"These nodes still need to be processed: {unprocessed}")
                assert len(nodes) > 0, "there are no nodes"
                # Move the node to the back (NOTE: doesn't do loop detection)
                # TODO: need some way to detect if the unprocessed
                # nodes are actually never going to do anything. Maybe trim?
                #             nodes = list(set(
                #                 unprocessed + nodes
                #             ))
                nodes.insert(0, node)
                dbg(
                    f"  Moved {repr(nodes[0])} to front of queue, some dependants {unprocessed} haven't been processed"
                )
                continue
            assert node not in already_processed, f"Node {node} isn't in {already_processed=}"
            dbg(f"\nChecking node {repr(node)}")

            curr = G.nodes[node]
            proto = str2ent(node.split("\n@")[0])
            dbg(f"  {curr=}")
            # Don't bother checking the initial sources for input rates
            if len(curr["output"]) == 0:
                # Given all the predecessors' output *rates*, calculate this node's input rate
                curr["input_"] = {}
                for prev in G.predecessors(node):
                    for item, flow_rate in G.nodes[prev]["output"].items():
                        if item not in curr["input_"]:
                            curr["input_"][item] = 0
                        curr["input_"][item] += flow_rate
                dbg(f"  curr[input_] is now: {curr['input_']}")

                if "assembling_machine" in node:
                    dbg(f"  asm machine: {curr}")
                    if curr["recipe"] == "empty":
                        print(f"asesmbling machine {repr(node)} has {curr['recipe']=}, is not equal to empty")
                    min_ratio = 1
                    # TODO crafting speed???
                    # dbg(f"{recipes=}")
                    curr["output"] = {}
                    if curr['recipe'] in recipes:
                        for item, rate in recipes[curr["recipe"]].consumes.items():
                            ratio = curr["input_"].get(item, 0) / rate
                            min_ratio = min(min_ratio, ratio)
                        dbg(
                            f"    Recipe consumables: {recipes[curr['recipe']].consumes}"
                        )
                        dbg(f"    Recipe products: {recipes[curr['recipe']].produces}")
                        curr["output"] = {
                            k: v * min_ratio
                            for k, v in recipes[curr["recipe"]].produces.items()
                        }
                    dbg(f"  Minimum ratio for {curr} is {min_ratio}")
                else:
                    # Given this node's total input, calculate it's total output
                    for k, v in curr["input_"].items():
                        curr["output"][k] = min(v, proto.flow)
                    dbg(
                        f'  made input_ match output: {curr["input_"]=} {curr["output"]=}'
                    )
            dbg(f"  after: {curr=}")
            dbg(f"Calcs:")
            for n in G.nodes:
                dbg(f"- {repr(n)}: {G.nodes[n]}")

            nodes = list(
                set(
                    [n for n in G.neighbors(node) if n not in already_processed]
                    + nodes
                )
            )
            dbg(f"Nodes: {nodes}")
            already_processed.append(node)

        assert count > 0, '"Recursion" depth reached, halting'

        output = {}
        dbg("iterating G.nodes")
        for n in G.nodes:
            dbg(f"- {repr(n)}: {G.nodes[n]}")
            if "bulk_inserter" not in n:
                continue
            dbg(f"{repr(n)} is bulk inserter, examining")
            for k, v in G.nodes[n]["output"].items():
                if k not in output:
                    output[k] = 0
                output[k] += v
                dbg(
                    f"- Added {v} to output[{k}] to make {output[k]} from {repr(n)}"
                )

        sources = [n for n in G if "stack_inserter" in n]
        sinks = [n for n in G if "bulk_inserter" in n]

        can_reach_sink = set().union(*(nx.ancestors(G, s) | {s} for s in sinks))
        reachable_from_source = set().union(
            *(nx.descendants(G, s) | {s} for s in sources)
        )
        unreachable = set(G.nodes) - (
            can_reach_sink.intersection(reachable_from_source)
        )

        dbg(f"{can_reach_sink=}")
        dbg(f"{reachable_from_source=}")
        dbg(
            f"source -> ({len(reachable_from_source)} nodes) ... ({len(can_reach_sink)} nodes) -> sink"
        )
        dbg(
            f"Final Throughput: {output}, {len(unreachable)} unreachable nodes: {unreachable}"
        )
        # NOTE: there's a subtle bug here that doesn't have much affect on the
        # reward being able to go down. The "unreachable" calculation tries to
        # not count the sink/source, since the model can't place those, but it
        # does so in a way that makes the end result weird if the sink and the
        # source are the only entities on the map. Not planning on fixing this
        # one anytime soon.
        return output, len(unreachable)


    def show_two_factories(
        one, two, title_one="First Factory", title_two="Second Factory"
    ):
        return mo.Html(f"""<table>
        <th>{title_one}</th>
        <th>{title_two}</th>
        <tr>
        <td>{world2html(one).data}</td>
        <td>{world2html(two).data}</td>
        </tr></table>""")


    def plot_loss_history(loss_history):
        # Create a figure
        fig = go.Figure()
        # Add traces for each key in loss_history
        for k in loss_history[-1].keys():
            fig.add_trace(
                go.Scatter(
                    x=list(
                        range(len(loss_history))
                    ),  # X-axis: range of iterations
                    y=[
                        float(v[k]) if k in v else np.nan for v in loss_history
                    ],  # Y-axis: values for the current key
                    mode="lines",  # Plot as lines
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
        assert torch.is_tensor(world_T), (
            f"world_T is {type(world_T)}, not a tensor"
        )
        assert torch.is_tensor(og_world), (
            f"og_world is {type(og_world)}, not a tensor"
        )
        assert len(world_T.shape) == 3, (
            f"Expected world_T to have 3 dimensions, but is of shape {world_T.shape}"
        )
        assert len(og_world.shape) == 3, (
            f"Expected og_world to have 3 dimensions, but is of shape {og_world.shape}"
        )
        assert world_T.shape[0] == world_T.shape[1], (
            f"Expected world_T to be square, but is of shape {world_T.shape}"
        )
        assert og_world.shape[0] == og_world.shape[1], (
            f"Expected og_world to be square, but is of shape {og_world.shape}"
        )

        empty_entity_value = str2ent("empty").value

        bulk_inserter_mask = (
            world_T[:, :, Channel.ENTITIES.value] == str2ent("bulk_inserter").value
        )
        world_T[:, :, Channel.ENTITIES.value][bulk_inserter_mask] = (
            empty_entity_value
        )

        stack_inserter_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == str2ent("stack_inserter").value
        )
        world_T[:, :, Channel.ENTITIES.value][stack_inserter_mask] = (
            empty_entity_value
        )

        green_circ_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == str2ent("electronic_circuit").value
        )
        world_T[:, :, Channel.ENTITIES.value][green_circ_mask] = empty_entity_value

        # Remove all transport belts without direction
        belt_mask = (
            world_T[:, :, Channel.ENTITIES.value]
            == str2ent("transport_belt").value
        )
        no_direction_mask = (
            world_T[:, :, Channel.DIRECTION.value] == Direction.NONE.value
        )
        world_T[:, :, Channel.ENTITIES.value][belt_mask & no_direction_mask] = (
            empty_entity_value
        )

        # # Ensure belts don't have recipes
        # belt_entity_value = str2ent('transport_belt').value
        # belt_entities = (world_T[:, :, Channel.ENTITIES.value] == belt_entity_value)
        # world_T[:, :, Channel.ITEMS.value][belt_entities] = empty_entity_value

        # # Ensure all empty entities have no recipe, no direction
        # no_entity = (world_T[:, :, Channel.ENTITIES.value] == empty_entity_value)
        # world_T[:, :, Channel.ITEMS.value][no_entity] = empty_entity_value
        # world_T[:, :, Channel.DIRECTION.value][no_entity] = Direction.NONE.value

        # Ensure the model can't just overwrite existing factories with a simpler thing.
        tworld = og_world.clone().detach().to(torch.int64)
        # tworld = torch.tensor(og_world, dtype=torch.int64)
        original_had_something = (
            tworld[:, :, Channel.ENTITIES.value] != empty_entity_value
        )
        for ch in list(Channel):
            replacements = tworld[:, :, ch.value][original_had_something]
            world_T[:, :, ch.value][original_had_something] = replacements
        return world_T


    def get_min_belts(world_CWH):
        assert world_CWH.shape[1] == world_CWH.shape[2], (
            "Wrong shape: {world_CWH.shape}"
        )
        C, W, H = world_CWH.shape

        stack_inserter_id = str2ent("stack_inserter").value
        bulk_inserter_id = str2ent("bulk_inserter").value
        coords1 = torch.where(
            world_CWH[Channel.ENTITIES.value] == bulk_inserter_id
        )
        assert len(coords1[0]) == len(coords1[1]) == 1, (
            f"Expected 1 bulk inserter, found {coords1} in world {world_CWH}"
        )
        w1, h1 = coords1[0][0], coords1[1][0]

        coords2 = torch.where(
            world_CWH[Channel.ENTITIES.value] == stack_inserter_id
        )
        assert len(coords2[0]) == len(coords2[1]) == 1, (
            f"Expected 1 stack inserter, found {coords2} in world {world_CWH}"
        )
        w2, h2 = coords2[0][0], coords2[1][0]

        # we want an estimate for how many belts are required, so get the
        # coords of the transport belt tile closest to the source/sink
        w1 = torch.clamp(w1, 1, W - 2)
        h1 = torch.clamp(h1, 1, H - 2)
        w2 = torch.clamp(w2, 1, W - 2)
        h2 = torch.clamp(h2, 1, H - 2)

        manhat_dist = torch.abs(w1 - w2) + torch.abs(h1 - h2)
        min_belts = manhat_dist + 1
        return min_belts


    def get_new_world(seed, n=6, min_belts=None, source_item=None, sink_item=None):
        stack_inserter_value = str2ent("stack_inserter").value
        bulk_inserter_value = str2ent("bulk_inserter").value
        empty_value = str2ent("empty").value

        if seed is not None:
            np.random.seed(seed)
        assert min_belts != [1], f"min_belts of [1] is sometimes unsatisfiable"
        if min_belts is None:
            min_belts = list(range(0, 64))
        w = new_world(width=n, height=n)
        boundary_tiles = []
        for i in range(n):
            for j in range(n):
                if i in (0, n - 1) and j in (0, n - 1):
                    continue
                if i in (0, n - 1) or j in (0, n - 1):
                    boundary_tiles.append((i, j))

        # Put a source and a sink on one of the boundaries
        source = boundary_tiles[np.random.choice(len(boundary_tiles))]
        w[source[0], source[1], Channel.ENTITIES.value] = stack_inserter_value
        # TODO not the most efficient, but it'll be okay for now
        limit = 1000
        while limit > 0:
            # Find random location for the sink
            sink = boundary_tiles[np.random.choice(len(boundary_tiles))]
            # Ensure the sink isn't on top of the source
            if source == sink:
                continue
            # Add the sink to the world
            w[sink[0], sink[1], Channel.ENTITIES.value] = bulk_inserter_value
            # Calculate the manhatten distance
            min_belt = get_min_belts(torch.tensor(w).permute(2, 0, 1))
            # If manhatten distance is acceptable and source != sink, we've got
            # our world
            if (source != sink) and (min_belt in min_belts):
                break
            # else, remove the sink from the world and try again
            w[sink[0], sink[1], Channel.ENTITIES.value] = empty_value
        assert limit > 0, "Infinite loop blocked"

        # Add the source + sink to the world
        # w[source[0], source[1], Channel.ITEMS.value] = str2ent('electronic_circuit').value
        # w[sink[0], sink[1], Channel.ITEMS.value] = str2ent('electronic_circuit').value
        if source_item is not None:
            w[source[0], source[1], Channel.ITEMS.value] = source_item
        if sink_item is not None:
            w[sink[0], sink[1], Channel.ITEMS.value] = sink_item

        # Figure out the direction of the source + sink
        for x, y, is_source in [(*source, True), (*sink, False)]:
            if x == 0:
                w[x, y, Channel.DIRECTION.value] = (
                    Direction.EAST if is_source else Direction.WEST
                ).value
            if x == n - 1:
                w[x, y, Channel.DIRECTION.value] = (
                    Direction.WEST if is_source else Direction.EAST
                ).value
            if y == 0:
                w[x, y, Channel.DIRECTION.value] = (
                    Direction.SOUTH if is_source else Direction.NORTH
                ).value
            if y == n - 1:
                w[x, y, Channel.DIRECTION.value] = (
                    Direction.NORTH if is_source else Direction.SOUTH
                ).value

        return torch.tensor(w).to(torch.float)


    def sample_world(probabilities):
        assert torch.is_tensor(probabilities), (
            f"probabilities is {type(probabilities)} not torch.Tensor"
        )
        distribution = Categorical(probs=probabilities)
        samples = distribution.sample()
        # make the directions fit the expected values
        d_direction = samples[:, :, Channel.DIRECTION.value]
        mask = d_direction > 0
        d_direction[mask] = d_direction[mask] * 4 - 4
        d_direction[~mask] = -1
        return samples


    def eval_model(actor, critic, pars, num_evaluations=1_000, pbar=False):
        torch.manual_seed(42)
        evals = []
        iterator = torch.randint(0, 2**16 - 1, (num_evaluations,)).tolist()
        if pbar:
            iterator = mo.status.progress_bar(iterator)
        for seed in iterator:
            original_world = get_new_world(seed, n=4)
            probabilities = actor(original_world)
            normalised_world = normalise_world(
                sample_world(probabilities), original_world
            )
            # value = critic(normalised_world)
            value = critic(normalised_world.to(torch.float))
            # Maybe having throughput being calculated as a black box is the problem?
            throughput = torch.tensor(
                funge_throughput(normalised_world)[0] / 15.0,
                dtype=value.dtype,
            )
            num_entities = (
                normalised_world[:, :, Channel.ENTITIES.value]
                != str2ent("empty").value
            ).sum()
            evals.append(
                {
                    "seed": seed,
                    "original_world": original_world,
                    "normalised_world": normalised_world,
                    "throughput": throughput,
                    "num_entities": num_entities,
                }
            )

        avg_throughput = sum([eval["throughput"] for eval in evals]) / len(evals)
        avg_num_entities = sum([eval["num_entities"] for eval in evals]) / len(
            evals
        )

        return evals, avg_throughput, float(avg_num_entities)


    def funge_throughput(world, debug=False):
        assert torch.is_tensor(world), f"world is {type(world)}, not a tensor"
        assert len(world.shape) == 3, (
            f"Expected world to have 3 dimensions, but is of shape {world.shape}"
        )
        assert world.shape[0] == world.shape[1], (
            f"Expected world to be square, but is of shape {world.shape}"
        )
        try:
            throughput, num_unreachable = calc_throughput(world2graph(world, debug=debug), debug=debug)
            if len(throughput) == 0:
                return 0, num_unreachable
            actual_throughput = list(throughput.values())[0]
            assert actual_throughput < float("inf"), (
                f"throughput is +inf, probably a bug, world is: {torch.tensor(world).permute(2, 0, 1)}"
            )
            return actual_throughput, num_unreachable
        except AssertionError:
            breakpoint()
            traceback.print_exc()
            return 0, 0


    def world2graph(world_WHC, debug=False):
        assert torch.is_tensor(world_WHC), (
            f"world is {type(world_WHC)}, not a tensor"
        )
        assert len(world_WHC.shape) == 3, (
            f"Expected world to have 3 dimensions, but is of shape {world_WHC.shape}"
        )
        assert world_WHC.shape[0] == world_WHC.shape[1], (
            f"Expected world to be square, but is of shape {world_WHC.shape}"
        )
        world_WHC = world_WHC.numpy()
        G = nx.DiGraph()

        def dbg(s):
            if debug:
                print(s)

        W, H, C = world_WHC.shape
        for x in range(W):
            for y in range(H):
                e = entities[world_WHC[x, y, Channel.ENTITIES.value]]
                if e.name == "empty":
                    continue

                # TODO somehow `item` is 0 even though it should be disallowed
                item = items[world_WHC[x, y, Channel.ITEMS.value]]
                d = Direction(world_WHC[x, y, Channel.DIRECTION.value])

                input_ = {}
                output = {}
                if e.name == "stack_inserter":
                    output = {item.name: float("inf")}

                self_name = f"{e.name}\n@{x},{y}"
                G.add_node(
                    self_name,
                    input_=input_,
                    output=output,
                    recipe=item.name if "assembling_machine" in e.name else None,
                )
                dbg(
                    f"Created node {repr(self_name)}: {G.nodes[self_name]}, direction is {d}, recipe is {item.name}"
                )

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
                elif d == Direction.NONE:
                    # If there's no direction, logic will be handled in the handlers
                    src = [x, y]
                    dst = [x, y]
                else:
                    assert False, f"Can't handle direction {d} for entity {e}"
                # Connect the inserters' & belts' nodes
                # Here we connect nodes twice, once on source->me and again on me->destination.
                x_src_valid = 0 <= src[0] < len(world_WHC)
                y_src_valid = 0 <= src[1] < len(world_WHC[0])
                x_dst_valid = 0 <= dst[0] < len(world_WHC)
                y_dst_valid = 0 <= dst[1] < len(world_WHC[0])
                if "inserter" in e.name:
                    if x_src_valid and y_src_valid:
                        src_entity = entities[
                            world_WHC[src[0], src[1], Channel.ENTITIES.value]
                        ]
                        src_direction = Direction(
                            world_WHC[src[0], src[1], Channel.DIRECTION.value]
                        )
                        src_not_empty = src_entity.name != "empty"
                        if src_not_empty:
                            G.add_edge(
                                f"{src_entity.name}\n@{src[0]},{src[1]}",
                                f"{e.name}\n@{x},{y}",
                            )
                            dbg(
                                f"{src_entity.name}@{src[0]},{src[1]} -> {e.name}@{x},{y}"
                            )
                    if x_dst_valid and y_dst_valid:
                        dst_entity = entities[
                            world_WHC[dst[0], dst[1], Channel.ENTITIES.value]
                        ]
                        # TODO: This doesn't allow for the case where
                        # an inserter can put things on the ground to
                        # be picked up by another inserter
                        dst_is_insertable = (
                            "belt" in dst_entity.name
                            or "assembling_machine" in dst_entity.name
                        )
                        if dst_is_insertable:
                            G.add_edge(
                                f"{e.name}\n@{x},{y}",
                                f"{dst_entity.name}\n@{dst[0]},{dst[1]}",
                            )
                            dbg(
                                f"{e.name}@{x},{y} -> {dst_entity.name}@{dst[0]},{dst[1]}"
                            )

                elif "transport_belt" in e.name:
                    if x_src_valid and y_src_valid:
                        src_entity = entities[
                            world_WHC[src[0], src[1], Channel.ENTITIES.value]
                        ]
                        src_direction = Direction(
                            world_WHC[src[0], src[1], Channel.DIRECTION.value]
                        )
                        src_misc = Misc(
                            world_WHC[src[0], src[1], Channel.MISC.value]
                        )
                        src_is_beltish = (
                            "belt" in src_entity.name
                            # Check the other belt is directly behind me and
                            # pointing the same direction
                            and src_direction == d
                            # Check that the other is not a downwards underground belt
                            and not (
                               "underground_belt" in src_entity.name
                                and src_misc.value == Misc.UNDERGROUND_DOWN
                            )
                        )
                        if src_is_beltish:
                            G.add_edge(
                                f"{src_entity.name}\n@{src[0]},{src[1]}",
                                f"{e.name}\n@{x},{y}",
                            )
                            dbg(
                                f"{src_entity.name}@{src[0]},{src[1]} -> {e.name}@{x},{y}",
                            )

                    if x_dst_valid and y_dst_valid:
                        dst_entity = entities[
                            world_WHC[dst[0], dst[1], Channel.ENTITIES.value]
                        ]
                        dst_direction = Direction(
                            world_WHC[dst[0], dst[1], Channel.DIRECTION.value]
                        )
                        dst_misc = Misc(
                            world_WHC[dst[0], dst[1], Channel.MISC.value]
                        )
                        dst_not_empty = dst_entity.name != "empty"
                        dst_is_belt = "belt" in dst_entity.name
                        opposite = Direction.SOUTH.value - Direction.NORTH.value
                        dst_opposing_belt = (
                            dst_is_belt and abs(dst_direction.value - d.value) == opposite
                        )
                        # various underground belt checks
                        # TODO figure out these checks
                        dest_underground_ok = (
                            True
                            # "underground_belt" not in dst_entity.name
                            # or (
                            #     (dst_direction.value == d.value and dst_misc.value == Misc.UNDERGROUND_DOWN)
                            # )
                        )
                        if dst_is_belt and not dst_opposing_belt and dest_underground_ok:
                            G.add_edge(
                                f"{e.name}\n@{x},{y}",
                                f"{dst_entity.name}\n@{dst[0]},{dst[1]}",
                            )
                            dbg(
                                f"{e.name}@{x},{y} -> {dst_entity.name}@{dst[0]},{dst[1]}",
                            )

                # connect up the assembling machines
                elif "assembling_machine" in e.name:
                    dbg(f"Connecting assembler {e}")
                    # search the blocks around the 3x3 assembling machine for inputs
                    for dx in range(-1, 4):
                        if not (0 <= x + dx < W):
                            # omit tiles outside the world
                            continue
                        for dy in range(-1, 4):
                            if not (0 <= y + dy < H):
                                # omit tiles outside the world
                                continue
                            if 0 <= dx < 3 and 0 <= dy < 3:
                                # omit tiles inside the assembler
                                continue
                            if dx in (-1, 3) and dy in (-1, 3):
                                # Omit corners
                                continue
                            other_e = entities[
                                world_WHC[x + dx, y + dy, Channel.ENTITIES.value]
                            ]
                            other_d = Direction(
                                world_WHC[x + dx, y + dy, Channel.DIRECTION.value]
                            )
                            # Only inserters can insert into an assembling machine
                            if "inserter" not in other_e.name:
                                continue
                            #                         if f"{other_e.name}\n@{x + dx},{y + dy}" == 'inserter\n@2,0':
                            #                             print(other_e, e, other_d, dy, dx)

                            # dbg(f"{dx=},{dy=}")
                            # dbg(f"{x+dx=},{y+dy=},{other_e=}")
                            other_str = f"{other_e.name}\n@{x + dx},{y + dy}"
                            self_str = f"{e.name}\n@{x},{y}"

                            # Direction is self -> other
                            if (
                                (other_d == Direction.NORTH and dy < 0)
                                or (other_d == Direction.SOUTH and dy > 0)
                                or (other_d == Direction.WEST and dx < 0)
                                or (other_d == Direction.EAST and dx > 0)
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
                            dbg(f"{repr(src)} -> {repr(dst)}")

                elif "underground_belt" in e.name:
                    m = Misc(world_WHC[x, y, Channel.MISC.value])
                    # Only down-undergrounds look for their upgoing counterparts,
                    # not the other way aroud
                    assert e.name == "underground_belt", (
                        "don't know how to handle other undergrounds yet"
                    )
                    if m == Misc.UNDERGROUND_DOWN:
                        max_delta = 6
                    elif m == Misc.UNDERGROUND_UP:
                        max_delta = 1
                    else:
                        assert False, f"Underground belts must be either UP or DOWN, not {m}"
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
                        x_valid = 0 <= dst[0] < len(world_WHC)
                        y_valid = 0 <= dst[1] < len(world_WHC[0])
                        if x_valid and y_valid:
                            dst_entity = entities[
                                world_WHC[dst[0], dst[1], Channel.ENTITIES.value]
                            ]
                            going_underground = (
                                dst_entity.name == "underground_belt"
                                and m == Misc.UNDERGROUND_DOWN
                            )
                            cxn_to_belt = (
                                "transport_belt" in dst_entity.name
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


    def generate_lesson(
        size=5,
        kind=LessonKind.MOVE_ONE_ITEM,
        num_missing_entities=float('inf'),
        seed=None,
        random_item=False,
        max_entities=float('inf')
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        min_entities_required = None
        world_CWH = torch.tensor(new_world(width=size, height=size)).permute(
            2, 0, 1
        )
        C, W, H = world_CWH.shape
        # No idea why, but there doing kind == LessonKind.MOVE_ONE_ITEM doesn't evaluate to true...
        if kind.value == LessonKind.MOVE_ONE_ITEM.value:
            # Choose a random source/sink
            original_count = 500 # usually require ~50, sometimes up to 100
            count = original_count
            while count > 0:
                count -= 1
                pos1 = torch.randint(0, H * W, (1,))
                pos2 = torch.randint(0, H * W, (1,))
                if pos1 == pos2:
                    # restart the loop until we find non-equal source/sink
                    continue

                source_WH = divmod(pos1.item(), W)
                sink_WH = divmod(pos2.item(), W)
                source_dir = random.choice(
                    [d for d in Direction if d != Direction.NONE]
                )
                sink_dir = random.choice(
                    [d for d in Direction if d != Direction.NONE]
                )

                if random_item:
                    item_value = random.choice([v.value for k, v in items.items()])
                else:
                    item_value = str2item('electronic_circuit').value

                world_CWH[Channel.ENTITIES.value, source_WH[0], source_WH[1]] = (
                    str2ent("source").value
                )
                world_CWH[Channel.ENTITIES.value, sink_WH[0], sink_WH[1]] = (
                    str2ent("sink").value
                )

                world_CWH[Channel.ITEMS.value, source_WH[0], source_WH[1]] = (
                    item_value
                )
                world_CWH[Channel.ITEMS.value, sink_WH[0], sink_WH[1]] = item_value

                world_CWH[Channel.DIRECTION.value, source_WH[0], source_WH[1]] = (
                    source_dir.value
                )
                world_CWH[Channel.DIRECTION.value, sink_WH[0], sink_WH[1]] = (
                    sink_dir.value
                )
                # print(world_CWH)

                paths = find_belt_paths_with_source_sink_orient(
                    entities=world_CWH[Channel.ENTITIES.value],
                    directions=world_CWH[Channel.DIRECTION.value],
                )
                # Remove all paths that would require placing too many entities
                paths = list(filter(lambda p: len(p) <= max_entities, paths))

                if len(paths) == 0:
                    world_CWH = torch.tensor(
                        new_world(width=size, height=size)
                    ).permute(2, 0, 1)
                    # Restart the loop until we get a source+sink that can be connected
                    continue
                else:
                    # Choose a valid path at random and add it to the map
                    if num_missing_entities != float('inf'):
                        chosen_path = random.choice(paths)
                        min_entities_required = len(chosen_path)
                        for x, y, d in chosen_path:
                            world_CWH[Channel.ENTITIES.value, x, y] = str2ent(
                                "transport_belt"
                            ).value
                            world_CWH[Channel.DIRECTION.value, x, y] = d.value
                    else:
                        min_entities_required = min([len(p) for p in paths])

                # Randomly remove some number of transport belts from the map

                if num_missing_entities != float('inf'):
                    entity_locs = (
                        (world_CWH[Channel.ENTITIES.value] != str2ent("source").value)
                        & (world_CWH[Channel.ENTITIES.value] != str2ent("sink").value)
                        & (world_CWH[Channel.ENTITIES.value] != str2ent("empty").value)
                    ).nonzero(as_tuple=False)
                    num_samples = min(num_missing_entities, len(entity_locs))
                    samples = [] if num_samples == 0 else random.sample(list(entity_locs), num_samples)
                    min_entities_required = num_samples
                    # print(f"Removing {num_samples} entities")
                    # if num_samples != 0:
                    #     breakpoint()
                    for x, y in samples:
                        # print(f"  Removing entity from {x},{y}, num_samples: {num_samples}")
                        world_CWH[Channel.ENTITIES.value, x, y] = str2ent(
                            "empty"
                        ).value
                        world_CWH[Channel.DIRECTION.value, x, y] = Direction.NONE.value
                        world_CWH[Channel.ITEMS.value, x, y] = str2item("empty").value
                        world_CWH[Channel.MISC.value, x, y] = Misc.NONE.value
                break
            if count == 0:
                raise Exception(f"Failed to find valid lesson after {original_count} attempts")
        else:
            raise Exception(f"Can't handle {kind}")
        # print(f"Required {original_count-count} (of {original_count}) attempts to generate world ({100- count/original_count*100:.2f}%)")

        return world_CWH, min_entities_required


    def find_belt_paths_with_source_sink_orient(
        entities: torch.Tensor,
        directions: torch.Tensor,
        source_value: int = str2ent("source").value,
        sink_value: int = str2ent("sink").value,
    ) -> List[List[Tuple[int, int, Direction]]]:
        """
        Find all shortest belt‐placement paths that start immediately in front of the source
        (in its facing direction) and end immediately behind the sink (opposite its facing),
        without overwriting the source or sink cells themselves.  Returns a list of paths,
        each path being a list of (row, col, Direction) tuples describing where to place
        each belt and which way it should face.  If no valid path exists, returns [].
        """
        # 1. sanity checks
        if (
            entities.ndim != 2
            or directions.ndim != 2
            or entities.shape != directions.shape
        ):
            raise ValueError(
                "entities and directions must be 2D tensors of the same shape"
            )
        H, W = entities.shape

        # 2. locate source & sink
        src_pos = (entities == source_value).nonzero(as_tuple=False)
        sink_pos = (entities == sink_value).nonzero(as_tuple=False)
        if len(src_pos) != 1 or len(sink_pos) != 1:
            raise ValueError("must have exactly one source and one sink")
        src = tuple(src_pos[0].tolist())
        sink = tuple(sink_pos[0].tolist())

        # 3. get facing directions
        src_dir = Direction(directions[src].item())
        sink_dir = Direction(directions[sink].item())
        if src_dir == Direction.NONE or sink_dir == Direction.NONE:
            return []  # cannot start or end if orientation is NONE

        # 4. compute start/end cells
        dr_s, dc_s = DIR_TO_DELTA[src_dir]
        start = (src[0] + dr_s, src[1] + dc_s)
        dr_k, dc_k = DIR_TO_DELTA[sink_dir]
        end = (sink[0] - dr_k, sink[1] - dc_k)
        # print(f'source is at {src} facing {Direction(directions[src].item())}, start is at {start}')
        # print(f'sink is at {sink} facing {Direction(directions[sink].item())}, end is at {end}')

        # 5. check bounds & not overlapping source/sink
        def in_bounds(cell):
            r, c = cell
            return 0 <= r < H and 0 <= c < W

        if not in_bounds(start) or not in_bounds(end):
            return []
        if start == src or start == sink or end == src or end == sink:
            return []

        # 6. BFS from start to end, avoiding src & sink cells
        deltas = list(DIR_TO_DELTA.values())
        dist = torch.full((H, W), -1, dtype=torch.int64)
        parents = defaultdict(list)
        q = deque([start])
        dist[start] = 0

        while q:
            r, c = q.popleft()
            if (r, c) == end:
                # add end to the thing
                break
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if not in_bounds((nr, nc)):
                    continue
                if (nr, nc) in (src, sink):
                    continue
                if dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    parents[(nr, nc)].append((r, c))
                    q.append((nr, nc))
                elif dist[nr, nc] == dist[r, c] + 1:
                    parents[(nr, nc)].append((r, c))

        if dist[end] < 0:
            return []

        # 7. backtrack all shortest paths
        all_paths: List[List[Tuple[int, int, Direction]]] = []

        def backtrack(cell, rev_path):
            if cell == start:
                path = [start] + list(reversed(rev_path))
                belts: List[Tuple[int, int, Direction]] = []
                # build belt placements along the path
                for (r1, c1), (r2, c2) in zip(path, path[1:]):
                    dr, dc = (r2 - r1, c2 - c1)
                    # find which Direction matches this delta
                    for d, delta in DIR_TO_DELTA.items():
                        if delta == (dr, dc):
                            belts.append((r1, c1, d))
                            break
                all_paths.append(belts + [(end[0], end[1], sink_dir)])
                return
            for p in parents[cell]:
                backtrack(p, rev_path + [cell])

        backtrack(end, [])
        return all_paths


    mo.md("Functions")
    return (
        calc_throughput,
        funge_throughput,
        generate_lesson,
        get_new_world,
        new_world,
        normalise_world,
        plot_flow_network,
        str2ent,
        str2item,
        world2graph,
        world2html,
    )


@app.cell
def _(LessonKind, generate_lesson, world2html):
    def __():
        # np.random.seed(42)
        # torch.manual_seed(42)
        # random.seed(42)

        world_CWH, _ = generate_lesson(
            size=16, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=float('inf')
        )

        # print(world_CWH[Channel.ENTITIES.value])
        # print(world_CWH[Channel.DIRECTION.value])

        return world2html(world_WHC=world_CWH.permute(1, 2, 0))


    __()
    return


@app.cell
def _(
    Channel,
    Direction,
    Misc,
    calc_throughput,
    new_world,
    np,
    plot_flow_network,
    str2ent,
    str2item,
    torch,
    traceback,
    world2graph,
    world2html,
):
    def blank2(n=7, seed=42):
        asm_mach_value = str2ent("assembling_machine_1").value
        belt_value = str2ent("transport_belt").value
        bulk_inserter_value = str2ent("bulk_inserter").value
        empty_value = str2ent("empty").value
        inserter_value = str2ent("inserter").value
        stack_inserter_value = str2ent("stack_inserter").value
        underground_value = str2ent("underground_belt").value

        cable_value = str2item("copper_cable").value
        copper_value = str2item("copper_plate").value
        iron_value = str2item("iron_plate").value
        green_circuit_value = str2item("electronic_circuit").value

        np.random.seed(seed)
        world_WHC = torch.tensor(new_world(width=12, height=12))

        world_WHC[0, 0, Channel.ENTITIES.value] = stack_inserter_value
        world_WHC[0, 0, Channel.ITEMS.value] = copper_value
        world_WHC[0, 0, Channel.DIRECTION.value] = Direction.EAST.value
        world_WHC[0, 10, Channel.ENTITIES.value] = stack_inserter_value
        world_WHC[0, 10, Channel.ITEMS.value] = iron_value
        world_WHC[0, 10, Channel.DIRECTION.value] = Direction.EAST.value
        world_WHC[0, 11, Channel.ENTITIES.value] = bulk_inserter_value
        world_WHC[0, 11, Channel.ITEMS.value] = green_circuit_value
        world_WHC[0, 11, Channel.DIRECTION.value] = Direction.WEST.value
        print(world_WHC.permute(2, 0, 1))

        world_WHC[0, 2, Channel.ENTITIES.value] = asm_mach_value
        world_WHC[3, 2, Channel.ENTITIES.value] = asm_mach_value
        world_WHC[6, 2, Channel.ENTITIES.value] = asm_mach_value
        world_WHC[0, 2, Channel.ITEMS.value] = cable_value
        world_WHC[3, 2, Channel.ITEMS.value] = cable_value
        world_WHC[6, 2, Channel.ITEMS.value] = cable_value

        world_WHC[1, 6, Channel.ENTITIES.value] = asm_mach_value
        world_WHC[1, 6, Channel.ITEMS.value] = green_circuit_value
        world_WHC[5, 6, Channel.ENTITIES.value] = asm_mach_value
        world_WHC[5, 6, Channel.ITEMS.value] = green_circuit_value

        world_WHC[2, 10, Channel.ENTITIES.value] = underground_value
        world_WHC[2, 10, Channel.DIRECTION.value] = Direction.EAST.value
        world_WHC[2, 10, Channel.MISC.value] = Misc.UNDERGROUND_DOWN.value

        world_WHC[6, 10, Channel.ENTITIES.value] = underground_value
        world_WHC[6, 10, Channel.DIRECTION.value] = Direction.EAST.value
        world_WHC[6, 10, Channel.MISC.value] = Misc.UNDERGROUND_UP.value

        inserter_locs = [
            (1, 1, Direction.SOUTH),
            (4, 1, Direction.SOUTH),
            (7, 1, Direction.SOUTH),
            (2, 5, Direction.SOUTH),
            (3, 5, Direction.SOUTH),
            (5, 5, Direction.SOUTH),
            (6, 5, Direction.SOUTH),
            (2, 9, Direction.NORTH),
            (3, 9, Direction.SOUTH),
            (5, 9, Direction.SOUTH),
            (6, 9, Direction.NORTH),
        ]
        for x, y, d in inserter_locs:
            world_WHC[x, y, Channel.ENTITIES.value] = inserter_value
            world_WHC[x, y, Channel.DIRECTION.value] = d.value

        belt_locs = [
            (1, 0, Direction.EAST),
            (2, 0, Direction.EAST),
            (3, 0, Direction.EAST),
            (4, 0, Direction.EAST),
            (5, 0, Direction.EAST),
            (6, 0, Direction.EAST),
            (7, 0, Direction.EAST),
            (1, 10, Direction.EAST),
            (7, 10, Direction.EAST),
            (1, 11, Direction.WEST),
            (2, 11, Direction.WEST),
            (3, 11, Direction.WEST),
            (3, 10, Direction.SOUTH),
            (4, 10, Direction.WEST),
            (5, 10, Direction.WEST),
        ]
        for x, y, d in belt_locs:
            world_WHC[x, y, Channel.ENTITIES.value] = belt_value
            world_WHC[x, y, Channel.DIRECTION.value] = d.value

        G = world2graph(world_WHC, debug=True)
        try:
            thput, _num_unreachable = calc_throughput(G, debug=False)
            print(thput)
        except Exception as e:
            print("err")
            print(traceback.format_exc())

        print(world_WHC.permute(2, 0, 1))
        return world2html(world_WHC), plot_flow_network(G)

        # return , world_WHC.permute(2, 1, 0)


    blank2()
    return


@app.cell
def _(items):
    items
    return


@app.cell
def _():
    import gymnasium as gym
    from ppo import AgentCNN, FactorioEnv

    path = "cleanrl/artifacts/agent-1.000000-factorion-FactorioEnv-v0__ppo__1__1745516456.pt"
    path = "cleanrl/artifacts/agent-0.994240-factorion-FactorioEnv-v0__ppo__1__2025-04-24T22-15-15.pt"

    path = "cleanrl/artifacts/agent-1.023271-factorion-FactorioEnv-v0__ppo__1__2025-04-25T04-00-29.pt"

    path = "cleanrl/artifacts/agent-0.966447-factorion-FactorioEnv-v0__ppo__1__2025-04-25T11-17-30.pt"
    path = "cleanrl/agent-1.023271-factorion-FactorioEnv-v0__ppo__1__2025-04-25T04-00-29.pt"


    def make_env():
        def _thunk():
            return FactorioEnv()

        return _thunk


    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(4)])

    # agent = AgentCNN(envs,
    #     chan1=32,
    #     chan2=32,
    #     chan3=32,
    #     flat_dim=32,
    # )
    # agent.load_state_dict(torch.load(path, weights_only=False))
    # agent.eval()
    return


@app.cell
def _(
    Channel,
    agent,
    funge_throughput,
    get_new_world,
    normalise_world,
    torch,
    world2html,
):
    size = 5
    world_CWH = get_new_world(seed=None, n=size).permute(2, 0, 1)
    world_CWH


    example_input = torch.randn(1, agent.channels, agent.width, agent.height)
    with torch.no_grad():
        action_BCWH, logprob, entropy, value = agent.get_action_and_value(
            world_CWH.unsqueeze(0)
        )

    action_CWH = action_BCWH[0, :, :, :]
    dir_CWH = action_CWH[Channel.DIRECTION.value, :, :]
    mask = dir_CWH > 0
    dir_CWH[mask] = dir_CWH[mask] * 4 - 4
    dir_CWH[~mask] = -1
    action_WHC = torch.tensor(action_CWH).permute(1, 2, 0)
    world_WHC = world_CWH.permute(1, 2, 0)
    normalised_world_WHC = normalise_world(action_WHC, world_WHC)
    throughput, num_unreachable = funge_throughput(
        normalised_world_WHC, debug=False
    )
    throughput /= 15.0
    frac_reachable = 1.0 - float(num_unreachable) / (size * size)
    normalised_world_CWH = normalised_world_WHC.permute(2, 0, 1)

    hallucination_rate = (
        normalised_world_WHC != action_CWH.permute(1, 2, 0)
    ).sum() / action_CWH.numel()

    (
        "(white inserter is source, green inserter is sink)",
        "---Initial world---",
        world2html(world_CWH.permute(1, 2, 0)),
        "---Before normalisation---",
        world2html(action_CWH.permute(1, 2, 0)),
        "---Predicted world---",
        world2html(normalised_world_WHC),
        f"Throughput: {throughput * 100:.0f}%, hallu: {hallucination_rate:.2f}, frac: {frac_reachable}",
    )
    return


if __name__ == "__main__":
    app.run()
