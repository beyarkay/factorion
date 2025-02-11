# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "factorio-draftsman",
#     "tqdm",
# ]
# ///
from draftsman.entity import new_entity
from draftsman.blueprintable import Blueprint
from draftsman.constants import Direction
import os
import json
import itertools
import tqdm

def dict_product(**kwargs):
    keys, values = zip(*kwargs.items())
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))

def direction_to_str(direction):
    if direction == Direction.NORTH:
        return "North"
    elif direction == Direction.EAST:
        return "East"
    elif direction == Direction.SOUTH:
        return "South"
    elif direction == Direction.WEST:
        return "West"



def pluralize(singular, plural):
    return (lambda x: singular if x == 1 else plural)

lane_s = pluralize("lane", "lanes")

def belts_in_a_line(
    num_lanes=1,
    belts_per_lane=16,
    belt_type="transport-belt",
    direction=Direction.EAST,
    verbose=False
):
    blueprint = Blueprint()
    belt_s = pluralize(belt_type, belt_type + "s")

    for lane_idx in range(0, num_lanes):
        # Add i transport belts in a row
        for belt_idx in range(belts_per_lane):
            if direction == Direction.EAST or direction == Direction.WEST:
                pos = (belt_idx, lane_idx)
            elif direction == Direction.NORTH or direction == Direction.SOUTH:
                pos = (lane_idx, belt_idx)
            else:
                raise Exception(f"Direction {direction} invalid")

            blueprint.entities.append(new_entity(
                belt_type,
                position=pos,
                direction=direction,
            ))

        direction_str = direction_to_str(direction)

        belt_type_str = belt_s(belts_per_lane).replace("-", " ")
        # Set the description field
        blueprint.description = f"A JSON factorio blueprint: {num_lanes} {lane_s(num_lanes)} of {belts_per_lane} {belt_type_str} going {direction_str}."
    if verbose:
        print(blueprint.description, '\n', blueprint.to_string())
    return (
        f"num_lanes={num_lanes},belts_per_lane={belts_per_lane},belt_type={belt_type},direction={direction}.json",
        blueprint
    )


def generate_belts_in_a_line():
    # Make sure the output directory exists
    if not os.path.exists('blueprints/autogen/belts_in_a_line'):
        os.makedirs('blueprints/autogen/belts_in_a_line')

    num_lanes_iter = range(1, 9)
    belts_per_lane_iter = range(1, 33)
    belt_types_iter = ['transport-belt', 'fast-transport-belt']
    direction_iter = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]

    iterator = dict_product(
        num_lanes=num_lanes_iter,
        belts_per_lane=belts_per_lane_iter,
        belt_type=belt_types_iter,
        direction=direction_iter
    )
    total = len(num_lanes_iter) * len(belts_per_lane_iter) * len(belt_types_iter) * len(direction_iter)

    print("generating belts_in_a_line...\n")
    for kwargs in tqdm.tqdm(iterator, total=total):
        filename, blueprint = belts_in_a_line(**kwargs)
        with open(f"blueprints/autogen/belts_in_a_line/{filename}", 'w') as f:
            f.write(json.dumps(blueprint.to_dict(), indent=2))


def belts_at_a_corner(
    num_lanes=1,
    belts_before_corner=4,
    belts_after_corner=8,
    belt_type="transport-belt",
    start_direction=Direction.EAST,
    turn_left=True,
    verbose=False
):
    blueprint = Blueprint()

    finsh_direction = Direction((int(start_direction) + (-4 if turn_left else +4)) % 16)

    for lane_idx in range(num_lanes):
        pos = [0, 0]
        for belt_idx in range(belts_before_corner+1 + lane_idx): # +1 for the corner
            if start_direction == Direction.EAST:
                pos[0] = belt_idx
                pos[1] = lane_idx if turn_left else -lane_idx
            elif start_direction == Direction.WEST:
                pos[0] = -belt_idx
                pos[1] = -lane_idx if turn_left else lane_idx
            elif start_direction == Direction.NORTH:
                pos[0] = lane_idx if turn_left else -lane_idx
                pos[1] = -belt_idx
            elif start_direction == Direction.SOUTH:
                pos[0] = -lane_idx if turn_left else lane_idx
                pos[1] = belt_idx
            else:
                raise Exception(f"Direction {start_direction} invalid")

            if belt_idx != belts_before_corner + lane_idx:
                blueprint.entities.append(new_entity(
                    belt_type,
                    position=pos,
                    direction=start_direction,
                ))
            else:
                blueprint.entities.append(new_entity(
                    belt_type,
                    position=pos,
                    direction=finsh_direction,
                ))

        for belt_idx in range(1, belts_after_corner + 1 + lane_idx):
            if finsh_direction == Direction.EAST:
                pos[0] = belt_idx - lane_idx
                pos[1] = (1 if turn_left else -1) * (belts_before_corner + lane_idx)
            elif finsh_direction == Direction.WEST:
                pos[0] = -belt_idx + lane_idx
                pos[1] = (-1 if turn_left else 1) * (belts_before_corner + lane_idx)
            elif finsh_direction == Direction.NORTH:
                pos[1] = -belt_idx + lane_idx
                pos[0] = (1 if turn_left else -1) * (belts_before_corner + lane_idx)
            elif finsh_direction == Direction.SOUTH:
                pos[1] = belt_idx -lane_idx
                pos[0] = (-1 if turn_left else 1) * (belts_before_corner + lane_idx)
            else:
                raise Exception(f"Direction {finsh_direction} invalid")
            blueprint.entities.append(new_entity(
                belt_type,
                position=pos,
                direction=finsh_direction,
            ))

    belt_type_str = belt_type.replace("-", " ")
    turn_str = "left" if turn_left else "right"
    start_direction_str = direction_to_str(start_direction)
    finsh_direction_str = direction_to_str(finsh_direction)

    blueprint.description = (
        f"A JSON factorio blueprint: {num_lanes} {lane_s(num_lanes)} of "
        f"{belt_type_str}s going {start_direction_str} for {belts_before_corner} tiles "
        f"and then turning {turn_str} to continue {finsh_direction_str} for "
        f"{belts_after_corner} tiles."
    )
    if verbose:
        print(f"{blueprint.description}\n{blueprint.to_string()}")
    return (
        f"{num_lanes=},"
        f"{belts_before_corner=},"
        f"{belts_after_corner=},"
        f"{belt_type=},"
        f"start_direction={start_direction_str},"
        f"{turn_left=}.json",
        blueprint
    )


def generate_belts_at_a_corner():
    # Make sure the output directory exists
    if not os.path.exists('blueprints/autogen/belts_at_a_corner'):
        os.makedirs('blueprints/autogen/belts_at_a_corner')

    num_lanes_iter = range(1, 5)
    belts_before_corner_iter = range(1, 33, 4)
    belts_after_corner_iter = range(1, 33, 4)
    belt_types_iter = ['transport-belt', 'fast-transport-belt']
    start_direction_iter = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
    turn_left_iter = [True, False]

    iterator = dict_product(
        num_lanes=num_lanes_iter,
        belts_before_corner=belts_before_corner_iter,
        belts_after_corner=belts_after_corner_iter,
        belt_type=belt_types_iter,
        start_direction=start_direction_iter,
        turn_left=turn_left_iter,
    )
    total = (
        len(num_lanes_iter)
        * len(belts_before_corner_iter)
        * len(belts_after_corner_iter)
        * len(belt_types_iter)
        * len(start_direction_iter)
        * len(turn_left_iter)
    )

    print("generating belts_at_a_corner...\n")
    for kwargs in tqdm.tqdm(iterator, total=total):
        filename, blueprint = belts_at_a_corner(**kwargs)
        with open(f"blueprints/autogen/belts_at_a_corner/{filename}", 'w') as f:
            f.write(json.dumps(blueprint.to_dict(), indent=2))


if __name__ == "__main__":
    # generate_belts_in_a_line()
    generate_belts_at_a_corner()
