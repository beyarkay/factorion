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


def pluralize(singular, plural):
    return (lambda x: singular if x == 1 else plural)

def belts_in_a_line(
    num_lanes=1,
    belts_per_lane=16,
    belt_type="transport-belt",
    direction=Direction.EAST,
    verbose=False
):
    blueprint = Blueprint()

    belt_s = pluralize(belt_type, belt_type + "s")
    lane_s = pluralize("lane", "lanes")

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

        if direction == Direction.NORTH:
            direction_str = "North"
        elif direction == Direction.EAST:
            direction_str = "East"
        elif direction == Direction.SOUTH:
            direction_str = "South"
        elif direction == Direction.WEST:
            direction_str = "West"

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
    num_belts_iter = range(1, 33)
    belt_types_iter = ['transport-belt', 'fast-transport-belt']
    direction_iter = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]

    iterator = itertools.product(
        num_lanes_iter,
        num_belts_iter,
        belt_types_iter,
        direction_iter
    )
    total = len(num_lanes_iter) * len(num_belts_iter) * len(belt_types_iter) * len(direction_iter)

    for num_lanes, num_belts, belt_type, direction in tqdm.tqdm(iterator, total=total):
        filename, blueprint = belts_in_a_line(
            num_lanes=num_lanes,
            belts_per_lane=num_belts,
            belt_type=belt_type,
            direction=direction
        )
        with open(f"blueprints/autogen/belts_in_a_line/{filename}", 'w') as f:
            f.write(json.dumps(blueprint.to_dict(), indent=2))


if __name__ == "__main__":
    generate_belts_in_a_line()
