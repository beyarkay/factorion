"""Exhaustive tests for find_belt_path and BFS pathfinding."""

import pytest

from helpers import Direction, find_belt_path


class TestFindBeltPath:
    """Test find_belt_path across many grid sizes, positions, and blocked patterns."""

    DIRS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]

    def test_all_start_end_pairs_on_small_grids(self):
        """For grid sizes 1..8, test every (start, end) pair with no blockers."""
        for size in range(1, 9):
            for sx in range(size):
                for sy in range(size):
                    for ex in range(size):
                        for ey in range(size):
                            start = (sx, sy)
                            end = (ex, ey)
                            for end_dir in self.DIRS:
                                path = find_belt_path(
                                    size, size, start, end, end_dir, set()
                                )
                                if start == end:
                                    # Single cell: path is just the cell itself
                                    assert path is not None
                                    assert len(path) == 1
                                    assert path[0][:2] == start
                                    continue

                                assert path is not None, (
                                    f"No path on {size}x{size} from {start} to {end}"
                                )

                                # Path starts at start, ends at end
                                assert path[0][:2] == start
                                assert path[-1][:2] == end
                                assert path[-1][2].value == end_dir.value

                                # Path length = manhattan distance (no blockers)
                                manhattan = abs(ex - sx) + abs(ey - sy)
                                assert len(path) == manhattan + 1, (
                                    f"{size}x{size} {start}->{end}: "
                                    f"len={len(path)}, manhattan+1={manhattan + 1}"
                                )

                                # Each step is manhattan distance 1 from previous
                                for i in range(len(path) - 1):
                                    ax, ay, _ = path[i]
                                    bx, by, _ = path[i + 1]
                                    assert abs(ax - bx) + abs(ay - by) == 1

                                # Direction of each belt matches the step direction
                                for i in range(len(path) - 1):
                                    ax, ay, d = path[i]
                                    bx, by, _ = path[i + 1]
                                    dx, dy = bx - ax, by - ay
                                    expected_val = {
                                        (0, -1): Direction.NORTH.value,
                                        (1, 0): Direction.EAST.value,
                                        (0, 1): Direction.SOUTH.value,
                                        (-1, 0): Direction.WEST.value,
                                    }[(dx, dy)]
                                    assert d.value == expected_val

                                # No duplicate positions
                                positions = [(x, y) for x, y, _ in path]
                                assert len(set(positions)) == len(positions)

    def test_blocked_cells(self):
        """Test that blocked cells are avoided on various grid sizes."""
        for size in range(3, 10):
            start = (0, 0)
            end = (size - 1, size - 1)

            # Block the entire second row except (1, size-1) — forces detour
            blocked = {(1, y) for y in range(size - 1)}
            path = find_belt_path(size, size, start, end, Direction.EAST, blocked)
            assert path is not None, f"No path on {size}x{size} with row block"

            positions = {(x, y) for x, y, _ in path}
            assert positions.isdisjoint(blocked), "Path goes through blocked cell"

            # Block everything except the border — should still find a path
            blocked = {
                (x, y)
                for x in range(1, size - 1)
                for y in range(1, size - 1)
            }
            path = find_belt_path(size, size, start, end, Direction.EAST, blocked)
            assert path is not None, f"No border path on {size}x{size}"
            positions = {(x, y) for x, y, _ in path}
            assert positions.isdisjoint(blocked)

    def test_no_path_when_fully_blocked(self):
        """Block all cells around start so no path exists."""
        for size in range(3, 8):
            start = (size // 2, size // 2)
            end = (0, 0)
            # Block all 4 neighbors of start
            sx, sy = start
            blocked = set()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < size and 0 <= ny < size:
                    blocked.add((nx, ny))

            path = find_belt_path(size, size, start, end, Direction.EAST, blocked)
            if end in blocked:
                assert path is None
            elif start == end:
                assert path is not None
            else:
                assert path is None, (
                    f"Expected no path from {start} on {size}x{size} "
                    f"with all neighbors blocked"
                )

    def test_out_of_bounds(self):
        """Start or end out of bounds returns None."""
        for size in range(1, 6):
            oob_positions = [(-1, 0), (0, -1), (size, 0), (0, size)]
            for oob in oob_positions:
                assert find_belt_path(
                    size, size, oob, (0, 0), Direction.EAST, set()
                ) is None, f"start={oob} should be OOB on {size}x{size}"
                assert find_belt_path(
                    size, size, (0, 0), oob, Direction.EAST, set()
                ) is None, f"end={oob} should be OOB on {size}x{size}"

    def test_blocked_start_or_end(self):
        """Blocked start or end returns None."""
        for size in range(2, 6):
            start = (0, 0)
            end = (size - 1, size - 1)
            assert find_belt_path(
                size, size, start, end, Direction.EAST, {start}
            ) is None
            assert find_belt_path(
                size, size, start, end, Direction.EAST, {end}
            ) is None

    def test_path_optimality_with_obstacles(self):
        """Verify paths are shortest even with complex obstacle patterns."""
        size = 10
        for seed in range(100):
            import random

            rng = random.Random(seed)

            # Random start/end
            start = (rng.randint(0, size - 1), rng.randint(0, size - 1))
            end = (rng.randint(0, size - 1), rng.randint(0, size - 1))
            if start == end:
                continue

            # Random blocked cells (30% density), never blocking start/end
            blocked = set()
            for x in range(size):
                for y in range(size):
                    if (x, y) not in (start, end) and rng.random() < 0.3:
                        blocked.add((x, y))

            path = find_belt_path(
                size, size, start, end, Direction.EAST, blocked
            )
            if path is None:
                continue

            # Verify it's a valid path
            positions = [(x, y) for x, y, _ in path]
            assert positions[0] == start
            assert positions[-1] == end
            assert set(positions).isdisjoint(blocked)

            # Verify optimality: try a second path with one fewer cell blocked
            # — should be same length or shorter
            path_len = len(path)
            for bx, by in list(blocked)[:5]:
                shorter_blocked = blocked - {(bx, by)}
                alt = find_belt_path(
                    size, size, start, end, Direction.EAST, shorter_blocked
                )
                if alt is not None:
                    assert len(alt) <= path_len, (
                        f"Removing block ({bx},{by}) gave longer path: "
                        f"{len(alt)} > {path_len}"
                    )
