"""Tests for manet.visibility (PLAN.md §5, §13)."""
from __future__ import annotations

import time

import networkx as nx
import numpy as np
import pytest

from manet.visibility import (
    FovCache,
    compute_fov,
    coverage,
    is_connected,
    los,
    observed_walls,
    visibility_graph,
)
from manet.world import grid_from_ascii

# --------------------------------------------------------------------------- fixtures

ROOM = [  # open room with pillars, no 2x2 checkerboard
    "############",
    "#..........#",
    "#..##......#",
    "#..##...#..#",
    "#.......#..#",
    "#..........#",
    "#.....###..#",
    "#..........#",
    "#.#........#",
    "############",
]

L_CORRIDOR = [
    "########",
    "#......#",
    "#.######",
    "#.######",
    "#.######",
    "########",
]

STRAIGHT = [
    "#########",
    "#.......#",
    "#########",
]

BLOCKED = [
    "#########",
    "#...#...#",
    "#########",
]

OPEN = [
    "#######",
    "#.....#",
    "#.....#",
    "#.....#",
    "#.....#",
    "#######",
]

# single wall cell (2,3) inside a room: the diagonal (1,2)-(3,4) passes exactly through its corner
GRAZE = [
    "#######",
    "#.....#",
    "#..#..#",
    "#.....#",
    "#######",
]


def g(rows):
    return grid_from_ascii(rows)[0]


def free_cells(grid):
    return [tuple(map(int, rc)) for rc in np.argwhere(grid)]


def random_grid(rng, h, w, p_wall):
    grid = rng.random((h, w)) >= p_wall
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
    return grid


# --------------------------------------------------------------------------- los basics


def test_los_symmetric_random_pairs():
    grid = g(ROOM)
    cells = free_cells(grid)
    rng = np.random.default_rng(0)
    for _ in range(500):
        a = cells[rng.integers(len(cells))]
        b = cells[rng.integers(len(cells))]
        assert los(grid, a, b) == los(grid, b, a), (a, b)


def test_los_symmetric_all_pairs_random_grids():
    rng = np.random.default_rng(1)
    for _ in range(3):
        grid = random_grid(rng, 12, 14, 0.25)
        cells = free_cells(grid)
        for i, a in enumerate(cells):
            for b in cells[i + 1:]:
                assert los(grid, a, b) == los(grid, b, a), (a, b)


def test_los_self_true():
    grid = g(ROOM)
    for c in free_cells(grid):
        assert los(grid, c, c)


def test_los_wall_cell_false():
    grid = g(ROOM)
    assert not los(grid, (0, 0), (1, 1))
    assert not los(grid, (1, 1), (2, 3))  # (2,3) is a pillar
    assert not los(grid, (2, 3), (2, 3))
    assert not los(grid, (2, 3), (2, 4))


def test_four_neighbours_see_each_other():
    for rows in (ROOM, L_CORRIDOR, GRAZE):
        grid = g(rows)
        for r, c in free_cells(grid):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if grid[n]:
                    assert los(grid, (r, c), n), ((r, c), n)


def test_straight_corridor_fully_visible():
    grid = g(STRAIGHT)
    cells = free_cells(grid)
    for a in cells:
        for b in cells:
            assert los(grid, a, b)


def test_wall_between_blocks():
    grid = g(BLOCKED)
    assert not los(grid, (1, 1), (1, 7))
    assert not los(grid, (1, 3), (1, 5))
    assert los(grid, (1, 1), (1, 3))


def test_l_corridor():
    grid = g(L_CORRIDOR)
    corner = (1, 1)
    for c in free_cells(grid):
        assert los(grid, corner, c)  # corner sees both arms
    # far ends do not see around the corner
    assert not los(grid, (1, 6), (4, 1))
    assert not los(grid, (1, 6), (3, 1))
    assert not los(grid, (1, 6), (2, 1))
    assert not los(grid, (4, 1), (1, 6))
    assert not los(grid, (4, 1), (1, 3))
    # diagonal neighbours around the reflex corner see each other (grazing, no checkerboard)
    assert los(grid, (1, 2), (2, 1))


def test_exact_diagonal_open_room_visible():
    grid = g(OPEN)
    assert los(grid, (1, 1), (4, 4))
    assert los(grid, (1, 5), (4, 2))
    assert los(grid, (2, 2), (3, 3))


def test_diagonal_grazing_single_wall_corner_visible():
    # Spec: samples exactly on lattice points are ignored -> a 45° line through the corner of a
    # single wall cell (no checkerboard) is NOT blocked.
    # Wall cell (2,3) spans rows [1.5,2.5] x cols [2.5,3.5].
    grid = g(GRAZE)
    assert los(grid, (1, 1), (3, 3))  # touches only corner (2.5, 2.5) of the wall
    assert los(grid, (3, 3), (1, 1))
    assert los(grid, (3, 5), (1, 3))  # touches only corner (1.5, 3.5)
    assert los(grid, (1, 3), (3, 5))
    # lines through the wall's interior are blocked
    assert not los(grid, (1, 2), (3, 4))  # runs along the wall cell's diagonal
    assert not los(grid, (1, 4), (3, 2))
    assert not los(grid, (2, 1), (2, 5))
    assert not los(grid, (1, 3), (3, 3))


def test_non_diagonal_line_clipping_wall_blocked():
    grid = g(GRAZE)
    # from (1,1) to (3,4): slope 2/3, passes through the interior of wall cell (2,3)? p(t)=(1+2t,1+3t)
    # at t=0.5 -> (2, 2.5) edge between (2,2) and (2,3); at t=0.6 -> (2.2, 2.8) -> cell (2,3) wall.
    assert not los(grid, (1, 1), (3, 4))
    assert not los(grid, (3, 4), (1, 1))


# --------------------------------------------------------------------------- FOV


def brute_fov(grid, src):
    h, w = grid.shape
    m = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            m[r, c] = los(grid, src, (r, c))
    return m


@pytest.mark.parametrize("rows", [ROOM, L_CORRIDOR, GRAZE, OPEN, BLOCKED])
def test_compute_fov_equals_bruteforce_ascii(rows):
    grid = g(rows)
    for src in free_cells(grid):
        fov = compute_fov(grid, src)
        assert fov.shape == grid.shape and fov.dtype == bool
        np.testing.assert_array_equal(fov, brute_fov(grid, src), err_msg=str(src))


def test_compute_fov_equals_bruteforce_random():
    rng = np.random.default_rng(7)
    for _ in range(4):
        grid = random_grid(rng, 18, 23, 0.2)
        cells = free_cells(grid)
        for k in rng.choice(len(cells), size=6, replace=False):
            src = cells[k]
            np.testing.assert_array_equal(compute_fov(grid, src), brute_fov(grid, src), err_msg=str(src))


def test_compute_fov_only_free_and_src_true():
    grid = g(ROOM)
    for src in [(1, 1), (5, 5), (8, 10)]:
        fov = compute_fov(grid, src)
        assert fov[src]
        assert not (fov & ~grid).any()


def test_compute_fov_performance():
    h = w = 80
    grid = np.ones((h, w), dtype=bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False
    for r in range(8, h - 4, 10):
        for c in range(8, w - 4, 10):
            grid[r:r + 2, c:c + 2] = False
    compute_fov(grid, (1, 1))  # warm-up
    t0 = time.perf_counter()
    fov = compute_fov(grid, (40, 40))
    dt = time.perf_counter() - t0
    assert fov[40, 40]
    assert dt < 0.5, dt


# --------------------------------------------------------------------------- cache


def test_fov_cache():
    grid = g(ROOM)
    cache = FovCache(grid)
    for src in [(1, 1), (4, 6), (8, 3)]:
        m1 = cache.fov(src)
        np.testing.assert_array_equal(m1, compute_fov(grid, src))
        assert cache.fov(src) is m1
    cells = free_cells(grid)
    rng = np.random.default_rng(3)
    for _ in range(200):
        a = cells[rng.integers(len(cells))]
        b = cells[rng.integers(len(cells))]
        assert cache.sees(a, b) == los(grid, a, b)


# --------------------------------------------------------------------------- coverage


def test_coverage_empty():
    grid = g(ROOM)
    cov = coverage(FovCache(grid), [])
    assert cov.shape == grid.shape and cov.dtype == bool
    assert not cov.any()


def test_coverage_union():
    grid = g(ROOM)
    cache = FovCache(grid)
    cells = [(1, 1), (8, 10), (4, 6)]
    expected = compute_fov(grid, cells[0]) | compute_fov(grid, cells[1]) | compute_fov(grid, cells[2])
    np.testing.assert_array_equal(coverage(cache, cells), expected)
    np.testing.assert_array_equal(coverage(cache, iter(cells)), expected)
    # coverage must not mutate cached masks
    np.testing.assert_array_equal(cache.fov((1, 1)), compute_fov(grid, (1, 1)))


def test_coverage_l_corridor_corner_covers_all():
    grid = g(L_CORRIDOR)
    np.testing.assert_array_equal(coverage(FovCache(grid), [(1, 1)]), grid)


# --------------------------------------------------------------------------- visibility graph


def test_visibility_graph():
    grid = g(L_CORRIDOR)
    cache = FovCache(grid)
    nodes = {"a": (1, 6), "b": (4, 1), "c": (1, 1), "d": (1, 6), 7: (1, 3)}
    G = visibility_graph(cache, nodes)
    assert isinstance(G, nx.Graph) and not G.is_directed()
    assert set(G.nodes) == set(nodes)
    expected = {
        frozenset(e)
        for e in [("a", "c"), ("a", "d"), ("a", 7), ("b", "c"), ("c", "d"), ("c", 7), ("d", 7)]
    }
    assert {frozenset(e) for e in G.edges} == expected
    assert G.number_of_edges() == len(expected)


def test_visibility_graph_empty():
    G = visibility_graph(FovCache(g(ROOM)), {})
    assert G.number_of_nodes() == 0


def test_is_connected():
    grid = g(L_CORRIDOR)
    cache = FovCache(grid)
    assert is_connected(cache, [])
    assert is_connected(cache, [(4, 1)])
    assert not is_connected(cache, [(1, 6), (4, 1)])
    assert is_connected(cache, [(1, 6), (4, 1), (1, 1)])
    assert is_connected(cache, [(1, 6), (1, 6)])
    assert not is_connected(cache, [(1, 6), (1, 6), (4, 1)])


# --------------------------------------------------------------------------- observed walls


def test_observed_walls():
    grid = g(STRAIGHT)
    covered = np.zeros_like(grid)
    covered[1, 1] = True
    ow = observed_walls(grid, covered)
    expected = np.zeros_like(grid)
    expected[0:3, 0:3] = True
    expected[1, 1:3] = False  # free cells are never walls
    np.testing.assert_array_equal(ow, expected)


def test_observed_walls_full_coverage():
    grid = g(L_CORRIDOR)
    ow = observed_walls(grid, grid.copy())
    assert not (ow & grid).any()
    # every wall touching a free cell (8-neighbourhood) is observed; far walls are not
    assert ow[0, 0] and ow[2, 2] and ow[5, 2] and ow[0, 7]
    assert not ow[4, 4] and not ow[5, 7]


def test_observed_walls_empty():
    grid = g(ROOM)
    assert not observed_walls(grid, np.zeros_like(grid)).any()
