"""Tests for manet.pathing (PLAN.md §6, §13)."""
import time

import numpy as np
import pytest

from manet.pathing import border, bfs_dist, closer_neighbors, neighbors, shortest_path
from manet.world import grid_from_ascii

ROOM = [
    "#######",
    "#.....#",
    "#.....#",
    "#.....#",
    "#.....#",
    "#######",
]

# U-shaped corridor: going from (1,1) to (1,3) must detour around the middle wall.
U_MAP = [
    "#####",
    "#.#.#",
    "#.#.#",
    "#...#",
    "#####",
]


def _room():
    g, _ = grid_from_ascii(ROOM)
    return g


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------- neighbors
def test_neighbors_order_and_bounds():
    assert neighbors((5, 5), (2, 2)) == [(1, 2), (2, 3), (3, 2), (2, 1)]
    assert neighbors((5, 5), (0, 0)) == [(0, 1), (1, 0)]
    assert neighbors((5, 5), (4, 4)) == [(3, 4), (4, 3)]
    assert neighbors((1, 1), (0, 0)) == []


# ---------------------------------------------------------------- bfs_dist
def test_bfs_open_room_is_manhattan():
    g = _room()
    src = (2, 3)
    d = bfs_dist(g, src)
    assert d.shape == g.shape
    assert np.issubdtype(d.dtype, np.integer)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c]:
                assert d[r, c] == _manhattan(src, (r, c))
            else:
                assert d[r, c] == -1


def test_bfs_unreachable_is_minus_one():
    g, _ = grid_from_ascii([
        "######",
        "#..#.#",
        "#..#.#",
        "######",
    ])
    d = bfs_dist(g, (1, 1))
    assert d[1, 4] == -1 and d[2, 4] == -1
    assert d[2, 2] == 2
    assert d[0, 0] == -1


def test_bfs_detour():
    g, _ = grid_from_ascii(U_MAP)
    d = bfs_dist(g, (1, 1))
    assert d[1, 3] == 6
    assert d[3, 2] == 3


def test_bfs_nonpassable_src_still_expands():
    g = _room()
    src = (0, 3)  # wall on the frame
    d = bfs_dist(g, src)
    assert d[src] == 0
    assert d[1, 3] == 1
    assert d[2, 4] == 3
    assert d[0, 2] == -1  # other walls stay unreachable


def test_bfs_directed_blocked_edge():
    g, _ = grid_from_ascii(U_MAP)
    # block the step (3,1)->(3,2): forward from (1,1) must be cut off
    blocked = frozenset({((3, 1), (3, 2))})
    d = bfs_dist(g, (1, 1), blocked)
    assert d[3, 2] == -1 and d[1, 3] == -1
    # the reverse direction is still usable
    d2 = bfs_dist(g, (1, 3), blocked)
    assert d2[1, 1] == 6
    assert d2[3, 1] == 4


def test_bfs_blocked_edge_detour_in_room():
    g = _room()
    blocked = {((1, 1), (1, 2))}
    d = bfs_dist(g, (1, 1), blocked)
    assert d[1, 2] == 3  # (1,1)->(2,1)->(2,2)->(1,2)
    assert d[1, 3] == 4  # (1,1)->(2,1)->(2,2)->(1,2)->(1,3)
    d_rev = bfs_dist(g, (1, 2), blocked)
    assert d_rev[1, 1] == 1


def test_bfs_performance_80x80():
    g = np.ones((80, 80), dtype=bool)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = False
    bfs_dist(g, (40, 40))  # warm-up
    t0 = time.perf_counter()
    d = bfs_dist(g, (1, 1))
    assert time.perf_counter() - t0 < 0.1
    assert d[78, 78] == 154


# ---------------------------------------------------------------- shortest_path
def _check_path(path, src, dst):
    assert path[0] == src and path[-1] == dst
    for a, b in zip(path, path[1:]):
        assert _manhattan(a, b) == 1


def test_shortest_path_valid_and_minimal():
    g, _ = grid_from_ascii(U_MAP)
    p = shortest_path(g, (1, 1), (1, 3))
    _check_path(p, (1, 1), (1, 3))
    assert len(p) == bfs_dist(g, (1, 1))[1, 3] + 1
    assert all(g[c] for c in p)


def test_shortest_path_lengths_match_bfs_everywhere():
    g = _room()
    src = (4, 1)
    d = bfs_dist(g, src)
    for r, c in zip(*np.nonzero(g)):
        p = shortest_path(g, src, (int(r), int(c)))
        _check_path(p, src, (r, c))
        assert len(p) == d[r, c] + 1


def test_shortest_path_deterministic_tiebreak():
    g = _room()
    # From (1,1) to (3,3): prefer E before S at each step (order N,E,S,W).
    assert shortest_path(g, (1, 1), (3, 3)) == [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    # From (3,3) to (1,1): prefer N before W.
    assert shortest_path(g, (3, 3), (1, 1)) == [(3, 3), (2, 3), (1, 3), (1, 2), (1, 1)]
    # repeated calls identical
    assert shortest_path(g, (4, 5), (1, 1)) == shortest_path(g, (4, 5), (1, 1))


def test_shortest_path_src_equals_dst():
    g = _room()
    assert shortest_path(g, (2, 2), (2, 2)) == [(2, 2)]


def test_shortest_path_unreachable():
    g, _ = grid_from_ascii([
        "######",
        "#..#.#",
        "######",
    ])
    assert shortest_path(g, (1, 1), (1, 4)) == []


def test_shortest_path_nonpassable_dst():
    g = _room()
    passable = np.zeros_like(g)
    passable[1:3, 1:4] = True  # "covered" part of the room
    dst = (3, 3)  # free but uncovered neighbour
    p = shortest_path(passable, (1, 1), dst)
    _check_path(p, (1, 1), dst)
    assert len(p) == 5
    assert all(passable[c] for c in p[:-1])
    # non-passable dst not adjacent to the passable region -> unreachable
    assert shortest_path(passable, (1, 1), (4, 5)) == []


def test_shortest_path_respects_blocked_edges():
    g, _ = grid_from_ascii(U_MAP)
    blocked = frozenset({((3, 1), (3, 2))})
    assert shortest_path(g, (1, 1), (1, 3), blocked) == []
    p = shortest_path(g, (1, 3), (1, 1), blocked)
    _check_path(p, (1, 3), (1, 1))
    assert len(p) == 7
    g2 = _room()
    p2 = shortest_path(g2, (1, 1), (1, 2), {((1, 1), (1, 2))})
    _check_path(p2, (1, 1), (1, 2))
    assert len(p2) == 4
    for a, b in zip(p2, p2[1:]):
        assert (a, b) != ((1, 1), (1, 2))


# ---------------------------------------------------------------- closer_neighbors
def test_closer_neighbors_all_one_closer():
    g = _room()
    target = (2, 3)
    d = bfs_dist(g, target)
    for r, c in zip(*np.nonzero(g)):
        cell = (int(r), int(c))
        cn = closer_neighbors(d, cell, g)
        for w in cn:
            assert d[w] == d[cell] - 1
        if cell != target:
            assert cn
    assert closer_neighbors(d, target, g) == []


def test_closer_neighbors_order():
    g = _room()
    d = bfs_dist(g, (1, 1))
    assert closer_neighbors(d, (3, 3), g) == [(2, 3), (3, 2)]  # N then W
    d2 = bfs_dist(g, (4, 5))
    assert closer_neighbors(d2, (2, 2), g) == [(2, 3), (3, 2)]  # E then S


def test_closer_neighbors_blocked_and_nonpassable():
    g = _room()
    d = bfs_dist(g, (1, 1))
    assert closer_neighbors(d, (3, 3), g, {((3, 3), (2, 3))}) == [(3, 2)]
    # blocking the reverse direction has no effect
    assert closer_neighbors(d, (3, 3), g, {((2, 3), (3, 3))}) == [(2, 3), (3, 2)]
    passable = g.copy()
    passable[3, 2] = False
    assert closer_neighbors(d, (3, 3), passable) == [(2, 3)]


# ---------------------------------------------------------------- border
def test_border_full_coverage_empty():
    g = _room()
    assert border(g, g.copy()) == []


def test_border_empty_coverage_empty():
    g = _room()
    assert border(g, np.zeros_like(g)) == []


def test_border_partial():
    g = _room()
    cov = np.zeros_like(g)
    cov[1:3, 1:4] = True
    # uncovered free cells: row 3-4 all, cols 4-5 in rows 1-2
    expected = [(1, 3), (2, 1), (2, 2), (2, 3)]
    assert border(g, cov) == expected


def test_border_walls_never_included_and_sorted():
    g = _room()
    cov = np.ones_like(g)  # covered includes walls
    cov[4, 5] = False
    b = border(g, cov)
    assert b == [(3, 5), (4, 4)]
    assert all(g[c] for c in b)
    assert b == sorted(b)


def test_border_wall_neighbour_not_frontier():
    g, _ = grid_from_ascii(U_MAP)
    cov = g.copy()
    cov[1, 3] = cov[2, 3] = False
    # (1,1),(2,1) are next to walls only/covered; (3,3) touches uncovered (2,3)
    assert border(g, cov) == [(3, 3)]
