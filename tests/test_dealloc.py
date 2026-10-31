"""Tests for manet.dealloc (PLAN.md §9, §13)."""
from __future__ import annotations

import numpy as np
import pytest

from manet.dealloc import greedy_dealloc
from manet.visibility import FovCache, coverage, is_connected
from manet.world import generate_world, grid_from_ascii

L_MAP = [
    "#######",
    "#D....#",
    "#####.#",
    "#####.#",
    "#######",
]

ROOM = [
    "#######",
    "#D....#",
    "#.....#",
    "#######",
]


@pytest.fixture
def lmap():
    grid, xd = grid_from_ascii(L_MAP)
    return FovCache(grid), xd


def test_empty_input(lmap):
    cache, xd = lmap
    assert greedy_dealloc(cache, xd, [], []) == []
    assert greedy_dealloc(cache, xd, [], [(1, 3)]) == []


def test_redundant_agent_in_open_room_removed():
    grid, xd = grid_from_ascii(ROOM)
    cache = FovCache(grid)
    # xd alone sees the whole convex room -> every terminal agent is redundant
    assert greedy_dealloc(cache, xd, [(1, 3), (2, 5)], []) == [0, 1]


def test_same_cell_lowest_index_first(lmap):
    cache, xd = lmap
    assert greedy_dealloc(cache, xd, [(1, 5), (1, 5)], []) == [0]


def test_relay_in_l_corridor_kept(lmap):
    cache, xd = lmap
    # (1,5) is the relay between xd and (3,5); (3,5) sees nothing (1,5) does not
    assert greedy_dealloc(cache, xd, [(1, 5), (3, 5)], []) == [1]
    # relay as only terminal, end agent fixed in `others`: relay must stay
    assert greedy_dealloc(cache, xd, [(1, 5)], [(3, 5)]) == []


def test_needed_for_coverage_kept(lmap):
    cache, xd = lmap
    assert greedy_dealloc(cache, xd, [(1, 5)], []) == []


def test_others_never_removed(lmap):
    cache, xd = lmap
    # the terminal agent duplicates an `others` agent -> terminal removed, others untouched
    assert greedy_dealloc(cache, xd, [(1, 5)], [(1, 5)]) == [0]
    assert greedy_dealloc(cache, xd, [], [(1, 5), (1, 5), (1, 2)]) == []


def _random_connected_set(cache, grid, xd, rng, k):
    free = [tuple(map(int, c)) for c in np.argwhere(grid)]
    nodes = [xd]
    tries = 0
    while len(nodes) < k + 1 and tries < 5000:
        tries += 1
        c = free[int(rng.integers(len(free)))]
        if any(cache.sees(c, n) for n in nodes):
            nodes.append(c)
    return nodes[1:]


@pytest.mark.parametrize("seed", range(6))
def test_property_coverage_and_connectivity_preserved(seed):
    world = generate_world(20, 20, seed)
    cache = FovCache(world.grid)
    rng = np.random.default_rng(seed)
    agents = _random_connected_set(cache, world.grid, world.xd, rng, 10)
    n_others = int(rng.integers(0, 3))
    others, terminal = agents[:n_others], agents[n_others:]
    base = coverage(cache, [world.xd] + agents)
    removed = greedy_dealloc(cache, world.xd, terminal, others)
    assert len(set(removed)) == len(removed)
    assert all(0 <= i < len(terminal) for i in removed)
    keep = [c for i, c in enumerate(terminal) if i not in set(removed)]
    rest = [world.xd] + keep + others
    assert np.array_equal(coverage(cache, rest), base)
    assert is_connected(cache, rest)
    # greedy fixed point: no remaining terminal agent is removable
    for j in range(len(keep)):
        r2 = [world.xd] + keep[:j] + keep[j + 1:] + others
        assert not (is_connected(cache, r2) and np.array_equal(coverage(cache, r2), base))
