"""Tests for the ISDA baseline (PLAN.md §8.3; paper V-B.3, Howard et al. 2002)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from manet.algorithms import get_algorithm
from manet.pathing import border
from manet.sim import Status, run
from manet.visibility import FovCache, coverage, is_connected
from manet.world import analyze, generate_world, grid_from_ascii

L_MAP = [
    "#########",
    "#D......#",
    "#######.#",
    "#######.#",
    "#######.#",
    "#########",
]

CORRIDOR_MAP = [
    "###########",
    "#D........#",
    "#########.#",
    "#########.#",
    "#####.....#",
    "#####.#####",
    "#####.#####",
    "###########",
]

ROOM_MAP = [
    "######",
    "#D...#",
    "#....#",
    "######",
]


def _world(rows):
    grid, xd = grid_from_ascii(rows)
    return analyze(grid, xd)


def _isda():
    from manet.algorithms.isda import Isda
    return Isda


def _make(world, **kw):
    kw.setdefault("n_max", max(world.n_max, 1))
    return _isda()(world, **kw)


# --------------------------------------------------------------------------- registry / trivial

def test_registry():
    cls = get_algorithm("ISDA")
    assert cls is _isda()
    assert cls.__name__ == "Isda"
    assert cls.name == "ISDA"


def test_fully_visible_world_needs_no_agents():
    algo = _make(_world(ROOM_MAP), n_max=3)
    snaps = list(run(algo))
    assert algo.phase == "done"
    assert algo.metrics.success
    assert algo.metrics.final_agents == 0
    assert algo.metrics.spawned_total == 0
    assert algo.metrics.coverage_pct == pytest.approx(100.0)
    assert snaps[-1].finished


def test_targets_empty_before_start():
    algo = _make(_world(L_MAP), n_max=5)
    assert algo._targets() == []


# --------------------------------------------------------------------------- first deployment

@pytest.mark.parametrize("rows", [L_MAP, CORRIDOR_MAP])
def test_first_agent_goes_to_border_and_coverage_grows(rows):
    world = _world(rows)
    algo = _make(world, n_max=10, seed=3)
    cov0 = algo.current_coverage().copy()
    b0 = set(border(world.grid, cov0))
    assert b0
    assert algo.step()
    movers = [a for a in algo.active_agents() if a.status is Status.MOVING]
    settled = [a for a in algo.active_agents() if a.status is Status.SETTLED]
    assert len(movers) + len(settled) == 1
    a = (movers or settled)[0]
    target = a.target
    assert target in b0
    assert algo._targets() == ([target] if movers else [])
    prev = world.xd
    # first move: at most one cell from xd
    assert abs(a.pos[0] - prev[0]) + abs(a.pos[1] - prev[1]) <= 1
    prev = a.pos
    guard = 0
    while a.status is Status.MOVING:
        assert algo._targets() == [target]
        assert algo.step()
        assert abs(a.pos[0] - prev[0]) + abs(a.pos[1] - prev[1]) <= 1
        assert cov0[a.pos]  # walks through the coverage of xd
        prev = a.pos
        guard += 1
        assert guard < 200
    assert a.pos == target
    assert a.status is Status.SETTLED
    assert algo._targets() == []
    cov1 = coverage(algo.cache, [world.xd, a.pos])
    assert (cov1 & ~cov0).any()          # strictly grew
    assert (cov1 | cov0).sum() > cov0.sum()


# --------------------------------------------------------------------------- invariants

def _check_run(world, seed=0, dealloc=True, n_max=None, max_snaps=None):
    cache = FovCache(world.grid)
    algo = _isda()(world, cache, seed=seed, dealloc=dealloc,
                   n_max=n_max if n_max is not None else world.n_max)
    xd = tuple(world.xd)
    prev_pos: dict[int, tuple] = {}
    prev_settled_cov = coverage(cache, [xd])
    prev_moving = 0
    seen_ids: set[int] = set()
    for snap in run(algo):
        active = [(i, p, s) for i, p, s in snap.agents if s is not Status.RETURNING]
        moving = [x for x in active if x[2] is Status.MOVING]
        settled = [x for x in active if x[2] is Status.SETTLED]
        # one mover at a time; spawn only when nobody was moving before
        assert len(moving) <= 1
        new_ids = {i for i, _, _ in snap.agents} - seen_ids
        assert len(new_ids) <= 1
        if new_ids:
            assert prev_moving == 0
        seen_ids |= new_ids
        # connectivity of xd + active agents
        assert is_connected(cache, [xd] + [p for _, p, _ in active])
        # settled coverage never shrinks
        scov = coverage(cache, [xd] + [p for _, p, _ in settled])
        assert not (prev_settled_cov & ~scov).any()
        prev_settled_cov = scov
        # each agent moves at most one 4-neighbour cell per step
        for i, p, _ in snap.agents:
            if i in prev_pos:
                q = prev_pos[i]
                assert abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 1
        prev_pos = {i: p for i, p, _ in snap.agents}
        assert snap.n_active <= algo.n_max
        sp = [p for _, p, _ in settled]
        assert len(sp) == len(set(sp))
        assert xd not in sp
        # targets hook: the mover's target or nothing
        if moving and snap.step > 0 and not snap.finished:
            assert len(snap.targets) == 1
        if not moving:
            assert snap.targets == []
        prev_moving = len(moving)
    return algo


WORLDS = [(n, s) for n in (20, 30, 40) for s in (0, 1, 2, 3, 42)]


@pytest.mark.parametrize("size,seed", WORLDS)
def test_invariants_and_success_on_generated_worlds(size, seed):
    world = generate_world(size, size, seed)
    algo = _check_run(world, seed=seed)
    m = algo.metrics
    assert algo.phase == "done", algo.message
    assert m.success, algo.message
    assert m.coverage_pct == pytest.approx(100.0)
    assert m.steps <= algo.t_max
    assert m.peak_agents <= algo.n_max
    assert m.final_agents <= m.peak_agents
    assert all(a.status is Status.SETTLED for a in algo.agents)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dealloc_never_uses_more_final_agents(seed):
    world = generate_world(30, 30, seed)
    on = _check_run(world, seed=seed, dealloc=True)
    off = _check_run(world, seed=seed, dealloc=False)
    assert on.metrics.success and off.metrics.success
    assert on.metrics.final_agents <= off.metrics.final_agents
    assert off.metrics.final_agents == off.metrics.spawned_total


def test_determinism_same_seed():
    world = generate_world(30, 30, 5)

    def metrics(seed):
        algo = _isda()(world, seed=seed)
        while algo.step():
            pass
        m = algo.metrics
        return (m.steps, m.steps_total, m.spawned_total, m.peak_agents, m.final_agents,
                m.coverage_pct, m.success, [h["agents"] for h in m.history],
                sorted(a.pos for a in algo.agents))

    assert metrics(7) == metrics(7)
    results = {repr(metrics(s)) for s in range(6)}
    assert len(results) > 1   # random frontier choice depends on the seed


def test_speed_40x40():
    world = generate_world(40, 40, 42)
    algo = _isda()(world, seed=42)
    t0 = time.perf_counter()
    for _ in run(algo):
        pass
    dt = time.perf_counter() - t0
    assert algo.metrics.success
    assert dt < 15.0
