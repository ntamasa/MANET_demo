"""Tests for CADENCE (PLAN.md §8.1, paper Alg. 1 + Section IV-A, Proposition 1)."""
from __future__ import annotations

from time import perf_counter

import pytest

from manet.algorithms import get_algorithm
from manet.sim import Status, run
from manet.visibility import FovCache, coverage, is_connected
from manet.world import analyze, generate_world, grid_from_ascii

L_ROOM = [
    "######",
    "#....#",
    "#....#",
    "#..###",
    "#D.###",
    "######",
]

PILLAR = [
    "#########",
    "#D......#",
    "#.......#",
    "#..##...#",
    "#..##...#",
    "#.......#",
    "#.......#",
    "#########",
]

SIZES = [20, 30, 40]
SEEDS = [0, 1, 2, 3, 42]
GEN_CASES = [(s, seed) for s in SIZES for seed in SEEDS]

_WORLDS: dict = {}


def _gen(size: int, seed: int):
    key = (size, seed)
    if key not in _WORLDS:
        w = generate_world(size, size, seed)
        _WORLDS[key] = (w, FovCache(w.grid))
    return _WORLDS[key]


def _ascii_world(rows, xd=None):
    grid, d = grid_from_ascii(rows)
    return analyze(grid, xd if xd is not None else d)


def _cadence():
    return get_algorithm("CADENCE")


def _run(world, cache=None, **kw):
    algo = _cadence()(world, cache, **kw)
    for _ in run(algo):
        pass
    return algo


# --------------------------------------------------------------------------- registry

def test_registry():
    cls = get_algorithm("CADENCE")
    assert cls.__name__ == "Cadence"
    assert cls.name == "CADENCE"
    assert get_algorithm("cadence") is cls


# --------------------------------------------------------------------------- hand-made rooms

def test_l_room_single_agent_walks_to_corner():
    w = _ascii_world(L_ROOM)
    assert w.valid_corner_cells == {(2, 2)}
    algo = _cadence()(w)
    seen_positions = []
    while algo.step():
        seen_positions += [a.pos for a in algo.agents]
    assert algo.metrics.spawned_total == 1
    assert algo.phase == "done"
    assert algo.metrics.success
    assert algo.metrics.coverage_pct == 100.0
    assert (2, 2) in seen_positions


def test_l_room_no_dealloc_agent_settles_on_corner():
    w = _ascii_world(L_ROOM)
    algo = _run(w, dealloc=False)
    assert algo.metrics.success and algo.metrics.final_agents == 1
    [a] = algo.agents
    assert a.pos == (2, 2) and a.status is Status.SETTLED


def test_pillar_room_never_targets_top_left_hole_corner():
    w = _ascii_world(PILLAR)
    assert (2, 2) not in w.valid_corner_cells
    algo = _cadence()(w, dealloc=False)
    targets_seen = set()
    while algo.step():
        targets_seen |= {a.target for a in algo.agents if a.status is not Status.RETURNING}
        targets_seen |= set(algo._targets())
    assert (2, 2) not in targets_seen
    settled = [a.pos for a in algo.active_agents()]
    assert all(a.status is Status.SETTLED for a in algo.active_agents())
    assert len(settled) <= 3
    assert set(settled) <= w.valid_corner_cells
    assert algo.metrics.success and algo.metrics.coverage_pct == 100.0


def test_pillar_room_with_dealloc_success():
    w = _ascii_world(PILLAR)
    algo = _run(w)
    assert algo.metrics.success
    assert algo.metrics.final_agents <= 3
    assert algo.metrics.peak_agents <= w.n_max


def test_xd_is_only_valid_corner():
    w = _ascii_world(L_ROOM, xd=(2, 2))
    algo = _run(w)
    assert algo.phase == "done"
    assert algo.metrics.success
    assert algo.metrics.spawned_total == 0
    assert algo.metrics.final_agents == 0
    assert algo.metrics.coverage_pct == 100.0


# --------------------------------------------------------------------------- invariants

@pytest.mark.parametrize("size,seed", GEN_CASES)
def test_invariants_generated(size, seed):
    world, cache = _gen(size, seed)
    algo = _cadence()(world, cache)
    xd = world.xd
    prev_settled_cov = coverage(cache, [xd])
    prev_pos = {}
    prev_spawned = 0
    reached = set()
    while True:
        cont = algo.step()
        active = algo.active_agents()
        # connectivity of xd + active agents
        assert is_connected(cache, [xd, *[a.pos for a in active]]), f"disconnected at {algo.step_count}"
        # coverage of xd + settled agents never shrinks
        settled = [a for a in active if a.status is Status.SETTLED]
        cov = coverage(cache, [xd, *[a.pos for a in settled]])
        assert not (prev_settled_cov & ~cov).any(), f"settled coverage shrank at {algo.step_count}"
        prev_settled_cov = cov
        # at most one spawn per step
        assert algo.metrics.spawned_total - prev_spawned <= 1
        prev_spawned = algo.metrics.spawned_total
        # each agent moves at most one 4-neighbour cell
        for a in algo.agents:
            p = prev_pos.get(a.id, xd)
            assert abs(a.pos[0] - p[0]) + abs(a.pos[1] - p[1]) <= 1
        prev_pos = {a.id: a.pos for a in algo.agents}
        assert len(active) <= algo.n_max
        # settled positions unique, valid corner cells
        sp = [a.pos for a in settled]
        assert len(sp) == len(set(sp))
        assert set(sp) <= world.valid_corner_cells
        reached |= set(sp)
        # targets unique
        tg = [a.target for a in active]
        assert len(tg) == len(set(tg))
        assert world.xd not in tg
        if not cont:
            break
    m = algo.metrics
    assert algo.phase == "done"
    assert m.success, algo.message
    assert m.coverage_pct == 100.0
    assert m.steps <= algo.t_max
    assert m.peak_agents <= algo.n_max
    assert m.peak_agents <= world.n_max


@pytest.mark.parametrize("size,seed", GEN_CASES)
def test_dealloc_reduces_agents(size, seed):
    world, cache = _gen(size, seed)
    on = _run(world, cache, dealloc=True)
    off = _run(world, cache, dealloc=False)
    assert on.metrics.success and off.metrics.success
    assert on.metrics.final_agents <= off.metrics.final_agents
    # without dealloc every assigned target is reached and kept
    assert off.metrics.final_agents == off.metrics.spawned_total
    assert all(a.status is Status.SETTLED and a.pos == a.target for a in off.agents)
    assert {a.pos for a in off.agents} == set(off._known_valid()) - {world.xd}


def test_determinism():
    world, _ = _gen(30, 1)
    a = _run(world, FovCache(world.grid), seed=7)
    b = _run(world, FovCache(world.grid), seed=7)
    ma, mb = a.metrics, b.metrics
    for f in ("steps", "steps_total", "spawned_total", "peak_agents", "final_agents", "coverage_pct",
              "success", "history"):
        assert getattr(ma, f) == getattr(mb, f), f


# --------------------------------------------------------------------------- snapshot hooks

def test_snapshot_hooks():
    world, cache = _gen(30, 2)
    algo = _cadence()(world, cache)
    saw_target = False
    for snap in run(algo):
        moving = sorted(a.target for a in algo.active_agents() if a.status is Status.MOVING)
        assert sorted(snap.targets) == moving  # assigned targets not yet reached
        saw_target |= bool(snap.targets)
        kv = set(snap.known_valid)
        assert kv <= world.valid_corner_cells
        assert kv <= {tuple(c) for c in zip(*snap.explored.nonzero())}
    assert saw_target
    assert set(algo._known_valid()) == world.valid_corner_cells  # every valid corner is eventually seen
    assert algo._targets() == []


# --------------------------------------------------------------------------- speed

def test_speed_40():
    world = generate_world(40, 40, 42)
    t0 = perf_counter()
    algo = _run(world)
    assert perf_counter() - t0 < 10.0
    assert algo.metrics.success
