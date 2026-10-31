"""Tests for DADENCE (PLAN.md §8.2, paper Alg. 2, Fig. 4-5, Section IV-B)."""
from __future__ import annotations

import time

import numpy as np
import pytest

from manet.algorithms import get_algorithm
from manet.pathing import bfs_dist, border
from manet.sim import Agent, Status
from manet.visibility import FovCache, coverage, is_connected
from manet.world import analyze, generate_world, grid_from_ascii

ROOM = [
    "######",
    "#D...#",
    "#....#",
    "######",
]

# 1-wide corridor with turns: xd at the west end of the top row.
CORRIDOR = [
    "###########",
    "#D.....####",
    "######.####",
    "######.####",
    "######....#",
    "#########.#",
    "#########.#",
    "###########",
]

# Two-turn corridor: agent 0 relays at the first corner, agent 1 at the second one.
RELAY = [
    "##########",
    "#D....####",
    "#####.####",
    "#####.####",
    "#####....#",
    "########.#",
    "########.#",
    "##########",
]


def _world(rows):
    grid, xd = grid_from_ascii(rows)
    return analyze(grid, xd)


def _dadence():
    return get_algorithm("DADENCE")


def _place(algo, cells):
    algo.agents = [Agent(id=i, pos=c) for i, c in enumerate(cells)]
    algo._next_id = len(cells)


def _run(world, **kw):
    algo = _dadence()(world, **kw)
    while algo.step():
        pass
    return algo


# --------------------------------------------------------------------------- registry / basics

def test_registry_and_options():
    cls = _dadence()
    assert cls.__name__ == "Dadence"
    assert cls.name == "DADENCE"
    w = _world(ROOM)
    assert cls(w).early_spawn is True
    assert cls(w, early_spawn=False).early_spawn is False


def test_first_step_spawns_one_agent_at_xd():
    w = generate_world(30, 30, seed=3)
    algo = _dadence()(w, seed=3)
    assert algo.step()
    act = algo.active_agents()
    assert len(act) == 1 and act[0].pos == w.xd
    assert algo.metrics.spawned_total == 1


def test_fully_visible_world_finishes_without_agents():
    algo = _run(_world(ROOM))
    assert algo.phase == "done"
    assert algo.metrics.success
    assert algo.metrics.spawned_total == 0
    assert algo.metrics.final_agents == 0
    assert algo.metrics.coverage_pct == pytest.approx(100.0)


def test_targets_hook():
    algo = _dadence()(_world(CORRIDOR), early_spawn=False)
    assert algo._targets() == []
    _place(algo, [(1, 1)])
    algo.step()
    assert algo.x_star is not None
    assert algo._targets() == [algo.x_star]


# --------------------------------------------------------------------------- queue jump

def test_queue_jump_shifts_chain_forward():
    w = _world(CORRIDOR)
    algo = _dadence()(w, early_spawn=False)
    old = [(1, 1), (1, 2), (1, 3)]
    _place(algo, old)
    algo.step()
    by_id = {a.id: a.pos for a in algo.active_agents()}
    # every agent of the chain moved exactly one cell east (toward x*)
    for i, (r, c) in enumerate(old):
        assert by_id[i] == (r, c + 1)
    assert {by_id[i] for i in range(3)} == (set(old) - {w.xd}) | {(1, 4)}
    cells = [a.pos for a in algo.active_agents()]
    assert len(cells) == len(set(cells))


def test_queue_jump_extends_past_target_when_path_full():
    w = _world(CORRIDOR)
    algo = _dadence()(w, early_spawn=False)
    chain = [(1, c) for c in range(1, 7)]  # xd .. x* = (1, 6) all occupied
    _place(algo, chain)
    F = coverage(algo.cache, [w.xd])  # stale coverage: (2, 6) still uncovered
    assert not F[2, 6]
    path = algo._queue_path(F, (1, 6))
    assert path[:6] == chain
    assert path[6:] == [(2, 6)]


# --------------------------------------------------------------------------- revert + pruning

def test_revert_breaking_move_and_prune_edge():
    w = _world(RELAY)
    algo = _dadence()(w, n_max=5, early_spawn=False)
    _place(algo, [(1, 5), (4, 5)])
    F = algo.current_coverage().copy()
    assert is_connected(algo.cache, [w.xd, (1, 5), (4, 5)])
    algo.step()
    assert algo.x_star == (4, 8)
    a0, a1 = algo.agents[0], algo.agents[1]
    # moving agent 0 down or agent 1 east would disconnect the visibility graph -> both reverted
    assert a0.pos == (1, 5) and a1.pos == (4, 5)
    # nobody moved -> a new agent is spawned at xd
    assert len(algo.active_agents()) == 3
    new = algo.active_agents()[-1]
    assert new.pos == w.xd
    # a spawn resets the per-agent edge pruning (IV-B)
    assert not a0.pruned and not a1.pruned
    cov = algo.current_coverage()
    assert not (F & ~cov).any()


def test_greedy_prunes_reverted_edges():
    """The greedy/revert phase alone (no spawn afterwards, which would reset the pruning)."""
    w = _world(RELAY)
    algo = _dadence()(w, n_max=2, early_spawn=False)
    _place(algo, [(1, 5), (4, 5)])
    algo.x_star = (4, 8)
    F = algo.current_coverage().copy()
    assert not algo._greedy(F)
    a0, a1 = algo.agents
    assert a0.pos == (1, 5) and a1.pos == (4, 5)
    assert ((1, 5), (2, 5)) in a0.pruned
    assert ((4, 5), (4, 6)) in a1.pruned


def test_train_move_follows_into_vacated_cell():
    """Paper Alg. 2 lines 16-22: a cell vacated by an agent processed earlier may be taken in the same step."""
    w = _world(CORRIDOR)
    algo = _dadence()(w, early_spawn=False)
    _place(algo, [(1, 3), (1, 2)])          # xd free, both agents in a row toward x*
    algo.step()
    by_id = {a.id: a.pos for a in algo.active_agents()}
    assert by_id[0] == (1, 4) and by_id[1] == (1, 3)


# --------------------------------------------------------------------------- target selection

def _brute_target(world, cache, agents_cells, F):
    B = border(world.grid, F)
    srcs = [world.xd, *agents_cells]
    ds = [bfs_dist(F, s) for s in srcs]
    big = world.grid.size
    best = None
    for x in B:
        vals = [d[x] if d[x] >= 0 else big for d in ds]
        key = (sum(vals) / len(vals), x)
        if best is None or key < best:
            best = key
    return best[1], B


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_target_minimises_mean_distance(seed):
    w = generate_world(25, 25, seed=seed)
    algo = _dadence()(w, seed=seed)
    for _ in range(40):
        F = algo.current_coverage().copy()
        cells = algo.active_cells()
        if not algo.step():
            break
        if algo.x_star is not None and algo._target_changed:
            exp, B = _brute_target(w, algo.cache, cells, F)
            assert algo.x_star in B
            assert algo.x_star == exp


FORK = [
    "###########",
    "#....D....#",
    "#.#######.#",
    "#.#######.#",
    "###########",
]


def test_target_kept_while_on_border():
    w = _world(FORK)
    algo = _dadence()(w, n_max=5, early_spawn=False)
    _place(algo, [(1, 4)])
    F = algo.current_coverage().copy()
    B = border(w.grid, F)
    best, _ = _brute_target(w, algo.cache, [(1, 4)], F)
    other = [b for b in B if b != best]
    assert other, "test map needs at least two border cells"
    algo.x_star = other[0]
    algo.step()
    assert algo.x_star == other[0]


# --------------------------------------------------------------------------- full-run invariants

CASES = [(s, seed) for s in (20, 30, 40) for seed in (0, 1, 2, 3, 42)]


def _check_run(world, **kw):
    algo = _dadence()(world, seed=kw.pop("seed", 0), **kw)
    cache = FovCache(world.grid)
    prev_pos: dict[int, tuple] = {}
    prev_cov = None
    prev_spawned = 0
    while True:
        cont = algo.step()
        running = algo.phase == "running" or algo.metrics.steps == algo.step_count
        act = algo.active_agents()
        cells = [a.pos for a in act]
        assert len(cells) == len(set(cells)), "collision"
        assert len(act) <= algo.n_max
        assert algo.metrics.spawned_total - prev_spawned <= 1
        prev_spawned = algo.metrics.spawned_total
        for a in algo.agents:
            if a.id in prev_pos:
                p, q = prev_pos[a.id], a.pos
                assert abs(p[0] - q[0]) + abs(p[1] - q[1]) <= 1
            else:
                assert a.pos == world.xd
        prev_pos = {a.id: a.pos for a in algo.agents}
        if running:
            assert is_connected(cache, [world.xd, *cells])
            cov = coverage(cache, [world.xd, *cells])
            if prev_cov is not None:
                assert not (prev_cov & ~cov).any(), "coverage shrank"
            prev_cov = cov
        if not cont:
            break
    return algo


@pytest.mark.parametrize("size,seed", CASES)
def test_invariants_and_success(size, seed):
    w = generate_world(size, size, seed=seed)
    algo = _check_run(w, seed=seed)
    m = algo.metrics
    assert algo.phase == "done"
    assert m.success, algo.message
    assert m.coverage_pct == pytest.approx(100.0)
    assert m.steps <= algo.t_max
    assert m.peak_agents <= algo.n_max
    assert m.final_agents <= m.peak_agents
    assert m.steps_total >= m.steps


@pytest.mark.parametrize("seed", [0, 42])
def test_without_early_spawn_succeeds(seed):
    w = generate_world(30, 30, seed=seed)
    algo = _check_run(w, seed=seed, early_spawn=False)
    assert algo.metrics.success, algo.message


def test_without_dealloc_keeps_agents():
    w = generate_world(30, 30, seed=1)
    algo = _run(w, seed=1, dealloc=False)
    assert algo.metrics.success
    assert algo.metrics.final_agents == len(algo.active_agents())


def test_deterministic():
    w = generate_world(30, 30, seed=7)
    a = _run(w, seed=7)
    b = _run(w, seed=7)
    for f in ("steps", "steps_total", "spawned_total", "peak_agents", "final_agents", "coverage_pct",
              "success", "history"):
        assert getattr(a.metrics, f) == getattr(b.metrics, f)


def test_speed_40x40():
    w = generate_world(40, 40, seed=42)
    t0 = time.perf_counter()
    algo = _run(w, seed=42)
    assert time.perf_counter() - t0 < 20.0
    assert algo.metrics.success


def test_agents_settled_at_end():
    w = generate_world(25, 25, seed=5)
    algo = _run(w, seed=5)
    assert all(a.status is Status.SETTLED for a in algo.active_agents())
