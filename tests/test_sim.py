"""Tests for manet.sim (PLAN.md §7)."""
from __future__ import annotations

import numpy as np
import pytest

from manet.sim import Agent, Algorithm, Metrics, Snapshot, Status, run
from manet.visibility import FovCache, coverage
from manet.world import analyze, grid_from_ascii

L_MAP = [
    "#######",
    "#D....#",
    "#####.#",
    "#####.#",
    "#######",
]
XD = (1, 1)
TO_CORNER = [(1, 2), (1, 3), (1, 4), (1, 5)]


@pytest.fixture
def world():
    grid, xd = grid_from_ascii(L_MAP)
    return analyze(grid, xd)


# --------------------------------------------------------------------------- test algorithms

class Idle(Algorithm):
    name = "idle"

    def _step(self) -> bool:
        return True


class Scripted(Algorithm):
    """Spawns one agent per plan (whenever xd is free), walks each along its plan, settles at the end;
    when all plans are settled: deallocate all active agents and finish."""
    name = "scripted"

    def __init__(self, world, plans, **kw):
        super().__init__(world, **kw)
        self.plans = [list(p) for p in plans]
        self.paths: dict[int, list] = {}

    def _step(self) -> bool:
        for a in self.active_agents():
            if a.status is Status.MOVING:
                p = self.paths[a.id]
                if p:
                    a.pos = p.pop(0)
                if not p:
                    a.status = Status.SETTLED
        if self.plans and self.can_spawn():
            a = self.spawn()
            self.paths[a.id] = self.plans.pop(0)
        if not self.plans and all(a.status is Status.SETTLED for a in self.active_agents()):
            self.deallocate(self.active_agents())
            return False
        return True

    def _targets(self):
        return [(3, 5)]

    def _known_valid(self):
        return [(1, 5)]


class Oscillate(Algorithm):
    name = "osc"

    def _step(self) -> bool:
        if not self.agents:
            self.spawn()
        else:
            a = self.agents[0]
            a.pos = (1, 2) if a.pos != (1, 2) else (1, 3)
        return True


class SelfFail(Algorithm):
    name = "selffail"

    def _step(self) -> bool:
        if self.step_count == 3:
            self.finish("boom", failed=True)
        return True


def two_agents(world, **kw):
    """A and B both end at (1,5): A redundant (lowest index) -> returns 4 cells to xd."""
    kw.setdefault("n_max", 5)
    return Scripted(world, [TO_CORNER, TO_CORNER], **kw)


def run_all(algo, limit=10_000):
    for _ in range(limit):
        if not algo.step():
            return
    raise AssertionError("did not terminate")


# --------------------------------------------------------------------------- __init__

def test_init_defaults(world):
    algo = Idle(world)
    assert algo.world is world
    assert isinstance(algo.cache, FovCache)
    assert algo.n_max == world.n_max
    h, w = world.grid.shape
    assert algo.t_max == 100 * max(h, w)
    assert algo.stall_limit == 4 * (h + w)
    assert algo.phase == "running"
    assert algo.step_count == 0
    assert algo.agents == []
    assert algo.dealloc is True
    assert isinstance(algo.metrics, Metrics)
    assert algo.explored.dtype == bool and algo.explored.shape == world.grid.shape
    assert np.array_equal(algo.explored, algo.cache.fov(XD))


def test_init_overrides_and_opts(world):
    cache = FovCache(world.grid)
    algo = Idle(world, cache, n_max=5, t_max=7, seed=3, dealloc=False, foo=1)
    assert algo.cache is cache and algo.n_max == 5 and algo.t_max == 7
    assert algo.dealloc is False and algo.opts == {"foo": 1}


def test_rng_deterministic(world):
    a, b, c = Idle(world, seed=42), Idle(world, seed=42), Idle(world, seed=43)
    x, y, z = a.rng.random(5), b.rng.random(5), c.rng.random(5)
    assert np.array_equal(x, y) and not np.array_equal(x, z)


# --------------------------------------------------------------------------- helpers

def test_spawn_respects_xd_and_n_max(world):
    algo = Idle(world, n_max=2)
    a0 = algo.spawn()
    assert isinstance(a0, Agent) and a0.id == 0 and a0.pos == XD and a0.status is Status.MOVING
    assert algo.xd_occupied() and not algo.can_spawn()
    assert algo.spawn() is None
    a0.pos = (1, 2)
    assert algo.can_spawn()
    a1 = algo.spawn()
    assert a1.id == 1
    a1.pos = (1, 3)
    assert not algo.can_spawn() and algo.spawn() is None  # n_max reached
    assert algo.metrics.spawned_total == 2
    assert algo.metrics.peak_agents == 2
    assert algo.active_cells() == [(1, 2), (1, 3)]


def test_returning_excluded(world):
    algo = Idle(world, n_max=1)
    a = algo.spawn()
    a.status = Status.RETURNING
    assert algo.active_agents() == []
    assert algo.active_cells() == []
    assert not algo.xd_occupied()
    assert algo.can_spawn()
    b = algo.spawn()
    assert b.id == 1 and algo.metrics.spawned_total == 2 and algo.metrics.peak_agents == 1


def test_current_coverage_includes_xd(world):
    algo = Idle(world)
    assert np.array_equal(algo.current_coverage(), algo.cache.fov(XD))
    a = algo.spawn()
    a.pos = (3, 5)
    assert np.array_equal(algo.current_coverage(), coverage(algo.cache, [XD, (3, 5)]))
    a.status = Status.RETURNING
    assert np.array_equal(algo.current_coverage(), algo.cache.fov(XD))


def test_deallocate_sets_returning_with_path(world):
    algo = Idle(world, n_max=5)
    a = algo.spawn()
    a.pos, a.status = (1, 5), Status.SETTLED
    b = algo.spawn()
    b.pos, b.status = (1, 5), Status.SETTLED
    removed = algo.deallocate([a, b])
    assert removed == [a]
    assert a.status is Status.RETURNING and b.status is Status.SETTLED
    assert a.path == [(1, 4), (1, 3), (1, 2), (1, 1)]
    assert algo.active_agents() == [b]


def test_deallocate_keeps_non_terminal_as_others(world):
    algo = Idle(world, n_max=5)
    a = algo.spawn()
    a.pos, a.status = (1, 5), Status.SETTLED
    b = algo.spawn()
    b.pos = (3, 5)  # MOVING, not terminal -> fixed graph member, makes a a necessary relay
    assert algo.deallocate([a]) == []
    assert a.status is Status.SETTLED


def test_deallocate_agent_on_xd_vanishes(world):
    algo = Idle(world)
    a = algo.spawn()
    a.status = Status.SETTLED
    removed = algo.deallocate([a])
    assert removed == [a]
    assert a not in algo.agents


def test_deallocate_disabled_is_noop(world):
    algo = Idle(world, dealloc=False)
    a = algo.spawn()
    a.status = Status.SETTLED
    assert algo.deallocate([a]) == []
    assert a in algo.agents and a.status is Status.SETTLED


# --------------------------------------------------------------------------- driver

def test_scripted_full_lifecycle(world):
    algo = two_agents(world)
    phases = []
    returning_pos = []
    for _ in range(100):
        cont = algo.step()
        phases.append(algo.phase)
        ret = [a for a in algo.agents if a.status is Status.RETURNING]
        returning_pos.append(ret[0].pos if ret else None)
        if not cont:
            break
    # steps 1..5 running, 6 finishes (dealloc of A), 7..9 returning, 9 done
    assert phases[:5] == ["running"] * 5
    assert phases[5:8] == ["returning"] * 3
    assert phases[8] == "done"
    assert len(phases) == 9
    assert returning_pos[5:9] == [(1, 4), (1, 3), (1, 2), None]
    m = algo.metrics
    assert m.steps == 6 and m.steps_total == 9
    assert m.spawned_total == 2 and m.peak_agents == 2 and m.final_agents == 1
    assert m.coverage_pct == pytest.approx(100.0)
    assert m.success is True
    assert [a.id for a in algo.agents] == [1]
    assert len(m.history) == 9
    assert [r["step"] for r in m.history] == list(range(1, 10))
    assert set(m.history[0]) >= {"step", "agents", "coverage_pct"}
    assert m.history[-1]["agents"] == 1
    # step after done
    assert algo.step() is False and algo.step_count == 9


def test_no_dealloc_ends_without_returning(world):
    algo = two_agents(world, dealloc=False)
    run_all(algo)
    assert algo.phase == "done"
    assert algo.metrics.steps == algo.metrics.steps_total == 6
    assert algo.metrics.final_agents == 2


def test_incomplete_coverage_not_success(world):
    algo = Scripted(world, [[(1, 2)]])
    run_all(algo)
    assert algo.metrics.coverage_pct < 100
    assert algo.metrics.success is False


def test_coverage_pct_formula(world):
    algo = Scripted(world, [[(1, 2)]])
    run_all(algo)
    free = world.grid.sum()
    cov = algo.current_coverage() & world.grid
    assert algo.metrics.coverage_pct == pytest.approx(100 * cov.sum() / free)


def test_explored_cumulative(world):
    algo = two_agents(world)
    prev = algo.explored.copy()
    while algo.step():
        assert np.all(algo.explored >= prev)
        assert np.all(algo.explored >= algo.current_coverage())
        prev = algo.explored.copy()
    assert algo.explored[world.grid].all()


def test_tmax_exceeded(world):
    algo = Oscillate(world, t_max=10)
    run_all(algo)
    m = algo.metrics
    assert algo.phase == "done"
    assert m.steps == 10 and m.success is False
    assert algo.message == "Tmax exceeded"


def test_stalled(world):
    algo = Idle(world, t_max=10_000)
    run_all(algo)
    assert algo.message == "stalled"
    assert algo.metrics.success is False
    assert algo.metrics.steps == algo.stall_limit


def test_finish_called_by_subclass(world):
    algo = SelfFail(world)
    run_all(algo)
    assert algo.phase == "done" and algo.message == "boom"
    assert algo.metrics.steps == 3 and algo.metrics.success is False


# --------------------------------------------------------------------------- snapshot / run

def test_initial_snapshot(world):
    algo = two_agents(world)
    s = algo.snapshot()
    assert isinstance(s, Snapshot)
    assert s.step == 0 and s.agents == [] and s.edges == [] and s.n_active == 0
    assert not s.finished
    assert np.array_equal(s.covered, algo.cache.fov(XD))
    assert s.coverage_pct == pytest.approx(100 * (algo.cache.fov(XD) & world.grid).sum() / world.grid.sum())
    assert s.targets == [(3, 5)] and s.known_valid == [(1, 5)]


def test_snapshot_edges_and_agents(world):
    algo = two_agents(world)
    for _ in range(6):
        algo.step()
    s = algo.snapshot()
    ids = {i: (p, st) for i, p, st in s.agents}
    assert ids[0][1] is Status.RETURNING and ids[1] == ((1, 5), Status.SETTLED)
    assert s.n_active == 1
    edges = {frozenset(e) for e in s.edges}
    assert edges == {frozenset({XD, (1, 5)})}  # returning agent excluded from graph
    assert s.message == algo.message
    # snapshot is a copy
    s.explored[:] = False
    assert algo.explored.any()


def test_snapshot_edges_same_cell(world):
    algo = Idle(world, n_max=5)
    a = algo.spawn()
    a.pos = (1, 5)
    b = algo.spawn()
    b.pos = (1, 5)
    s = algo.snapshot()
    assert len(s.edges) == 3
    assert ((1, 5), (1, 5)) in s.edges


def test_run_generator(world):
    algo = two_agents(world)
    snaps = list(run(algo))
    assert snaps[0].step == 0
    assert len(snaps) == algo.metrics.steps_total + 1
    assert [s.step for s in snaps[:-1]] == list(range(0, algo.metrics.steps_total))
    assert snaps[-1].finished and not any(s.finished for s in snaps[:-1])
    assert snaps[-1].step == algo.metrics.steps_total
    assert algo.metrics.wall_time_s > 0
