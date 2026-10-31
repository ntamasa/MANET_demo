"""Tests for manet.summary (T9). Also hosts the dummy algorithms reused by test_cli / test_app.

Hook: the real algorithms are looked up lazily through manet.algorithms._REGISTRY at call time
(manet.summary.make_algorithm -> manet.algorithms.get_algorithm), so tests swap in dummies with
monkeypatch.setitem(manet.algorithms._REGISTRY, NAME, ("tests.test_summary", "DummyGreedy")).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import manet.algorithms as algorithms
from manet.pathing import border, neighbors, shortest_path
from manet.sim import Algorithm, Status, run
from manet.summary import (compact_snapshot, expand_snapshot, format_summary, history_frame,
                           make_algorithm, summary_dict)
from manet.visibility import FovCache
from manet.world import generate_world

# --------------------------------------------------------------------------- dummy algorithms


class DummyGreedy(Algorithm):
    """One agent at a time: spawn, walk to an uncovered free cell next to the border, settle.
    Returns False when the border is empty (=> 100% coverage). Unlimited agents."""
    name = "dummy"

    def __init__(self, world, cache=None, **kw):
        super().__init__(world, cache, **kw)
        self.n_max = 10 ** 6

    def _step(self) -> bool:
        for a in self.active_agents():
            if a.status is Status.MOVING:
                if a.path:
                    a.pos = a.path.pop(0)
                if not a.path:
                    a.status = Status.SETTLED
                return True
        cov = self.current_coverage()
        b = border(self.world.grid, cov)
        if not b:
            self.deallocate(self.active_agents())
            return False
        shape = self.world.grid.shape
        goal = next(n for n in neighbors(shape, b[0]) if self.world.grid[n] and not cov[n])
        a = self.spawn()
        if a is None:
            return True
        a.target = goal
        a.path = shortest_path(self.world.grid, a.pos, goal)[1:]
        if not a.path:
            a.status = Status.SETTLED
        return True

    def _targets(self):
        return [a.target for a in self.active_agents() if a.status is Status.MOVING and a.target]


class DummyFail(Algorithm):
    """Spawns one agent and gives up after 3 steps (coverage incomplete => failure)."""
    name = "dummyfail"

    def _step(self) -> bool:
        if not self.agents:
            self.spawn()
        return self.step_count < 3


class DummyEcho(DummyFail):
    """Records the opts it was built with (to test make_algorithm's argument passing)."""
    last_opts: dict = {}

    def __init__(self, world, cache=None, **kw):
        super().__init__(world, cache, **kw)
        DummyEcho.last_opts = dict(kw, cache=cache)


def register_dummies(monkeypatch, ok="CADENCE", fail="ISDA", echo=None):
    monkeypatch.setitem(algorithms._REGISTRY, ok, ("tests.test_summary", "DummyGreedy"))
    if fail:
        monkeypatch.setitem(algorithms._REGISTRY, fail, ("tests.test_summary", "DummyFail"))
    if echo:
        monkeypatch.setitem(algorithms._REGISTRY, echo, ("tests.test_summary", "DummyEcho"))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def world():
    return generate_world(15, 15, 3)


@pytest.fixture
def finished(world, monkeypatch):
    register_dummies(monkeypatch)
    algo = make_algorithm("CADENCE", world, seed=3, dealloc=True)
    snaps = list(run(algo))
    return algo, snaps


# --------------------------------------------------------------------------- make_algorithm


def test_make_algorithm_uses_registry_case_insensitive(world, monkeypatch):
    register_dummies(monkeypatch)
    algo = make_algorithm("cadence", world, seed=1, dealloc=False)
    assert isinstance(algo, DummyGreedy)
    assert algo.dealloc is False
    assert algo.t_max == 100 * 15


def test_make_algorithm_passes_limits_and_cache(world, monkeypatch):
    register_dummies(monkeypatch, echo="ISDA", fail=None)
    cache = FovCache(world.grid)
    algo = make_algorithm("ISDA", world, seed=5, dealloc=True, n_max=7, t_max=99, cache=cache)
    assert algo.n_max == 7 and algo.t_max == 99 and algo.cache is cache
    assert "early_spawn" not in algo.opts


def test_make_algorithm_zero_limits_mean_auto(world, monkeypatch):
    register_dummies(monkeypatch, echo="ISDA", fail=None)
    algo = make_algorithm("ISDA", world, seed=5, dealloc=True, n_max=0, t_max=0)
    assert algo.n_max == world.n_max and algo.t_max == 100 * 15


def test_make_algorithm_early_spawn_only_for_dadence(world, monkeypatch):
    register_dummies(monkeypatch, echo="DADENCE", fail=None)
    algo = make_algorithm("DADENCE", world, seed=5, dealloc=True, early_spawn=False)
    assert algo.opts == {"early_spawn": False}


# --------------------------------------------------------------------------- summary_dict

KEYS = {"algorithm", "size", "seed", "success", "message", "steps", "steps_total", "wall_time_s",
        "spawned_total", "peak_agents", "final_agents", "coverage_pct", "n_max", "t_max",
        "n_corners", "n_holes", "n_valid_corners"}


def test_summary_dict_consistent(world, finished):
    algo, _ = finished
    d = summary_dict("CADENCE", world, 3, algo)
    assert set(d) == KEYS
    m = algo.metrics
    assert d["algorithm"] == "CADENCE"
    assert d["size"] == "15x15"
    assert d["seed"] == 3
    assert d["success"] is True and m.success
    assert d["message"] == algo.message == "coverage complete"
    assert d["steps"] == m.steps and d["steps_total"] == m.steps_total
    assert d["steps_total"] >= d["steps"] > 0
    assert d["wall_time_s"] == m.wall_time_s > 0
    assert d["spawned_total"] == m.spawned_total >= 1
    assert d["peak_agents"] == m.peak_agents
    assert d["final_agents"] == m.final_agents
    assert d["coverage_pct"] == pytest.approx(100.0)
    assert d["n_max"] == algo.n_max and d["t_max"] == algo.t_max
    assert d["n_corners"] == world.n_corners
    assert d["n_holes"] == world.n_holes
    assert d["n_valid_corners"] == len(world.valid_corners)


def test_summary_dict_failure(world, monkeypatch):
    register_dummies(monkeypatch)
    algo = make_algorithm("ISDA", world, seed=3, dealloc=True)
    list(run(algo))
    d = summary_dict("ISDA", world, 3, algo)
    assert d["success"] is False
    assert d["coverage_pct"] < 100


# --------------------------------------------------------------------------- history_frame


def test_history_frame(finished):
    algo, _ = finished
    df = history_frame(algo.metrics)
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "step"
    assert list(df.columns) == ["agents", "coverage_pct"]
    assert len(df) == len(algo.metrics.history) == algo.metrics.steps_total
    assert df["coverage_pct"].iloc[-1] == pytest.approx(100.0)


def test_history_frame_empty():
    from manet.sim import Metrics
    df = history_frame(Metrics())
    assert len(df) == 0 and list(df.columns) == ["agents", "coverage_pct"]


# --------------------------------------------------------------------------- format_summary


def test_format_summary_contains_numbers(world, finished):
    algo, _ = finished
    d = summary_dict("CADENCE", world, 3, algo)
    s = format_summary(d)
    assert "\n" in s
    for part in ("CADENCE", "15x15", "Lépések", "Futási idő", "Lefedettség", "+1 pseudo-agent (xd)",
                 f"{d['spawned_total']}/{d['peak_agents']}/{d['final_agents']}",
                 str(d["steps"]), str(d["steps_total"]), "100.0", str(d["n_max"])):
        assert part in s, part


def test_format_summary_failure_marked(world, monkeypatch):
    register_dummies(monkeypatch)
    algo = make_algorithm("ISDA", world, seed=3, dealloc=True)
    list(run(algo))
    s = format_summary(summary_dict("ISDA", world, 3, algo))
    assert "✘" in s and algo.message in s


# --------------------------------------------------------------------------- compact / expand


def test_compact_expand_roundtrip(finished):
    _, snaps = finished
    for snap in snaps[:: max(1, len(snaps) // 7)] + [snaps[-1]]:
        c = compact_snapshot(snap)
        e = expand_snapshot(c)
        assert e.covered.dtype == bool and e.explored.dtype == bool
        assert np.array_equal(e.covered, snap.covered)
        assert np.array_equal(e.explored, snap.explored)
        for f in ("step", "agents", "targets", "known_valid", "edges", "n_active", "coverage_pct",
                  "message", "finished"):
            assert getattr(e, f) == getattr(snap, f), f


def test_compact_is_smaller(finished):
    _, snaps = finished
    snap = snaps[-1]
    c = compact_snapshot(snap)
    assert c.covered_bits.nbytes < snap.covered.nbytes
