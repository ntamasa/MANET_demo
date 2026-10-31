"""Cross-algorithm invariants (M2): every registered algorithm must satisfy the POCGAGP rules."""
import pytest

from manet.algorithms import ALGORITHM_NAMES, get_algorithm
from manet.sim import Status, run
from manet.visibility import is_connected
from manet.world import generate_world

WORLDS = [(25, 25, 1), (30, 40, 7), (40, 40, 42)]


def _algo_cls(name):
    try:
        return get_algorithm(name)
    except ModuleNotFoundError:
        pytest.skip(f"{name} not implemented yet")


def _run(name, h, w, seed, **kw):
    world = generate_world(h, w, seed)
    algo = _algo_cls(name)(world, seed=seed, **kw)
    return world, algo, list(run(algo))


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
@pytest.mark.parametrize("h,w,seed", WORLDS)
def test_step_invariants(name, h, w, seed):
    world, algo, snaps = _run(name, h, w, seed)
    for prev, cur in zip(snaps, snaps[1:]):
        before = {i: pos for i, pos, _ in prev.agents}
        for i, pos, _ in cur.agents:
            if i in before:
                d = abs(pos[0] - before[i][0]) + abs(pos[1] - before[i][1])
                assert d <= 1, f"{name}: agent {i} moved {d} cells at step {cur.step}"
            else:  # spawned at xd, may already take its first step in the same time step
                d = abs(pos[0] - world.xd[0]) + abs(pos[1] - world.xd[1])
                assert d <= 1, f"{name}: agent {i} appeared {d} cells from xd"
        new = [i for i, _, _ in cur.agents if i not in before]
        assert len(new) <= 1, f"{name}: {len(new)} spawns in one step"
        active = [pos for _, pos, st in cur.agents if st != Status.RETURNING]
        assert len(active) <= algo.n_max
        assert is_connected(algo.cache, [world.xd, *active]), f"{name}: disconnected at step {cur.step}"


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
@pytest.mark.parametrize("h,w,seed", WORLDS)
def test_final_state(name, h, w, seed):
    world, algo, snaps = _run(name, h, w, seed)
    m = algo.metrics
    assert algo.phase == "done"
    assert snaps[-1].finished
    assert m.success, algo.message
    assert m.coverage_pct == pytest.approx(100.0)
    assert m.steps <= algo.t_max
    assert m.peak_agents <= algo.n_max
    assert m.final_agents <= m.peak_agents <= m.spawned_total
    assert m.wall_time_s > 0
    settled = [pos for _, pos, st in snaps[-1].agents if st == Status.SETTLED]
    assert len(settled) == len(set(settled))


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_deterministic(name):
    a = _run(name, 30, 30, 3)[1].metrics
    b = _run(name, 30, 30, 3)[1].metrics
    assert (a.steps, a.spawned_total, a.peak_agents, a.final_agents) == \
           (b.steps, b.spawned_total, b.peak_agents, b.final_agents)


@pytest.mark.parametrize("name", ALGORITHM_NAMES)
def test_dealloc_never_increases_final_agents(name):
    on = _run(name, 30, 30, 5, dealloc=True)[1].metrics
    off = _run(name, 30, 30, 5, dealloc=False)[1].metrics
    assert on.success and off.success
    assert on.final_agents <= off.final_agents
