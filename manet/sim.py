"""Simulation framework shared by all algorithms (PLAN.md §7).

Data classes below are final (T0). The Algorithm base class behaviour is implemented in T5.

Lifecycle of Algorithm.step() (base class, T5):
  phase "running":   step_count += 1; cont = self._step() (algorithm-specific);
                     returning agents advance one cell (removed on reaching xd);
                     metrics/history/explored updated; if not cont -> finish("...") ;
                     if step_count >= t_max -> finish(fail, "Tmax exceeded");
                     if no coverage growth and no position change for stall_limit steps -> finish(fail, "stalled").
  phase "returning": only RETURNING agents move; when none remain -> phase "done", return False.
  phase "done":      return False.
finish(): metrics.steps = step_count, coverage_pct from active agents + xd, success = (coverage == 100%
and not failed), final_agents = #active agents.
On reaching "done": metrics.steps_total = step_count.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import ClassVar

import numpy as np

from manet.dealloc import greedy_dealloc
from manet.pathing import shortest_path
from manet.visibility import FovCache, coverage, visibility_graph
from manet.world import Cell, World


class Status(Enum):
    MOVING = "moving"
    SETTLED = "settled"
    RETURNING = "returning"


@dataclass
class Agent:
    id: int
    pos: Cell
    status: Status = Status.MOVING
    target: Cell | None = None
    path: list[Cell] = field(default_factory=list)          # remaining cells to walk (excluding pos)
    pruned: set[tuple[Cell, Cell]] = field(default_factory=set)


@dataclass
class Snapshot:
    step: int
    agents: list[tuple[int, Cell, Status]]
    covered: np.ndarray            # current F~(active agents ∪ {xd})
    explored: np.ndarray           # cumulative covered
    targets: list[Cell]
    known_valid: list[Cell]
    edges: list[tuple[Cell, Cell]]  # visibility-graph edges among xd and active agents
    n_active: int                   # agents excluding RETURNING, excluding xd
    coverage_pct: float             # 100 * covered free cells / free cells
    message: str = ""
    finished: bool = False


@dataclass
class Metrics:
    steps: int = 0
    steps_total: int = 0
    wall_time_s: float = 0.0
    spawned_total: int = 0
    peak_agents: int = 0
    final_agents: int = 0
    coverage_pct: float = 0.0
    success: bool = False
    history: list[dict] = field(default_factory=list)   # {"step", "agents", "coverage_pct"}


class Algorithm(ABC):
    """Base class. Subclasses implement _step() and may override _targets()/_known_valid().

    Public attributes after __init__:
      world, cache, n_max, t_max, rng (np.random.Generator(seed)), dealloc (bool), opts (dict),
      agents (list[Agent], includes RETURNING), step_count, phase ("running"|"returning"|"done"),
      explored (bool mask), metrics (Metrics), message (str), stall_limit = 4*(H+W).
    Defaults: n_max = world.n_max, t_max = 100 * max(H, W), cache = FovCache(world.grid).
    """

    name: ClassVar[str] = "base"

    def __init__(self, world: World, cache: FovCache | None = None, *, n_max: int | None = None,
                 t_max: int | None = None, seed: int = 0, dealloc: bool = True, **opts):
        h, w = world.grid.shape
        self.world = world
        self.cache = cache if cache is not None else FovCache(world.grid)
        self.n_max = int(world.n_max if n_max is None else n_max)
        self.t_max = int(100 * max(h, w) if t_max is None else t_max)
        self.rng = np.random.default_rng(seed)
        self.dealloc = bool(dealloc)
        self.opts = dict(opts)
        self.agents: list[Agent] = []
        self.step_count = 0
        self.phase = "running"
        self.metrics = Metrics()
        self.message = ""
        self.stall_limit = 4 * (h + w)
        self._xd: Cell = (int(world.xd[0]), int(world.xd[1]))
        self._free = np.asarray(world.grid, dtype=bool)
        self._n_free = max(1, int(self._free.sum()))
        self._next_id = 0
        self._cov_key: tuple | None = None
        self._cov: np.ndarray | None = None
        self.explored = self.current_coverage().copy()
        self._explored_count = int(self.explored.sum())
        self._last_sig = self._signature()
        self._stall = 0

    # --- to implement in subclasses -------------------------------------------------------------
    @abstractmethod
    def _step(self) -> bool:
        """One step of the algorithm proper. Return False when the coverage phase is complete."""

    def _targets(self) -> list[Cell]:
        return []

    def _known_valid(self) -> list[Cell]:
        return []

    # --- helpers available to subclasses (T5) ---------------------------------------------------
    def active_agents(self) -> list[Agent]:
        """Agents with status != RETURNING."""
        return [a for a in self.agents if a.status is not Status.RETURNING]

    def active_cells(self) -> list[Cell]:
        return [tuple(a.pos) for a in self.agents if a.status is not Status.RETURNING]

    def xd_occupied(self) -> bool:
        """True if any active agent stands on xd."""
        return self._xd in self.active_cells()

    def can_spawn(self) -> bool:
        """xd free (no active agent there) and len(active_agents()) < n_max."""
        return not self.xd_occupied() and len(self.active_agents()) < self.n_max

    def spawn(self) -> Agent | None:
        """Create an agent at xd (status MOVING) if can_spawn(); updates spawned_total/peak. Else None."""
        if not self.can_spawn():
            return None
        a = Agent(id=self._next_id, pos=self._xd)
        self._next_id += 1
        self.agents.append(a)
        self.metrics.spawned_total += 1
        self.metrics.peak_agents = max(self.metrics.peak_agents, len(self.active_agents()))
        return a

    def current_coverage(self) -> np.ndarray:
        """coverage(cache, [xd] + active_cells())."""
        key = tuple(sorted(set(self.active_cells())))
        if self._cov is None or key != self._cov_key:  # memoised: positions rarely all change
            cov = coverage(self.cache, [self._xd, *key])
            cov.flags.writeable = False
            self._cov, self._cov_key = cov, key
        return self._cov

    def deallocate(self, terminal: list[Agent]) -> list[Agent]:
        """Run greedy_dealloc on `terminal` (other active agents are kept as fixed graph members),
        set removed agents to RETURNING with a path to xd through `explored`. Returns removed agents.
        No-op (returns []) if self.dealloc is False."""
        if not self.dealloc:
            return []
        terminal = [a for a in terminal if a.status is not Status.RETURNING]
        term_ids = {id(a) for a in terminal}
        others = [tuple(a.pos) for a in self.active_agents() if id(a) not in term_ids]
        idx = greedy_dealloc(self.cache, self._xd, [tuple(a.pos) for a in terminal], others)
        removed = [terminal[i] for i in idx]
        for a in removed:
            self._send_home(a)
        return removed

    # --- internals ------------------------------------------------------------------------------
    def _send_home(self, a: Agent) -> None:
        a.status = Status.RETURNING
        a.target = self._xd
        a.path = []
        pos = tuple(a.pos)
        if pos == self._xd:
            self.agents.remove(a)
            return
        path = shortest_path(self.explored | self.current_coverage(), pos, self._xd)
        if not path:  # should not happen (pos is always explored); fall back to the full map
            path = shortest_path(self._free, pos, self._xd)
        a.path = path[1:]

    def _advance_returning(self) -> None:
        for a in [a for a in self.agents if a.status is Status.RETURNING]:
            if tuple(a.pos) != self._xd:
                if not a.path:
                    a.path = shortest_path(self._free, tuple(a.pos), self._xd)[1:]
                if not a.path:  # unreachable: vanish rather than block termination
                    self.agents.remove(a)
                    continue
                a.pos = a.path.pop(0)
            if tuple(a.pos) == self._xd:
                self.agents.remove(a)

    def _any_returning(self) -> bool:
        return any(a.status is Status.RETURNING for a in self.agents)

    def _signature(self) -> tuple:
        return tuple((a.id, tuple(a.pos), a.status) for a in self.agents)

    def _pct(self, cov: np.ndarray) -> float:
        return 100.0 * int((cov & self._free).sum()) / self._n_free

    def _record(self) -> None:
        cov = self.current_coverage()
        self.explored |= cov
        n_active = len(self.active_agents())
        self.metrics.peak_agents = max(self.metrics.peak_agents, n_active)
        self.metrics.history.append({"step": self.step_count, "agents": n_active,
                                     "coverage_pct": self._pct(cov)})

    def _set_done(self) -> None:
        self.phase = "done"
        self.metrics.steps_total = self.step_count

    # --- driver ---------------------------------------------------------------------------------
    def step(self) -> bool:
        if self.phase == "done":
            return False
        self.step_count += 1
        if self.phase == "running":
            cont = self._step()
            self._advance_returning()
            self._record()
            if self.phase == "running":  # the subclass may have called finish() itself
                count = int(self.explored.sum())
                sig = self._signature()
                self._stall = 0 if (count > self._explored_count or sig != self._last_sig) else self._stall + 1
                self._explored_count, self._last_sig = count, sig
                if not cont:
                    self.finish("coverage complete")
                elif self.step_count >= self.t_max:
                    self.finish("Tmax exceeded", failed=True)
                elif self._stall >= self.stall_limit:
                    self.finish("stalled", failed=True)
            return self.phase != "done"
        # phase "returning": only RETURNING agents move
        self._advance_returning()
        self._record()
        if not self._any_returning():
            self._set_done()
            return False
        return True

    def finish(self, message: str = "", failed: bool = False) -> None:
        """End the coverage phase (idempotent: ignored unless phase == "running")."""
        if self.phase != "running":
            return
        self.message = message
        m = self.metrics
        cov = self.current_coverage()
        m.steps = self.step_count
        m.coverage_pct = self._pct(cov)
        m.success = (not failed) and int((cov & self._free).sum()) == int(self._free.sum())
        m.final_agents = len(self.active_agents())
        if self._any_returning():
            self.phase = "returning"
        else:
            self._set_done()

    def snapshot(self) -> Snapshot:
        cov = self.current_coverage()
        nodes: dict = {"xd": self._xd}
        for a in self.active_agents():
            nodes[a.id] = tuple(a.pos)
        graph = visibility_graph(self.cache, nodes)
        return Snapshot(
            step=self.step_count,
            agents=[(a.id, tuple(a.pos), a.status) for a in self.agents],
            covered=np.array(cov),
            explored=self.explored.copy(),
            targets=list(self._targets()),
            known_valid=list(self._known_valid()),
            edges=[(nodes[u], nodes[v]) for u, v in graph.edges()],
            n_active=len(nodes) - 1,
            coverage_pct=self._pct(cov),
            message=self.message,
            finished=self.phase == "done",
        )


def run(algo: Algorithm) -> Iterator[Snapshot]:
    """Yield the initial snapshot, one per step, and a final one with finished=True;
    sets metrics.wall_time_s."""
    t0 = perf_counter()
    yield algo.snapshot()
    while algo.step():
        yield algo.snapshot()
    algo.metrics.wall_time_s = max(perf_counter() - t0, 1e-9)
    yield algo.snapshot()
