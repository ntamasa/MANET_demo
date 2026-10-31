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
from typing import ClassVar

import numpy as np

from manet.visibility import FovCache
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
        raise NotImplementedError

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
        raise NotImplementedError

    def active_cells(self) -> list[Cell]:
        raise NotImplementedError

    def xd_occupied(self) -> bool:
        """True if any active agent stands on xd."""
        raise NotImplementedError

    def can_spawn(self) -> bool:
        """xd free (no active agent there) and len(active_agents()) < n_max."""
        raise NotImplementedError

    def spawn(self) -> Agent | None:
        """Create an agent at xd (status MOVING) if can_spawn(); updates spawned_total/peak. Else None."""
        raise NotImplementedError

    def current_coverage(self) -> np.ndarray:
        """coverage(cache, [xd] + active_cells())."""
        raise NotImplementedError

    def deallocate(self, terminal: list[Agent]) -> list[Agent]:
        """Run greedy_dealloc on `terminal` (other active agents are kept as fixed graph members),
        set removed agents to RETURNING with a path to xd through `explored`. Returns removed agents.
        No-op (returns []) if self.dealloc is False."""
        raise NotImplementedError

    # --- driver ---------------------------------------------------------------------------------
    def step(self) -> bool:
        raise NotImplementedError

    def finish(self, message: str = "", failed: bool = False) -> None:
        raise NotImplementedError

    def snapshot(self) -> Snapshot:
        raise NotImplementedError


def run(algo: Algorithm) -> Iterator[Snapshot]:
    """Yield the initial snapshot, one per step, and a final one with finished=True;
    sets metrics.wall_time_s."""
    raise NotImplementedError
