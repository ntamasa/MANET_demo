"""CADENCE – Centralized Algorithm (paper Alg. 1 + Section IV-A modified loop, PLAN.md §8.1).

Agents are sent from xd to observed valid 270° corners (Def. 8). At most one spawn per step; all
MOVING agents advance in parallel along their own shortest path (through explored terrain) and settle
on arrival. The coverage phase ends when every observed valid corner is assigned and reached.
Deallocated corners stay assigned (never re-targeted); xd counts as occupying its own cell.
"""
from __future__ import annotations

from manet.pathing import bfs_dist, shortest_path
from manet.sim import Algorithm, Status
from manet.world import Cell


class Cadence(Algorithm):
    name = "CADENCE"

    def __init__(self, world, cache=None, **kw):
        super().__init__(world, cache, **kw)
        self._valid: set[Cell] = {tuple(c) for c in world.valid_corner_cells}
        self.known_valid: set[Cell] = set()
        self.assigned: dict[Cell, int] = {}        # target cell -> agent id
        self._pending_dealloc = False
        self._update_known()

    # --- helpers --------------------------------------------------------------------------------
    def _seen(self):
        return self.explored | self.current_coverage()

    def _update_known(self) -> None:
        seen = self._seen()
        self.known_valid |= {c for c in self._valid if seen[c]}

    def _unassigned(self) -> list[Cell]:
        return sorted(c for c in self.known_valid if c not in self.assigned and c != self._xd)

    def _moving(self):
        return [a for a in self.agents if a.status is Status.MOVING]

    # --- algorithm ------------------------------------------------------------------------------
    def _step(self) -> bool:
        self._update_known()
        # 0. deallocation / termination act on the configuration reached at the end of the previous
        #    step, so agents that arrived then do not also start walking home in the same step
        #    (each agent moves at most one cell per step).
        if not self._moving():
            settled = [a for a in self.agents if a.status is Status.SETTLED]
            if not self._unassigned():  # every observed valid corner assigned and reached
                self.deallocate(settled)
                return False
            if self._pending_dealloc:   # safety rule: only when nobody is MOVING
                self.deallocate(settled)
                self._pending_dealloc = False
        # 1. at most one spawn: closest (BFS over explored terrain) unassigned observed valid corner
        unassigned = self._unassigned()
        if unassigned and self.can_spawn():
            seen = self._seen()
            dist = bfs_dist(seen, self._xd)
            reachable = [c for c in unassigned if dist[c] >= 0]
            if reachable:
                target = min(reachable, key=lambda c: (int(dist[c]), c))
                a = self.spawn()
                a.target = target
                a.path = shortest_path(seen, self._xd, target)[1:]
                self.assigned[target] = a.id
        # 2. every MOVING agent advances one cell; arrival -> SETTLED
        for a in self._moving():
            if a.path:
                a.pos = a.path.pop(0)
            if not a.path:
                a.status = Status.SETTLED
                self._pending_dealloc = True
        self._update_known()  # register newly observed corners promptly
        return True

    # --- snapshot hooks -------------------------------------------------------------------------
    def _targets(self) -> list[Cell]:
        """Assigned targets not yet reached (targets of MOVING agents)."""
        return sorted(a.target for a in self._moving())

    def _known_valid(self) -> list[Cell]:
        return sorted(self.known_valid)
