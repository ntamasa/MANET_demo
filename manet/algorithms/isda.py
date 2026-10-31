"""ISDA – Incremental Self-Deployment Algorithm (Howard et al. 2002; paper V-B.3; PLAN.md §8.3).

Agents are deployed one at a time. When no agent is moving, a new agent is spawned at xd and sent to a
uniformly random cell of the border (Def. 13) of the current coverage, along a shortest 4-neighbour path
through the covered region. On arrival it settles (becomes terminal) and, since nothing else moves,
greedy deallocation runs on the settled agents. The coverage phase ends when no border remains.
"""
from __future__ import annotations

from manet.pathing import bfs_dist, border, shortest_path
from manet.sim import Agent, Algorithm, Status
from manet.world import Cell


class Isda(Algorithm):
    name = "ISDA"

    def _mover(self) -> Agent | None:
        for a in self.agents:
            if a.status is Status.MOVING:
                return a
        return None

    def _settled(self) -> list[Agent]:
        return [a for a in self.agents if a.status is Status.SETTLED]

    def _plan(self, target: Cell) -> list[Cell]:
        """Shortest path xd -> target through the coverage of xd + settled agents (keeps the mover
        always visible from the network); falls back to explored / free space if needed."""
        cov = self.current_coverage()
        for passable in (cov, self.explored | cov, self._free):
            path = shortest_path(passable, self._xd, target)
            if path:
                return path[1:]
        return []

    def _step(self) -> bool:
        mover = self._mover()
        if mover is None:
            cov = self.current_coverage()
            occupied = set(self.active_cells())
            frontier = [c for c in border(self.world.grid, cov) if c not in occupied and c != self._xd]
            if not frontier:
                self.deallocate(self._settled())
                return False
            # Prefer border cells reachable from xd inside the coverage, so the mover stays visible
            # to the network on its whole way (discrete LOS can leave covered pockets unconnected).
            dist = bfs_dist(cov, self._xd)
            reachable = [c for c in frontier if dist[c] >= 0]
            frontier = reachable or frontier
            if not self.can_spawn():
                return True  # wait (n_max reached / xd busy); base handles stall / Tmax
            target = frontier[int(self.rng.integers(len(frontier)))]
            mover = self.spawn()
            mover.target = target
            mover.path = self._plan(target)
        # the single moving agent advances one cell
        if mover.path:
            mover.pos = mover.path.pop(0)
        if not mover.path:
            mover.status = Status.SETTLED
            self.deallocate(self._settled())  # safe: no agent is MOVING now
        return True

    def _targets(self) -> list[Cell]:
        mover = self._mover()
        return [mover.target] if mover is not None and mover.target is not None else []
