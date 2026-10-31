"""DADENCE – decentralised deployment toward a shared border target (PLAN.md §8.2, paper Alg. 2, Fig. 4-5,
Section IV-B).

Per step (Fig. 4 blocks):
  F = coverage(xd ∪ A), B = border(F). B empty -> settle, deallocate, done.
  A empty -> spawn the first agent at xd.
  x* (shared target) = argmin over B of the mean BFS distance (over F) to A ∪ {xd}; recomputed only when it
  leaves the border; a new target resets every agent's pruned edges.
  Agent on xd -> queue jump: the occupied prefix of the shortest path xd -> x* shifts one cell forward.
  Otherwise every agent declares a greedy move to a strictly closer neighbour not in A' (paper lines 16-22:
  agents processed closest-to-x* first, seeded random tie-break; a cell vacated by an agent processed
  earlier may be taken -> "train" moves); declared moves are reverted (largest id first, cascading to agents
  that followed into the reverted agent's cell) while the visibility graph of A' ∪ {xd} is disconnected or
  F ⊄ F~(A' ∪ {xd}). Reverted moves (with their trains) that are valid after all are re-applied; only the
  remaining invalid leader moves get their edge pruned (IV-B). Nobody moved -> spawn (+ pruning reset).
  Early spawn (IV-B, optional): spawn when coverage has not grown for th = mean dist(A, x*) steps.
Connectivity, non-shrinking coverage and collision freedom hold every step; deallocation only at the end.

Documented deviations (PLAN §15 latitude): train moves into cells vacated in the same step (as in the paper's
A' bookkeeping, needed against deadlocks of agents pinned by thin diagonal sight lines); re-apply pass after
the revert loop; pruning also reset when nobody moved and no spawn is possible (n_max); random tie-breaks.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from manet.pathing import bfs_dist, border, closer_neighbors, neighbors, shortest_path
from manet.sim import Agent, Algorithm, Status
from manet.world import Cell


class Dadence(Algorithm):
    name = "DADENCE"

    def __init__(self, world, cache=None, *, early_spawn: bool = True, **kw):
        super().__init__(world, cache, early_spawn=early_spawn, **kw)
        self.early_spawn = bool(early_spawn)
        self.x_star: Cell | None = None
        self._target_changed = False   # True if x* was (re)computed in the last step (for tests/UI)
        self._no_growth = 0            # steps without coverage growth (early spawning)

    # ------------------------------------------------------------------------------------------ hooks
    def _targets(self) -> list[Cell]:
        return [self.x_star] if self.x_star is not None else []

    # ------------------------------------------------------------------------------------------ step
    def _step(self) -> bool:
        self._target_changed = False
        F = self.current_coverage().copy()
        B = border(self.world.grid, F)
        act = self.active_agents()
        if not B:                                   # [2] full coverage
            for a in act:
                a.status = Status.SETTLED
                a.target = None
            self.x_star = None
            self.deallocate(act)
            return False
        if not act:                                 # first iteration (line 4)
            self.spawn()
            return True

        if self.x_star is None or self.x_star not in set(B):   # [3] only when x* left the border
            self.x_star = self._choose_target(F, B, act)
            self._target_changed = True
            self._reset_pruning()
        for a in act:
            a.target = self.x_star

        n_before = int(F.sum())
        moved = False
        if self.xd_occupied():                      # [4] queue jump (lines 7-13)
            moved = self._queue_jump(F)
        if not moved:                               # greedy moves + revert (lines 15-28)
            moved = self._greedy(F)
        spawned = False
        if not moved:                               # line 30
            if self.spawn() is not None:
                spawned = True
            # spawn resets pruning (IV-B); when spawning is impossible (n_max) the pruned edges are
            # also forgotten so agents retry in the (possibly changed) configuration
            self._reset_pruning()

        # early spawning (IV-B)
        F2 = self.current_coverage()
        self._no_growth = 0 if (spawned or int(F2.sum()) > n_before) else self._no_growth + 1
        if self.early_spawn and not spawned and self._no_growth >= 1 and self.can_spawn():
            d = bfs_dist(F2, self.x_star)
            ds = [int(d[a.pos]) for a in self.active_agents() if d[a.pos] >= 0]
            th = max(1, round(sum(ds) / len(ds))) if ds else 1
            if self._no_growth >= th and self.spawn() is not None:
                self._no_growth = 0
                self._reset_pruning()
        return True

    # ------------------------------------------------------------------------------------------ parts
    def _reset_pruning(self) -> None:
        for a in self.active_agents():
            a.pruned.clear()

    def _choose_target(self, F: np.ndarray, B: list[Cell], act: list[Agent]) -> Cell:
        """argmin over border cells of the mean BFS distance (over F) to agents ∪ {xd};
        unreachable counts as H*W; ties -> smallest (r, c)."""
        big = self.world.grid.size
        total = np.zeros(F.shape, dtype=np.int64)
        for s in [self._xd, *(tuple(a.pos) for a in act)]:
            d = bfs_dist(F, s)
            total += np.where(d < 0, big, d)
        rows, cols = np.array(B).T
        i = int(np.argmin(total[rows, cols]))     # B is sorted -> first minimum = smallest (r, c)
        return B[i]

    def _queue_path(self, F: np.ndarray, x_star: Cell) -> list[Cell]:
        """Shortest path xd -> x* over F; if all its cells are occupied, extended by one free neighbour of x*
        (uncovered preferred). [] if x* is unreachable."""
        path = shortest_path(F, self._xd, x_star)
        if not path:
            return []
        occ = set(self.active_cells())
        if all(p in occ for p in path):
            on_path = set(path)
            cands = [nb for nb in neighbors(F.shape, x_star)
                     if self.world.grid[nb] and nb not in occ and nb not in on_path]
            cands.sort(key=lambda nb: bool(F[nb]))  # uncovered first (stable: N, E, S, W)
            if cands:
                path.append(cands[0])
        return path

    def _queue_jump(self, F: np.ndarray) -> bool:
        path = self._queue_path(F, self.x_star)
        if not path:
            return False
        by_pos = {tuple(a.pos): a for a in self.active_agents()}
        k = next((i for i, p in enumerate(path) if p not in by_pos), None)
        if k is None:
            return False
        for i in range(k - 1, -1, -1):             # Fig. 5: shift the occupied prefix forward
            by_pos[path[i]].pos = path[i + 1]
        return True

    def _greedy(self, F: np.ndarray) -> bool:
        act = sorted(self.active_agents(), key=lambda a: a.id)
        x = self.x_star
        base = bfs_dist(F, x)
        dists: dict[frozenset, np.ndarray] = {}
        # A' (paper lines 16-22): an agent's own cell leaves A' when it is processed, so a later agent may
        # follow into a cell vacated earlier in the same step (train move). Agents are processed closest to
        # x* first (ties: id), so a whole queue can advance in one step.
        planned = {tuple(a.pos) for a in act}
        declared: dict[int, Cell] = {}
        unreach = self.world.grid.size
        tie = self.rng.permutation(len(act))       # random (seeded) tie-break: avoids repeated deadlocks
        keyed = sorted(range(len(act)),
                       key=lambda k: (base[act[k].pos] if base[act[k].pos] >= 0 else unreach, tie[k]))
        for a in (act[k] for k in keyed):
            pos = tuple(a.pos)
            if pos == x:
                continue
            if a.pruned:
                key = frozenset(a.pruned)
                d = dists.get(key)
                if d is None:   # pruned edges removed (both directions) from this agent's graph
                    d = bfs_dist(F, x, a.pruned | {(v, u) for u, v in a.pruned})
                    dists[key] = d
            else:
                d = base
            cands = [w for w in closer_neighbors(d, pos, F, a.pruned) if w not in planned]
            if len(cands) > 1:
                cands = [cands[k] for k in self.rng.permutation(len(cands))]
            if cands:
                declared[a.id] = cands[0]
                planned.discard(pos)
                planned.add(cands[0])
        if not declared:
            return False
        by_id = {a.id: a for a in act}
        orig = {a.id: tuple(a.pos) for a in act}
        new = {i: declared.get(i, orig[i]) for i in orig}
        follower = {declared[i]: i for i in declared}   # cell -> agent declaring a move into it

        def revert(i: int) -> list[int]:
            """Put agent i back; agents that declared into its cell are reverted too (no collisions)."""
            out = []
            while i is not None and new[i] != orig[i]:
                new[i] = orig[i]
                out.append(i)
                i = follower.get(orig[i])
            return out

        reverted: list[int] = []
        for i in sorted(declared, reverse=True):     # revert largest id first (lines 23-28)
            if self._valid(list(new.values()), F):
                break
            reverted += revert(i)
        # Re-apply reverted moves that are valid after all (smallest id first). A move is re-tried together
        # with the train of reverted agents following into its vacated cell (longest train first). Only a
        # leader whose move is invalid even with its whole train gets its edge pruned (IV-B); followers
        # blocked by a leader that stays are not pruned.
        pending = set(reverted)
        occ = set(new.values())
        for i in sorted(pending):
            if i not in pending:
                continue
            if declared[i] in occ:   # target still occupied: blocked (may still join its leader's train)
                continue
            pending.discard(i)
            train = [i]
            j = follower.get(orig[i])
            while j is not None and j in pending:
                train.append(j)
                j = follower.get(orig[j])
            for k in range(len(train), 0, -1):
                for t in train[:k]:
                    new[t] = declared[t]
                if self._valid(list(new.values()), F):
                    for t in train[:k]:
                        pending.discard(t)
                    break
                for t in train[:k]:
                    new[t] = orig[t]
            else:
                by_id[i].pruned.add((orig[i], declared[i]))   # IV-B edge pruning
            occ = set(new.values())
        moved = False
        for a in act:
            if new[a.id] != tuple(a.pos):
                a.pos = new[a.id]
                moved = True
        return moved

    def _valid(self, cells: list[Cell], F: np.ndarray) -> bool:
        """Visibility graph of cells ∪ {xd} connected and F ⊆ F~(cells ∪ {xd})."""
        nodes = [self._xd, *cells]
        fovs = [self.cache.fov(c) for c in nodes]
        cov = np.logical_or.reduce(fovs)
        if (F & ~cov).any():
            return False
        seen = {0}
        q = deque([0])
        while q:
            i = q.popleft()
            fi = fovs[i]
            for j, c in enumerate(nodes):
                if j not in seen and (c == nodes[i] or fi[c]):
                    seen.add(j)
                    q.append(j)
        return len(seen) == len(nodes)
