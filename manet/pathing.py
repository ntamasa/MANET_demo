"""4-neighbour grid path planning (PLAN.md §6, paper Def. 11–13).

`passable` is a bool mask (H, W): the cells an agent may plan through. `blocked_edges` holds directed
edges (u, v) that may not be traversed from u to v (DADENCE per-agent edge pruning).
Neighbour order is fixed: N, E, S, W.
"""
from __future__ import annotations

import numpy as np

from manet.world import Cell

NEIGHBORS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def neighbors(shape: tuple[int, int], cell: Cell) -> list[Cell]:
    """In-bounds 4-neighbours in N, E, S, W order."""
    h, w = shape
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS if 0 <= r + dr < h and 0 <= c + dc < w]


def bfs_dist(passable: np.ndarray, src: Cell, blocked_edges: frozenset | set = frozenset()) -> np.ndarray:
    """int ndarray of BFS distances from src over passable cells; -1 = unreachable.
    If src itself is not passable it still gets distance 0 and expansion starts from it."""
    raise NotImplementedError


def shortest_path(passable: np.ndarray, src: Cell, dst: Cell,
                  blocked_edges: frozenset | set = frozenset()) -> list[Cell]:
    """Cells from src to dst inclusive (deterministic: N,E,S,W tie-break). [src] if src == dst.
    [] if unreachable. dst need not be passable (it is allowed as the final cell)."""
    raise NotImplementedError


def closer_neighbors(dist: np.ndarray, cell: Cell, passable: np.ndarray,
                     blocked_edges: frozenset | set = frozenset()) -> list[Cell]:
    """Def. 12: passable neighbours w of cell with 0 <= dist[w] < dist[cell], excluding blocked (cell, w),
    in N, E, S, W order. dist is a distance-to-target array (e.g. bfs_dist from the target)."""
    raise NotImplementedError


def border(grid: np.ndarray, covered: np.ndarray) -> list[Cell]:
    """Def. 13: covered free cells having a free, uncovered 4-neighbour. Sorted (row, col)."""
    raise NotImplementedError
