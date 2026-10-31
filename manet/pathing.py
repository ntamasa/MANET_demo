"""4-neighbour grid path planning (PLAN.md §6, paper Def. 11–13).

`passable` is a bool mask (H, W): the cells an agent may plan through. `blocked_edges` holds directed
edges (u, v) that may not be traversed from u to v (DADENCE per-agent edge pruning).
Neighbour order is fixed: N, E, S, W.
"""
from __future__ import annotations

from collections import deque

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
    pas = np.asarray(passable, dtype=bool)
    h, w = pas.shape
    pl = pas.tolist()  # list access is much faster than ndarray indexing in a Python loop
    dist = [[-1] * w for _ in range(h)]
    sr, sc = src
    dist[sr][sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        nd = dist[r][c] + 1
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and pl[nr][nc] and dist[nr][nc] < 0:
                if blocked_edges and ((r, c), (nr, nc)) in blocked_edges:
                    continue
                dist[nr][nc] = nd
                q.append((nr, nc))
    return np.array(dist, dtype=np.int64).reshape(h, w)


def _dist_to(passable: np.ndarray, src: Cell, dst: Cell, blocked_edges) -> list[list[int]]:
    """Reverse BFS from dst: dist[u] = length of shortest src-compatible path u -> dst.
    Intermediate cells must be passable; src may be non-passable (only as start); dst is the root."""
    pl = np.asarray(passable, dtype=bool).tolist()
    h, w = len(pl), len(pl[0])
    dist = [[-1] * w for _ in range(h)]
    dist[dst[0]][dst[1]] = 0
    q = deque([dst])
    while q:
        v = q.popleft()
        r, c = v
        nd = dist[r][c] + 1
        for dr, dc in NEIGHBORS:
            u = (r + dr, c + dc)
            ur, uc = u
            if not (0 <= ur < h and 0 <= uc < w) or dist[ur][uc] >= 0:
                continue
            if u == src:
                if blocked_edges and (u, v) in blocked_edges:
                    continue
                dist[ur][uc] = nd
                return dist  # src reached; nothing further needed
            if pl[ur][uc] and not (blocked_edges and (u, v) in blocked_edges):
                dist[ur][uc] = nd
                q.append(u)
    return dist


def shortest_path(passable: np.ndarray, src: Cell, dst: Cell,
                  blocked_edges: frozenset | set = frozenset()) -> list[Cell]:
    """Cells from src to dst inclusive (deterministic: N,E,S,W tie-break). [src] if src == dst.
    [] if unreachable. dst need not be passable (it is allowed as the final cell)."""
    src, dst = tuple(src), tuple(dst)
    if src == dst:
        return [src]
    dist = _dist_to(passable, src, dst, blocked_edges)
    h, w = len(dist), len(dist[0])
    if dist[src[0]][src[1]] < 0:
        return []
    # Greedy descent: at each step take the first N,E,S,W neighbour one step closer to dst.
    path = [src]
    cur = src
    while cur != dst:
        d = dist[cur[0]][cur[1]]
        for dr, dc in NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr][nc] == d - 1 \
                    and not (blocked_edges and (cur, (nr, nc)) in blocked_edges):
                cur = (nr, nc)
                break
        else:  # pragma: no cover - cannot happen with a consistent dist field
            raise RuntimeError("shortest_path: inconsistent distance field")
        path.append(cur)
    return path


def closer_neighbors(dist: np.ndarray, cell: Cell, passable: np.ndarray,
                     blocked_edges: frozenset | set = frozenset()) -> list[Cell]:
    """Def. 12: passable neighbours w of cell with 0 <= dist[w] < dist[cell], excluding blocked (cell, w),
    in N, E, S, W order. dist is a distance-to-target array (e.g. bfs_dist from the target)."""
    cell = tuple(cell)
    d = int(dist[cell])
    out = []
    for nb in neighbors(dist.shape, cell):
        dw = int(dist[nb])
        if 0 <= dw < d and passable[nb] and (cell, nb) not in blocked_edges:
            out.append(nb)
    return out


def border(grid: np.ndarray, covered: np.ndarray) -> list[Cell]:
    """Def. 13: covered free cells having a free, uncovered 4-neighbour. Sorted (row, col)."""
    g = np.asarray(grid, dtype=bool)
    cov = np.asarray(covered, dtype=bool) & g
    frontier = np.pad(g & ~cov, 1, constant_values=False)
    has_unc_nb = frontier[:-2, 1:-1] | frontier[2:, 1:-1] | frontier[1:-1, :-2] | frontier[1:-1, 2:]
    rows, cols = np.nonzero(cov & has_unc_nb)  # row-major -> already sorted
    return [(int(r), int(c)) for r, c in zip(rows, cols)]
