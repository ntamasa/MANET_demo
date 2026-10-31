"""Line of sight, field of view, coverage and visibility graph (PLAN.md §5, paper Def. 6, 10, 14).

LOS(a, b): segment between cell centres sampled with K = 4*max(|dr|,|dc|)+1 points, each mapped to
cell floor(p + 0.5); samples lying exactly on a lattice point (both fractional parts 0.5 ± 1e-6) are
ignored. Visible iff every mapped cell is free. Symmetric by construction.

Implementation note: with m = max(|dr|,|dc|) and t_i = i / (4m), every sample is the rational
p_i = (4m*a + i*(b-a)) / (4m). Cell and lattice tests are done in exact integer arithmetic
(floor((num + 2m) / 4m); lattice iff (num + 2m) % 4m == 0 in both coordinates), which equals the
float rule above without rounding error. Reversing a, b maps sample i to 4m - i, so LOS is exactly
symmetric, and compute_fov agrees bit-for-bit with los.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np

from manet.world import Cell

_CHUNK_SAMPLES = 1 << 21  # max N*K samples per vectorised batch


def _visible_batch(grid: np.ndarray, src: np.ndarray, targets: np.ndarray, m: int) -> np.ndarray:
    """Visibility from src to each target (all at Chebyshev distance m >= 1). targets: (N, 2) int64."""
    d = 4 * m
    i = np.arange(d + 1, dtype=np.int64)  # (K,)
    s = (d * src + 2 * m)[None, None, :] + i[None, :, None] * (targets - src)[:, None, :]  # (N, K, 2)
    cells = s // d
    lattice = (s % d == 0).all(axis=2)
    ok = grid[cells[..., 0], cells[..., 1]] | lattice
    return ok.all(axis=1)


def los(grid: np.ndarray, a: Cell, b: Cell) -> bool:
    """Line of sight between two cells (both must be free, else False)."""
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    if not (grid[a] and grid[b]):
        return False
    m = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    if m == 0:
        return True
    src = np.asarray(a, dtype=np.int64)
    tgt = np.asarray([b], dtype=np.int64)
    return bool(_visible_batch(grid, src, tgt, m)[0])


def compute_fov(grid: np.ndarray, src: Cell) -> np.ndarray:
    """Bool mask (H, W) of free cells b with los(grid, src, b). Vectorised (chunked). src itself is True."""
    grid = np.asarray(grid, dtype=bool)
    src = (int(src[0]), int(src[1]))
    out = np.zeros(grid.shape, dtype=bool)
    if not grid[src]:
        return out
    out[src] = True
    s = np.asarray(src, dtype=np.int64)
    targets = np.argwhere(grid).astype(np.int64)
    dist = np.abs(targets - s).max(axis=1)
    order = np.argsort(dist, kind="stable")
    targets, dist = targets[order], dist[order]
    bounds = np.flatnonzero(np.diff(dist)) + 1
    # one group per Chebyshev distance m -> common K = 4m+1 inside the group (identical to los)
    for grp_t, grp_d in zip(np.split(targets, bounds), np.split(dist, bounds)):
        m = int(grp_d[0])
        if m == 0:
            continue
        step = max(1, _CHUNK_SAMPLES // (4 * m + 1))
        for k in range(0, len(grp_t), step):
            chunk = grp_t[k:k + step]
            vis = _visible_batch(grid, s, chunk, m)
            out[chunk[vis, 0], chunk[vis, 1]] = True
    return out


class FovCache:
    """Memoises compute_fov per source cell. Returned masks must be treated as read-only."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self._cache: dict[Cell, np.ndarray] = {}

    def fov(self, cell: Cell) -> np.ndarray:
        cell = (int(cell[0]), int(cell[1]))
        mask = self._cache.get(cell)
        if mask is None:
            mask = compute_fov(self.grid, cell)
            mask.flags.writeable = False
            self._cache[cell] = mask
        return mask

    def sees(self, a: Cell, b: Cell) -> bool:
        return bool(self.fov(a)[b[0], b[1]])


def coverage(cache: FovCache, cells: Iterable[Cell]) -> np.ndarray:
    """OR of the FOVs of the given cells; all-False mask if cells is empty."""
    out = np.zeros(np.shape(cache.grid), dtype=bool)
    for c in cells:
        out |= cache.fov(c)
    return out


def visibility_graph(cache: FovCache, nodes: dict[Hashable, Cell]) -> nx.Graph:
    """Undirected graph, one node per key, edge iff the two cells see each other.
    Nodes on the same cell are connected."""
    G = nx.Graph()
    items = list(nodes.items())
    G.add_nodes_from(k for k, _ in items)
    for i, (ka, ca) in enumerate(items):
        fa = cache.fov(ca)
        for kb, cb in items[i + 1:]:
            if tuple(ca) == tuple(cb) or fa[cb[0], cb[1]]:
                G.add_edge(ka, kb)
    return G


def is_connected(cache: FovCache, cells: list[Cell]) -> bool:
    """Whether the visibility graph of the cells is connected (True for 0 or 1 cell)."""
    cells = list(cells)
    if len(cells) <= 1:
        return True
    return nx.is_connected(visibility_graph(cache, dict(enumerate(cells))))


def observed_walls(grid: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Wall cells having at least one covered 8-neighbour (rendering only)."""
    grid = np.asarray(grid, dtype=bool)
    h, w = grid.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = covered
    near = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                near |= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return near & ~grid
