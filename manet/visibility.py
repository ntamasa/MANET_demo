"""Line of sight, field of view, coverage and visibility graph (PLAN.md §5, paper Def. 6, 10, 14).

LOS(a, b): segment between cell centres sampled with K = 4*max(|dr|,|dc|)+1 points, each mapped to
cell floor(p + 0.5); samples lying exactly on a lattice point (both fractional parts 0.5 ± 1e-6) are
ignored. Visible iff every mapped cell is free. Symmetric by construction.
"""
from __future__ import annotations

from collections.abc import Hashable, Iterable

import networkx as nx
import numpy as np

from manet.world import Cell


def los(grid: np.ndarray, a: Cell, b: Cell) -> bool:
    """Line of sight between two cells (both must be free, else False)."""
    raise NotImplementedError


def compute_fov(grid: np.ndarray, src: Cell) -> np.ndarray:
    """Bool mask (H, W) of free cells b with los(grid, src, b). Vectorised (chunked). src itself is True."""
    raise NotImplementedError


class FovCache:
    """Memoises compute_fov per source cell. Returned masks must be treated as read-only."""

    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self._cache: dict[Cell, np.ndarray] = {}

    def fov(self, cell: Cell) -> np.ndarray:
        raise NotImplementedError

    def sees(self, a: Cell, b: Cell) -> bool:
        raise NotImplementedError


def coverage(cache: FovCache, cells: Iterable[Cell]) -> np.ndarray:
    """OR of the FOVs of the given cells; all-False mask if cells is empty."""
    raise NotImplementedError


def visibility_graph(cache: FovCache, nodes: dict[Hashable, Cell]) -> nx.Graph:
    """Undirected graph, one node per key, edge iff the two cells see each other.
    Nodes on the same cell are connected."""
    raise NotImplementedError


def is_connected(cache: FovCache, cells: list[Cell]) -> bool:
    """Whether the visibility graph of the cells is connected (True for 0 or 1 cell)."""
    raise NotImplementedError


def observed_walls(grid: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Wall cells having at least one covered 8-neighbour (rendering only)."""
    raise NotImplementedError
