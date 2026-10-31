"""Grid world: geometry analysis (corners, holes, valid corners, Nmax) and dungeon generation.

Conventions (PLAN.md §3): grid is a bool ndarray (H, W), True = free cell. Cells are (row, col).
Vertex (i, j) is the lattice point between rows i-1, i and columns j-1, j (1 <= i <= H-1, 1 <= j <= W-1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Cell = tuple[int, int]


@dataclass(frozen=True)
class Corner:
    vertex: tuple[int, int]      # lattice point (i, j)
    agent_cell: Cell             # free cell diagonally opposite the wall cell in the 2x2 window
    wall_cell: Cell              # the single wall cell of the 2x2 window
    hole_id: int | None          # wall-component label if that component is a hole, else None


@dataclass
class World:
    grid: np.ndarray
    xd: Cell
    n_corners: int                               # n = #90° + #270° corners
    n_holes: int                                 # h
    reflex_corners: list[Corner]                 # all 270° corners
    valid_corners: list[Corner]                  # reflex minus top-left corner of each hole (Def. 8)
    valid_corner_cells: set[Cell] = field(default_factory=set)  # dedup of valid_corners' agent_cell
    n_max: int = 0                               # (n + 2h - 4) // 2

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape


def grid_from_ascii(rows: list[str]) -> tuple[np.ndarray, Cell | None]:
    """'#' = wall, '.' = free, 'D' = free deployment point. Returns (grid, xd or None)."""
    h, w = len(rows), len(rows[0])
    grid = np.zeros((h, w), dtype=bool)
    xd = None
    for r, line in enumerate(rows):
        assert len(line) == w, "ragged ascii map"
        for c, ch in enumerate(line):
            if ch in ".D":
                grid[r, c] = True
            if ch == "D":
                xd = (r, c)
    return grid, xd


def has_checkerboard(grid: np.ndarray) -> bool:
    """True if any 2x2 window is a diagonal pattern [[1,0],[0,1]] or [[0,1],[1,0]] (violates Assumption 3)."""
    raise NotImplementedError


def count_convex_corners(grid: np.ndarray) -> int:
    """Number of vertices whose 2x2 window has exactly 1 free cell (90° corners)."""
    raise NotImplementedError


def label_walls(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    """4-connected labelling of wall cells. Returns (labels, hole_ids).
    labels: int ndarray (H, W), 0 for free cells, 1.. for wall components.
    hole_ids: labels of components that do NOT touch the outer border of the array."""
    raise NotImplementedError


def find_reflex_corners(grid: np.ndarray) -> list[Corner]:
    """All 270° corners (2x2 window with exactly 3 free cells), with hole_id filled in.
    Sorted by vertex (i, j)."""
    raise NotImplementedError


def select_valid_corners(reflex: list[Corner]) -> list[Corner]:
    """Def. 8: drop, for each hole, its top-left reflex corner = smallest vertex row i, tie -> smallest j."""
    raise NotImplementedError


def analyze(grid: np.ndarray, xd: Cell) -> World:
    """Build a World from a grid: corners, holes, valid corners, n_max = (n + 2h - 4) // 2 (at least 1)."""
    raise NotImplementedError


def enforce_assumption3(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy without checkerboard 2x2 windows (open one interior wall cell of the pattern;
    if both wall cells are on the outer frame, close the free cells instead). Outer frame stays wall."""
    raise NotImplementedError


def largest_component(grid: np.ndarray) -> np.ndarray:
    """Copy keeping only the largest 4-connected free component."""
    raise NotImplementedError


def generate_world(h: int, w: int, seed: int, room_min: int = 5, corridor_width: int = 2,
                   pillar_prob: float = 0.3) -> World:
    """BSP dungeon (PLAN.md §4.2). Deterministic for a seed. Guarantees: frame is wall, no checkerboard,
    free space 4-connected, at least one reflex corner, xd is free."""
    raise NotImplementedError
