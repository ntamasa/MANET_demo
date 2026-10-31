"""Grid world: geometry analysis (corners, holes, valid corners, Nmax) and dungeon generation.

Conventions (PLAN.md §3): grid is a bool ndarray (H, W), True = free cell. Cells are (row, col).
Vertex (i, j) is the lattice point between rows i-1, i and columns j-1, j (1 <= i <= H-1, 1 <= j <= W-1).
"""
from __future__ import annotations

import math
from collections import deque
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


# ---------------------------------------------------------------- 2x2 window helpers

def _windows(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four cells of every 2x2 window; element [i-1, j-1] belongs to vertex (i, j)."""
    g = np.asarray(grid, dtype=bool)
    return g[:-1, :-1], g[:-1, 1:], g[1:, :-1], g[1:, 1:]


def _checkerboard_mask(grid: np.ndarray) -> np.ndarray:
    a, b, c, d = _windows(grid)
    return (a & d & ~b & ~c) | (b & c & ~a & ~d)


def _free_count(grid: np.ndarray) -> np.ndarray:
    a, b, c, d = _windows(grid)
    return a.astype(np.int8) + b + c + d


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling of True cells (BFS). Returns (labels, count); 0 = background."""
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    flat = mask.ravel().tolist()
    lab = [0] * (H * W)
    n = 0
    for start in np.flatnonzero(mask).tolist():
        if lab[start]:
            continue
        n += 1
        lab[start] = n
        q = deque([start])
        while q:
            k = q.popleft()
            r, c = divmod(k, W)
            if r > 0 and flat[k - W] and not lab[k - W]:
                lab[k - W] = n
                q.append(k - W)
            if r < H - 1 and flat[k + W] and not lab[k + W]:
                lab[k + W] = n
                q.append(k + W)
            if c > 0 and flat[k - 1] and not lab[k - 1]:
                lab[k - 1] = n
                q.append(k - 1)
            if c < W - 1 and flat[k + 1] and not lab[k + 1]:
                lab[k + 1] = n
                q.append(k + 1)
    labels[:] = np.asarray(lab, dtype=np.int32).reshape(H, W)
    return labels, n


def _set_frame(grid: np.ndarray) -> None:
    grid[0, :] = grid[-1, :] = False
    grid[:, 0] = grid[:, -1] = False


# ---------------------------------------------------------------- geometry

def has_checkerboard(grid: np.ndarray) -> bool:
    """True if any 2x2 window is a diagonal pattern [[1,0],[0,1]] or [[0,1],[1,0]] (violates Assumption 3)."""
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        return False
    return bool(_checkerboard_mask(grid).any())


def count_convex_corners(grid: np.ndarray) -> int:
    """Number of vertices whose 2x2 window has exactly 1 free cell (90° corners)."""
    return int((_free_count(grid) == 1).sum())


def label_walls(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    """4-connected labelling of wall cells. Returns (labels, hole_ids).
    labels: int ndarray (H, W), 0 for free cells, 1.. for wall components.
    hole_ids: labels of components that do NOT touch the outer border of the array."""
    labels, n = _label(~np.asarray(grid, dtype=bool))
    border = set(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]).tolist())
    holes = {k for k in range(1, n + 1) if k not in border}
    return labels, holes


# window cells relative to vertex (i, j): (-1,-1), (-1,0), (0,-1), (0,0).
# wall-cell offset -> offset of the diagonally opposite (agent) cell
_OPPOSITE = {(-1, -1): (0, 0), (-1, 0): (0, -1), (0, -1): (-1, 0), (0, 0): (-1, -1)}


def _reflex(grid: np.ndarray, labels: np.ndarray, holes: set[int]) -> list[Corner]:
    a, b, c, _ = _windows(grid)
    out: list[Corner] = []
    for i0, j0 in np.argwhere(_free_count(grid) == 3).tolist():  # row-major => sorted by vertex
        i, j = i0 + 1, j0 + 1
        if not a[i0, j0]:
            off = (-1, -1)
        elif not b[i0, j0]:
            off = (-1, 0)
        elif not c[i0, j0]:
            off = (0, -1)
        else:
            off = (0, 0)
        wall = (i + off[0], j + off[1])
        ao = _OPPOSITE[off]
        lab = int(labels[wall])
        out.append(Corner((i, j), (i + ao[0], j + ao[1]), wall, lab if lab in holes else None))
    return out


def find_reflex_corners(grid: np.ndarray) -> list[Corner]:
    """All 270° corners (2x2 window with exactly 3 free cells), with hole_id filled in.
    Sorted by vertex (i, j)."""
    labels, holes = label_walls(grid)
    return _reflex(grid, labels, holes)


def select_valid_corners(reflex: list[Corner]) -> list[Corner]:
    """Def. 8: drop, for each hole, its top-left reflex corner = smallest vertex row i, tie -> smallest j."""
    top_left: dict[int, tuple[int, int]] = {}
    for c in reflex:
        if c.hole_id is not None:
            cur = top_left.get(c.hole_id)
            if cur is None or c.vertex < cur:
                top_left[c.hole_id] = c.vertex
    return [c for c in reflex if c.hole_id is None or top_left[c.hole_id] != c.vertex]


def analyze(grid: np.ndarray, xd: Cell) -> World:
    """Build a World from a grid: corners, holes, valid corners, n_max = (n + 2h - 4) // 2 (at least 1)."""
    grid = np.asarray(grid, dtype=bool)
    assert not has_checkerboard(grid), "grid violates Assumption 3 (checkerboard 2x2 window)"
    labels, holes = label_walls(grid)
    reflex = _reflex(grid, labels, holes)
    valid = select_valid_corners(reflex)
    n = count_convex_corners(grid) + len(reflex)
    h = len(holes)
    return World(
        grid=grid,
        xd=(int(xd[0]), int(xd[1])),
        n_corners=n,
        n_holes=h,
        reflex_corners=reflex,
        valid_corners=valid,
        valid_corner_cells={c.agent_cell for c in valid},
        n_max=max(1, (n + 2 * h - 4) // 2),
    )


# ---------------------------------------------------------------- post-processing

def enforce_assumption3(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return a copy without checkerboard 2x2 windows (open one interior wall cell of the pattern;
    if both wall cells are on the outer frame, close the free cells instead). Outer frame stays wall."""
    g = np.array(grid, dtype=bool, copy=True)
    H, W = g.shape
    if H < 2 or W < 2:
        return g
    _set_frame(g)

    def interior(cell: Cell) -> bool:
        return 0 < cell[0] < H - 1 and 0 < cell[1] < W - 1

    while True:
        bad = np.argwhere(_checkerboard_mask(g)).tolist()
        if not bad:
            return g
        for i0, j0 in bad:
            cells = [(i0, j0), (i0, j0 + 1), (i0 + 1, j0), (i0 + 1, j0 + 1)]
            vals = [bool(g[x]) for x in cells]
            if not (vals[0] == vals[3] and vals[1] == vals[2] and vals[0] != vals[1]):
                continue  # already fixed by an earlier change in this pass
            walls = [x for x, v in zip(cells, vals) if not v]
            opts = [x for x in walls if interior(x)]
            if opts:
                g[opts[int(rng.integers(len(opts)))]] = True
            else:
                for x, v in zip(cells, vals):
                    if v:
                        g[x] = False


def largest_component(grid: np.ndarray) -> np.ndarray:
    """Copy keeping only the largest 4-connected free component."""
    labels, n = _label(np.asarray(grid, dtype=bool))
    if n == 0:
        return np.zeros(np.shape(grid), dtype=bool)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


# ---------------------------------------------------------------- dungeon generator

Rect = tuple[int, int, int, int]  # r0, c0, r1, c1 inclusive


def _bsp(rect: Rect, depth: int, min_leaf: int, rng: np.random.Generator) -> dict:
    """Binary space partition. Sibling leaves are separated by a 1-cell wall line."""
    r0, c0, r1, c1 = rect
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    can_h = hh >= 2 * min_leaf + 1   # cut across rows
    can_v = ww >= 2 * min_leaf + 1   # cut across columns
    if depth <= 0 or not (can_h or can_v):
        return {"rect": rect}
    horiz = bool(rng.random() < hh / (hh + ww)) if (can_h and can_v) else can_h
    if horiz:
        s = int(rng.integers(r0 + min_leaf, r1 - min_leaf + 1))  # separator row
        a, b = (r0, c0, s - 1, c1), (s + 1, c0, r1, c1)
    else:
        s = int(rng.integers(c0 + min_leaf, c1 - min_leaf + 1))  # separator column
        a, b = (r0, c0, r1, s - 1), (r0, s + 1, r1, c1)
    return {"rect": rect, "kids": (_bsp(a, depth - 1, min_leaf, rng), _bsp(b, depth - 1, min_leaf, rng))}


def _place_rooms(node: dict, room_min: int, rng: np.random.Generator) -> list[Rect]:
    if "kids" in node:
        rooms: list[Rect] = []
        for k in node["kids"]:
            rooms += _place_rooms(k, room_min, rng)
        node["rooms"] = rooms
        return rooms
    r0, c0, r1, c1 = node["rect"]
    hh, ww = r1 - r0 + 1, c1 - c0 + 1
    rh = int(rng.integers(min(room_min, hh), hh + 1))
    rw = int(rng.integers(min(room_min, ww), ww + 1))
    rr = int(rng.integers(r0, r1 - rh + 2))
    cc = int(rng.integers(c0, c1 - rw + 2))
    room = (rr, cc, rr + rh - 1, cc + rw - 1)
    node["rooms"] = [room]
    return [room]


def _point_in(room: Rect, cw: int, rng: np.random.Generator) -> Cell:
    r0, c0, r1, c1 = room
    r = int(rng.integers(r0, max(r0, r1 - cw + 1) + 1))
    c = int(rng.integers(c0, max(c0, c1 - cw + 1) + 1))
    return r, c


def _connect(node: dict, grid: np.ndarray, corr: np.ndarray, cw: int, rng: np.random.Generator) -> None:
    """L-shaped corridors (width cw) between a random room of each pair of sibling subtrees."""
    if "kids" not in node:
        return
    ka, kb = node["kids"]
    _connect(ka, grid, corr, cw, rng)
    _connect(kb, grid, corr, cw, rng)
    ra = ka["rooms"][int(rng.integers(len(ka["rooms"])))]
    rb = kb["rooms"][int(rng.integers(len(kb["rooms"])))]
    (y0, x0), (y1, x1) = _point_in(ra, cw, rng), _point_in(rb, cw, rng)
    H, W = grid.shape
    if rng.random() < 0.5:   # horizontal leg first, then vertical
        segs = [(y0, min(x0, x1), y0 + cw - 1, max(x0, x1) + cw - 1),
                (min(y0, y1), x1, max(y0, y1) + cw - 1, x1 + cw - 1)]
    else:                    # vertical leg first, then horizontal
        segs = [(min(y0, y1), x0, max(y0, y1) + cw - 1, x0 + cw - 1),
                (y1, min(x0, x1), y1 + cw - 1, max(x0, x1) + cw - 1)]
    for r0, c0, r1, c1 in segs:
        r0, c0 = max(1, r0), max(1, c0)
        r1, c1 = min(H - 2, r1), min(W - 2, c1)
        grid[r0:r1 + 1, c0:c1 + 1] = True
        corr[r0:r1 + 1, c0:c1 + 1] = True


def _add_pillar(grid: np.ndarray, corr: np.ndarray, room: Rect, rng: np.random.Generator) -> None:
    """One rectangular wall pillar (2x2..4x4), >= 2 cells from the room walls, not touching corridors."""
    r0, c0, r1, c1 = room
    for _ in range(8):
        ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        lo_r, hi_r = r0 + 2, r1 - 2 - ph + 1
        lo_c, hi_c = c0 + 2, c1 - 2 - pw + 1
        if hi_r < lo_r or hi_c < lo_c:
            continue
        pr, pc = int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))
        ring = (slice(pr - 1, pr + ph + 1), slice(pc - 1, pc + pw + 1))
        if corr[ring].any() or not grid[ring].all():
            continue
        grid[pr:pr + ph, pc:pc + pw] = False
        return


def _postprocess(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Frame -> Assumption 3 -> largest component, repeated until stable."""
    g = grid
    while True:
        prev = g
        g = g.copy()
        _set_frame(g)
        g = enforce_assumption3(g, rng)
        g = largest_component(g)
        if np.array_equal(g, prev):
            return g


def _carve(h: int, w: int, rng: np.random.Generator, room_min: int, corridor_width: int,
           pillar_prob: float) -> np.ndarray:
    grid = np.zeros((h, w), dtype=bool)
    depth = max(1, int(round(math.log2(max(2.0, h * w / 100.0)))))
    root = _bsp((1, 1, h - 2, w - 2), depth, max(1, room_min), rng)
    rooms = _place_rooms(root, max(1, room_min), rng)
    for r0, c0, r1, c1 in rooms:
        grid[r0:r1 + 1, c0:c1 + 1] = True
    corr = np.zeros_like(grid)
    _connect(root, grid, corr, max(1, corridor_width), rng)
    for room in rooms:
        big = room[2] - room[0] + 1 >= 9 and room[3] - room[1] + 1 >= 9
        if big and rng.random() < pillar_prob:
            _add_pillar(grid, corr, room, rng)
    return grid


def _force_reflex(grid: np.ndarray) -> np.ndarray:
    """Fallback for Assumption 2: wall one free cell whose 8 neighbours are all free (1x1 hole)."""
    g = grid.copy()
    H, W = g.shape
    full = np.ones((H - 2, W - 2), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            full &= g[1 + dr:H - 1 + dr, 1 + dc:W - 1 + dc]
    cand = np.argwhere(full)
    if len(cand):
        r, c = cand[len(cand) // 2]
        g[r + 1, c + 1] = False
    return g


def generate_world(h: int, w: int, seed: int, room_min: int = 5, corridor_width: int = 2,
                   pillar_prob: float = 0.3) -> World:
    """BSP dungeon (PLAN.md §4.2). Deterministic for a seed. Guarantees: frame is wall, no checkerboard,
    free space 4-connected, at least one reflex corner, xd is free."""
    if h < 4 or w < 4:
        raise ValueError("world must be at least 4x4")
    grid = np.zeros((h, w), dtype=bool)
    rng = np.random.default_rng(seed)
    for k in range(20):  # Assumption 2: regenerate with seed + 1000*k until non-convex
        rng = np.random.default_rng(seed + 1000 * k)
        grid = _postprocess(_carve(h, w, rng, room_min, corridor_width, pillar_prob), rng)
        if grid.any() and (_free_count(grid) == 3).any():
            break
    else:
        grid = _force_reflex(grid)
        if not (_free_count(grid) == 3).any():
            raise ValueError(f"cannot generate a non-convex world of size {h}x{w}")
    free = np.argwhere(grid)
    r, c = free[int(rng.integers(len(free)))]
    return analyze(grid, (int(r), int(c)))
