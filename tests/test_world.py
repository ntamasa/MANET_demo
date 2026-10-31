"""Tests for manet.world (PLAN.md §4, §13)."""
from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

from manet.world import (
    Corner,
    World,
    analyze,
    count_convex_corners,
    enforce_assumption3,
    find_reflex_corners,
    generate_world,
    grid_from_ascii,
    has_checkerboard,
    label_walls,
    largest_component,
    select_valid_corners,
)

# ---------------------------------------------------------------- hand-made maps

RECT = [
    "#####",
    "#...#",
    "#.D.#",
    "#####",
]

L_ROOM = [
    "######",
    "#D...#",
    "#....#",
    "#..###",
    "#..###",
    "######",
]

PILLAR = [
    "#########",
    "#D......#",
    "#.......#",
    "#..##...#",
    "#..##...#",
    "#.......#",
    "#.......#",
    "#########",
]

TWO_HOLES = [
    "############",
    "#D.........#",
    "#..........#",
    "#..##......#",
    "#..##......#",
    "#.......#..#",
    "#..........#",
    "#..........#",
    "############",
]


def _free_connected(grid: np.ndarray) -> bool:
    cells = list(zip(*np.nonzero(grid)))
    if not cells:
        return False
    seen = {cells[0]}
    q = deque([cells[0]])
    H, W = grid.shape
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _check_lemma1(world: World) -> None:
    n, h = world.n_corners, world.n_holes
    assert (n + 4 * h - 4) % 2 == 0
    assert len(world.reflex_corners) == (n + 4 * h - 4) // 2
    assert len(world.valid_corners) == (n + 2 * h - 4) // 2
    assert world.n_max == (n + 2 * h - 4) // 2


# ---------------------------------------------------------------- checkerboard / convex


def test_has_checkerboard_false_on_plain_room():
    g, _ = grid_from_ascii(RECT)
    assert not has_checkerboard(g)


def test_has_checkerboard_true_main_diagonal():
    g, _ = grid_from_ascii([
        "#####",
        "#.#.#",
        "##..#",
        "#####",
    ])
    # window rows 1-2, cols 1-2: [[1,0],[0,1]]
    assert has_checkerboard(g)


def test_has_checkerboard_true_anti_diagonal():
    g = np.array([[False, True], [True, False]])
    assert has_checkerboard(g)
    g2 = np.array([[True, False], [False, True]])
    assert has_checkerboard(g2)


def test_has_checkerboard_false_small_patterns():
    for g in (np.array([[True, True], [False, False]]),
              np.array([[True, False], [True, False]]),
              np.array([[True, True], [True, False]]),
              np.zeros((3, 3), bool), np.ones((3, 3), bool)):
        assert not has_checkerboard(g)


def test_count_convex_corners_rectangle():
    g, _ = grid_from_ascii(RECT)
    assert count_convex_corners(g) == 4


def test_count_convex_corners_l_room():
    g, _ = grid_from_ascii(L_ROOM)
    assert count_convex_corners(g) == 5


def test_count_convex_corners_pillar_room():
    g, _ = grid_from_ascii(PILLAR)
    assert count_convex_corners(g) == 4  # pillar corners are reflex w.r.t. free space


# ---------------------------------------------------------------- labelling


def test_label_walls_frame_not_hole_pillar_is_hole():
    g, _ = grid_from_ascii(PILLAR)
    labels, holes = label_walls(g)
    assert labels.shape == g.shape
    assert np.all(labels[g] == 0)
    assert np.all(labels[~g] > 0)
    frame = labels[0, 0]
    assert frame not in holes
    assert np.all(labels[0, :] == frame) and np.all(labels[:, -1] == frame)
    assert labels[3, 3] in holes
    assert labels[3, 3] == labels[4, 4] == labels[3, 4] == labels[4, 3]
    assert holes == {labels[3, 3]}


def test_label_walls_no_holes_in_l_room():
    g, _ = grid_from_ascii(L_ROOM)
    labels, holes = label_walls(g)
    assert holes == set()
    assert len(set(labels[~g].tolist())) == 1


def test_label_walls_two_holes():
    g, _ = grid_from_ascii(TWO_HOLES)
    labels, holes = label_walls(g)
    assert len(holes) == 2
    assert labels[3, 3] in holes and labels[5, 8] in holes
    assert labels[3, 3] != labels[5, 8]


# ---------------------------------------------------------------- corners / analyze


def test_rectangle_no_reflex():
    g, xd = grid_from_ascii(RECT)
    assert find_reflex_corners(g) == []
    w = analyze(g, xd)
    assert w.n_corners == 4 and w.n_holes == 0
    assert w.valid_corners == [] and w.valid_corner_cells == set()


def test_l_room():
    g, xd = grid_from_ascii(L_ROOM)
    w = analyze(g, xd)
    assert w.xd == (1, 1)
    assert w.n_corners == 6
    assert w.n_holes == 0
    assert len(w.reflex_corners) == 1
    assert len(w.valid_corners) == 1
    assert w.n_max == 1
    c = w.reflex_corners[0]
    assert c.vertex == (3, 3)
    assert c.wall_cell == (3, 3)
    # free cell diagonally opposite the wall cell within the 2x2 window
    assert c.agent_cell == (2, 2)
    assert c.hole_id is None
    assert w.valid_corner_cells == {(2, 2)}
    _check_lemma1(w)


@pytest.mark.parametrize("rot", [0, 1, 2, 3])
def test_agent_cell_diagonal_opposite_all_orientations(rot):
    g, _ = grid_from_ascii(L_ROOM)
    g = np.rot90(g, rot).copy()
    corners = find_reflex_corners(g)
    assert len(corners) == 1
    c = corners[0]
    i, j = c.vertex
    window = {(i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j)}
    assert c.wall_cell in window and c.agent_cell in window
    assert not g[c.wall_cell] and g[c.agent_cell]
    assert abs(c.agent_cell[0] - c.wall_cell[0]) == 1
    assert abs(c.agent_cell[1] - c.wall_cell[1]) == 1


def test_pillar_room():
    g, xd = grid_from_ascii(PILLAR)
    w = analyze(g, xd)
    assert w.n_corners == 8
    assert w.n_holes == 1
    assert len(w.reflex_corners) == 4
    assert len(w.valid_corners) == 3
    assert w.n_max == 3
    assert {c.vertex for c in w.reflex_corners} == {(3, 3), (3, 5), (5, 3), (5, 5)}
    hole_ids = {c.hole_id for c in w.reflex_corners}
    assert len(hole_ids) == 1 and None not in hole_ids
    excluded = [c for c in w.reflex_corners if c not in w.valid_corners]
    assert len(excluded) == 1
    ex = excluded[0]
    assert ex.vertex == (3, 3)          # top-left of the hole
    assert ex.wall_cell == (3, 3)
    assert ex.agent_cell == (2, 2)
    assert (2, 2) not in w.valid_corner_cells
    assert w.valid_corner_cells == {(2, 5), (5, 2), (5, 5)}
    _check_lemma1(w)


def test_reflex_sorted_by_vertex():
    g, _ = grid_from_ascii(TWO_HOLES)
    corners = find_reflex_corners(g)
    verts = [c.vertex for c in corners]
    assert verts == sorted(verts)


def test_two_holes_lemma1():
    g, xd = grid_from_ascii(TWO_HOLES)
    w = analyze(g, xd)
    assert w.n_holes == 2
    assert w.n_corners == 12
    assert len(w.reflex_corners) == 8
    assert len(w.valid_corners) == 6
    _check_lemma1(w)
    excluded = {c.vertex for c in w.reflex_corners if c not in w.valid_corners}
    assert excluded == {(3, 3), (5, 8)}


def test_select_valid_corners_tiebreak_by_column():
    # hole whose topmost reflex corners share the row: choose smallest j
    reflex = [
        Corner((5, 7), (4, 6), (5, 7), 2),
        Corner((5, 3), (4, 2), (5, 3), 2),
        Corner((8, 3), (8, 2), (7, 3), 2),
        Corner((2, 2), (1, 1), (2, 2), None),
        Corner((2, 9), (1, 10), (2, 9), None),
    ]
    valid = select_valid_corners(reflex)
    assert Corner((5, 3), (4, 2), (5, 3), 2) not in valid
    assert len(valid) == 4
    # frame (non-hole) corners are never removed
    assert all(c in valid for c in reflex if c.hole_id is None)


def test_valid_corner_cells_dedup_in_narrow_corridor():
    # 1-wide crossing: all four reflex corners share the centre agent cell
    g, xd = grid_from_ascii([
        "#######",
        "###D###",
        "###.###",
        "#.....#",
        "###.###",
        "###.###",
        "#######",
    ])
    w = analyze(g, xd)
    assert w.n_corners == 12 and w.n_holes == 0
    assert len(w.valid_corners) == 4
    assert w.valid_corner_cells == {(3, 3)}
    _check_lemma1(w)


# ---------------------------------------------------------------- post-processing


def test_enforce_assumption3_removes_checkerboard():
    g, _ = grid_from_ascii([
        "#######",
        "#.....#",
        "#..#..#",
        "#...#.#",
        "#.....#",
        "#######",
    ])
    assert has_checkerboard(g)
    out = enforce_assumption3(g, np.random.default_rng(0))
    assert out is not g
    assert has_checkerboard(g)  # input untouched
    assert not has_checkerboard(out)
    assert not out[0, :].any() and not out[-1, :].any()
    assert not out[:, 0].any() and not out[:, -1].any()
    assert np.all(out[g])  # free cells stay free (a wall cell was opened)


def test_enforce_assumption3_many_random_grids():
    rng = np.random.default_rng(1)
    for _ in range(30):
        g = rng.random((12, 14)) < 0.55
        g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = False
        out = enforce_assumption3(g, rng)
        assert not has_checkerboard(out)
        assert not out[0, :].any() and not out[-1, :].any()
        assert not out[:, 0].any() and not out[:, -1].any()


def test_largest_component_keeps_biggest():
    g, _ = grid_from_ascii([
        "##########",
        "#....#..##",
        "#....#..##",
        "#....#####",
        "######.###",
        "##########",
    ])
    out = largest_component(g)
    assert out.sum() == 12
    assert out[1:4, 1:5].all()
    assert not out[1:3, 6:8].any()
    assert not out[4, 6]
    assert g.sum() == 12 + 4 + 1  # input untouched


# ---------------------------------------------------------------- generator

SIZES = [(15, 15), (25, 40), (40, 40), (60, 60)]
SEEDS = list(range(10))


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("seed", SEEDS)
def test_generate_world_invariants(size, seed):
    h, w = size
    world = generate_world(h, w, seed)
    g = world.grid
    assert g.shape == (h, w) and g.dtype == bool
    assert world.shape == (h, w)
    assert not g[0, :].any() and not g[-1, :].any()
    assert not g[:, 0].any() and not g[:, -1].any()
    assert not has_checkerboard(g)
    assert _free_connected(g)
    assert len(world.reflex_corners) >= 1
    assert g[world.xd]
    _check_lemma1(world)
    assert world.n_max >= 1
    assert world.valid_corner_cells == {c.agent_cell for c in world.valid_corners}
    for c in world.reflex_corners:
        assert g[c.agent_cell] and not g[c.wall_cell]


@pytest.mark.parametrize("size", SIZES)
def test_generate_world_deterministic(size):
    a = generate_world(*size, seed=42)
    b = generate_world(*size, seed=42)
    assert np.array_equal(a.grid, b.grid)
    assert a.xd == b.xd
    assert a.reflex_corners == b.reflex_corners


def test_generate_world_seeds_differ():
    grids = [generate_world(40, 40, s).grid for s in range(5)]
    distinct = {g.tobytes() for g in grids}
    assert len(distinct) >= 4


def test_generate_world_pillars_make_holes():
    found = False
    for seed in range(20):
        world = generate_world(40, 40, seed, pillar_prob=1.0)
        if world.n_holes >= 1:
            found = True
            _check_lemma1(world)
            assert any(c.hole_id is not None for c in world.reflex_corners)
            break
    assert found


def test_generate_world_fast():
    best = min(_timed(lambda s=s: generate_world(60, 60, s)) for s in range(3))
    assert best < 0.5


def _timed(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t
