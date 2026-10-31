import math

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from PIL import Image, ImageSequence

from manet.render import COLORS, background_rgb, make_gif, render
from manet.sim import Snapshot, Status
from manet.world import World, grid_from_ascii

ASCII = [
    "########",
    "#D.....#",
    "#......#",
    "#..##..#",
    "#......#",
    "########",
]


@pytest.fixture
def world():
    grid, xd = grid_from_ascii(ASCII)
    return World(grid=grid, xd=xd, n_corners=8, n_holes=1, reflex_corners=[], valid_corners=[],
                 valid_corner_cells=set(), n_max=3)


def make_snap(world, step=3, **kw):
    h, w = world.shape
    covered = np.zeros((h, w), bool)
    covered[1, 1:4] = True
    explored = covered.copy()
    explored[2, 1:4] = True
    d = dict(
        step=step,
        agents=[(0, (1, 3), Status.SETTLED), (1, (2, 2), Status.MOVING), (2, (4, 5), Status.RETURNING),
                (3, (4, 1), Status.MOVING)],
        covered=covered,
        explored=explored,
        targets=[(1, 6), (4, 6)],
        known_valid=[(2, 3)],
        edges=[((1, 1), (1, 3)), ((1, 3), (2, 2))],
        n_active=3,
        coverage_pct=12.345,
    )
    d.update(kw)
    return Snapshot(**d)


@pytest.fixture(autouse=True)
def close_all():
    yield
    plt.close("all")


def by_label(ax, label):
    return [c for c in ax.collections if c.get_label() == label]


def offsets(coll):
    return {tuple(np.round(p, 6)) for p in coll.get_offsets()}


# --- background ---------------------------------------------------------------------------------

def test_background_shape_range(world):
    img = background_rgb(world, make_snap(world))
    assert img.shape == (6, 8, 3)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_background_colours(world):
    img = background_rgb(world, make_snap(world))
    c = lambda k: np.array(to_rgb(COLORS[k]))  # noqa: E731
    assert np.allclose(img[1, 2], c("covered"))
    assert np.allclose(img[2, 2], c("explored"))      # explored, not covered
    assert np.allclose(img[4, 4], c("unseen_free"))
    assert np.allclose(img[0, 2], c("seen_wall"))     # neighbours explored (1,2)
    assert np.allclose(img[3, 3], c("seen_wall"))     # 8-neighbour (2,2)/(2,3) explored
    assert np.allclose(img[3, 0], c("seen_wall"))     # diagonal (2,1) explored
    assert np.allclose(img[5, 5], c("unseen_wall"))
    assert np.allclose(img[3, 4], c("seen_wall"))     # diagonal of (2,3)
    assert np.allclose(img[0, 7], c("unseen_wall"))


def test_background_covered_beats_explored(world):
    s = make_snap(world)
    s.explored[:] = False  # covered but not in explored -> still covered colour
    img = background_rgb(world, s)
    assert np.allclose(img[1, 2], to_rgb(COLORS["covered"]))


# --- static render ------------------------------------------------------------------------------

def test_render_static(world):
    fig = render(world, None, title="hello map")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert len(ax.images) == 1
    assert ax.images[0].get_array().shape[:2] == (6, 8)
    assert "hello map" in ax.get_title()
    xd = by_label(ax, "xd")
    assert len(xd) == 1 and offsets(xd[0]) == {(1.0, 1.0)}  # (r,c)=(1,1) -> x=c, y=r
    assert not by_label(ax, "edges")
    # everything unseen
    img = ax.images[0].get_array()
    assert np.allclose(img[2, 2][:3], to_rgb(COLORS["unseen_free"]))
    plt.close(fig)


def test_no_figure_leak(world):
    plt.close("all")
    fig = render(world, make_snap(world))
    plt.close(fig)
    assert plt.get_fignums() == []


# --- snapshot render ----------------------------------------------------------------------------

def test_agent_markers(world):
    fig = render(world, make_snap(world))
    ax = fig.axes[0]
    exp = {Status.SETTLED: {(3.0, 1.0)}, Status.MOVING: {(2.0, 2.0), (1.0, 4.0)},
           Status.RETURNING: {(5.0, 4.0)}}
    for st, pts in exp.items():
        colls = by_label(ax, st.value)
        assert len(colls) == 1
        assert offsets(colls[0]) == pts
        fc = colls[0].get_facecolors()
        assert np.allclose(fc[0][:3], to_rgb(COLORS[st.value]))
    xd = by_label(ax, "xd")
    assert len(xd) == 1
    assert np.allclose(xd[0].get_facecolors()[0][:3], to_rgb(COLORS["xd"]))
    plt.close(fig)


def test_missing_status_not_drawn(world):
    s = make_snap(world, agents=[(0, (1, 3), Status.SETTLED)])
    fig = render(world, s)
    ax = fig.axes[0]
    assert len(by_label(ax, "settled")) == 1
    assert by_label(ax, "moving") == [] or len(by_label(ax, "moving")[0].get_offsets()) == 0
    plt.close(fig)


def test_edges(world):
    s = make_snap(world)
    fig = render(world, s, show_edges=True)
    lc = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    assert len(lc) == 1
    assert len(lc[0].get_segments()) == len(s.edges)
    seg = np.asarray(lc[0].get_segments()[0])
    assert np.allclose(seg, [[1, 1], [3, 1]])
    plt.close(fig)
    fig = render(world, s, show_edges=False)
    assert not [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
    plt.close(fig)


def test_targets_and_corners(world):
    s = make_snap(world)
    fig = render(world, s, show_corners=True)
    ax = fig.axes[0]
    t = by_label(ax, "targets")
    assert len(t) == 1 and len(t[0].get_offsets()) == 2
    assert offsets(t[0]) == {(6.0, 1.0), (6.0, 4.0)}
    kv = by_label(ax, "valid corners")
    assert len(kv) == 1 and offsets(kv[0]) == {(3.0, 2.0)}
    plt.close(fig)
    fig = render(world, s, show_corners=False)
    ax = fig.axes[0]
    assert by_label(ax, "valid corners") == []
    assert len(by_label(ax, "targets")) == 1
    plt.close(fig)


def test_default_title(world):
    fig = render(world, make_snap(world, step=17))
    t = fig.axes[0].get_title()
    assert "step 17" in t
    assert "agents 3" in t
    assert "12.3%" in t
    plt.close(fig)


def test_custom_title_with_snapshot(world):
    fig = render(world, make_snap(world, step=5), title="CADENCE")
    t = fig.axes[0].get_title()
    assert "CADENCE" in t and "step 5" in t
    plt.close(fig)


def test_legend_present(world):
    fig = render(world, make_snap(world))
    assert fig.legends or fig.axes[0].get_legend() is not None
    plt.close(fig)


# --- GIF ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("n,every", [(5, 1), (7, 3), (5, 2), (1, 4)])
def test_make_gif(world, tmp_path, n, every):
    snaps = [make_snap(world, step=i, coverage_pct=10.0 * i) for i in range(n)]
    path = tmp_path / "out.gif"
    plt.close("all")
    ret = make_gif(world, snaps, str(path), fps=5, every=every)
    assert ret == str(path)
    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        frames = sum(1 for _ in ImageSequence.Iterator(im))
    assert frames == math.ceil(n / every)
    assert plt.get_fignums() == []


def test_make_gif_includes_last(world, tmp_path):
    # n=6, every=4 -> frames 0, 4 and the last (5)
    snaps = [make_snap(world, step=i, coverage_pct=10.0 * i) for i in range(6)]
    path = tmp_path / "out.gif"
    make_gif(world, snaps, str(path), every=4)
    with Image.open(path) as im:
        frames = sum(1 for _ in ImageSequence.Iterator(im))
    assert frames == 3  # spec: every-th incl. the last (one more than ceil(6/4))
