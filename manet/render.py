"""Snapshot -> matplotlib figure, GIF export (PLAN.md §10)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from manet.sim import Snapshot, Status  # noqa: E402
from manet.world import World  # noqa: E402

COLORS = {
    "unseen_free": "#3a3a3a",
    "covered": "#fff3a0",
    "explored": "#d9cf8a",
    "seen_wall": "#5b6b7f",
    "unseen_wall": "#111111",
    "xd": "#1a9e3f",
    "settled": "#d62728",
    "moving": "#ff8c00",
    "returning": "#9e9e9e",
    "edge": "#2ca02c",
    "target": "#ffd700",
}


def _observed_walls(grid: np.ndarray, explored: np.ndarray) -> np.ndarray:
    """Wall cells with at least one explored (free) 8-neighbour."""
    h, w = grid.shape
    p = np.zeros((h + 2, w + 2), dtype=bool)
    p[1:-1, 1:-1] = explored & grid
    near = np.zeros((h, w), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                near |= p[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
    return ~grid & near


def background_rgb(world: World, snap: Snapshot) -> np.ndarray:
    """(H, W, 3) float array in [0, 1] using COLORS (covered > explored > unseen for free cells;
    walls 'seen_wall' if observed_walls(grid, explored) else 'unseen_wall')."""
    grid = np.asarray(world.grid, dtype=bool)
    h, w = grid.shape
    if snap is None:
        covered = explored = np.zeros((h, w), dtype=bool)
    else:
        covered = np.asarray(snap.covered, dtype=bool)
        explored = np.asarray(snap.explored, dtype=bool) | covered
    img = np.empty((h, w, 3), dtype=float)
    img[:] = to_rgb(COLORS["unseen_wall"])
    img[_observed_walls(grid, explored)] = to_rgb(COLORS["seen_wall"])
    img[grid] = to_rgb(COLORS["unseen_free"])
    img[grid & explored] = to_rgb(COLORS["explored"])
    img[grid & covered] = to_rgb(COLORS["covered"])
    return img


def _xy(cells) -> np.ndarray:
    """Cells (r, c) -> points (x=c, y=r) for imshow(origin='upper')."""
    return np.array([(c, r) for r, c in cells], dtype=float).reshape(-1, 2)


def _figsize(world: World) -> tuple[float, float]:
    h, w = world.shape
    s = 7.0 / max(h, w)
    return max(w * s, 3.0), max(h * s, 3.0) + 0.9   # extra room for the legend below


def _legend_handles(show_edges: bool, show_corners: bool) -> list:
    def m(marker, color, label, ms=8, **kw):
        return Line2D([], [], linestyle="none", marker=marker, color=color, markersize=ms,
                      label=label, **kw)
    hs = [m("s", COLORS["xd"], "xd"),
          m("o", COLORS["settled"], "settled"),
          m("o", COLORS["moving"], "moving"),
          m("o", COLORS["returning"], "returning"),
          m("*", COLORS["target"], "target", ms=11, markeredgecolor="k", markeredgewidth=0.5)]
    if show_corners:
        hs.append(m("o", COLORS["settled"], "valid corner", markerfacecolor="none"))
    if show_edges:
        hs.append(Line2D([], [], color=COLORS["edge"], lw=1.2, label="link"))
    return hs


def _title(snap: Snapshot | None, title: str | None) -> str:
    if snap is None:
        return title or ""
    stats = f"step {snap.step} | agents {snap.n_active} | coverage {snap.coverage_pct:.1f}%"
    return f"{title} | {stats}" if title else stats


def _draw(fig: plt.Figure, world: World, snap: Snapshot | None, show_edges: bool,
          show_corners: bool, title: str | None) -> None:
    fig.clf()
    ax = fig.add_subplot(111)
    h, w = world.shape
    ax.imshow(background_rgb(world, snap), interpolation="nearest", origin="upper")
    # marker area (pt^2) roughly proportional to the cell area
    cell_pt = 7.0 * 72.0 / max(h, w)
    ms = float(np.clip((0.6 * cell_pt) ** 2, 6.0, 120.0))

    if snap is not None:
        if show_edges and snap.edges:
            segs = [_xy([a, b]) for a, b in snap.edges]
            ax.add_collection(LineCollection(segs, colors=COLORS["edge"], linewidths=1.0,
                                             zorder=2, label="edges"))
        if show_corners and snap.known_valid:
            p = _xy(snap.known_valid)
            ax.scatter(p[:, 0], p[:, 1], s=ms * 1.6, facecolors="none", edgecolors=COLORS["settled"],
                       linewidths=1.0, zorder=3, label="valid corners")
        if snap.targets:
            p = _xy(snap.targets)
            ax.scatter(p[:, 0], p[:, 1], s=ms * 2.2, marker="*", c=COLORS["target"], edgecolors="k",
                       linewidths=0.4, zorder=5, label="targets")

    p = _xy([world.xd])
    ax.scatter(p[:, 0], p[:, 1], s=ms * 1.3, marker="s", c=COLORS["xd"], edgecolors="k",
               linewidths=0.5, zorder=3, label="xd")

    if snap is not None:
        for st in (Status.RETURNING, Status.MOVING, Status.SETTLED):
            cells = [pos for _, pos, s in snap.agents if s == st]
            if cells:
                p = _xy(cells)
                ax.scatter(p[:, 0], p[:, 1], s=ms, marker="o", c=COLORS[st.value], edgecolors="k",
                           linewidths=0.4, zorder=4, label=st.value)

    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(_title(snap, title), fontsize=10)
    fig.legend(handles=_legend_handles(show_edges, show_corners), loc="lower center",
               ncol=4, fontsize=7, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.12)


def render(world: World, snap: Snapshot | None = None, show_edges: bool = True,
           show_corners: bool = True, title: str | None = None) -> plt.Figure:
    """Draw the world (and the snapshot if given). snap=None -> static map with only xd marked,
    everything unseen. Caller is responsible for plt.close(fig)."""
    fig = plt.figure(figsize=_figsize(world))
    _draw(fig, world, snap, show_edges, show_corners, title)
    return fig


def _frame_indices(n: int, every: int) -> list[int]:
    every = max(1, int(every))
    idx = list(range(0, n, every))
    if n and idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def make_gif(world: World, snapshots: list[Snapshot], path: str, fps: int = 10, every: int = 1) -> str:
    """Write an animated GIF (PillowWriter) of every `every`-th snapshot (always incl. the last). Returns path."""
    if not snapshots:
        raise ValueError("make_gif: no snapshots")
    idx = _frame_indices(len(snapshots), every)
    fig = plt.figure(figsize=_figsize(world))
    try:
        def update(i):
            _draw(fig, world, snapshots[i], True, True, None)
            return []
        anim = FuncAnimation(fig, update, frames=idx, blit=False, repeat=False,
                             cache_frame_data=False)
        anim.save(path, writer=PillowWriter(fps=fps))
    finally:
        plt.close(fig)
    return path
