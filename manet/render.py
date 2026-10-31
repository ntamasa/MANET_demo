"""Snapshot -> matplotlib figure, GIF export (PLAN.md §10)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from manet.sim import Snapshot  # noqa: E402
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


def background_rgb(world: World, snap: Snapshot) -> np.ndarray:
    """(H, W, 3) float array in [0, 1] using COLORS (covered > explored > unseen for free cells;
    walls 'seen_wall' if observed_walls(grid, explored) else 'unseen_wall')."""
    raise NotImplementedError


def render(world: World, snap: Snapshot | None = None, show_edges: bool = True,
           show_corners: bool = True, title: str | None = None) -> plt.Figure:
    """Draw the world (and the snapshot if given). snap=None -> static map with only xd marked,
    everything unseen. Caller is responsible for plt.close(fig)."""
    raise NotImplementedError


def make_gif(world: World, snapshots: list[Snapshot], path: str, fps: int = 10, every: int = 1) -> str:
    """Write an animated GIF (PillowWriter) of every `every`-th snapshot (always incl. the last). Returns path."""
    raise NotImplementedError
