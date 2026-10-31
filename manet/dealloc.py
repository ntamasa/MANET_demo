"""Algorithm 3 – Greedy Deallocation (PLAN.md §9)."""
from __future__ import annotations

from manet.visibility import FovCache
from manet.world import Cell


def greedy_dealloc(cache: FovCache, xd: Cell, terminal: list[Cell], others: list[Cell]) -> list[int]:
    """Return indices into `terminal` of the agents to remove.

    Repeatedly (fixed order, lowest index first) remove a terminal agent if, with xd + remaining
    terminal + others, the visibility graph stays connected and coverage stays equal to the coverage
    of the original configuration. `others` are never removed. Restart the scan after every removal.
    """
    raise NotImplementedError
