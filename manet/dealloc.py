"""Algorithm 3 – Greedy Deallocation (PLAN.md §9)."""
from __future__ import annotations

from manet.visibility import FovCache, coverage, is_connected
from manet.world import Cell


def greedy_dealloc(cache: FovCache, xd: Cell, terminal: list[Cell], others: list[Cell]) -> list[int]:
    """Return indices into `terminal` of the agents to remove.

    Repeatedly (fixed order, lowest index first) remove a terminal agent if, with xd + remaining
    terminal + others, the visibility graph stays connected and coverage stays equal to the coverage
    of the original configuration. `others` are never removed. Restart the scan after every removal.
    """
    terminal = [tuple(c) for c in terminal]
    others = [tuple(c) for c in others]
    xd = tuple(xd)
    base = int(coverage(cache, [xd, *terminal, *others]).sum())
    keep = list(range(len(terminal)))
    removed: list[int] = []
    changed = True
    while changed:
        changed = False
        for i in keep:
            rest = [terminal[j] for j in keep if j != i]
            cells = [xd, *rest, *others]
            # coverage of a subset is a subset of `base`, so equal counts <=> equal masks
            if int(coverage(cache, cells).sum()) == base and is_connected(cache, cells):
                keep.remove(i)
                removed.append(i)
                changed = True
                break
    return removed
