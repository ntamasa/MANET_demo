"""Run summary helpers shared by app.py and cli.py (PLAN.md §11-12). No Streamlit here.

Algorithms are resolved through manet.algorithms.get_algorithm at call time (module attribute
lookup), so tests can swap in dummy algorithms by patching manet.algorithms._REGISTRY.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from manet import algorithms
from manet.sim import Algorithm, Metrics, Snapshot
from manet.visibility import FovCache
from manet.world import World


def make_algorithm(name: str, world: World, *, seed: int, dealloc: bool, n_max: int | None = None,
                   t_max: int | None = None, early_spawn: bool = True,
                   cache: FovCache | None = None) -> Algorithm:
    """Instantiate the algorithm registered under `name` (case-insensitive).
    n_max / t_max of None or 0 mean "auto" (world default). early_spawn is passed only to DADENCE."""
    cls = algorithms.get_algorithm(name)
    opts = {"early_spawn": bool(early_spawn)} if name.upper() == "DADENCE" else {}
    return cls(world, cache, n_max=int(n_max) if n_max else None,
               t_max=int(t_max) if t_max else None, seed=int(seed), dealloc=bool(dealloc), **opts)


def summary_dict(algo_name: str, world: World, seed: int, algo: Algorithm) -> dict:
    m = algo.metrics
    h, w = world.grid.shape
    return {
        "algorithm": algo_name.upper(),
        "size": f"{h}x{w}",
        "seed": int(seed),
        "success": bool(m.success),
        "message": algo.message,
        "steps": int(m.steps),
        "steps_total": int(m.steps_total),
        "wall_time_s": float(m.wall_time_s),
        "spawned_total": int(m.spawned_total),
        "peak_agents": int(m.peak_agents),
        "final_agents": int(m.final_agents),
        "coverage_pct": float(m.coverage_pct),
        "n_max": int(algo.n_max),
        "t_max": int(algo.t_max),
        "n_corners": int(world.n_corners),
        "n_holes": int(world.n_holes),
        "n_valid_corners": len(world.valid_corners),
    }


def history_frame(metrics: Metrics) -> pd.DataFrame:
    """History as a DataFrame indexed by step, columns [agents, coverage_pct]."""
    df = pd.DataFrame(metrics.history, columns=["step", "agents", "coverage_pct"])
    return df.set_index("step")[["agents", "coverage_pct"]]


def format_summary(d: dict) -> str:
    ok = "✔ siker" if d["success"] else "✘ bukás"
    return "\n".join([
        f"Algoritmus: {d['algorithm']}   Térkép: {d['size']} (seed {d['seed']})",
        f"Eredmény: {ok} ({d['message']})",
        f"Lépések (lefedésig / összes visszasétálással): {d['steps']} / {d['steps_total']}",
        f"Futási idő: {d['wall_time_s']:.3f} s",
        f"Ágensek (összes/csúcs/végső): {d['spawned_total']}/{d['peak_agents']}/{d['final_agents']}"
        "  (+1 pseudo-agent (xd))",
        f"Lefedettség: {d['coverage_pct']:.1f}%",
        f"Elméleti korlát: Nmax = {d['n_max']}, Tmax = {d['t_max']}"
        f"   (n = {d['n_corners']}, h = {d['n_holes']}, valid sarkok = {d['n_valid_corners']})",
    ])


# --------------------------------------------------------------------------- replay storage

@dataclass(frozen=True)
class CompactSnapshot:
    """Snapshot with covered/explored bit-packed (np.packbits); other fields kept as-is."""
    shape: tuple[int, int]
    covered_bits: np.ndarray
    explored_bits: np.ndarray
    rest: dict


_PACKED = ("covered", "explored")


def compact_snapshot(snap: Snapshot) -> CompactSnapshot:
    rest = {f.name: getattr(snap, f.name) for f in fields(Snapshot) if f.name not in _PACKED}
    cov = np.asarray(snap.covered, dtype=bool)
    return CompactSnapshot(shape=cov.shape, covered_bits=np.packbits(cov),
                           explored_bits=np.packbits(np.asarray(snap.explored, dtype=bool)), rest=rest)


def expand_snapshot(c: CompactSnapshot) -> Snapshot:
    n = c.shape[0] * c.shape[1]

    def unpack(bits):
        return np.unpackbits(bits, count=n).astype(bool).reshape(c.shape)

    return Snapshot(covered=unpack(c.covered_bits), explored=unpack(c.explored_bits), **c.rest)
