"""MANET coverage demo – Streamlit UI (PLAN.md §11).

Usage:
    streamlit run app.py

Sidebar: generate a random dungeon map (size, seed, pillar probability), pick an algorithm
(CADENCE / DADENCE / ISDA), set options, press Start. The run is animated live; afterwards a summary,
the agents/coverage history, a replay slider and GIF/CSV downloads appear. Every run is appended to a
results table, so the three algorithms can be compared on the same map (the World and its FovCache
are kept in st.session_state and reused across runs).

Algorithms are built via manet.summary.make_algorithm, which resolves manet.algorithms.get_algorithm
(and its _REGISTRY) at call time – tests patch the registry with dummy algorithms.
"""
from __future__ import annotations

import os
import tempfile
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from manet import summary
from manet.algorithms import ALGORITHM_NAMES
from manet.render import make_gif, render
from manet.sim import Snapshot, run
from manet.visibility import FovCache
from manet.world import World, generate_world

SIZE_MIN, SIZE_MAX = 15, 80
GIF_MAX_FRAMES = 200


# --------------------------------------------------------------------------- state

def init_state() -> None:
    st.session_state.setdefault("results", [])
    st.session_state.setdefault("last_run", None)


def new_world(h: int, w: int, seed: int, pillar_prob: float) -> World:
    """Generate a world, reset its FovCache and the per-map run state; Nmax/Tmax inputs follow it.
    Must be called before the n_max / t_max widgets are instantiated in the current run."""
    world = generate_world(int(h), int(w), int(seed), pillar_prob=float(pillar_prob))
    ss = st.session_state
    ss.world = world
    ss.cache = FovCache(world.grid)
    ss.world_params = {"h": int(h), "w": int(w), "seed": int(seed), "pillar_prob": float(pillar_prob)}
    ss.last_run = None
    ss.n_max = int(world.n_max)
    ss.t_max = 100 * max(int(h), int(w))
    return world


# --------------------------------------------------------------------------- sidebar

def sidebar() -> dict:
    sb = st.sidebar
    sb.header("Térkép")
    w = sb.number_input("Szélesség", SIZE_MIN, SIZE_MAX, 40, key="width")
    h = sb.number_input("Magasság", SIZE_MIN, SIZE_MAX, 40, key="height")
    seed = sb.number_input("Seed", 0, 2 ** 31 - 1, 42, key="seed")
    pillar = sb.slider("Lyukak (pillars) gyakorisága", 0.0, 1.0, 0.3, 0.05, key="pillar_prob")
    if sb.button("Generate map", key="generate") or "world" not in st.session_state:
        new_world(h, w, seed, pillar)

    sb.header("Algoritmus")
    algo = sb.selectbox("Algoritmus", ALGORITHM_NAMES, index=0, key="algo")
    dealloc = sb.checkbox("Deallocation", True, key="dealloc")
    early_spawn = True
    if algo == "DADENCE":
        early_spawn = sb.checkbox("DADENCE early spawn", True, key="early_spawn")
    world = st.session_state.world
    n_max = sb.number_input(f"Nmax (0 = auto: {world.n_max})", min_value=0, step=1, key="n_max")
    t_max = sb.number_input("Tmax (0 = auto: 100·max(H,W))", min_value=0, step=100, key="t_max")

    sb.header("Megjelenítés")
    every = sb.slider("Lépés / frame", 1, 50, 1, key="every")
    delay = sb.slider("Késleltetés (s)", 0.0, 0.2, 0.03, 0.01, key="delay")
    show_corners = sb.checkbox("Valid sarkok mutatása", True, key="show_corners")
    show_edges = sb.checkbox("Élek mutatása", True, key="show_edges")
    start = sb.button("Start", key="start", type="primary")
    return dict(algo=algo, dealloc=dealloc, early_spawn=early_spawn, n_max=int(n_max),
                t_max=int(t_max), every=int(every), delay=float(delay), show_corners=show_corners,
                show_edges=show_edges, start=start)


# --------------------------------------------------------------------------- drawing helpers

def draw(ph, world: World, snap: Snapshot | None, cfg: dict, title: str) -> None:
    fig = render(world, snap, show_edges=cfg["show_edges"], show_corners=cfg["show_corners"], title=title)
    ph.pyplot(fig)
    plt.close(fig)


def world_info(world: World) -> str:
    p = st.session_state.world_params
    return (f"**Térkép** {p['h']}×{p['w']}, seed {p['seed']} — n = {world.n_corners}, "
            f"h = {world.n_holes}, valid sarkok = {len(world.valid_corners)}, "
            f"Nmax = {world.n_max}, xd = {tuple(world.xd)}")


def live_metrics(ph, snap: Snapshot) -> None:
    with ph.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Step", snap.step)
        c2.metric("Agents", snap.n_active)
        c3.metric("Coverage %", f"{snap.coverage_pct:.1f}")


# --------------------------------------------------------------------------- simulation

def run_simulation(world: World, cfg: dict, ph_map, ph_stats, prog) -> dict:
    """Animate one run into the placeholders; return the stored run record."""
    ss = st.session_state
    algo = summary.make_algorithm(cfg["algo"], world, seed=ss.world_params["seed"],
                                  dealloc=cfg["dealloc"], n_max=cfg["n_max"], t_max=cfg["t_max"],
                                  early_spawn=cfg["early_spawn"], cache=ss.cache)
    snaps = []
    for snap in run(algo):
        snaps.append(summary.compact_snapshot(snap))
        if snap.step % cfg["every"] == 0 or snap.finished:
            draw(ph_map, world, snap, cfg, cfg["algo"])
            live_metrics(ph_stats, snap)
            prog.progress(min(snap.step / max(algo.t_max, 1), 1.0))
            if cfg["delay"] > 0:
                time.sleep(cfg["delay"])
    prog.progress(1.0)
    d = summary.summary_dict(cfg["algo"], world, ss.world_params["seed"], algo)
    ss.results.append(d)
    ss.pop("replay", None)  # new run -> replay slider restarts at the last frame
    return {"algorithm": cfg["algo"], "summary": d, "history": summary.history_frame(algo.metrics),
            "snaps": snaps, "gif": None}


# --------------------------------------------------------------------------- result panels

def show_summary(last: dict) -> None:
    d = last["summary"]
    st.subheader("Összegzés")
    status = "✔ siker" if d["success"] else "✘ bukás"
    st.markdown(f"**{d['algorithm']}** — {d['size']}, seed {d['seed']} — {status} ({d['message']})")
    c = st.columns(4)
    c[0].metric("Lépések (lefedésig / összes)", f"{d['steps']} / {d['steps_total']}")
    c[1].metric("Futási idő (s)", f"{d['wall_time_s']:.2f}")
    c[2].metric("Ágensek (összes/csúcs/végső)",
                f"{d['spawned_total']}/{d['peak_agents']}/{d['final_agents']}", help="+1 pseudo-agent (xd)")
    c[3].metric("Lefedettség", f"{d['coverage_pct']:.1f}%")
    st.caption(f"+1 pseudo-agent (xd) — elméleti korlát: Nmax = {d['n_max']}, Tmax = {d['t_max']}")
    with st.expander("Szöveges összegzés"):
        st.text(summary.format_summary(d))
    st.line_chart(last["history"])


def show_replay(ph_map, world: World, last: dict, cfg: dict) -> None:
    snaps = last["snaps"]
    idx = len(snaps) - 1
    if len(snaps) > 1:
        st.subheader("Visszajátszás")
        idx = st.slider("Step", 0, len(snaps) - 1, len(snaps) - 1, key="replay")
    draw(ph_map, world, summary.expand_snapshot(snaps[idx]), cfg, last["algorithm"])


def gif_bytes(world: World, last: dict) -> bytes:
    snaps = [summary.expand_snapshot(s) for s in last["snaps"]]
    every = max(1, -(-len(snaps) // GIF_MAX_FRAMES))
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        make_gif(world, snaps, path, every=every)
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.unlink(path)


def show_downloads(world: World, last: dict) -> None:
    name = f"{last['algorithm']}_{last['summary']['size']}_seed{last['summary']['seed']}"
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", last["history"].to_csv().encode(), f"{name}.csv",
                       "text/csv", key="dl_csv")
    if last["gif"] is None:
        if c2.button("GIF készítése", key="make_gif"):
            with st.spinner("GIF készül…"):
                last["gif"] = gif_bytes(world, last)
    if last["gif"] is not None:
        c2.download_button("Download GIF", last["gif"], f"{name}.gif", "image/gif", key="dl_gif")


def show_results() -> None:
    if st.session_state.results:
        st.subheader("Futások összehasonlítása")
        st.dataframe(pd.DataFrame(st.session_state.results))


# --------------------------------------------------------------------------- main

def main() -> None:
    st.set_page_config(page_title="MANET coverage demo", layout="wide")
    init_state()
    cfg = sidebar()
    world = st.session_state.world
    st.title("MANET coverage demo")
    st.markdown(world_info(world))
    ph_map, ph_stats, prog = st.empty(), st.empty(), st.empty()
    if cfg["start"]:
        st.session_state.last_run = run_simulation(world, cfg, ph_map, ph_stats, prog.progress(0.0))
    last = st.session_state.last_run
    if last is None:
        draw(ph_map, world, None, cfg, "Térkép (xd)")
    else:
        show_summary(last)
        show_replay(ph_map, world, last, cfg)
        show_downloads(world, last)
    show_results()


main()
