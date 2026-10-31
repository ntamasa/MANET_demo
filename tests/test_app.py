"""Smoke tests for the Streamlit app (T9) via streamlit.testing.v1.AppTest.

AppTest runs the script in-process, so the dummy algorithms registered with monkeypatch in
manet.algorithms._REGISTRY are picked up by app.py (it builds algorithms through
manet.summary.make_algorithm, which resolves the registry at call time).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from manet.algorithms import ALGORITHM_NAMES
from tests.test_summary import register_dummies

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture
def at(monkeypatch):
    register_dummies(monkeypatch, ok="CADENCE", fail="ISDA")
    app = AppTest.from_file(APP, default_timeout=TIMEOUT)
    app.run()
    assert not app.exception
    return app


def small_map(at, h=15, w=15, seed=3):
    at.number_input(key="height").set_value(h)
    at.number_input(key="width").set_value(w)
    at.number_input(key="seed").set_value(seed)
    at.button(key="generate").click().run()
    assert not at.exception


def test_loads_and_generates_default_world(at):
    world = at.session_state["world"]
    assert world.grid.shape == (40, 40)
    assert any("Nmax" in m.value for m in at.markdown)


def test_selectbox_options(at):
    sb = at.selectbox(key="algo")
    assert list(sb.options) == ALGORITHM_NAMES
    assert sb.value == "CADENCE"


def test_generate_map_stores_world(at):
    small_map(at, 16, 20, 7)
    world = at.session_state["world"]
    assert world.grid.shape == (16, 20)
    # Nmax / Tmax inputs follow the new world
    assert at.number_input(key="n_max").value == world.n_max
    assert at.number_input(key="t_max").value == 100 * 20


def test_early_spawn_only_for_dadence(at):
    keys = [c.key for c in at.checkbox]
    assert "early_spawn" not in keys
    at.selectbox(key="algo").set_value("DADENCE").run()
    assert "early_spawn" in [c.key for c in at.checkbox]
    at.selectbox(key="algo").set_value("ISDA").run()
    assert "early_spawn" not in [c.key for c in at.checkbox]


def _start(at, algo="CADENCE"):
    at.selectbox(key="algo").set_value(algo)
    at.slider(key="delay").set_value(0.0)
    at.slider(key="every").set_value(50)
    at.button(key="start").click().run()
    assert not at.exception


def test_start_produces_summary_and_results(at):
    small_map(at)
    _start(at)
    last = at.session_state["last_run"]
    assert last["summary"]["algorithm"] == "CADENCE"
    assert last["summary"]["success"] is True
    assert len(at.session_state["results"]) == 1
    tables = [d.value for d in at.dataframe]
    assert any(len(t) == 1 and "algorithm" in t.columns for t in tables)
    assert any("Összegzés" in s.value for s in at.subheader)
    assert len(at.metric) >= 3


def test_results_accumulate_and_replay(at):
    small_map(at)
    _start(at, "CADENCE")
    _start(at, "ISDA")
    results = at.session_state["results"]
    assert [r["algorithm"] for r in results] == ["CADENCE", "ISDA"]
    assert [r["success"] for r in results] == [True, False]
    # replay slider survives a rerun and redraws without error
    rep = at.slider(key="replay")
    rep.set_value(rep.min).run()
    assert not at.exception


def test_world_and_cache_reused_across_runs(at):
    small_map(at)
    _start(at)
    world, cache = at.session_state["world"], at.session_state["cache"]
    _start(at, "ISDA")
    assert at.session_state["world"] is world and at.session_state["cache"] is cache


def test_gif_button_prepares_gif(at):
    small_map(at)
    _start(at, "ISDA")
    at.button(key="make_gif").click().run()
    assert not at.exception
    gif = at.session_state["last_run"]["gif"]
    assert isinstance(gif, bytes) and gif[:3] == b"GIF"
