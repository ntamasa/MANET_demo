"""Tests for cli.py (T9), using the dummy algorithms from test_summary."""
from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

import cli
from tests.test_summary import register_dummies

BASE = ["--size", "15", "15", "--seed", "3"]


@pytest.fixture(autouse=True)
def dummies(monkeypatch):
    register_dummies(monkeypatch, ok="CADENCE", fail="ISDA", echo="DADENCE")


def test_success_returns_0_and_prints_summary(capsys):
    assert cli.main(BASE + ["--algo", "CADENCE"]) == 0
    out = capsys.readouterr().out
    assert "CADENCE" in out and "15x15" in out and "Lefedettség" in out and "100.0" in out


def test_failure_returns_1(capsys):
    assert cli.main(BASE + ["--algo", "ISDA"]) == 1
    assert "✘" in capsys.readouterr().out


def test_algo_case_insensitive(capsys):
    assert cli.main(BASE + ["--algo", "cadence"]) == 0
    assert "CADENCE" in capsys.readouterr().out


def test_unknown_algo_rejected():
    with pytest.raises(SystemExit):
        cli.main(BASE + ["--algo", "FOO"])


def test_writes_csv(tmp_path):
    p = tmp_path / "hist.csv"
    assert cli.main(BASE + ["--algo", "CADENCE", "--csv", str(p)]) == 0
    df = pd.read_csv(p)
    assert list(df.columns) == ["step", "agents", "coverage_pct"]
    assert len(df) > 0 and df["coverage_pct"].iloc[-1] == pytest.approx(100.0)


def test_writes_gif(tmp_path):
    p = tmp_path / "run.gif"
    assert cli.main(BASE + ["--algo", "CADENCE", "--gif", str(p), "--every", "20"]) == 0
    with Image.open(p) as im:
        assert im.format == "GIF" and im.n_frames >= 2


def test_options_forwarded(capsys):
    from tests.test_summary import DummyEcho
    rc = cli.main(BASE + ["--algo", "DADENCE", "--no-dealloc", "--no-early-spawn", "--n-max", "4",
                          "--t-max", "50", "--pillar-prob", "0.0"])
    assert rc == 1
    assert DummyEcho.last_opts["early_spawn"] is False
    assert DummyEcho.last_opts["dealloc"] is False
    assert DummyEcho.last_opts["n_max"] == 4 and DummyEcho.last_opts["t_max"] == 50
    assert "50" in capsys.readouterr().out


def test_defaults_parse():
    args = cli.parse_args([])
    assert args.size == [40, 40] and args.seed == 42 and args.algo == "CADENCE"
    assert args.dealloc is True and args.early_spawn is True
    assert args.pillar_prob == pytest.approx(0.3) and args.every == 1
