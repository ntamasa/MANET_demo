# RUNBOOK – MANET demo implementáció

Alap: `PLAN.md`. Módszer: **TDD** – minden feladat: (1) tesztek megírása, (2) futtatás → piros, (3) implementáció, (4) zöld, (5) commit.
Interfész-szerződés: a `manet/` modulokban lévő stubok szignatúrái és docstringjei (T0). Ezeket a feladatok **nem** változtatják meg (csak bővíthetnek privát segédfüggvényekkel).

Python: `MANET_demo/.venv/Scripts/python.exe` (worktree-kből abszolút úttal). Tesztfuttatás: `python -m pytest -q` a repo/worktree gyökeréből (`pytest.ini`: `pythonpath = .`).

## Részfeladatok

| ID | Feladat | Fájlok (csak ezeket írja) | Függ | Párhuzamos? |
|---|---|---|---|---|
| T0 | Váz: git, interfész-stubok, `Cell`/dataclassok, `grid_from_ascii`, pytest config | `manet/*` stubok, `pytest.ini`, `.gitignore` | – | ❌ (fővonal, előfeltétel) |
| T1 | `world.py`: sarkok, lyukak, valid sarkok, Nmax, Assumption 3, generátor | `manet/world.py`, `tests/test_world.py` | T0 | ✅ hullám 1 |
| T2 | `visibility.py`: LOS, FOV, cache, coverage, visibility graph | `manet/visibility.py`, `tests/test_visibility.py` | T0 | ✅ hullám 1 |
| T3 | `pathing.py`: BFS, legrövidebb út, closer neighbors, border | `manet/pathing.py`, `tests/test_pathing.py` | T0 | ✅ hullám 1 |
| T4 | `render.py`: Snapshot → Figure, GIF | `manet/render.py`, `tests/test_render.py` | T0 | ✅ hullám 1 |
| M1 | Merge T1–T4 → main, teljes tesztkör | – | T1–T4 | ❌ |
| T5 | `sim.py` (Algorithm base, run) + `dealloc.py` | `manet/sim.py`, `manet/dealloc.py`, `tests/test_sim.py`, `tests/test_dealloc.py` | M1 | ❌ hullám 2 (egyedül; mindenki erre épít) |
| T6 | CADENCE | `manet/algorithms/cadence.py`, `tests/test_cadence.py` | T5 | ✅ hullám 3 |
| T7 | DADENCE | `manet/algorithms/dadence.py`, `tests/test_dadence.py` | T5 | ✅ hullám 3 |
| T8 | ISDA | `manet/algorithms/isda.py`, `tests/test_isda.py` | T5 | ✅ hullám 3 |
| T9 | UI + CLI (`app.py`, `cli.py`, `manet/summary.py`) | `app.py`, `cli.py`, `manet/summary.py`, `tests/test_cli.py`, `tests/test_app.py` | T5 | ✅ hullám 3 (dummy algoritmussal tesztel) |
| M2 | Merge T6–T9, közös invariáns-tesztek (`tests/test_algorithms.py`), app smoke teszt | `tests/test_algorithms.py` | T6–T9 | ❌ |

Miért így: T1–T4 csak a T0-ban rögzített adattípusokra (`np.ndarray` grid, `Cell`, `World`, `Snapshot`) épít, egymást nem hívják → párhuzamos. T5 mindhárom alsó modult használja és minden algoritmus őt használja → szűk keresztmetszet, soros. T6–T9 csak a T5 `Algorithm` API-ját használja, és külön fájlba ír → párhuzamos. A regisztráció (`manet/algorithms/__init__.py`) T0-ban kész: lusta import, így a hiányzó modul nem töri el a többi ágat.

## Állapot

| ID | Állapot | Megjegyzés |
|---|---|---|
| T0 | ✅ | commit a9a0dc2 |
| T1–T4 | ✅ | T1 f2e907d (71 teszt), T2 a706162 (29), T3 e998ca8 (23), T4 7f2d090 (17) |
| M1 | ✅ | 140 teszt zöld a main-en |
| T5 | ✅ | 053862f, 174 teszt zöld; API: docs/ALGORITHM_API.md |
| T6–T9 | ⏳ | hullám 3 fut, 4 párhuzamos subagent |
