# MANET coverage demo

Interaktív Python demo a *Partially Observable Cooperative Guard Art Gallery Problem* (POCGAGP) megoldására, a
Meriaux et al. cikk alapján: [On Mobile Ad Hoc Networks for Coverage of Partially Observable Worlds (arXiv:2512.09495)](https://arxiv.org/abs/2512.09495).

Véletlen ortogonális „dungeon” pályán mobil ágensek építenek összefüggő hálózatot, miközben felfedezik és teljesen lefedik az ismeretlen teret.

**Algoritmusok:** CADENCE (centralizált, valid 270°-os sarkok), DADENCE (decentralizált, közös cél + queue-jump), ISDA (frontier baseline) – opcionális greedy deallokációval.

## Telepítés

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Használat

```
.venv\Scripts\streamlit run app.py                                   # UI: pályaméret, algoritmus, Start, élő preview, összegzés
.venv\Scripts\python cli.py --size 40 40 --seed 42 --algo dadence --gif out.gif
.venv\Scripts\python -m pytest -q                                   # tesztek
```

## Példa eredmény (40×40, seed 42)

| Algoritmus | Lépések | Ágensek (összes/csúcs/végső) | Lefedés |
|---|---|---|---|
| CADENCE | 215 | 42/39/13 | 100% |
| DADENCE | 308 | 13/13/13 | 100% |
| ISDA | 787 | 16/15/15 | 100% |

## Dokumentáció

- `PLAN.md` – részletes implementációs terv és a cikktől való eltérések
- `RUNBOOK.md` – részfeladatok, párhuzamosítás, állapot
- `docs/ALGORITHM_API.md` – az algoritmus-alaposztály API-ja

Készítők: Horánszki Patrik, Nagy Tamás (ELTE IK, MSc).
