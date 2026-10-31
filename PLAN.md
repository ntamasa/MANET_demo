# MANET coverage demo – implementációs terv

Forrás: Meriaux et al., *On Mobile Ad Hoc Networks for Coverage of Partially Observable Worlds* (arXiv:2512.09495) = **fő forrás**; MANET kickoff prezentáció (Horánszki P., Nagy T.) = stack + scope. Ellentmondás esetén a cikk az irányadó.

---

## 0. Cél (scope)

Egyszerű, interaktív Python demo app, amely:

1. Generál egy `H × W` méretű, véletlen, ortogonális "dungeon" rácsvilágot (méret + seed megadható).
2. Drop-down menüből választható algoritmus: **CADENCE**, **DADENCE**, **ISDA**.
3. **Start** gombra lefuttatja a szimulációt a generált pályán, és élőben mutatja: ágensek mozgása, lefedett (observed) terület, kommunikációs élek (visibility graph), célpontok, deployment pont.
4. A végén összegzést ad: eltelt idő (szimulációs lépésszám + valós futási idő), felhasznált ágensek (összes spawnolt, csúcs, végső deallokáció után), lefedettség %, siker/bukás.

Nem cél: APF, lattice, MARL (IDQL/VDN), kvadfa-komplexitás benchmark, 1500 világos kísérletsorozat. (A kvadfa-mérőszám opcionális extra, lásd 12.)

---

## 1. Tech stack (kickoff 12. dia alapján)

| Rész | Könyvtár |
|---|---|
| Nyelv | Python 3.11+ (a gépen: 3.13.7, venv: `MANET_demo/.venv`) |
| Rács, FOV/LOS, maszkok | NumPy |
| Visibility graph, összefüggőség | NetworkX |
| Megjelenítés (frame-ek, GIF export) | matplotlib (+ `PillowWriter`) |
| Metrika-történet, összegző táblázat, CSV | pandas |
| UI (drop-down, gombok, élő preview) | **Streamlit** (kickoff: "Optional demo UI: matplotlib or Streamlit" – a drop-down miatt Streamlit) |
| Tesztek | pytest |

`requirements.txt`:
```
numpy>=1.26
networkx>=3.2
matplotlib>=3.8
pandas>=2.1
streamlit>=1.32
pillow>=10.0
pytest>=8.0
```
Nincs GPU, nincs RL, nincs scipy (saját BFS-címkézés).

---

## 2. Projektstruktúra

```
MANET_demo/
├── PLAN.md
├── requirements.txt
├── app.py                    # Streamlit UI (belépési pont: streamlit run app.py)
├── cli.py                    # headless futtatás + GIF export (debug/teszt)
├── manet/
│   ├── __init__.py
│   ├── world.py              # World dataclass, dungeon generátor, Assumption 3 javítás, sarkok, lyukak, Nmax
│   ├── visibility.py         # LOS, FOV-cache, coverage, visibility graph
│   ├── pathing.py            # BFS távolság, legrövidebb út, strictly closer neighbors, border
│   ├── sim.py                # Agent, Snapshot, Metrics, Algorithm base class, run() generátor
│   ├── dealloc.py            # Algorithm 3 – Greedy Deallocation + visszasétáltatás xd-be
│   ├── algorithms/
│   │   ├── __init__.py       # ALGORITHMS = {"CADENCE": Cadence, "DADENCE": Dadence, "ISDA": Isda}
│   │   ├── cadence.py
│   │   ├── dadence.py
│   │   └── isda.py
│   └── render.py             # Snapshot -> matplotlib Figure
└── tests/
    ├── test_world.py
    ├── test_visibility.py
    ├── test_pathing.py
    ├── test_dealloc.py
    └── test_algorithms.py
```

---

## 3. Konvenciók

- Rács: `grid: np.ndarray[bool]`, alak `(H, W)`, `True` = szabad cella (W része), `False` = fal/lyuk.
- Cella koordináta: `(r, c)` tuple; `r` = sor (felülről lefelé nő), `c` = oszlop.
- A külső keret mindig fal (`grid[0,:] = grid[-1,:] = grid[:,0] = grid[:,-1] = False`).
- Szomszédság mozgáshoz: 4-szomszéd (N/S/E/W, cikk Def. 11 – square graph).
- 1 szimulációs lépés = minden ágens legfeljebb 1 cellát lép (párhuzamos mozgás is 1 lépés, cikk IV-A).
- Pseudo-agent: a deployment pont `xd` (szabad cella); mindig látó csomópont a gráfban, de nem számít bele `A`-ba.
- Minden véletlenség egy `np.random.Generator`-ból jön (`seed` a UI-ból) → determinisztikus futás.

---

## 4. `world.py` – világ és geometria

### 4.1 `World` dataclass
```python
@dataclass
class World:
    grid: np.ndarray            # bool (H, W)
    xd: tuple[int, int]         # deployment point
    n_corners: int              # n  (90° + 270° sarkok)
    n_holes: int                # h
    reflex_corners: list[Corner]
    valid_corner_cells: set[tuple[int,int]]   # ágens-pozíciók a valid sarkokhoz
    n_max: int                  # (n + 2h - 4) // 2  (Eq. 1 / Eq. 3)
```
`Corner = (vertex(i,j), agent_cell(r,c), wall_cell(r,c), hole_id | None)`.

### 4.2 Dungeon generátor – `generate_world(h, w, seed, room_min=5, corridor_width=2, pillar_prob=0.3) -> World`

BSP alapú (kickoff: "NumPy · BSP rooms"):

1. `grid = zeros((h,w), bool)`.
2. Rekurzív BSP a belső téglalapon (`[1,h-2] × [1,w-2]`): vágj vízszintesen/függőlegesen véletlen helyen, amíg a levél ≥ `2*room_min+2`; max mélység ~ `log2(h*w/100)`.
3. Minden levélben szoba: véletlen téglalap legalább `room_min × room_min`, 1 cella margóval → `True`.
4. Testvér-levelek szobáit L alakú folyosóval kötöd össze (szélesség `corridor_width`, alapból 2 – a 2 széles folyosó kevesebb Assumption-3 problémát okoz).
5. **Lyukak (holes)**: `pillar_prob` eséllyel nagy (≥ 9×9) szobákba 1 db téglalap oszlop (2×2..4×4), legalább 2 cellára a szoba falaitól → ezek lesznek a lyukak (fontos a "top-left invalid corner" szabály bemutatásához).
6. **Utófeldolgozás ciklusban, amíg változik**:
   - keret falra állítása;
   - **Assumption 3** kikényszerítése: minden 2×2 ablakra, ha sakktábla minta (`[[1,0],[0,1]]` vagy `[[0,1],[1,0]]`), az egyik fal-cellát (rng) szabaddá teszed (de keret-cellát soha; ha mindkettő keret, a szabadokat falazod be);
   - csak a legnagyobb 4-összefüggő szabad komponens marad (többi → fal);
   - lyukak, amelyek csak 1 cellányira vannak egymástól/faltól, megmaradhatnak, de lyuk–keret érintkezés nem gond (akkor az nem lyuk, hanem keret része).
7. **Assumption 2** (nem konvex): ha nincs 270°-os sarok → új seed-del (`seed+1000*k`) újragenerál.
8. `xd`: véletlen szabad cella (rng). (Opcionálisan: "legbalfelsőbb szabad cella" mód.)
9. Sarkok, lyukak, `n_max` számítása (4.3–4.5).

Méret-korlát a UI-ban: 15…80 (alapértelmezés 40×40). 80 fölött a FOV és a matplotlib lassul.

### 4.3 Sarkok (vektorizált 2×2 ablak)
Rácspont (vertex) `(i, j)`, `1 ≤ i ≤ H-1`, `1 ≤ j ≤ W-1`, ablak: cellák `(i-1,j-1),(i-1,j),(i,j-1),(i,j)`.
- szabad cellák száma `== 1` → konvex (90°) sarok;
- `== 3` → reflex (270°) sarok; a fal-cella az ablakban `wall_cell`, **az ágens-cella az ablakban a fal-cellával átlósan szemközti szabad cella** (ez "öleli" a sarkot, mindkét folyosóba belát);
- sakktábla minta nem fordulhat elő (Assumption 3) – assert.
- `n = #konvex + #reflex`.

### 4.4 Lyukak
- Fal-cellák 4-összefüggő címkézése saját BFS-sel (Assumption 3 miatt 4/8 szomszédság ekvivalens).
- Az a komponens, amely érinti a keretet, a külső határ; minden más komponens = 1 lyuk. `h = #lyuk`.
- Reflex sarok `hole_id`-je = a `wall_cell` címkéje, ha az lyuk.

### 4.5 Valid sarkok (cikk Def. 8)
- Minden reflex sarok valid, **kivéve** lyukanként a "top-left" sarkot: a lyuk reflex sarkai közül a legnagyobb y (= **legkisebb vertex sorindex `i`**), egyenlőség esetén a legkisebb x (= legkisebb `j`).
- `valid_corner_cells` = a valid sarkok `agent_cell`-jeinek halmaza (deduplikálva; keskeny folyosóban több sarok eshet ugyanarra a cellára).
- `n_max = (n + 2h - 4) // 2` (Eq. 1 = Eq. 3, `M_valid`). UI-ban felülírható. Ellenőrzés: `|valid sarkok| == (n + 2h - 4)/2` (Lemma 1: reflex = `(n+4h-4)/2`, valid = reflex − h) → teszt.

---

## 5. `visibility.py` – látás (Def. 6, 10, 14)

### 5.1 LOS (line of sight)
Két szabad cella `a`, `b` látja egymást, ha a cellaközéppontokat összekötő szakasz csak szabad cellákon halad át.
- Szimmetrikus definíció → Assumption 1 (kölcsönös láthatóság) automatikusan teljesül.
- Mintavételezés: `t ∈ linspace(0,1,K)`, `K = 4·max(|dr|,|dc|)+1`; pont `p = a + t·(b-a)`; cella = `floor(p + 0.5)`.
- **Sarok-szabály**: ha egy mintapont mindkét koordinátájának tört része `0.5 ± 1e-6` (pontosan rácspont), a mintát figyelmen kívül hagyod. Mivel a generátor kizárja a sakktábla mintát, átlós "átlátás két fal között" nem fordulhat elő (Fig. 3).
- 4-szomszédok mindig látják egymást.

### 5.2 FOV – `fov(world, cell) -> np.ndarray[bool]`
- Egy forrásból az összes szabad célcellára egyszerre (vektorizált): célcellák távolság szerint rendezve, ~1024-es chunkokban, chunkonként közös `K`-val → `(N, K, 2)` tömb, `grid[idx]` mind `True` → látható.
- Eredmény: `(H,W)` bool maszk (csak szabad cellák lehetnek `True`).
- **Cache**: `dict[cell -> mask]` a `World`-höz kötve (`FovCache` osztály). Egy cellára csak egyszer számolunk.

### 5.3 Segédfüggvények
```python
coverage(cache, cells: Iterable[cell]) -> mask        # OR a FOV-okon  (F~(A ∪ {xd}))
sees(cache, a, b) -> bool                             # cache.fov(a)[b]
visibility_graph(cache, nodes: dict[id, cell]) -> nx.Graph   # él, ha sees()
is_connected(cache, cells) -> bool                    # nx.is_connected, 1 csúcsra True
observed_walls(world, covered) -> mask                # fal, aminek 8-szomszédja lefedett (csak rajzoláshoz)
```

---

## 6. `pathing.py`

```python
bfs_dist(passable: mask, src, blocked_edges=frozenset()) -> np.ndarray[int]  # -1 = elérhetetlen
shortest_path(passable, src, dst) -> list[cell]                               # src..dst, BFS + parent
closer_neighbors(dist_to_target, cell, passable, blocked_edges) -> list[cell] # Def. 12: L(w,x*) < L(u,x*)
border(world, covered) -> list[cell]   # Def. 13: lefedett szabad cella, amelynek van szabad, NEM lefedett 4-szomszédja
```
- `passable` = az a terület, amin az ágens tervezhet: **a lefedett (covered) szabad cellák** (ismert terep). Kivétel: DADENCE queue-jump kiterjesztés (7.2).
- A border számításhoz a ground-truth `grid`-et használjuk (az ágensek szenzorai a szomszédos falat is látják).
- Determinizmus: szomszédok sorrendje fix (N, E, S, W); tie-break rng helyett fix sorrend, kivéve ISDA.

---

## 7. `sim.py` – közös keret

```python
class Status(Enum): MOVING, SETTLED, RETURNING
@dataclass
class Agent:
    id: int; pos: cell; status: Status
    target: cell | None = None
    path: list[cell] = field(default_factory=list)
    pruned: set[tuple[cell,cell]] = field(default_factory=set)   # DADENCE edge pruning

@dataclass
class Snapshot:            # 1 lépés állapota a rajzoláshoz / replay-hez
    step: int
    agents: list[tuple[int, cell, Status]]
    covered: np.ndarray    # aktuális F~(A ∪ {xd})
    explored: np.ndarray   # kumulált, valaha látott cellák
    targets: list[cell]    # CADENCE: kiosztott sarkok; ISDA: aktuális cél; DADENCE: [x*]
    known_valid: list[cell]  # felfedezett valid sarkok (CADENCE)
    edges: list[tuple[cell,cell]]  # visibility graph élei (xd-vel együtt)
    n_active: int; coverage_pct: float; message: str

@dataclass
class Metrics:
    steps: int = 0                  # szimulációs lépések (amíg a lefedés el nem készül)
    steps_total: int = 0            # + visszasétálás xd-be deallokáció után
    wall_time_s: float = 0.0
    spawned_total: int = 0
    peak_agents: int = 0            # max egyidejű ágens (xd nélkül)
    final_agents: int = 0           # deallokáció után (xd nélkül)
    coverage_pct: float = 0.0
    success: bool = False           # 100% lefedés, Nmax és Tmax betartva
    history: list[dict]             # step, agents, coverage_pct → pandas DataFrame

class Algorithm(ABC):
    name: str
    def __init__(self, world, cache, *, n_max, t_max, rng, dealloc=True, **opts): ...
    @abstractmethod
    def step(self) -> bool: ...        # True = fut tovább, False = kész
    def snapshot(self) -> Snapshot: ...
    def spawn(self) -> Agent: ...      # xd-ben, ha xd üres és |A| < n_max; metrikák frissítése
    def active_cells(self) -> list[cell]  # ágensek pozíciói (RETURNING is látó csomópont, amíg él)

def run(algo: Algorithm) -> Iterator[Snapshot]:
    t0 = perf_counter(); yield algo.snapshot()
    while algo.step(): yield algo.snapshot()
    algo.metrics.wall_time_s = perf_counter() - t0; yield algo.snapshot()
```

Közös szabályok:
- `Tmax` elérésekor a futás leáll, `success=False`, üzenet: "Tmax exceeded".
- Ha `|A| == n_max` és nincs előrelépés `stall_limit = 4·(H+W)` lépésig → leáll, `success=False`.
- `RETURNING` ágens (deallokált): a `shortest_path(explored, pos, xd)` mentén visszasétál, `xd`-be érve eltűnik. A visszatérő ágensek nem kellenek az összefüggőséghez/lefedéshez (a deallokáció enélkül is helyes volt), csak rajzolva vannak. A cikk szerint a visszasétálás túllépheti a Tmax-ot.
- Ütközés: CADENCE és ISDA esetén az átmenő (MOVING/RETURNING) ágensek ugyanazon cellán áthaladhatnak (kis robotok – dokumentált egyszerűsítés); végső (SETTLED) pozíciók mindig egyediek. DADENCE-ben szigorú ütközésmentesség (7.2).
- Metrikák minden lépés végén: `peak_agents = max(peak, |A|)`, `history.append(...)`.

---

## 8. Algoritmusok

### 8.1 CADENCE (`cadence.py`) – cikk Alg. 1 + IV-A módosított ciklus

Állapot: `A` ágensek, `T` (kiosztott valid-sarok-cellák → ágens id), `explored` (kumulált), `known_valid` (felfedezett valid sarok-cellák).

`step()`:
```
1. covered  = coverage(xd ∪ {a.pos | a ∈ A, status != RETURNING})
   explored |= covered
   known_valid |= valid_corner_cells ∩ explored            # S = φ(A ∪ T ∪ {xd})
2. unassigned = known_valid − T − {xd}
   if unassigned and xd nem foglalt (nincs ágens a cellán) and |A| < n_max:
        c = argmin_{c ∈ unassigned} bfs_dist(explored, xd)[c]   # determinisztikus "pick"
        a = spawn(); a.target = c; a.path = shortest_path(explored, xd, c)[1:]
        T[c] = a.id                                            # max 1 spawn / lépés
3. minden MOVING ágens: pos = path.pop(0)
   ha path üres → status = SETTLED
4. ha valaki most lett SETTLED és dealloc: greedy_dealloc(SETTLED ágensek)   (9. pont)
5. RETURNING ágensek léptetése
6. vége, ha: unassigned üres AND nincs MOVING ágens  (→ még egy covered frissítés, végső dealloc)
   ekkor metrics.steps = aktuális lépés; utána csak RETURNING ágensek mozognak, amíg el nem fogynak
```
Megjegyzések:
- A sarok `xd`-re is eshet → az "foglalt" a pseudo-agent által (nem kell ágens).
- Bizonyítás szerint ha minden látott valid sarok foglalt, a világ le van fedve; diszkrét rácson LOS-diszkretizáció miatt ez ritkán sérülhet → a végső `coverage_pct`-t mindig ténylegesen kiszámoljuk és őszintén kiírjuk.
- Deallokált sarok `T`-ben marad ("kész"), nem kerül újra kiosztásra.

### 8.2 DADENCE (`dadence.py`) – cikk Alg. 2 + Fig. 4 + IV-B optimalizációk

Állapot: `A` (mind MOVING-nak számít; nincs fix cél, csak közös `x*`), `x_star`, `stall` (lépés lefedés-növekedés nélkül), `F` = aktuális lefedés.

`step()` (Fig. 4 blokkok):
```
F = coverage(xd ∪ A);  B = border(F)
[2] if B üres: dealloc(A) ha engedélyezett; return False (kész)
[1st iter] if A üres: spawn(); return True                                # line 4
[3] if x_star is None or x_star ∉ B:                                      # IV-B: csak ha kiesett a borderből
        D_s = bfs_dist(F, s) minden s ∈ A ∪ {xd}
        x_star = argmin_{x ∈ B} mean_s D_s[x]   (tie: kisebb (r,c))
        minden ágens pruned = ∅                                           # edge-pruning reset
[4] if van ágens xd-ben:                                                   # line 7–13, queue-jump
        P = shortest_path(F, xd, x_star)
        ha P minden cellája foglalt (x_star is): P += [x_star egy szabad, nem lefedett 4-szomszédja]
        k = első index, ahol P[k] nem foglalt
        for i = k-1 .. 0: az ágens P[i]-ből P[i+1]-be lép (lánc-eltolás, Fig. 5)
        → elmozdulás csak "előre a sor mentén"; konnektivitás/lefedés bizonyítottan megmarad (Prop. 2–3)
   else:                                                                   # line 15–33
        dist = bfs_dist(F, x_star)
        declared = {}
        for a in A (id szerinti sorrendben):
            if a.pos == x_star: continue
            d = dist, vagy ha a.pruned nem üres: bfs_dist(F, x_star, a.pruned)
            cand = [w ∈ closer_neighbors(d, a.pos, F, a.pruned)
                    if w nem foglalt egyetlen ágens AKTUÁLIS pozíciója által sem
                    and w ∉ declared.values()]
            if cand: declared[a] = cand[0]
        # revert ciklus (line 23–28)
        A' = A a declared lépésekkel
        while not is_connected(A' ∪ {xd}) or not (F ⊆ coverage(A' ∪ {xd})):
            a = a még nem visszavont declared ágensek közül a legnagyobb id-jű
            visszavon: A'[a] = a.pos;  a.pruned.add((a.pos, declared[a]))      # IV-B edge pruning
        if A' == A (senki nem mozdult):
            if xd szabad and |A| < n_max: spawn()                              # line 30, block 11
        else:
            A = A'                                                             # line 32, block 8
[early spawn, IV-B] új F; ha |F| nem nőtt: stall += 1 else stall = 0
        th = max(1, round(mean_a bfs_dist(F, x_star)[a.pos]))
        if early_spawn and stall >= th and xd szabad and |A| < n_max and ebben a lépésben nem volt spawn:
            spawn(); stall = 0; minden pruned = ∅
return True
```
Részletek:
- Ütközésmentesség: foglalt cellába senki nem deklarálhat lépést (még ha a foglaló el is lépne) → a visszavonás sosem okoz ütközést; a "vonat" így lépésenként halad.
- Ha minden ágens összes kimenő éle elfogyott (pruned) és nincs spawn lehetőség → `stall` nő → stall_limit-nél leáll (7.).
- Deallokáció csak a végén (cikk IV-C: DADENCE-ben nincs terminális ágens a lefedés előtt).

### 8.3 ISDA (`isda.py`) – Howard et al. 2002, a cikk V-B.3 szerint

"Új ágens a lefedés határára (frontier), véletlenszerűen kiválasztott pontra; egyszerre egy ágens mozog."

`step()`:
```
covered = coverage(xd ∪ SETTLED ∪ MOVING pozíciók); B = border(covered)
if nincs MOVING ágens:
    if B üres: végső dealloc; return False
    if xd szabad and |A| < n_max:
        x = rng.choice(B)                                   # véletlen frontier cella
        a = spawn(); a.target = x; a.path = shortest_path(covered, xd, x)[1:]
MOVING ágens: 1 cellát lép; célba érve SETTLED (+ dealloc a SETTLED-ekre, ha engedélyezett)
RETURNING ágensek léptetése
```
- Helyesség: frontier cella lefedett → látható valakiből (összefüggő); célba érve a nem lefedett szomszédja látható lesz → szigorú haladás; a letelepedett ágensek nem mozdulnak → lefedés nem csökken.

---

## 9. `dealloc.py` – Algorithm 3 (Greedy Deallocation)

```python
def greedy_dealloc(cache, xd, terminal: list[Agent], others: list[cell]) -> list[Agent]:
    """Visszaadja az eltávolítandó ágenseket. others = nem-terminális ágensek pozíciói (pl. MOVING),
    ezek nem kerülnek ellenőrzésre, de a gráfban szerepelnek."""
    keep = list(terminal); removed = []
    base = coverage(xd ∪ keep ∪ others)
    changed = True
    while changed:
        changed = False
        for a in keep:                                       # fix sorrend: legkisebb id előre
            rest = [b for b in keep if b is not a]
            if is_connected(xd ∪ rest ∪ others) and coverage(xd ∪ rest ∪ others) == base:
                keep = rest; removed.append(a); changed = True; break
    return removed
```
Eltávolított ágens → `status = RETURNING`, útvonal `xd`-be. CADENCE/ISDA: minden letelepedéskor + végén; DADENCE: csak a végén (Def. 15). UI-checkbox kapcsolja ki/be.

---

## 10. `render.py` – megjelenítés

`render(world, snap, show_edges=True, show_corners=True) -> Figure` (figsize arányos, max ~7 inch):
- Háttér RGB tömb (`imshow`, `interpolation="nearest"`):
  - nem látott szabad cella: sötétszürke `#3a3a3a`
  - lefedett (covered) szabad cella: világos sárga `#fff3a0` (cikk Fig. 1)
  - explored, de jelenleg nem lefedett: halványabb sárga `#d9cf8a`
  - látott fal: szürkéskék `#5b6b7f`; nem látott fal: fekete `#111`
- `xd`: zöld négyzet marker.
- Ágensek: SETTLED piros kör, MOVING narancs kör, RETURNING szürke kör.
- Kommunikációs élek: vékony zöld vonalak (`LineCollection`).
- Célok: arany csillag (CADENCE: kiosztott sarkok, ISDA: aktuális cél, DADENCE: `x*`).
- Felfedezett valid sarkok (opcionális): piros üres kör.
- Cím: `"{alg} | step {k} | agents {n} | coverage {p:.1f}%"`.
- Jelmagyarázat a figura alatt (egyszer).
- Minden frame után `plt.close(fig)` (memóriaszivárgás ellen).

`make_gif(world, snapshots, path, fps=10, every=1)` – `FuncAnimation` + `PillowWriter`.

---

## 11. `app.py` – Streamlit UI

### Sidebar
| Vezérlő | Típus | Alapérték |
|---|---|---|
| Szélesség / Magasság | `number_input` 15–80 | 40 / 40 |
| Seed | `number_input` | 42 |
| Lyukak (pillars) gyakorisága | `slider` 0–1 | 0.3 |
| **Generate map** | `button` | – |
| **Algoritmus** | `selectbox` ["CADENCE", "DADENCE", "ISDA"] | CADENCE |
| Deallocation | `checkbox` | ✔ |
| DADENCE early spawn | `checkbox` (csak DADENCE-nél látszik) | ✔ |
| Nmax | `number_input`, alap = `world.n_max` | auto |
| Tmax | `number_input`, alap = `100 · max(H,W)` (cikk: 50×50 → 5000) | auto |
| Animáció sebesség | `slider` "lépés/frame" 1–50 + késleltetés 0–0.2 s | 1 / 0.03 |
| Valid sarkok / élek mutatása | `checkbox` | ✔ / ✔ |
| **Start** | `button` | – |

### Működés
1. `st.session_state.world` – Generate gombra (vagy első betöltéskor) `generate_world(...)`; megjelenik a statikus térkép (`xd`-vel) + info: `n`, `h`, `#valid corner`, `Nmax`.
2. Start → `algo = ALGORITHMS[name](world, FovCache(world), ...)`; ciklus:
   ```python
   ph_map, ph_stats, prog = st.empty(), st.empty(), st.progress(0)
   for snap in run(algo):
       snaps.append(compact(snap))                    # replay-hez
       if snap.step % every == 0 or done:
           ph_map.pyplot(render(world, snap))
           ph_stats: st.metric × 3 (Step, Agents, Coverage %)
           prog.progress(min(snap.step / t_max, 1.0))
           time.sleep(delay)
   ```
   Stop: Streamlit natív "Stop" (újrafuttatás megszakítja a ciklust) – külön gomb nem kell.
3. Vége → **Összegzés** panel:
   - Algoritmus, térkép (H×W, seed), siker ✔/✘ + ok
   - Szimulációs lépések (lefedésig) és összes (visszasétálással)
   - Valós futási idő (s)
   - Spawnolt ágensek összesen, csúcs (egyidejű), végső (deallokáció után) – megjegyzés: "+1 pseudo-agent (xd)"
   - Lefedettség %
   - Elméleti korlát: `Nmax = (n+2h−4)/2`, `Tmax`
   - `st.line_chart(pd.DataFrame(history).set_index("step")[["agents","coverage_pct"]])`
   - Replay: `st.slider("Step", 0, last)` → az adott snapshot kirajzolása
   - "Download GIF" (`make_gif` → `st.download_button`) és "Download CSV" (history)
4. `session_state.results` listában az eddigi futások összegző sorai → `st.dataframe` (így egy térképen a 3 algoritmus egymás után futtatva összehasonlítható).

### Teljesítmény
- Snapshot-tárolás: `covered`/`explored` `np.packbits`-szel tömörítve.
- 80×80 fölött ne engedjen (UI-limit).
- FOV cache a teljes futás alatt; ugyanarra a világra több futásnál is újrahasznosítható (`session_state.cache`).

---

## 12. `cli.py`

```
python cli.py --size 40 40 --seed 42 --algo DADENCE [--no-dealloc] [--gif out.gif] [--csv hist.csv]
```
Kiírja az összegzést (ugyanazok a mezők, mint a UI-ban). Debughoz és a tesztekhez.

Opcionális extra (nem kötelező): `--quadtree` – kvadfa-csomópontszám (cikk V-A.1) kiírása komplexitásként.

---

## 13. Tesztek (pytest)

`test_world.py`
- Generált világok (10 seed, 3 méret): keret fal; nincs sakktábla 2×2; szabad tér 4-összefüggő; van reflex sarok; `xd` szabad.
- Kézi L-alakú szoba: `n = 6`, reflex = 1, `h = 0`, valid = 1, `Nmax = 1`.
- Kézi téglalap szoba 1 téglalap lyukkal: `n = 8`, `h = 1`, reflex = 4, valid = 3, `Nmax = 3`; a kizárt sarok a lyuk bal-felső sarka.
- Általános: `|valid sarok| == (n + 2h − 4)/2` és `|reflex| == (n + 4h − 4)/2` (Lemma 1) minden generált világon (sarkok száma, nem a deduplikált cellák).

`test_visibility.py`
- LOS szimmetrikus (véletlen párok); 4-szomszéd mindig lát; fal blokkol; átlós rácsponton áthaladó egyenes nem blokkolt, ha nincs sakktábla.
- FOV-cache ugyanazt adja, mint a nem cache-elt számítás.

`test_pathing.py`
- BFS távolság üres szobában = Manhattan; `closer_neighbors` mind 1-gyel közelebb; `border` üres teljes lefedésnél.

`test_dealloc.py`
- Eltávolítás után lefedés változatlan és gráf összefüggő; redundáns ágens (ugyanazon cellán/ugyanazt látó) eltávolításra kerül.

`test_algorithms.py` (mindhárom algoritmus, 5 seed, 25×25 és 40×40)
- Minden lépésben: `G(A ∪ {xd})` összefüggő (RETURNING nélkül).
- DADENCE, ISDA: lefedés monoton nem csökken lépésről lépésre. CADENCE: SETTLED ∪ xd lefedése nem csökken.
- Futás leáll `Tmax`-on belül; `peak_agents ≤ Nmax`; végső lefedés 100% (DADENCE, ISDA; CADENCE-nél ≥ 99%, és log figyelmeztetés, ha < 100%).
- Determinizmus: azonos seed → azonos `Metrics`.
- Kvalitatív (nem hard assert, csak riport): átlagosan CADENCE lépésszáma < DADENCE, DADENCE végső ágensszáma ≤ CADENCE (cikk VI-B).

---

## 14. Implementációs sorrend (mérföldkövek)

1. `world.py` + generátor + `test_world.py` (statikus térkép kirajzolása `render.py`-vel).
2. `visibility.py` + `pathing.py` + tesztek.
3. `sim.py` keret + **ISDA** (legegyszerűbb, validálja a keretet) + `dealloc.py`.
4. **CADENCE**.
5. **DADENCE** (queue-jump, revert, edge pruning, early spawn).
6. `test_algorithms.py` invariánsok.
7. `app.py` (Streamlit): generálás → futtatás → élő preview → összegzés → replay/GIF/CSV.
8. `cli.py`, README-szerű súgó az `app.py` tetején (`streamlit run app.py`).

---

## 15. Dokumentált feltételezések / eltérések a cikktől

| Téma | Döntés | Indok |
|---|---|---|
| LOS diszkrét rácson | cellaközép–cellaközép szakasz mintavételezve, rácspont-minták ignorálva | Def. 10 egyszerűsítése; szimmetrikus → Assumption 1 |
| Valid sarok ágens-cellája | a 2×2 ablakban a fal-cellával átlós szabad cella | diszkrét megfelelője a 270°-os csúcsnak |
| CADENCE "pick x" | `xd`-hez BFS-ben legközelebbi kiosztatlan sarok | a cikk tetszőlegest enged; determinisztikus |
| Áthaladó ágensek ütközése (CADENCE/ISDA) | engedett | cikk: "max 1 spawn/lépés" az ütközés ellen; egyszerűsítés |
| DADENCE revert sorrend | legnagyobb id-jű mozgót vonja vissza először | a cikk tetszőleges választást enged |
| DADENCE queue-jump, ha a teljes út foglalt | x* nem lefedett szomszédjába hosszabbít | elakadás elkerülése, invariánsok megmaradnak |
| ISDA frontier | véletlen border cella (seed) | cikk V-B.3 |
| Ágensszám | `xd` pseudo-agent nélkül; UI-ban "+1 xd" jelölve | cikk ábráiban beleszámolják |
| Tmax alapérték | `100 · max(H, W)` | cikk: 50→5k, 100→10k, 250→30k (közelítés) |
| Kickoff vs. cikk | ütközés nincs érdemben; minden képlet (Eq. 1/3, Def. 8) a cikkből | felhasználói utasítás |
