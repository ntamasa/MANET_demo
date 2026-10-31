# Algorithm subclass API (manet.sim, T5)

- `class X(Algorithm): name = "CADENCE"`; `__init__(world, cache=None, *, n_max=None, t_max=None, seed=0, dealloc=True, **opts)` → call `super().__init__(...)` first, then own state. Extra kwargs in `self.opts`. All randomness via `self.rng`.
- `_step() -> bool` (only required method): one time unit. Move each active agent at most one 4-neighbour cell by assigning `a.pos`; set `a.status` MOVING/SETTLED; `a.target`, `a.path`, `a.pruned` free to use. Do NOT touch RETURNING agents, `step_count`, `explored`, `metrics`, `phase`.
- Return True = continue; return False = coverage phase complete (base calls `finish("coverage complete")`; success = 100% coverage of active agents + xd). Call `self.deallocate(...)` yourself before returning False if the algorithm deallocates at the end.
- `self.finish(msg, failed=True)` only for algorithm-specific failure. Never for Tmax/stall (base does it: Tmax → "Tmax exceeded", no progress for `stall_limit = 4*(H+W)` steps → "stalled"; progress = explored grew or any agent (id,pos,status) changed).
- Helpers: `active_agents()`, `active_cells()` (non-RETURNING), `xd_occupied()`, `can_spawn()` (xd free of active agents and active count < n_max), `spawn()` → new MOVING Agent at xd or None (ids from 0; updates spawned_total/peak), `current_coverage()` (xd + active cells; cached, READ-ONLY → copy before mutating), `deallocate(terminal_agents)` → removed agents become RETURNING with a path to xd, moved automatically; other active agents are kept as fixed graph members (their coverage counts!); returns [] if dealloc=False.
- **Safety rule for deallocation:** call `deallocate()` only when no agent is MOVING (transient coverage of a moving agent would otherwise justify removing a settled agent whose coverage is then lost when the mover leaves).
- State: `self.explored` (cumulative seen mask; usable as `passable`), `self.cache.fov(cell)`, `self.cache.sees(a,b)`, `self.world.valid_corner_cells`, `self.world.xd`, `self.n_max`, `self.t_max`, `self.step_count`.
- Hooks: `_targets()`, `_known_valid()` → cell lists for `Snapshot.targets` / `Snapshot.known_valid`.
- Driving: `for snap in run(algo): ...` or `while algo.step(): pass`; then `algo.metrics`, `algo.message`.
- Step order in base: `_step()` → returning agents move → explored/peak/history update → checks (complete → Tmax → stall).
- Pathing: `manet.pathing.bfs_dist/shortest_path/closer_neighbors/border/neighbors`; visibility: `manet.visibility.coverage/is_connected/visibility_graph/los`.
