# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Existing trained agents produce identical behavior after upgrade while unlocking 100x+ throughput via JAX JIT and vmap.
**Current focus:** Phase 1 - Dual Backend & Vectorized Core Rewrite

## Current Position

Phase: 1 of 4 (Dual Backend & Vectorized Core Rewrite)
Plan: 7 of 7 in current phase (COMPLETE)
Status: Phase 1 Complete
Last activity: 2026-02-11 -- Completed 01-07-PLAN.md (Integration)

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 7min
- Total execution time: 0.85 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 7 | 51min | 7min |

**Recent Trend:**
- Last 5 plans: 6min, 9min, 7min, 2min, 6min
- Trend: consistent, fast

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [01-01]: Used __getattr__ lazy resolution in cogrid/backend/__init__.py so 'from cogrid.backend import xp' always returns current backend
- [01-01]: Separate 1D int32 arrays per property in build_lookup_tables (CAN_PICKUP, CAN_OVERLAP, etc.) rather than single matrix
- [01-01]: free_space handled as hardcoded overlappable entry in build_lookup_tables since it is not in OBJECT_REGISTRY
- [01-02]: Empty grid cells use type_id=0 in object_type_map (matching object_to_idx(None)==0), NOT -1
- [01-02]: pot_contents and agent_inv use -1 sentinel for empty slots per locked encoding conventions
- [01-02]: DIR_VEC_TABLE lazily initialized via get_dir_vec_table() to avoid import-time backend dependency
- [01-02]: Agent arrays sorted by agent_id for deterministic ordering; agent_ids list maps index to AgentID
- [01-03]: ACTION_TO_DIR lazy-initialized as xp.array([3, 1, 2, 0, -1, -1, -1]) mapping CardinalActions indices to Directions enum values
- [01-03]: Collision resolution and swap detection use Python loops marked PHASE2 for lax.fori_loop conversion
- [01-03]: Parity test uses RNG bit_generator.state forking for identical priority ordering between original and vectorized
- [01-04]: Dynamic can_pickup_from evaluation: static lookup table + inline instance-level condition check prevents false-positive elif matching
- [01-04]: OvercookedAgent.can_pickup() pot special override replicated as separate sub-case in branch 2 (plate required for pot, empty inv for stacks)
- [01-04]: Counter placed-on tracked in object_state_map[r,c] as type_id integer (0=empty); delivery zone consumes soup silently
- [01-05]: Agent overlays in full_map_encoding use scope='global' matching Grid.encode() behavior where grid_agent.encode() called without scope
- [01-05]: CAN_OVERLAP static lookup table used for can_move_direction -- sufficient for Overcooked (no Door objects with dynamic overlap)
- [01-05]: Channel 1 (color) left as 0 in array full_map_encoding -- Pot.encode() tomato flag not replicated from arrays
- [01-06]: Reward functions use prev_state dict exclusively (matching existing pattern where state=self.prev_grid)
- [01-06]: Pot index lookup uses linear scan over pot_positions list; int() casts on array elements for numpy/JAX scalar compatibility
- [01-07]: Vectorized movement as primary path with sync back to Agent objects; existing interact/features/rewards run on synced objects
- [01-07]: Array state fully rebuilt from objects after interact() each step for Phase 1 safety
- [01-07]: Type IDs computed defensively with -1 sentinel for non-existent types to support non-Overcooked scopes
- [01-07]: Interaction tables only built for overcooked scope (None for other scopes)
- [01-07]: Shadow parity validation disabled by default (_validate_array_parity = False)
- [Roadmap revision]: Restructured from 8 phases to 4 -- front-loading the vectorization rewrite (movement, interactions, obs, rewards) into Phase 1 alongside backend dispatch, rather than deferring it to phases 3-6
- [Roadmap revision]: Phase 1 includes 21 requirements covering backend dispatch, array state representation, and all simulation logic vectorization -- this is intentionally the largest phase as it is the core work
- [Roadmap revision]: Functional state model (EnvState pytree) and JIT compatibility deferred to Phase 2 -- vectorized array ops come first, immutable pytree wrapping comes second
- [Roadmap]: Integration constraint honored -- all phases refactor existing code, new files limited to backend module and EnvState definition

### Pending Todos

None yet.

### Blockers/Concerns

- Research flags interactions (pot cooking state machine) as MEDIUM research confidence -- may need phase-specific research during Phase 1 planning
- Byte-identity contract may need relaxation to allclose(atol=1e-7) for float values -- to be determined empirically in Phase 3

## Session Continuity

Last session: 2026-02-11
Stopped at: Completed 01-07-PLAN.md (Phase 1 Complete)
Resume file: None
