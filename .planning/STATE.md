# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-10)

**Core value:** Existing trained agents produce identical behavior after upgrade while unlocking 100x+ throughput via JAX JIT and vmap.
**Current focus:** Phase 3 - End-to-End Integration & Parity

## Current Position

Phase: 3 of 4 (End-to-End Integration & Parity)
Plan: 2 of 3 in current phase (COMPLETE)
Status: Executing Phase 3
Last activity: 2026-02-11 -- Completed 03-02 (PettingZoo JAX backend integration)

Progress: [███████░░░] 63%

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 6min
- Total execution time: 1.39 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 7 | 51min | 7min |
| 01.1 | 3 | 14min | 5min |
| 02 | 3 | 9min | 3min |
| 03 | 2 | 10min | 5min |

**Recent Trend:**
- Last 5 plans: 8min, 4min, 5min, 5min, 5min
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
- [01.1-01]: Scope config uses callable builder (not dict) so tables are built lazily at first get_scope_config() call, avoiding import-time backend dependency
- [01.1-01]: xp imported inside functions (not at module level) in array_config.py matching existing codebase pattern
- [01.1-01]: interaction_handler takes action_type string ('pickup_from' or 'place_on') to dispatch scope-specific sub-cases from generic priority chain
- [01.1-02]: Backward-compatible re-exports in core/array_rewards.py with TODO marker for removal after Plan 03 wiring
- [01.1-02]: test_interaction_parity kept as standalone callable (not pytest) to match existing usage pattern
- [01.1-03]: Handler delegation with fallthrough: interaction_handler returns True/False, False allows generic fallback (e.g. counter place_on)
- [01.1-03]: tick_objects_array() removed from core entirely -- Overcooked tick handler accessed directly via scope_config["tick_handler"]
- [01.1-03]: extra_state passed as **kwargs to process_interactions_array for scope-specific arrays (pot_contents, pot_timer, etc.)
- [01.1-03]: Docstring references to Overcooked kept as informational examples -- only code logic removed from core
- [Roadmap revision]: Restructured from 8 phases to 4 -- front-loading the vectorization rewrite (movement, interactions, obs, rewards) into Phase 1 alongside backend dispatch, rather than deferring it to phases 3-6
- [Roadmap revision]: Phase 1 includes 21 requirements covering backend dispatch, array state representation, and all simulation logic vectorization -- this is intentionally the largest phase as it is the core work
- [Roadmap revision]: Functional state model (EnvState pytree) and JIT compatibility deferred to Phase 2 -- vectorized array ops come first, immutable pytree wrapping comes second
- [Roadmap]: Integration constraint honored -- all phases refactor existing code, new files limited to backend module and EnvState definition
- [02-01]: EnvState uses object type hints (not jax.Array) so class definition works without JAX installed
- [02-01]: register_envstate_pytree() is idempotent and separate from class definition -- called only when backend is jax
- [02-01]: move_agents_jax() returns (new_pos, new_dir, new_key) -- 3-tuple with consumed PRNG key, unlike numpy path's 2-tuple
- [02-01]: Direction vector table created inline in JAX path as jnp.array rather than using shared lazy-init global
- [02-01]: JAX 0.4.38 required for numpy 1.26.4 compatibility (JAX 0.9.0 requires numpy>=2.0)
- [02-02]: process_interactions_jax designed for call from JIT context, not direct jax.jit wrapping -- non-array args (scope_config, lookup_tables) closed over at trace time via functools.partial or closure
- [02-02]: All 4 interaction branches computed unconditionally with jnp.where selection -- no Python if/else on traced values
- [02-02]: Pot position lookup via jnp.all(pot_positions == target, axis=1) + jnp.argmax replaces dict-based pot_pos_to_idx
- [02-02]: Toggle dispatch uses lax.switch with n_types branches (default no-op, door toggle at door type ID) -- extensible via scope config toggle_branches_jax
- [02-02]: Static tables built at scope config init time, closed over by interaction body -- not passed as traced args
- [02-02]: Fixed _build_interaction_tables to use .at[].set() for JAX array compatibility
- [02-03]: full_map_encoding_feature_jax takes pre-computed agent_type_ids array instead of scope string to avoid string lookup under JIT
- [02-03]: compute_rewards_jax uses closure pattern for JIT (reward_config captured, not passed as arg) since dicts with strings not hashable
- [02-03]: Direction vector table created inline in JAX reward helper as jnp.array (matching 02-01 pattern)
- [02-03]: Shared _compute_fwd_positions_jax helper extracts forward position computation used by all three JAX reward functions
- [03-01]: JAX imports at function level in jax_step.py to avoid cogrid/core/typing.py shadowing stdlib typing
- [03-01]: Smoke test uses numpy env to extract layout then converts to JAX arrays, avoiding backend conflict
- [03-01]: scope_config static_tables numpy arrays must be explicitly converted to jnp.array before JIT tracing
- [03-01]: Step ordering verified: prev_state capture -> tick -> move -> interact -> obs -> rewards -> dones
- [03-02]: Layout parsing (layout_to_array_state, create_agent_arrays, _extract_overcooked_state) always uses numpy -- JAX arrays are immutable, can't do in-place assignment
- [03-02]: JAX feature names hard-coded to [agent_position, agent_dir, full_map_encoding, can_move_direction, inventory] regardless of config feature space name
- [03-02]: Reward name mapping: strip _reward suffix from config names (delivery_reward -> delivery) to match JAX fn_map
- [03-02]: Static tables and interaction tables converted from numpy to jnp at __init__ time for JIT compatibility

### Pending Todos

None yet.

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1: Fix environment separation of concerns (no environment-specific logic in core methods) (URGENT)

### Blockers/Concerns

- Research flags interactions (pot cooking state machine) as MEDIUM research confidence -- may need phase-specific research during Phase 1 planning
- Byte-identity contract may need relaxation to allclose(atol=1e-7) for float values -- to be determined empirically in Phase 3

## Session Continuity

Last session: 2026-02-11
Stopped at: Completed 03-02-PLAN.md (PettingZoo JAX backend integration)
Resume file: None
