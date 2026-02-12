---
phase: 06-core-algorithms
plan: 02
subsystem: core
tags: [interactions, xp, array-ops, overcooked, vectorized]

# Dependency graph
requires:
  - phase: 05-foundation-state-model-backend-helpers
    provides: "xp dispatch, array_ops.set_at/set_at_2d, EnvState with extra_state"
provides:
  - "Unified process_interactions() using xp -- single function replaces _array and _jax variants"
  - "Unified overcooked_tick() -- single pot cooking timer using xp"
  - "Unified overcooked_interaction_body() -- single per-agent interaction body using xp and array_ops"
  - "Scope config with unified function references (no _jax keys)"
affects: [06-core-algorithms, 08-step-pipeline]

# Tech tracking
tech-stack:
  added: []
  patterns: ["Per-agent sequential interaction body with xp.where cascading", "Vectorized agent-ahead detection via pairwise position matching"]

key-files:
  created: []
  modified:
    - cogrid/core/interactions.py
    - cogrid/envs/overcooked/array_config.py
    - cogrid/envs/overcooked/test_interactions.py
    - cogrid/cogrid_env.py

key-decisions:
  - "Static-range Python loop over n_agents instead of hardcoded 2-agent unroll -- handles any agent count while remaining JIT-compatible"
  - "Scope config drops interaction_handler/tick_handler_jax/interaction_body_jax keys -- single function per role"

patterns-established:
  - "Per-agent interaction body pattern: compute all branch conditions with xp.where cascading, apply via set_at/set_at_2d"
  - "Vectorized agent-ahead detection: pairwise position comparison matrix with self-exclusion mask"

# Metrics
duration: 7min
completed: 2026-02-12
---

# Phase 6 Plan 02: Unified Interactions Summary

**Single process_interactions() with xp.where cascading replaces both _array and _jax interaction pipelines, plus unified overcooked_tick() and overcooked_interaction_body()**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-12T22:01:30Z
- **Completed:** 2026-02-12T22:08:34Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Unified interaction pipeline: single process_interactions() with vectorized forward position computation and per-agent interaction body delegation
- Unified Overcooked handlers: overcooked_tick() and overcooked_interaction_body() replace 6 old functions (_overcooked_tick_handler, overcooked_tick_jax, _overcooked_interaction_handler, overcooked_interaction_body_jax, _place_on_pot, _place_on_delivery_zone)
- Zero int() casts in all step-path code, zero JAX-specific imports (lax, jnp) in either file
- All 8 parity tests pass with the unified code

## Task Commits

Each task was committed atomically:

1. **Task 1: Unify overcooked_tick() and overcooked_interaction_body()** - `9472edd` (feat)
2. **Task 2: Implement unified process_interactions()** - `15f648b` (feat)

## Files Created/Modified
- `cogrid/core/interactions.py` - Unified process_interactions() replacing both _array and _jax variants
- `cogrid/envs/overcooked/array_config.py` - Unified overcooked_tick() and overcooked_interaction_body(), deleted 6 old functions
- `cogrid/envs/overcooked/test_interactions.py` - Updated imports and call signatures for unified API
- `cogrid/cogrid_env.py` - Fixed stale import (move_agents_array -> move_agents)

## Decisions Made
- Used static-range Python loop over n_agents for sequential agent processing instead of hardcoding 2 agent indices. n_agents is a shape dimension (known at compile time in JAX), so this loop unrolls at trace time and is JIT-compatible. This approach handles 1-agent test scenarios and generalizes beyond Overcooked's 2-agent case.
- Dropped scope config keys: interaction_handler, tick_handler_jax, interaction_body_jax, toggle_branches_jax, place_on_handlers. Single function references per role.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed stale move_agents_array import in cogrid_env.py**
- **Found during:** Task 1 (verification step)
- **Issue:** cogrid_env.py imported move_agents_array which no longer exists (renamed to move_agents in plan 06-01). This blocked all imports from the cogrid package.
- **Fix:** Updated import to use move_agents, adapted call site to pass pre-computed priority array instead of RNG.
- **Files modified:** cogrid/cogrid_env.py
- **Verification:** Python import succeeds
- **Committed in:** 9472edd (Task 1 commit)

**2. [Rule 3 - Blocking] Updated test_interactions.py for unified API**
- **Found during:** Task 2 (verification step)
- **Issue:** test_interactions.py imported process_interactions_array and _overcooked_tick_handler, both deleted. Also passed old-format extra_state with pot_pos_to_idx and type_ids (now handled by static_tables inside interaction_body).
- **Fix:** Updated imports to unified names, simplified _build_extra_state to pass only pot arrays.
- **Files modified:** cogrid/envs/overcooked/test_interactions.py
- **Verification:** All 8 parity tests pass
- **Committed in:** 15f648b (Task 2 commit)

**3. [Rule 1 - Bug] Changed hardcoded 2-agent unroll to static-range loop**
- **Found during:** Task 2 (verification step)
- **Issue:** Plan specified hardcoded agent 0 and agent 1 indexing, but test suite has 1-agent scenarios that would fail with IndexError on agent_inv[1, 0].
- **Fix:** Replaced hardcoded 2-call pattern with `for i in range(n_agents)` loop (n_agents is a static shape dim, unrolls at trace time).
- **Files modified:** cogrid/core/interactions.py
- **Verification:** All 8 parity tests pass (1-agent and 2-agent scenarios)
- **Committed in:** 15f648b (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All auto-fixes necessary for correctness and test execution. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Interaction pipeline unified. Ready for feature extractor unification (plan 06-03).
- jax_step.py still imports old function names (process_interactions_jax) -- will need updating when the step pipeline is unified in Phase 8.

## Self-Check: PASSED

All files found, all commits verified.

---
*Phase: 06-core-algorithms*
*Completed: 2026-02-12*
