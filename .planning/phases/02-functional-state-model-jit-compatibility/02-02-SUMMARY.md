---
phase: 02-functional-state-model-jit-compatibility
plan: 02
subsystem: interactions
tags: [jax, jit, lax-fori-loop, lax-switch, jnp-where, overcooked, interactions, toggle]

# Dependency graph
requires:
  - phase: 01-backend-dispatch-array-state
    provides: "Vectorized process_interactions_array, lookup tables, scope config"
  - phase: 01.1-fix-environment-separation
    provides: "Scope config with interaction_handler, tick_handler, array_config.py"
  - phase: 02-01
    provides: "EnvState pytree, lax.fori_loop collision pattern, jnp.where masking pattern"
provides:
  - "process_interactions_jax() JIT-compatible interaction function in core/interactions.py"
  - "overcooked_interaction_body_jax() with jnp.where masking for all 4 priority branches"
  - "overcooked_tick_jax() JIT-compatible pot cooking tick"
  - "Toggle dispatch via jax.lax.switch on object type ID"
  - "Static tables pattern for scope config under JIT"
affects: [02-03-PLAN, observations-jax, rewards-jax, step-jit-composition]

# Tech tracking
tech-stack:
  added: [jax.lax.switch, functools.partial]
  patterns: [jnp-where-cascading-branches, array-based-position-matching, static-tables-closure, lax-switch-type-dispatch]

key-files:
  created: []
  modified:
    - cogrid/core/interactions.py
    - cogrid/envs/overcooked/array_config.py

key-decisions:
  - "process_interactions_jax designed for call from JIT context, not direct jax.jit wrapping -- non-array args (scope_config, lookup_tables) closed over at trace time via functools.partial or closure"
  - "All 4 interaction branches computed unconditionally with jnp.where selection -- no Python if/else on traced values"
  - "Pot position lookup via jnp.all(pot_positions == target, axis=1) + jnp.argmax replaces dict-based pot_pos_to_idx"
  - "Toggle dispatch uses lax.switch with n_types branches (default no-op, door toggle at door type ID) -- extensible via scope config toggle_branches_jax"
  - "Static tables built at scope config init time, closed over by interaction body -- not passed as traced args"
  - "Fixed _build_interaction_tables to use .at[].set() for JAX array compatibility (arrays created by xp are JAX arrays when backend is jax)"

patterns-established:
  - "Cascading jnp.where: compute all branch results unconditionally, select with cascading jnp.where(b1, r1, jnp.where(b2, r2, ...)) for mutually exclusive branches"
  - "Static tables closure: scope config provides lookup arrays and constants at init time, interaction body closes over them for zero-overhead JIT access"
  - "Array-based position matching: replace dict.get((r,c)) with jnp.all(positions == target, axis=1) for JIT-compatible spatial lookup"
  - "lax.switch type dispatch: build list of n_types handler functions, dispatch by type ID index -- all branches traced, runtime selects one"
  - "functools.partial for JIT binding: bind non-array args (scope_config, tables, constants) via functools.partial, JIT the resulting pure-array closure"

# Metrics
duration: 7min
completed: 2026-02-12
---

# Phase 2 Plan 2: JAX Interactions Summary

**JIT-compatible interaction processing with lax.fori_loop agent iteration, jnp.where branch masking, array-based pot lookup, and lax.switch toggle dispatch**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-12T03:58:35Z
- **Completed:** 2026-02-12T04:05:49Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- process_interactions_jax() JIT-compiles without ConcretizationTypeError for both Overcooked and generic scopes
- All 4 priority branches (pickup, pickup_from, drop, place_on) with Overcooked sub-cases (pot, delivery zone, stacks) implemented via jnp.where masking
- Toggle action dispatches via jax.lax.switch on object type ID with door toggle and extensible scope config branches
- Overcooked tick function (overcooked_tick_jax) JIT-compiles and produces correct timer decrements
- Existing numpy path (process_interactions_array) completely unchanged with all PHASE2 markers preserved

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement JAX-compatible Overcooked interaction and tick functions** - `98dda80` (feat)
2. **Task 2: Implement core process_interactions_jax with lax.fori_loop** - `bfaa83e` (feat)

## Files Created/Modified
- `cogrid/core/interactions.py` - Added process_interactions_jax() with lax.fori_loop, generic body, and lax.switch toggle dispatch
- `cogrid/envs/overcooked/array_config.py` - Added overcooked_interaction_body_jax(), overcooked_tick_jax(), _build_static_tables(); updated build_overcooked_scope_config() with JAX keys; fixed _build_interaction_tables() for JAX array compatibility

## Decisions Made
- **functools.partial for JIT:** process_interactions_jax uses Python-level control flow to build closures around scope config and lookup tables, designed to be called from within a JIT context (e.g., jitted step function) rather than being directly jax.jit-wrapped with static_argnames
- **Unconditional branch computation:** All 4 priority branches computed unconditionally with results selected via cascading jnp.where -- ensures uniform execution under JIT tracing
- **Array-based pot matching:** pot_pos_to_idx dict replaced with jnp.all(pot_positions == target, axis=1) + jnp.argmax for JIT-compatible spatial lookup
- **n_types toggle branches:** Toggle dispatch builds a list of n_types functions (default noop), indexed by object type ID via lax.switch -- extensible through scope config toggle_branches_jax list
- **Static tables at build time:** Scope config _build_static_tables() pre-builds all lookup arrays and integer constants at config creation time, avoiding runtime dict lookups under JIT

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] _build_interaction_tables uses direct array assignment incompatible with JAX**
- **Found during:** Task 2 (process_interactions_jax verification)
- **Issue:** `pickup_from_produces[onion_stack_id] = onion_id` raises TypeError on JAX arrays ("JAX arrays are immutable")
- **Fix:** Added .at[].set() path with `hasattr(arr, 'at')` guard for JAX, keeping direct assignment for numpy
- **Files modified:** cogrid/envs/overcooked/array_config.py
- **Verification:** Scope config builds successfully on JAX backend
- **Committed in:** bfaa83e (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix for JAX backend compatibility. No scope creep.

## Issues Encountered
None beyond the auto-fixed issue above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- process_interactions_jax() ready for integration into JIT-compiled step() function
- overcooked_tick_jax() ready for JIT-compiled tick processing
- Static tables pattern established for observations and rewards JAX conversion in Plan 03
- Toggle dispatch extensible for future scope-specific toggle types
- All interaction sub-functions (pickup, pickup_from, drop, place_on, pot, delivery zone, stacks) have JAX-path equivalents

## Self-Check: PASSED

- [x] cogrid/core/interactions.py exists (modified)
- [x] cogrid/envs/overcooked/array_config.py exists (modified)
- [x] 02-02-SUMMARY.md exists
- [x] Commit 98dda80 found (Task 1)
- [x] Commit bfaa83e found (Task 2)
- [x] process_interactions_jax importable
- [x] overcooked_interaction_body_jax importable
- [x] overcooked_tick_jax importable

---
*Phase: 02-functional-state-model-jit-compatibility*
*Completed: 2026-02-12*
