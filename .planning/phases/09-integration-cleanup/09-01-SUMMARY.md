---
phase: 09-integration-cleanup
plan: 01
subsystem: core
tags: [step-pipeline, scope-generic, refactor, callbacks]

# Dependency graph
requires:
  - phase: 08-step-pipeline
    provides: "Unified step/reset pipeline with build factories"
provides:
  - "Scope-generic step_pipeline.py with zero env-specific imports"
  - "overcooked_tick_state() wrapper conforming to generic tick handler signature"
  - "Generic extra_state prefix convention in reset() and step()"
  - "reward_config compute_fn callback pattern"
affects: [09-02, 09-03, cogrid_env]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "tick_handler(state, scope_config) -> state signature for scope-specific tick logic"
    - "reward_config['compute_fn'] callback instead of hardcoded import"
    - "Generic extra_state prefix-strip/re-prefix for process_interactions kwargs"
    - "layout_arrays base_keys convention for building extra_state in reset()"

key-files:
  created: []
  modified:
    - "cogrid/core/step_pipeline.py"
    - "cogrid/envs/overcooked/array_config.py"
    - "cogrid/cogrid_env.py"
    - "cogrid/tests/test_step_pipeline.py"

key-decisions:
  - "tick_handler signature changed from (pot_contents, pot_timer) to (state, scope_config) -> state for full generality"
  - "reward compute_fn provided via reward_config dict rather than imported at module level"
  - "extra_state keys auto-prefixed/stripped using scope.key convention in step() and reset()"
  - "pot_capacity and cooking_time removed from reset() signature (scope-specific, not core params)"

patterns-established:
  - "Scope config must include 'scope' key for prefix convention"
  - "reward_config must include 'compute_fn' key pointing to reward function"
  - "Tick handlers accept (state, scope_config) and return updated state"

# Metrics
duration: 4min
completed: 2026-02-12
---

# Phase 9 Plan 1: Scope-Generic Step Pipeline Summary

**Removed all Overcooked-specific hardcoding from step_pipeline.py via tick_handler state wrapper, reward compute_fn callback, and generic extra_state prefix convention**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-13T00:31:09Z
- **Completed:** 2026-02-13T00:35:06Z
- **Tasks:** 1
- **Files modified:** 4

## Accomplishments
- step_pipeline.py now has zero imports from `cogrid.envs.*` and zero hardcoded "overcooked" string literals
- Tick handler delegated to scope config with generic `(state, scope_config) -> state` signature
- Reward computation delegated via `reward_config["compute_fn"]` callback
- Extra_state keys handled generically via scope prefix convention in both step() and reset()
- All 4 existing step pipeline tests pass unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Make step_pipeline.py scope-generic** - `ead2e0f` (feat)

## Files Created/Modified
- `cogrid/core/step_pipeline.py` - Removed all env-specific imports and hardcoded keys; tick/reward/extra_state now fully generic
- `cogrid/envs/overcooked/array_config.py` - Added `overcooked_tick_state()` wrapper, added `"scope"` key to config, updated tick_handler reference
- `cogrid/cogrid_env.py` - Added `"compute_fn": compute_rewards` to reward config dict
- `cogrid/tests/test_step_pipeline.py` - Added `"compute_fn": compute_rewards` to test reward config

## Decisions Made
- tick_handler signature changed from `(pot_contents, pot_timer) -> (pot_contents, pot_timer, pot_state)` to `(state, scope_config) -> state` -- the wrapper in array_config.py handles the extraction/re-assembly
- reward compute_fn is provided in reward_config dict rather than imported at module level in step_pipeline.py -- callers (cogrid_env.py, tests) set this key
- extra_state keys use generic prefix-strip/re-prefix pattern: `"scope.key"` stripped to `"key"` for kwargs, re-prefixed on return
- Removed `pot_capacity` and `cooking_time` from `reset()` signature since they are Overcooked-specific concerns handled inside the tick handler closure

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed variable shadowing of `key` in step() and reset()**
- **Found during:** Task 1 (test_step_jax_backend_eager failure)
- **Issue:** The `for key, val in state.extra_state.items()` loop shadowed the `key` variable holding the JAX PRNG key, causing `jax.random.split(state.rng_key)` to receive a string on the next step
- **Fix:** Renamed loop variables to `es_key` (in step) and `la_key` (in reset) to avoid shadowing
- **Files modified:** cogrid/core/step_pipeline.py
- **Verification:** All 4 tests pass including JAX backend test
- **Committed in:** ead2e0f (part of task commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix was necessary for correctness. No scope creep.

## Issues Encountered
None beyond the variable shadowing bug caught and fixed during verification.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- step_pipeline.py is now fully scope-generic, ready for 09-02 (backward-compat alias cleanup) and 09-03 (final integration)
- All existing tests pass without modification

---
*Phase: 09-integration-cleanup*
*Completed: 2026-02-12*
