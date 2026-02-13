---
phase: 08-step-pipeline
plan: 02
subsystem: core
tags: [xp, step-pipeline, build-factories, jit, backward-compat, shim]

# Dependency graph
requires:
  - phase: 08-step-pipeline-01
    provides: "step(), reset(), envstate_to_dict() in step_pipeline.py"
  - phase: 06-simulation-functions
    provides: "move_agents(), process_interactions(), get_all_agent_obs(), build_feature_fn()"
  - phase: 07-rewards-scope-config
    provides: "compute_rewards(), build_overcooked_scope_config()"
provides:
  - "build_step_fn() factory: closes over static config, returns (state, actions) -> ... closure with auto-JIT on JAX"
  - "build_reset_fn() factory: closes over layout config, returns (rng) -> (state, obs) closure with auto-JIT on JAX"
  - "jax_step.py backward-compat shim re-exporting all names from step_pipeline.py"
  - "End-to-end test suite verifying both backends and JIT compilation"
affects: [09-pettingzoo-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Init-time function composition: build_step_fn/build_reset_fn close over static config and return pure closures"
    - "Auto-JIT pattern: jit_compile=None auto-detects from get_backend(), True/False forces"
    - "Backward-compat shim: thin module re-exports to maintain old import paths during transition"

key-files:
  created:
    - cogrid/tests/test_step_pipeline.py
  modified:
    - cogrid/core/step_pipeline.py
    - cogrid/core/jax_step.py
    - cogrid/core/movement.py

key-decisions:
  - "build_step_fn/build_reset_fn use jit_compile=None defaulting to auto-detect from get_backend() rather than always-JIT"
  - "jax_step.py reduced to 7-line shim with import aliases -- all implementation in step_pipeline.py"
  - "DIR_VEC_TABLE in move_agents() created inline via xp.array() instead of cached global to avoid stale numpy array under JIT"

patterns-established:
  - "Factory pattern for step/reset: init-time closure composition with auto-JIT wrapping"
  - "Backward-compat shim pattern: from module import X as old_name"

# Metrics
duration: 4min
completed: 2026-02-12
---

# Phase 8 Plan 02: Build Factories and Backward-Compat Shim Summary

**build_step_fn/build_reset_fn factories with auto-JIT wrapping, jax_step.py converted to 7-line backward-compat shim**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-12T23:57:25Z
- **Completed:** 2026-02-13T00:01:10Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `build_step_fn()` and `build_reset_fn()` to `step_pipeline.py` -- init-time factories that close over all static config and return pure `(state, actions) -> ...` / `(rng) -> (state, obs)` closures, with automatic `jax.jit` wrapping on JAX backend
- Replaced the 600-line `jax_step.py` with a 7-line backward-compat shim that re-exports all names from `step_pipeline.py`, ensuring zero breakage of `cogrid_env.py` and existing tests
- Created comprehensive test suite (`test_step_pipeline.py`) with 4 tests covering numpy step/reset, JAX eager step/reset, JIT-compiled factory execution, and backward-compat alias verification
- Fixed DIR_VEC_TABLE caching bug in `move_agents()` that caused `TracerArrayConversionError` under JIT when the cached numpy array was indexed with a JAX tracer

## Task Commits

Each task was committed atomically:

1. **Task 1: Add build_step_fn and build_reset_fn factories, convert jax_step.py to shim** - `e9fd7d6` (feat)
2. **Task 2: Create end-to-end test suite for unified step pipeline** - `2ce5822` (test)

## Files Created/Modified

- `cogrid/core/step_pipeline.py` - Added `build_step_fn()` and `build_reset_fn()` factory functions with auto-JIT detection
- `cogrid/core/jax_step.py` - Replaced with backward-compat shim: re-exports `step` as `jax_step`, `reset` as `jax_reset`, `build_step_fn` as `make_jitted_step`, `build_reset_fn` as `make_jitted_reset`, plus `envstate_to_dict`
- `cogrid/core/movement.py` - Fixed DIR_VEC_TABLE: inline `xp.array()` creation instead of cached global from `get_dir_vec_table()`
- `cogrid/tests/test_step_pipeline.py` - 4 end-to-end tests for the unified step pipeline on both backends

## Decisions Made

- **Auto-detect JIT wrapping:** `build_step_fn`/`build_reset_fn` default `jit_compile=None` which auto-detects from `get_backend()`. This allows the same factory code to work on both backends without caller changes.
- **jax_step.py as pure shim:** All implementation removed from `jax_step.py` -- it is now exclusively import aliases. This is the cleanest separation, with full cleanup (deletion) deferred to Phase 9.
- **Inline DIR_VEC_TABLE creation:** Changed `move_agents()` to create `DIR_VEC_TABLE` via `xp.array()` on every call instead of relying on the cached global from `get_dir_vec_table()`. The cached global breaks under JIT when the first call happened on numpy backend. The 4x2 constant is trivially cheap and JIT will constant-fold it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed DIR_VEC_TABLE caching bug in move_agents()**
- **Found during:** Task 2 (test_build_step_fn_jit_compiles)
- **Issue:** `get_dir_vec_table()` lazily caches a global `DIR_VEC_TABLE` using `xp.array()`. When first called on numpy backend, it caches a numpy array. When `move_agents()` is later called under JAX JIT, indexing the numpy array with a JAX tracer raises `TracerArrayConversionError`.
- **Fix:** Replaced `DIR_VEC_TABLE = get_dir_vec_table()` with inline `xp.array([[0,1],[1,0],[0,-1],[-1,0]], dtype=xp.int32)` in `move_agents()`. This ensures the table is always created with the active backend's array type.
- **Files modified:** `cogrid/core/movement.py`
- **Verification:** `test_build_step_fn_jit_compiles` passes -- `jax.jit(step_fn)(state, actions)` compiles and executes without error
- **Committed in:** `2ce5822` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential for JIT correctness. No scope creep.

## Issues Encountered

- Pre-existing test failures in `test_cross_backend_parity.py`: Several tests (`test_scripted_parity`, `test_eager_vs_jit`, `test_jax_step_determinism`, `test_move_agents_jax_eager_vs_jit`, `test_process_interactions_jax_eager_vs_jit`, `test_obs_jax_eager_vs_jit`) fail due to references to removed APIs (`state.pot_contents`, `move_agents_jax`, `process_interactions_jax`). These are pre-existing issues from Phase 5-7 refactoring, not caused by Plan 08-02. They will be addressed in Phase 9 cleanup.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 8 is now complete: unified `step_pipeline.py` with `step()`, `reset()`, `build_step_fn()`, `build_reset_fn()`, `envstate_to_dict()` -- all working on both numpy and JAX backends with JIT compilation verified
- `jax_step.py` is a clean shim ready for deletion in Phase 9
- `cogrid_env.py` continues working via backward-compat imports from `jax_step.py`
- Phase 9 can: (1) update `cogrid_env.py` to import from `step_pipeline.py` directly, (2) wire numpy backend through the functional pipeline, (3) delete `jax_step.py`, (4) fix pre-existing test failures in `test_cross_backend_parity.py`

---
*Phase: 08-step-pipeline*
*Completed: 2026-02-12*
