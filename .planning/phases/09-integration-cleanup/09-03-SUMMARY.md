---
phase: 09-integration-cleanup
plan: 03
subsystem: cleanup
tags: [dead-code-removal, backward-compat, imports, testing]

# Dependency graph
requires:
  - phase: 09-01
    provides: "scope-generic step pipeline with unified step/reset/envstate_to_dict"
  - phase: 08-02
    provides: "jax_step.py backward-compat shim, build_step_fn/build_reset_fn factories"
  - phase: 06-03
    provides: "build_feature_fn_jax, get_all_agent_obs_jax backward-compat aliases"
  - phase: 07-01
    provides: "compute_rewards_jax backward-compat alias"
provides:
  - "Clean codebase with zero backward-compat shims or _jax aliases"
  - "All test files updated to unified function names (step, reset, envstate_to_dict)"
  - "Stale tests deleted (test_backward_compat_aliases, test_move_agents_jax_eager_vs_jit, test_process_interactions_jax_eager_vs_jit)"
affects: [09-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single unified function names (step, reset, get_all_agent_obs, compute_rewards) -- no _jax suffixes"

key-files:
  created: []
  modified:
    - cogrid/feature_space/array_features.py
    - cogrid/envs/overcooked/array_rewards.py
    - cogrid/tests/test_step_pipeline.py
    - cogrid/tests/test_cross_backend_parity.py
  deleted:
    - cogrid/core/jax_step.py

key-decisions:
  - "cogrid_env.py changes deferred to 09-02 (parallel execution, no merge conflicts)"

patterns-established:
  - "No backward-compat aliases: all callers import directly from the source module"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 9 Plan 3: Dead Code Cleanup Summary

**Deleted jax_step.py shim, all _jax backward-compat aliases, and stale tests referencing removed Phase 1-7 functions**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-13T00:37:46Z
- **Completed:** 2026-02-13T00:40:59Z
- **Tasks:** 2
- **Files modified:** 5 (1 deleted, 2 source modified, 2 test modified)

## Accomplishments
- Deleted cogrid/core/jax_step.py (7-line backward-compat shim from Phase 8)
- Removed all _jax suffixed aliases: build_feature_fn_jax, get_all_agent_obs_jax, compute_rewards_jax
- Deleted 3 stale test functions that referenced removed Phase 1-7 functions (move_agents_jax, process_interactions_jax, test_backward_compat_aliases)
- Updated 4 test functions to import from step_pipeline instead of jax_step
- vmap at 1024 environments already correctly configured in test_vmap_correctness.py (BATCH_SIZE = 1024)

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete jax_step.py and backward-compat aliases** - `be4d84b` (chore)
2. **Task 2: Update tests to use unified imports and verify vmap** - `e894b15` (refactor)

## Files Created/Modified
- `cogrid/core/jax_step.py` - DELETED (backward-compat shim no longer needed)
- `cogrid/feature_space/array_features.py` - Removed build_feature_fn_jax, get_all_agent_obs_jax aliases
- `cogrid/envs/overcooked/array_rewards.py` - Removed compute_rewards_jax alias
- `cogrid/tests/test_step_pipeline.py` - Deleted test_backward_compat_aliases
- `cogrid/tests/test_cross_backend_parity.py` - Deleted 2 stale tests, updated 4 tests to unified imports

## Decisions Made
- cogrid_env.py not modified (09-02 handles those imports in parallel to avoid merge conflicts)
- Test functions renamed to remove _jax suffix (test_jax_step_determinism -> test_step_determinism, etc.)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- JAX-backend tests (test_cross_backend_parity.py, test_vmap_correctness.py) cannot run until 09-02 updates cogrid_env.py to use unified imports. The failure is `ImportError: cannot import name 'build_feature_fn_jax'` at cogrid_env.py:222, which is 09-02's responsibility. test_step_pipeline.py (numpy backend path) passes all 3 tests.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Once 09-02 updates cogrid_env.py imports, the full test suite (including JAX backend and vmap tests) should pass
- Zero backward-compat code remains in scope files
- Only cogrid_env.py retains references to deleted names (09-02 handles)

## Self-Check: PASSED

- All files exist/deleted as expected
- Both commits (be4d84b, e894b15) verified in git history
- Zero stale _jax references in test files

---
*Phase: 09-integration-cleanup*
*Completed: 2026-02-12*
