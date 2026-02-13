---
phase: 14-auto-wired-cogridenv-validation
plan: 01
subsystem: core
tags: [autowire, scope-config, reward-config, cogrid-env, overcooked, component-registry]

# Dependency graph
requires:
  - phase: 13-overcooked-migration/03
    provides: "build_scope_config_from_components auto-composes tick/interaction/extra_state_builder/static_tables"
  - phase: 13-overcooked-migration/02
    provides: "DeliveryReward, OnionInPotReward, SoupInDishReward registered as ArrayReward subclasses"
provides:
  - "CoGridEnv.__init__() uses build_scope_config_from_components() and build_reward_config_from_components() exclusively"
  - "Zero scope-specific branches in cogrid_env.py"
  - "overcooked/__init__.py triggers registration via decorator imports only (no manual register_scope_config/register_symbols)"
  - "extra_state_builder called in reset() after layout_to_array_state with scope-prefix stripping"
affects: [dead-code-cleanup, goal-finding-example, parity-tests]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Auto-wiring pattern: CoGridEnv uses build_scope_config_from_components and build_reward_config_from_components for all scope configs"
    - "Extra state builder integration: called after layout_to_array_state, scope prefix stripped before merge to avoid double-prefixing by step_pipeline.reset()"
    - "Import-only registration: overcooked/__init__.py imports trigger @register_object_type and @register_reward_type decorators"

key-files:
  created: []
  modified:
    - cogrid/cogrid_env.py
    - cogrid/envs/overcooked/__init__.py
    - cogrid/envs/overcooked/test_interactions.py
    - cogrid/tests/test_cross_backend_parity.py

key-decisions:
  - "Extra state builder keys are scope-prefixed by the builder; strip prefix in reset() before merging into _array_state to avoid double-prefixing by step_pipeline.reset()"
  - "test_rewards_eager_vs_jit updated to use auto-wired compute_fn instead of manual compute_rewards dispatcher"
  - "test_interactions.py updated to use build_scope_config_from_components and added _reset_backend_for_testing for test isolation"

patterns-established:
  - "Auto-wiring as default: CoGridEnv.__init__() uses auto-wiring exclusively, no manual scope_config/reward_config assembly"
  - "Scope-prefix stripping: when extra_state_builder returns prefixed keys, strip before passing to step_pipeline which re-adds prefix"

# Metrics
duration: 5min
completed: 2026-02-13
---

# Phase 14 Plan 01: Auto-Wired CoGridEnv Summary

**CoGridEnv.__init__() switched from manual get_scope_config/reward_config to auto-wiring via build_scope_config_from_components and build_reward_config_from_components with zero scope-specific branches**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-13T14:29:32Z
- **Completed:** 2026-02-13T14:35:27Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Replaced get_scope_config() call with build_scope_config_from_components() in CoGridEnv.__init__(), eliminating scope_config registry dependency
- Replaced manual reward_config assembly (including `if self.scope == "overcooked"` branch) with build_reward_config_from_components()
- Added extra_state_builder handling in reset() after layout_to_array_state with scope-prefix stripping
- Updated overcooked/__init__.py to import-only registration, removing register_scope_config and register_symbols calls
- Updated test_interactions.py and test_cross_backend_parity.py to use auto-wired configs
- All 86 tests pass

## Task Commits

Each task was committed atomically:

1. **Task 1+2: Rewire CoGridEnv.__init__() and reset() to use auto-wiring** - `cc6fce8` (feat)

## Files Created/Modified
- `cogrid/cogrid_env.py` - Replaced get_scope_config + manual reward assembly with auto-wiring calls; added extra_state_builder in reset()
- `cogrid/envs/overcooked/__init__.py` - Removed register_scope_config/register_symbols; added explicit imports for component/reward registration
- `cogrid/envs/overcooked/test_interactions.py` - Updated to use build_scope_config_from_components; added backend reset for test isolation
- `cogrid/tests/test_cross_backend_parity.py` - Updated test_rewards_eager_vs_jit to use auto-wired compute_fn

## Decisions Made
- Extra state builder returns scope-prefixed keys (e.g., "overcooked.pot_contents"). These must be stripped before merging into _array_state because step_pipeline.reset() re-adds the prefix when building EnvState.extra_state. This avoids double-prefixing ("overcooked.overcooked.pot_contents").
- Tasks 1 and 2 were combined into a single commit because the extra_state_builder fix (Task 2) was blocking Task 1's test verification -- the tests require pot arrays to be present in EnvState.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Double scope-prefix in extra_state keys**
- **Found during:** Task 1 (test verification)
- **Issue:** extra_state_builder returned scope-prefixed keys (overcooked.pot_contents), and step_pipeline.reset() added another prefix, resulting in overcooked.overcooked.pot_contents
- **Fix:** Strip scope prefix from extra_state_builder output before merging into _array_state
- **Files modified:** cogrid/cogrid_env.py
- **Verification:** All 86 tests pass; EnvState.extra_state keys are correctly single-prefixed
- **Committed in:** cc6fce8

**2. [Rule 1 - Bug] test_rewards_eager_vs_jit used old compute_rewards with new reward_config format**
- **Found during:** Task 1 (test verification)
- **Issue:** Test imported manual compute_rewards dispatcher which expects reward_config["rewards"] key, but auto-wired config uses compute_fn closure
- **Fix:** Updated test to use reward_config["compute_fn"] instead of manual compute_rewards
- **Files modified:** cogrid/tests/test_cross_backend_parity.py
- **Verification:** test_rewards_eager_vs_jit passes with JIT and eager parity
- **Committed in:** cc6fce8

**3. [Rule 3 - Blocking] test_interaction_parity backend mismatch after test ordering**
- **Found during:** Task 1 (test verification)
- **Issue:** test_interactions.py had no backend reset; after JAX tests set backend to "jax", numpy arrays failed with 'numpy.ndarray has no attribute at'
- **Fix:** Added _reset_backend_for_testing() call at test start
- **Files modified:** cogrid/envs/overcooked/test_interactions.py
- **Verification:** Full test suite passes in any execution order
- **Committed in:** cc6fce8

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All fixes necessary for correctness. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- CoGridEnv now uses auto-wiring exclusively for scope_config and reward_config
- Ready for Plan 02 (goal-finding example + JIT/vmap parity tests) and Plan 03 (dead code deletion)
- Manual build_overcooked_scope_config(), compute_rewards dispatcher, scope_config.py module are now dead code candidates for Plan 03

## Self-Check: PASSED

- All 4 modified files exist on disk
- Task commit (cc6fce8) found in git log
- 14-01-SUMMARY.md exists
- All 86 tests pass

---
*Phase: 14-auto-wired-cogridenv-validation*
*Completed: 2026-02-13*
