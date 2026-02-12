---
phase: 07-rewards-scope-config
plan: 02
subsystem: testing
tags: [cross-backend, parity, rewards, numpy, jax, pytest]

# Dependency graph
requires:
  - phase: 07-01
    provides: unified reward functions (delivery_reward, onion_in_pot_reward, soup_in_dish_reward, compute_rewards) using xp dispatch
provides:
  - cross-backend parity tests for all 3 unified reward functions + compute_rewards
  - updated eager-vs-JIT test using unified compute_rewards name
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "importlib.import_module for cross-backend test module loading (avoids package name collision)"
    - "Backend switch pattern: _reset_backend_for_testing() + set_backend() + importlib.reload()"

key-files:
  created:
    - cogrid/tests/test_reward_parity.py
  modified:
    - cogrid/tests/test_cross_backend_parity.py

key-decisions:
  - "Used importlib.import_module instead of 'import X as Y' to avoid package namespace collision with overcooked.py module"

patterns-established:
  - "Reward parity test pattern: build numpy state dict, run on numpy backend, reset, run on JAX backend, compare with atol=1e-7"

# Metrics
duration: 4min
completed: 2026-02-12
---

# Phase 7 Plan 2: Cross-Backend Reward Parity Tests Summary

**4 cross-backend parity tests verifying unified reward functions produce identical float32 results on numpy and JAX backends, plus eager-vs-JIT test updated to unified compute_rewards name**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-12T23:21:03Z
- **Completed:** 2026-02-12T23:25:04Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 4 new parity tests (delivery, onion_in_pot, soup_in_dish, compute_rewards) verify cross-backend numerical equivalence
- Each test builds a scripted state dict that triggers a specific reward, runs on both backends, compares with atol=1e-7
- Existing eager-vs-JIT test updated from compute_rewards_jax to compute_rewards (unified name)
- TEST-01 requirement satisfied: cross-backend parity verified for all unified reward functions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create cross-backend reward parity tests** - `d52ca13` (test)
2. **Task 2: Update existing test to use unified compute_rewards** - `ee7e0db` (feat)

## Files Created/Modified
- `cogrid/tests/test_reward_parity.py` - 4 cross-backend parity tests for unified reward functions
- `cogrid/tests/test_cross_backend_parity.py` - Updated eager-vs-JIT test to use compute_rewards (unified name)

## Decisions Made
- Used `importlib.import_module()` instead of `import cogrid.envs.overcooked.array_rewards as ar_mod` to avoid a package namespace collision where `cogrid.envs.overcooked` resolves to `overcooked.py` module instead of the package when using the `import X.Y.Z as alias` syntax.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Import path collision for cogrid.envs.overcooked.array_rewards**
- **Found during:** Task 1 (Create cross-backend reward parity tests)
- **Issue:** `import cogrid.envs.overcooked.array_rewards as ar_mod` fails because `cogrid.envs.overcooked` resolves to the `overcooked.py` module (due to `from cogrid.envs.overcooked import overcooked` in `cogrid/envs/__init__.py`), not the package directory.
- **Fix:** Used `importlib.import_module("cogrid.envs.overcooked.array_rewards")` which correctly resolves the module path.
- **Files modified:** cogrid/tests/test_reward_parity.py
- **Verification:** All 4 tests pass on both backends.
- **Committed in:** d52ca13 (Task 1 commit)

**2. [Rule 1 - Bug] Fixed stale EnvState attribute access in eager-vs-JIT test**
- **Found during:** Task 2 (Update existing test to use unified compute_rewards)
- **Issue:** Test accessed `state.pot_contents`, `state.pot_timer`, `state.pot_positions` directly on EnvState, but these moved to `extra_state` dict in Phase 5. The test was already broken before this plan.
- **Fix:** Used `envstate_to_dict()` output (already imported in the test) to access pot arrays via dict keys instead of direct attributes.
- **Files modified:** cogrid/tests/test_cross_backend_parity.py
- **Verification:** `test_rewards_jax_eager_vs_jit` passes.
- **Committed in:** ee7e0db (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Both fixes necessary for tests to run. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 7 complete: all reward functions unified, cross-backend parity verified
- No compute_rewards_jax references remain in test files
- Ready for Phase 8 (step pipeline unification)

---
*Phase: 07-rewards-scope-config*
*Completed: 2026-02-12*
