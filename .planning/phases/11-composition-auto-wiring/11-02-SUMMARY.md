---
phase: 11-composition-auto-wiring
plan: 02
subsystem: core
tags: [autowire, reward-config, compute-fn, composition, array-reward, tdd]

# Dependency graph
requires:
  - phase: 10-02
    provides: ArrayReward base class, register_reward_type decorator, RewardMetadata dataclass
  - phase: 11-01
    provides: autowire.py module with build_scope_config_from_components, test_autowire.py
provides:
  - build_reward_config_from_components function that auto-builds reward_config from registry metadata
  - Composed compute_fn closure that instantiates registered ArrayReward subclasses, applies coefficient weighting and common_reward broadcasting
  - Zero-reward fallback for scopes with no registered rewards
affects: [overcooked-migration, cogrid-env-simplification, phase-14-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [closure-based-reward-composition, composition-layer-coefficient-weighting, global-plus-scope-reward-merging]

key-files:
  created: []
  modified:
    - cogrid/core/autowire.py
    - cogrid/tests/test_autowire.py

key-decisions:
  - "Coefficient and common_reward handling is in the composition layer, not inside compute() -- compute() returns raw unweighted rewards"
  - "Global-scope rewards are merged after scope-specific rewards via get_reward_types('global')"
  - "compute_fn uses cogrid.backend.xp for JAX/numpy compatibility (same pattern as existing compose_rewards)"

patterns-established:
  - "Composition-time instantiation: ArrayReward instances created once and closed over by compute_fn, not recreated at step time"
  - "Composition-layer weighting: coefficient * raw_reward applied in compute_fn, common_reward broadcasting (sum then fill) also in compute_fn"

# Metrics
duration: 2min
completed: 2026-02-13
---

# Phase 11 Plan 02: Reward Config Auto-Wiring Summary

**build_reward_config_from_components auto-composes a compute_fn closure from registered ArrayReward subclasses with coefficient weighting and common_reward broadcasting in the composition layer**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-13T11:19:22Z
- **Completed:** 2026-02-13T11:21:59Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Implemented build_reward_config_from_components that queries registry for scope + global ArrayReward subclasses and composes a single compute_fn
- compute_fn instantiates rewards at composition time, calls each compute(), applies coefficient weighting and common_reward broadcasting, sums to (n_agents,) float32
- 7 new TDD tests covering: required keys, callable compute_fn, empty scope zeros, single reward with coefficient, common_reward broadcasting, multi-reward summation, reward_config passthrough
- All 77 tests pass (19 autowire + 58 existing) with zero regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- Write failing tests** - `93b2571` (test)
2. **Task 2: GREEN + REFACTOR -- Implement build_reward_config_from_components** - `8413fc1` (feat)

## Files Created/Modified
- `cogrid/core/autowire.py` - Added build_reward_config_from_components with composed compute_fn closure, updated module docstring (205 lines total)
- `cogrid/tests/test_autowire.py` - Added 7 reward_config tests with test ArrayReward subclasses (19 tests total)

## Decisions Made
- Coefficient and common_reward handling is in the composition layer (compute_fn), not inside ArrayReward.compute() -- compute() returns raw unweighted rewards (per Pitfall 5 in research)
- Global-scope rewards are merged after scope-specific rewards, maintaining consistency with build_scope_config_from_components global-then-scope pattern
- compute_fn uses cogrid.backend.xp (not raw numpy) for JAX/numpy backend compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Both autowire functions (scope_config and reward_config) are operational and tested
- Phase 11 (Composition & Auto-Wiring) is fully complete
- Ready for Phase 12 (Generic Interaction Signature) or Phase 14 (CoGridEnv integration)
- All 77 tests pass with zero regressions

## Self-Check: PASSED

- [x] cogrid/core/autowire.py exists
- [x] cogrid/tests/test_autowire.py exists
- [x] 11-02-SUMMARY.md exists
- [x] Commit 93b2571 exists
- [x] Commit 8413fc1 exists

---
*Phase: 11-composition-auto-wiring*
*Completed: 2026-02-13*
