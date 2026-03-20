---
phase: 02-base-class-refactoring
plan: 02
subsystem: api
tags: [python, class-refactoring, instance-methods, init-subclass, jax, numpy]

# Dependency graph
requires:
  - phase: 01-safety-baseline
    provides: Golden output tests for regression detection
  - phase: 02-base-class-refactoring (plan 01)
    provides: Test scaffold with xfail markers, re-exports in __init__.py
provides:
  - Refactored LocalView base class with instance-method API
  - __init_subclass__ validation for n_extra_channels
  - _scatter_to_grid static helper for JAX/NumPy transparency
  - Dual API path in build_feature_fn (new instance + old classmethod)
  - All new API tests passing without xfail markers
affects: [03-subclass-migration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Instance-as-closure-state bridge: build_feature_fn creates instance, captures in closure"
    - "__init_subclass__ for class attribute validation at definition time"
    - "Dual API detection via identity check (cls.method is not BaseClass.method)"

key-files:
  created: []
  modified:
    - cogrid/feature_space/local_view.py
    - cogrid/tests/test_base_class_refactor.py

key-decisions:
  - "Backward compat: __init_subclass__ skips validation when old API methods are overridden in cls.__dict__"
  - "API detection: uses cls.extra_channels is not LocalView.extra_channels for new API, cls.build_extra_channel_fn is not LocalView.build_extra_channel_fn for old API"
  - "Base LocalView does not define n_extra_channels; uses getattr(cls, 'n_extra_channels', 0) fallback"

patterns-established:
  - "Instance-as-closure-state: classmethod creates instance, wraps instance method in closure for generic_local_view_feature"
  - "Dual API path: check method identity to route between new instance API and old classmethod API"
  - "_scatter_to_grid: staticmethod hiding hasattr(ch, 'at') JAX/NumPy branching"

requirements-completed: [API-01, API-02, HELP-01, SAFE-02, SAFE-04]

# Metrics
duration: 9min
completed: 2026-03-20
---

# Phase 02 Plan 02: Base Class Refactoring Summary

**LocalView refactored with instance-method API (extra_channels + n_extra_channels), _scatter_to_grid helper, __init_subclass__ validation, and dual API path preserving OvercookedLocalView backward compatibility**

## Performance

- **Duration:** 9 min
- **Started:** 2026-03-20T03:30:57Z
- **Completed:** 2026-03-20T03:40:39Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Refactored LocalView with new instance-method API: `__init__`, `extra_channels(self, state, H, W)`, `_scatter_to_grid`, `__init_subclass__` validation
- Dual API path in `build_feature_fn` and `compute_obs_dim`: new subclasses use instance methods, OvercookedLocalView continues using old classmethods
- All 173 tests pass including 4 golden tests (bit-identical output), 7 new API tests, and 35 feature tests
- JAX JIT compatibility verified with instance-in-closure pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor LocalView base class with new instance-method API** - `76df0e9` (feat)
2. **Task 2: Remove xfail markers and verify full test suite** - `54d62d3` (test)

## Files Created/Modified
- `cogrid/feature_space/local_view.py` - Refactored LocalView with new API: __init_subclass__, __init__, extra_channels, _scatter_to_grid, dual API build_feature_fn
- `cogrid/tests/test_base_class_refactor.py` - Removed all xfail markers, fixed JIT test with pytree registration

## Decisions Made
- **__init_subclass__ backward compat:** Skip n_extra_channels validation when subclass has `extra_n_channels` or `build_extra_channel_fn` in its `__dict__` (old API). This allows OvercookedLocalView to continue working without modification until Phase 3 migrates it.
- **API detection strategy:** Use Python identity checks (`cls.extra_channels is not LocalView.extra_channels`) to determine which API path a subclass uses. Cleaner than hasattr checks and works correctly with inheritance.
- **Base class n_extra_channels:** Base LocalView intentionally does NOT define `n_extra_channels`. Uses `getattr(cls, 'n_extra_channels', 0)` fallback in `compute_obs_dim` and `build_feature_fn`. This follows the CONTEXT.md decision that only subclasses must declare it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] __init_subclass__ backward compatibility for OvercookedLocalView**
- **Found during:** Task 1 (LocalView refactoring)
- **Issue:** The plan's __init_subclass__ would reject OvercookedLocalView at class definition time since it doesn't have n_extra_channels (it uses the old extra_n_channels classmethod API)
- **Fix:** Added old-API detection in __init_subclass__: skip validation when cls.__dict__ contains extra_n_channels or build_extra_channel_fn
- **Files modified:** cogrid/feature_space/local_view.py
- **Verification:** OvercookedLocalView imports and golden tests pass
- **Committed in:** 76df0e9 (Task 1 commit)

**2. [Rule 1 - Bug] JAX JIT test missing pytree registration**
- **Found during:** Task 2 (xfail removal)
- **Issue:** test_jax_jit_tracing failed because StateView was not registered as a JAX pytree node before JIT tracing
- **Fix:** Added `register_stateview_pytree()` call to the test, matching the pattern used in other JAX tests (test_reward_parity.py)
- **Files modified:** cogrid/tests/test_base_class_refactor.py
- **Verification:** test_jax_jit_tracing passes
- **Committed in:** 54d62d3 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both fixes necessary for correctness. The __init_subclass__ fix ensures backward compatibility with OvercookedLocalView. The pytree fix follows established project patterns. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- LocalView base class refactoring complete with dual API path
- Ready for Phase 3: migrate OvercookedLocalView to new API (extra_channels instance method + n_extra_channels attribute)
- Old API classmethods (extra_n_channels, build_extra_channel_fn) can be removed after Phase 3 migration

## Self-Check: PASSED

- FOUND: cogrid/feature_space/local_view.py
- FOUND: cogrid/tests/test_base_class_refactor.py
- FOUND: .planning/phases/02-base-class-refactoring/02-02-SUMMARY.md
- FOUND: commit 76df0e9 (Task 1)
- FOUND: commit 54d62d3 (Task 2)

---
*Phase: 02-base-class-refactoring*
*Completed: 2026-03-20*
