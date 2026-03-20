---
phase: 03-subclass-update-and-cleanup
plan: 02
subsystem: feature-space
tags: [local-view, validation, api-cleanup, ndarray]

# Dependency graph
requires:
  - phase: 03-subclass-update-and-cleanup/plan-01
    provides: "OvercookedLocalView migrated to new instance-method API"
  - phase: 02-base-class-refactoring
    provides: "LocalView base class with __init_subclass__, instance constructor, extra_channels method"
provides:
  - "Clean LocalView with single API path (no deprecated classmethods)"
  - "Return-type validation with clear TypeError/ValueError messages"
  - "Module docstring with usage example for subclass authors"
  - "Tests for wrong return type, wrong ndim, wrong shape"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ndarray return-type validation (ndim check, shape check) in build_feature_fn closure"

key-files:
  created: []
  modified:
    - cogrid/feature_space/local_view.py
    - cogrid/tests/test_base_class_refactor.py

key-decisions:
  - "Three-tier validation: hasattr(ndim) check, ndim==3 check, exact shape check -- catches list/tuple, 2D arrays, and channel mismatches with specific error messages"
  - "Updated existing channel mismatch test (Test 5) to match new ValueError format since shape validation now replaces count-only validation"

patterns-established:
  - "Return-type validation pattern: check hasattr(ndim) then ndim then exact shape for clear progressive error messages"

requirements-completed: [API-05, SAFE-03, COMPAT-01]

# Metrics
duration: 3min
completed: 2026-03-20
---

# Phase 03 Plan 02: Deprecated API Removal and Return-Type Validation Summary

**Removed all deprecated classmethods from LocalView, added three-tier return-type validation with clear error messages, and updated docstring with subclass usage example**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-20T04:16:10Z
- **Completed:** 2026-03-20T04:19:43Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Removed extra_n_channels and build_extra_channel_fn classmethods entirely from LocalView
- Removed __init_subclass__ backward-compat guard and compute_obs_dim old-API fallback
- Simplified build_feature_fn from three branches (new/old/base) to two (new/base)
- Added three-tier return-type validation: TypeError for non-ndarray, TypeError for wrong ndim, ValueError for wrong shape
- Updated module and class docstrings to show only the new API with a usage example
- Added 3 new tests covering wrong return type, wrong ndim, and wrong shape error messages

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove deprecated API and add return-type validation in LocalView** - `ca50e5e` (feat)
2. **Task 2: Add tests for return-type validation error messages** - `f4855a5` (test)

## Files Created/Modified
- `cogrid/feature_space/local_view.py` - Cleaned-up LocalView with only new API, return-type validation, updated docstrings
- `cogrid/tests/test_base_class_refactor.py` - Updated Test 5 error message, added Tests 8-10 for validation errors

## Decisions Made
- Three-tier validation (hasattr(ndim), ndim==3, exact shape) provides clear progressive error messages for each common subclassing mistake
- Updated existing Test 5 to match new error message format (shape-based instead of count-based)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated Test 5 error message pattern**
- **Found during:** Task 1 (return-type validation implementation)
- **Issue:** Existing test_channel_count_mismatch_error expected old error format "n_extra_channels=3 but extra_channels() returned 5 channels" but new validation produces "extra_channels() returned shape (3, 3, 5), expected (3, 3, 3)"
- **Fix:** Updated pytest.raises match pattern in Test 5 to match new ValueError format
- **Files modified:** cogrid/tests/test_base_class_refactor.py
- **Verification:** All 10 tests pass
- **Committed in:** ca50e5e (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Necessary update since validation error message format changed. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 03 (subclass-update-and-cleanup) is now complete
- LocalView has a single, clean API with validated return types
- All 176 tests pass including 4 golden tests with exact equality
- No deprecated methods remain anywhere in the codebase

## Self-Check: PASSED

All files found, all commits verified.

---
*Phase: 03-subclass-update-and-cleanup*
*Completed: 2026-03-20*
