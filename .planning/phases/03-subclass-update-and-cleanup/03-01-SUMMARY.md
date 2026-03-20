---
phase: 03-subclass-update-and-cleanup
plan: 01
subsystem: feature-space
tags: [local-view, overcooked, instance-method-api, subclass-migration]

# Dependency graph
requires:
  - phase: 02-base-class-refactoring
    provides: "LocalView base class with new instance-method API (n_extra_channels, extra_channels, _scatter_to_grid)"
provides:
  - "OvercookedLocalView migrated to new instance-method API"
  - "Old classmethods (extra_n_channels, build_extra_channel_fn) no longer used by any subclass"
affects: [03-subclass-update-and-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Instance-method subclass pattern: n_extra_channels class attr + extra_channels(self, state, H, W) returning (H,W,E) array"
    - "Config in __init__: scope-dependent lookups (object_to_idx) cached as instance attributes"

key-files:
  created: []
  modified:
    - "cogrid/envs/overcooked/features.py"

key-decisions:
  - "Used loop with _scatter_to_grid instead of 8 explicit calls -- cleaner, same behavior"
  - "Return xp.stack(layers, axis=-1) for (H,W,8) ndarray -- bridge in build_feature_fn unstacks to list"

patterns-established:
  - "Subclass migration pattern: replace classmethods with class attr + instance method + __init__ config"

requirements-completed: [API-04]

# Metrics
duration: 2min
completed: 2026-03-20
---

# Phase 03 Plan 01: Subclass Migration Summary

**OvercookedLocalView migrated from old classmethod API to new instance-method API with n_extra_channels=8, __init__ config, and extra_channels returning (H,W,8) array**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-20T04:11:12Z
- **Completed:** 2026-03-20T04:13:17Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Replaced `extra_n_channels` classmethod with `n_extra_channels = 8` class attribute
- Replaced `build_extra_channel_fn` classmethod with `extra_channels(self, state, H, W)` instance method
- Added `__init__` to cache scope-dependent config (onion_id, tomato_id, capacity, cook_time) as instance attributes
- Used `_scatter_to_grid` helper instead of manual JAX/NumPy branching
- All 173 tests pass including 4 golden tests with bit-identical output

## Task Commits

Each task was committed atomically:

1. **Task 1: Migrate OvercookedLocalView to new instance-method API** - `2dc312c` (feat)

## Files Created/Modified
- `cogrid/envs/overcooked/features.py` - Migrated OvercookedLocalView from old classmethod API to new instance-method API

## Decisions Made
- Used loop with `_scatter_to_grid` instead of 8 explicit calls -- functionally identical, cleaner code
- Returns `xp.stack(layers, axis=-1)` producing `(H,W,8)` ndarray; the bridge in `build_feature_fn` unstacks to list of `(H,W)` arrays

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- OvercookedLocalView is the only subclass that was using the old API
- Old classmethods (extra_n_channels, build_extra_channel_fn) in LocalView base class are now unused by any subclass
- Ready for Plan 02: removal of deprecated classmethods from LocalView

## Self-Check: PASSED

- FOUND: 03-01-SUMMARY.md
- FOUND: commit 2dc312c
- FOUND: cogrid/envs/overcooked/features.py

---
*Phase: 03-subclass-update-and-cleanup*
*Completed: 2026-03-20*
