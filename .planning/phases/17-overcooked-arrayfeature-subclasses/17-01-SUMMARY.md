---
phase: 17-overcooked-arrayfeature-subclasses
plan: 01
subsystem: overcooked
tags: [array-features, subclasses, parity-tests, tdd, registration, overcooked]

# Dependency graph
requires:
  - phase: 15-01
    provides: "ArrayFeature base class, register_feature_type decorator, FeatureMetadata"
  - phase: 15-02
    provides: "compose_feature_fns, obs_dim_for_features"
  - phase: 16-01
    provides: "ArrayFeature subclass pattern (delegate to bare function), global scope registration"
provides:
  - "OvercookedInventory ArrayFeature subclass (per_agent=True, obs_dim=5)"
  - "NextToCounter ArrayFeature subclass (per_agent=True, obs_dim=4)"
  - "NextToPot ArrayFeature subclass (per_agent=True, obs_dim=16)"
  - "7 ClosestObj ArrayFeature variants via factory (onion, plate, plate_stack, onion_stack, onion_soup, delivery_zone, counter)"
  - "OrderedPotFeatures ArrayFeature subclass (per_agent=True, obs_dim=24)"
  - "DistToOtherPlayers ArrayFeature subclass (per_agent=True, obs_dim=2)"
  - "Parity tests verifying numerical identity with bare functions"
  - "Overcooked scope registration of all 12 per-agent features"
affects: [18-autowire-integration, 19-migration]

# Tech tracking
tech-stack:
  added: []
  patterns: ["ClosestObj factory function for parameterized ArrayFeature registration (7 variants from single factory)"]

key-files:
  created: []
  modified:
    - cogrid/envs/overcooked/overcooked_array_features.py
    - cogrid/envs/overcooked/__init__.py
    - cogrid/tests/test_array_features.py

key-decisions:
  - "ClosestObj uses factory function to create 7 separate registered subclasses with dynamic obs_dim"
  - "DistToOtherPlayers hardcodes n_agents=2 (Overcooked assumption, parameterizable later)"
  - "Overcooked __init__.py imports overcooked_array_features to trigger registration at import time"

patterns-established:
  - "Factory pattern for parameterized ArrayFeature registration: define class, set attributes, then apply register_feature_type decorator"
  - "Overcooked per-agent features delegate to bare functions with pre-computed type IDs from build_feature_fn"

# Metrics
duration: 2min
completed: 2026-02-14
---

# Phase 17 Plan 01: Overcooked Per-Agent ArrayFeature Subclasses Summary

**6 Overcooked per-agent ArrayFeature subclasses (12 registrations including 7 ClosestObj variants) registered to overcooked scope with parity-verified delegation to bare functions**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-14T14:37:49Z
- **Completed:** 2026-02-14T14:40:34Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- OvercookedInventory, NextToCounter, NextToPot, 7 ClosestObj variants, OrderedPotFeatures, DistToOtherPlayers ArrayFeature subclasses with overcooked scope registration
- All 12 produce numerically identical outputs to their bare function counterparts (verified by 7 parity tests)
- All discoverable via get_feature_types(scope="overcooked") with correct per_agent=True and obs_dim metadata
- ClosestObj factory creates 7 variants from a single parameterized factory function
- 25 total tests pass (18 existing + 7 new parity/registration tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: RED -- failing parity tests for 6 Overcooked per-agent ArrayFeature subclasses** - `ae6dac0` (test)
2. **Task 2: GREEN -- implement 6 Overcooked per-agent ArrayFeature subclasses** - `72a4805` (feat)

## Files Created/Modified
- `cogrid/envs/overcooked/overcooked_array_features.py` - Added 6 ArrayFeature subclasses (OvercookedInventory, NextToCounter, NextToPot, ClosestObj x7, OrderedPotFeatures, DistToOtherPlayers) with @register_feature_type decorators
- `cogrid/envs/overcooked/__init__.py` - Added import of overcooked_array_features to trigger registration
- `cogrid/tests/test_array_features.py` - Added 7 parity/registration tests for all Overcooked per-agent features

## Decisions Made
- ClosestObj uses a factory function (`_make_closest_obj_feature`) to create 7 separate registered subclasses with dynamic `obs_dim` values. The class is defined first with all attributes, then the `register_feature_type` decorator is applied programmatically (avoids the issue of decorator running before dynamic attribute assignment).
- DistToOtherPlayers hardcodes `n_agents=2` (standard Overcooked assumption). If multi-agent support is needed, Phase 18 can parameterize this.
- Added `overcooked_array_features` import to `cogrid/envs/overcooked/__init__.py` to ensure registration triggers at import time (same pattern as `overcooked_grid_objects` and `array_rewards`).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- All 12 Overcooked per-agent ArrayFeature subclasses registered and validated
- Phase 17 Plan 02 (global features: LayoutID, EnvironmentLayout) can follow the same pattern
- Phase 18 (autowire integration) can use compose_feature_fns with these registered features
- No blockers

## Self-Check: PASSED

- FOUND: cogrid/envs/overcooked/overcooked_array_features.py
- FOUND: cogrid/envs/overcooked/__init__.py
- FOUND: cogrid/tests/test_array_features.py
- FOUND: 17-01-SUMMARY.md
- FOUND: commit ae6dac0 (Task 1)
- FOUND: commit 72a4805 (Task 2)
- 25/25 tests pass

---
*Phase: 17-overcooked-arrayfeature-subclasses*
*Completed: 2026-02-14*
