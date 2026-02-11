---
phase: 01-dual-backend-vectorized-core-rewrite
plan: 04
subsystem: core
tags: [numpy, vectorized-interactions, lookup-tables, pot-cooking, interaction-priority]

# Dependency graph
requires:
  - "01-01: cogrid.backend module with xp dispatch, build_lookup_tables, object_to_idx"
  - "01-02: layout_to_array_state, create_agent_arrays, get_dir_vec_table"
provides:
  - "process_interactions_array() for all 4 interaction branches using integer lookup tables"
  - "tick_objects_array() for vectorized pot cooking timer"
  - "build_interaction_tables() for auxiliary tables (PICKUP_FROM_PRODUCES, LEGAL_POT_INGREDIENTS, type_ids)"
affects: [01-07]

# Tech tracking
tech-stack:
  added: []
  patterns: [dynamic-condition-evaluation, priority-branch-fallthrough, pot-cooking-state-machine]

key-files:
  created:
    - cogrid/core/interactions.py
  modified: []

key-decisions:
  - "Dynamic can_pickup_from evaluation prevents false-positive elif matching: static CAN_PICKUP_FROM flag checked first, then instance-level conditions (pot ready + plate) evaluated inline to allow correct fallthrough to place_on branch"
  - "OvercookedAgent.can_pickup() special override for pots replicated via separate pot sub-case in branch 2: agent with plate can pickup from pot even at capacity"
  - "Counter place_on tracks placed item in object_state_map[r,c] as type_id integer (0 = empty), matching existing obj_placed_on pattern"
  - "Delivery zone consumes soup silently (no state update) matching existing DeliveryZone.place_on behavior"

patterns-established:
  - "Interaction processing: process_interactions_array() as standalone pure function taking and returning array state"
  - "Pot state encoding: n_items + n_items * timer (matching existing Pot.tick() exactly)"
  - "Auxiliary table pattern: build_interaction_tables(scope) at init, pass to process function"
  - "PHASE2 mutation markers: all in-place array mutations annotated with # PHASE2: convert to .at[].set()"

# Metrics
duration: 9min
completed: 2026-02-11
---

# Phase 01 Plan 04: Vectorized Interaction Processing Summary

**process_interactions_array() with 4-branch priority dispatch using integer lookup tables, tick_objects_array() for vectorized pot cooking, and dynamic can_pickup_from evaluation matching OvercookedAgent special override for pot-plate-soup swap**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-11T15:35:18Z
- **Completed:** 2026-02-11T15:44:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Created process_interactions_array() handling all 4 interaction priority branches (pickup, pickup_from, drop, place_on) using integer lookup tables instead of isinstance() dispatch
- Implemented tick_objects_array() with fully vectorized pot cooking timer matching existing Pot.tick() state encoding exactly
- Built dynamic can_pickup_from evaluation that correctly handles the priority fallthrough issue where pots have both CAN_PICKUP_FROM and CAN_PLACE_ON flags
- Added comprehensive parity test suite validating array vs object behavior across all interaction types and edge cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement interaction processing with lookup tables** - `fa08885` (feat)
2. **Task 2: Add interaction parity test** - `bf488e8` (test)

## Files Created/Modified
- `cogrid/core/interactions.py` - New file containing process_interactions_array(), tick_objects_array(), build_interaction_tables(), and test_interaction_parity()

## Decisions Made
- **Dynamic can_pickup_from evaluation:** The static CAN_PICKUP_FROM lookup table marks pots as always having pickup-from capability, but the original Pot.can_pickup_from() is dynamic (requires dish_ready AND agent has plate). Using the static flag alone in the elif chain would cause the pot branch to consume the elif even when conditions aren't met, preventing fallthrough to the place_on branch. Solution: pre-compute the dynamic condition before entering the if/elif chain, so branch 2 only activates when the actual conditions are satisfied.
- **OvercookedAgent.can_pickup() special case:** The original code has a special override where OvercookedAgent.can_pickup(pot) returns True if agent has a plate, bypassing the capacity check. This is essential for the plate-to-soup swap. Replicated by splitting branch 2 into pot (plate required) and stack (empty inventory required) sub-cases.
- **Counter placed-on tracking via object_state_map:** Uses object_state_map[r,c] to store the type_id of a placed item (0 = nothing placed), avoiding the need for a separate placed_on_map array. This mirrors the existing obj_placed_on field on GridObj.
- **Delivery zone soup consumption:** DeliveryZone.place_on() in the original code simply deletes the cell parameter. The array version clears agent inventory without updating any grid state, matching this behavior.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed priority branch fallthrough for pot CAN_PICKUP_FROM**
- **Found during:** Task 1 (implementing interaction processing)
- **Issue:** Plan's elif structure used `CAN_PICKUP_FROM[fwd_type] == 1` as the branch 2 condition, which is a static property True for ALL pots. When an agent holding an onion faces a non-ready pot, branch 2 would fire (pot has CAN_PICKUP_FROM=True) but the sub-conditions (plate + ready) wouldn't match, and the elif chain would prevent branch 4 (place_on) from running. This means agents could never place ingredients into pots.
- **Fix:** Pre-compute the dynamic can_pickup_from condition (including pot readiness and plate check) before the if/elif chain. Branch 2 only enters when the actual instance-level conditions are satisfied, allowing correct fallthrough to branch 4 when conditions are not met.
- **Files modified:** cogrid/core/interactions.py
- **Verification:** Test 3 (place onion in pot) and 16 comprehensive tests all pass.
- **Committed in:** fa08885 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Critical correctness fix -- without it, placing ingredients in pots would be impossible. The static vs dynamic property distinction was a subtle but fundamental issue in translating object-oriented dispatch to lookup table dispatch.

## Issues Encountered
- Pre-existing test failures persist (FIXED_GRIDS import in test_gridworld_env.py, pygame requirement in test_overcooked_env.py) -- neither related to our changes. The one passing test in cogrid/tests/ continues to pass.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Interaction processing functions ready for integration in Plan 07
- build_interaction_tables() provides all auxiliary data needed by process_interactions_array()
- Pot cooking state machine fully tested and matching existing semantics
- All functions are standalone (take array state, return array state) -- ready for wiring into CoGridEnv.step()

## Self-Check: PASSED

- All files exist (cogrid/core/interactions.py, 01-04-SUMMARY.md)
- All commits found (fa08885, bf488e8)
- All functions importable (process_interactions_array, tick_objects_array, build_interaction_tables, test_interaction_parity)

---
*Phase: 01-dual-backend-vectorized-core-rewrite*
*Completed: 2026-02-11*
