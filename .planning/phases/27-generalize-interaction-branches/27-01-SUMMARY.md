---
phase: 27-generalize-interaction-branches
plan: 01
subsystem: config
tags: [numpy, recipes, branchless, overcooked, interaction-branches, prefix-match, IS_DELIVERABLE]

# Dependency graph
requires:
  - phase: 26-recipe-table-infrastructure
    provides: compile_recipes producing recipe_ingredients, recipe_result, recipe_cooking_time arrays in static_tables
  - phase: 25-interaction-cascade-refactor
    provides: accumulated-handled branch pattern with _BRANCHES list, ctx dict, and (handled, ctx) -> (cond, updates, handled) interface
provides:
  - Recipe-driven _branch_pickup_from_pot using sort-and-compare matching against recipe_ingredients
  - Recipe prefix-match in _branch_place_on_pot replacing same_type check, with per-recipe cook_time set when pot fills
  - IS_DELIVERABLE array lookup in _branch_place_on_delivery replacing hardcoded soup ID checks
  - IS_DELIVERABLE and max_ingredients added to _build_static_tables and ctx dict
  - overcooked_tick_state passing max_ingredients as capacity
  - test_mixed_recipe_end_to_end and test_per_recipe_cook_time test functions
affects: [28-stacks, 29-orders, 30-features-rewards]

# Tech tracking
tech-stack:
  added: []
  patterns: [sentinel-aware-sort, prefix-match-recipe-validation, IS_DELIVERABLE-lookup, per-recipe-cook-time]

key-files:
  created: []
  modified:
    - cogrid/envs/overcooked/config.py
    - cogrid/envs/overcooked/test_interactions.py

key-decisions:
  - "Sentinel-aware sort (replace -1 with INT32_MAX before xp.sort) to keep ingredient IDs at front of sorted arrays for recipe matching"
  - "IS_DELIVERABLE built from recipe_result at init time; backward-compat fallback hardcodes onion_soup and tomato_soup when recipe_tables is None"
  - "Per-recipe cook_time set in _branch_place_on_pot when pot fills, not in tick; pot_timer already carries the value"
  - "pot_contents shape stays (n_pots, 3); variable capacity supported in logic via max_ingredients but shape change deferred"

patterns-established:
  - "Sentinel-aware sort: replace -1 with INT32_MAX before xp.sort, restore after, to keep real values at front"
  - "Prefix-match for ingredient validation: slot_mask = arange < n_filled, match only filled positions"
  - "IS_DELIVERABLE array indexed by type ID for O(1) deliverability check"

# Metrics
duration: 5min
completed: 2026-02-17
---

# Phase 27 Plan 01: Generalize Interaction Branches Summary

**Recipe-driven interaction branches using sort-and-compare matching, prefix validation, IS_DELIVERABLE lookup, and per-recipe cook times -- replacing all hardcoded soup-type logic in _branch_place_on_pot, _branch_pickup_from_pot, and _branch_place_on_delivery**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-17T20:34:09Z
- **Completed:** 2026-02-17T20:39:46Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Rewired _branch_pickup_from_pot to use recipe table lookup (xp.sort + xp.all against recipe_ingredients, recipe_result[matched_idx]) replacing all_tomato/onion_soup_id/tomato_soup_id logic
- Rewired _branch_place_on_pot to use prefix-match against recipe table replacing same_type check, with per-recipe cook_time set in pot_timer when pot fills
- Rewired _branch_place_on_delivery to use IS_DELIVERABLE[inv_item] lookup replacing hardcoded soup ID checks
- Added IS_DELIVERABLE array and max_ingredients to _build_static_tables with backward-compat fallback
- Added recipe table entries to ctx dict in overcooked_interaction_body
- Passed max_ingredients as capacity in overcooked_tick_state
- Added test_mixed_recipe_end_to_end (2 onion + 1 tomato -> onion_soup, cook_time=20, full place-cook-pickup-deliver cycle)
- Added test_per_recipe_cook_time (onion=10 ticks, tomato=50 ticks, each produces correct soup)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewire init-time tables, branch functions, and tick to use recipe lookups** - `d07dc0c` (feat)
2. **Task 2: Add mixed-recipe and per-recipe cook time end-to-end tests** - `28a9070` (test)

## Files Created/Modified
- `cogrid/envs/overcooked/config.py` - Rewired _branch_pickup_from_pot, _branch_place_on_pot, _branch_place_on_delivery to use recipe tables; added IS_DELIVERABLE and max_ingredients to _build_static_tables; added recipe entries to ctx dict; updated overcooked_tick_state capacity
- `cogrid/envs/overcooked/test_interactions.py` - Added test_mixed_recipe_end_to_end and test_per_recipe_cook_time; updated invariant test ctx dict with recipe table entries

## Decisions Made
- Used sentinel-aware sort (replace -1 with INT32_MAX before xp.sort, restore after) because xp.sort places -1 sentinels at the beginning of the array, but recipe_ingredients has real values first and sentinels last. This ensures positional comparison works correctly.
- IS_DELIVERABLE built at init time from recipe_result when recipe_tables provided; falls back to hardcoded onion_soup/tomato_soup when recipe_tables is None, preserving backward compatibility for callers that don't supply recipe_tables.
- Per-recipe cook_time written into pot_timer by _branch_place_on_pot when it detects the pot just became full. No changes needed to overcooked_tick since it already decrements whatever timer value is present.
- pot_contents shape remains (n_pots, 3) -- the prefix-match and max_ingredients logic supports variable-length recipes in principle, but recipes with != 3 ingredients would need shape changes deferred to Phase 28+.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Sentinel-aware sort for recipe matching**
- **Found during:** Task 1 (branch function rewiring)
- **Issue:** xp.sort places -1 sentinels at the beginning of the sorted array (smallest values), but recipe_ingredients stores real ingredient IDs first and -1 sentinels at the end. Direct sort caused position-by-position comparison to fail: sorted_pot[0] = -1 never matched recipe_ingredients[i, 0] = onion_id.
- **Fix:** Before sorting, replace -1 with INT32_MAX so sentinels sort to the end, then restore -1 after sorting. Applied to both _branch_pickup_from_pot (exact match) and _branch_place_on_pot (prefix match).
- **Files modified:** cogrid/envs/overcooked/config.py
- **Verification:** All 8 parity tests pass, mixed-recipe test passes.
- **Committed in:** d07dc0c (Task 1 commit)

**2. [Rule 3 - Blocking] Added recipe table entries to invariant test ctx dict**
- **Found during:** Task 1 (branch function rewiring)
- **Issue:** test_at_most_one_branch_fires builds its own ctx dict manually and did not include the new recipe_ingredients, recipe_result, recipe_cooking_time, max_ingredients, IS_DELIVERABLE entries. Branch functions now access these from ctx, causing KeyError.
- **Fix:** Added the 5 recipe table entries to the ctx dict in test_at_most_one_branch_fires, sourced from static_tables.
- **Files modified:** cogrid/envs/overcooked/test_interactions.py
- **Verification:** Invariant test passes with 600 random states.
- **Committed in:** d07dc0c (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. Sentinel-aware sort is the key algorithmic insight missed by the plan's examples. No scope creep.

## Issues Encountered
None beyond the deviations documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All interaction branches are now fully recipe-driven from compile_recipes output
- Adding a new recipe requires only a config dict change -- no code changes to branch functions
- 13 total tests pass (8 parity + 1 invariant + 3 recipe infra + 2 new end-to-end)
- IS_DELIVERABLE, recipe_ingredients, recipe_result, recipe_cooking_time, max_ingredients all available in static_tables and ctx dict
- pot_contents shape is (n_pots, 3); recipes with != 3 ingredients require shape changes (Phase 28+ concern)
- Rewards and features still reference hardcoded soup IDs (Phase 30 concern)

## Self-Check: PASSED

All files found, all commits verified.

---
*Phase: 27-generalize-interaction-branches*
*Completed: 2026-02-17*
