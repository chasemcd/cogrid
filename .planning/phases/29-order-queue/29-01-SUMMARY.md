---
phase: 29-order-queue
plan: 01
subsystem: environment
tags: [overcooked, order-queue, tick-handler, extra-state, delivery]

# Dependency graph
requires:
  - phase: 27-generalize-interaction-branches
    provides: "Recipe table lookups, IS_DELIVERABLE, accumulated-handled branches"
  - phase: 25-interaction-cascade-refactor
    provides: "Sparse updates dict, merge loop, branch function pattern"
provides:
  - "Fixed-size order queue arrays in extra_state (order_recipe, order_timer, order_spawn_counter, order_recipe_counter, order_n_expired)"
  - "_build_order_tables for deterministic weighted round-robin spawn cycle"
  - "Order tick logic (decrement/expire/spawn) in overcooked_tick_state"
  - "Order consumption in _branch_place_on_delivery"
  - "order_config parameter in build_overcooked_extra_state"
affects: [30-reward-shaping]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Order queue as fixed-shape sentinel arrays (-1 = empty) in extra_state"
    - "Deterministic round-robin spawn via pre-built cycle array in static_tables"
    - "Order tick: decrement -> expire -> spawn sequence to free slots before fill"

key-files:
  created: []
  modified:
    - cogrid/envs/overcooked/config.py
    - cogrid/envs/overcooked/overcooked_grid_objects.py
    - cogrid/envs/overcooked/test_interactions.py

key-decisions:
  - "Deterministic spawn (weighted round-robin) avoids RNG-under-vmap concern from STATE.md"
  - "order_n_expired scalar in extra_state enables Phase 30 reward to detect expirations via prev/curr diff"
  - "Delivery without matching order still succeeds (soup consumed, no order consumed) for backward compat"
  - "Order arrays passed through kwargs + ctx dict, not positional args, to minimize signature disruption"

patterns-established:
  - "Order queue pattern: fixed-shape (max_active,) arrays with -1 sentinel, managed by tick handler"
  - "Config-absent = no order keys in extra_state (not shape-0 arrays)"

# Metrics
duration: 5min
completed: 2026-02-17
---

# Phase 29 Plan 01: Order Queue Summary

**Fixed-size timed order queue in extra_state with deterministic spawn, countdown/expiry, and delivery consumption -- fully backward-compatible when orders config absent**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-17T22:11:08Z
- **Completed:** 2026-02-17T22:16:03Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Order queue arrays (order_recipe, order_timer, order_spawn_counter, order_recipe_counter, order_n_expired) in extra_state with config-driven lifecycle
- Tick handler extends overcooked_tick_state with decrement/expire/spawn sequence using deterministic weighted round-robin
- Delivery branch consumes first matching active order on soup delivery
- Full backward compatibility: no order keys when config absent, all 8+ existing tests pass unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Order config, extra_state arrays, tick handler, and interaction wiring** - `bdf7d54` (feat)
2. **Task 2: Order queue lifecycle tests** - `f781e9c` (test)

## Files Created/Modified
- `cogrid/envs/overcooked/config.py` - Added _build_order_tables, extended build_overcooked_extra_state with order_config, added order tick logic to overcooked_tick_state, added order consumption to _branch_place_on_delivery, extended overcooked_interaction_body/fn for order array pass-through
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Extended Pot.extra_state_schema with order arrays, updated Pot.build_static_tables to call _build_order_tables
- `cogrid/envs/overcooked/test_interactions.py` - Added 5 tests: tick lifecycle, delivery consumption, delivery without match, backward compat, config validation

## Decisions Made
- Deterministic spawn (weighted round-robin cycle) instead of RNG, resolving the STATE.md blocker about RNG under vmap
- order_n_expired scalar tracked for Phase 30 reward consumption (prev/curr diff pattern)
- Delivery without matching active order still succeeds (backward compat; reward system handles distinction)
- Order arrays as keyword args to overcooked_interaction_body, passed through ctx dict to branches

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Order queue fully functional: spawn, countdown, expiry, and delivery consumption
- order_n_expired available for Phase 30 reward shaping (expired order penalty)
- Recipe counter and spawn cycle in static_tables ready for multi-recipe environments

---
*Phase: 29-order-queue*
*Completed: 2026-02-17*
