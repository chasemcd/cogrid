---
phase: 30-features-rewards-adaptation
plan: 01
subsystem: rewards
tags: [overcooked, rewards, IS_DELIVERABLE, recipe_reward, order_queue, expired_order, tip_bonus]

# Dependency graph
requires:
  - phase: 26-recipe-table-infrastructure
    provides: "recipe_reward array, compile_recipes, IS_DELIVERABLE table"
  - phase: 29-order-queue
    provides: "order_n_expired prev/curr diff, order_recipe/order_timer arrays, delivery consumption"
provides:
  - "DeliveryReward with IS_DELIVERABLE lookup, per-recipe values, and order-aware gating"
  - "ExpiredOrderPenalty reward class using prev/curr diff on order_n_expired"
  - "static_tables plumbed from scope_config through reward_config to reward classes"
  - "Tip bonus mechanism in delivery_reward (default disabled, coefficient=0.0)"
affects: [30-features-rewards-adaptation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "static_tables forwarded through reward_config for reward-layer access to recipe/order tables"
    - "Order-aware reward gating via prev/curr StateView comparison of order_recipe arrays"
    - "IS_DELIVERABLE table-driven deliverability check replacing hardcoded type comparison"

key-files:
  created: []
  modified:
    - cogrid/core/autowire.py
    - cogrid/cogrid_env.py
    - cogrid/envs/overcooked/rewards.py
    - cogrid/envs/overcooked/test_interactions.py

key-decisions:
  - "Backward compat fallback uses coefficient (not coefficient*20) when static_tables absent"
  - "Tip bonus integrated into delivery_reward (not separate class) to avoid duplicating order-consumption detection"
  - "ExpiredOrderPenalty uses common reward (all agents penalized equally)"
  - "Default tip_coefficient=0.0 (disabled) for backward compatibility"

patterns-established:
  - "Reward functions receive reward_config with static_tables for access to recipe/order lookup arrays"
  - "Order-aware rewards detect consumption by comparing prev_state vs state order_recipe arrays"

# Metrics
duration: 5min
completed: 2026-02-17
---

# Phase 30 Plan 01: Order-Aware Rewards Summary

**IS_DELIVERABLE-based delivery reward with per-recipe values, order-gated delivery, expired order penalty, and tip bonus mechanism**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-17T22:55:06Z
- **Completed:** 2026-02-17T23:00:37Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Delivery reward uses IS_DELIVERABLE table lookup instead of hardcoded onion_soup check
- Per-recipe reward values from recipe_reward array (e.g., onion_soup=20, tomato_soup=30)
- Delivery reward gated by order match when orders enabled, fires unconditionally when disabled
- ExpiredOrderPenalty class returns configurable penalty for newly expired orders via prev/curr diff
- static_tables flows end-to-end: scope_config -> cogrid_env.py -> reward_config -> reward classes
- 6 new tests covering all reward behaviors

## Task Commits

Each task was committed atomically:

1. **Task 1: Plumb static_tables and rewrite reward classes** - `304ff32` (feat)
2. **Task 2: Tests for order-aware rewards** - `2a44c3b` (test)

## Files Created/Modified
- `cogrid/core/autowire.py` - build_reward_config_from_components accepts optional static_tables parameter
- `cogrid/cogrid_env.py` - Passes static_tables from scope_config to reward_config at call site (1 line)
- `cogrid/envs/overcooked/rewards.py` - Updated DeliveryReward, new ExpiredOrderPenalty, tip bonus mechanism
- `cogrid/envs/overcooked/test_interactions.py` - 6 new tests for IS_DELIVERABLE, per-recipe values, order gating, expired penalty, backward compat, end-to-end wiring

## Decisions Made
- Backward compat fallback uses `coefficient` per earner (not `coefficient * 20.0`) when static_tables are absent, preserving the original reward magnitude for callers using the direct function API
- Tip bonus is integrated directly into `delivery_reward` rather than a separate `TipBonus` class, avoiding duplicated order-consumption detection logic
- ExpiredOrderPenalty broadcasts to all agents (common penalty), consistent with team-based Overcooked dynamics
- Default `tip_coefficient=0.0` ensures tip is disabled by default for backward compatibility; users enable via reward_config

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed backward compat fallback reward value**
- **Found during:** Task 1 (reward rewrite)
- **Issue:** Plan specified fallback as `coefficient * 20.0` but this breaks existing test expectations (coefficient=1.0 expected reward=1.0, not 20.0)
- **Fix:** Changed fallback to `coefficient` (matching original per-earner behavior)
- **Files modified:** cogrid/envs/overcooked/rewards.py
- **Verification:** All 4 test_reward_parity.py tests pass
- **Committed in:** 304ff32 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential for backward compat. Without this fix, all existing delivery reward tests would fail.

## Issues Encountered
- Pre-existing test isolation issue: `test_factory_registers_new_types` and `test_factory_stack_dispenses_item` (from Phase 28) register `test_mushroom` type in global state, changing `n_types` count and causing `obs_shape` tests to fail when run in same pytest session. Not caused by Phase 30 changes. Tests pass in isolation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Reward layer fully connected to Phase 26 recipe tables and Phase 29 order queue
- Plan 02 (features adaptation) can proceed independently
- All 40 relevant tests pass (4 reward parity + 22 interaction + 14 features, excluding factory tests with pre-existing isolation issue)

## Self-Check: PASSED

All files found: cogrid/core/autowire.py, cogrid/cogrid_env.py, cogrid/envs/overcooked/rewards.py, cogrid/envs/overcooked/test_interactions.py
All commits found: 304ff32, 2a44c3b

---
*Phase: 30-features-rewards-adaptation*
*Completed: 2026-02-17*
