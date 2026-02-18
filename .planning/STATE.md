# CoGrid -- State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-17)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core -- readable, simple, and fast.
**Current focus:** Phase 30 - Features & Rewards Adaptation

## Current Position

Phase: 30 of 30 (Features & Rewards Adaptation)
Plan: 3 of 3 in current phase (COMPLETE)
Status: Phase 30 plan 03 complete (gap closure)
Last activity: 2026-02-17 -- Completed 30-03 gap closure

Progress: [====================] 100% (Phase 30: 3/3 plans)

## Performance Metrics

**Velocity:**
- Total plans completed: 8
- Average duration: 4min
- Total execution time: 0.52 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 25-interaction-cascade-refactor | 1 | 5min | 5min |
| 26-recipe-table-infrastructure | 1 | 3min | 3min |
| 27-generalize-interaction-branches | 1 | 5min | 5min |
| 28-generic-stack-table-building | 1 | 3min | 3min |
| 29-order-queue | 1 | 5min | 5min |
| 30-features-rewards-adaptation | 3 | 13min | 4min |

**Recent Trend:**
- Last 5 plans: 3min, 5min, 5min, 5min, 3min
- Trend: stable

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 6 phases derived from 28 requirements; cascade refactor first per pitfall analysis
- [Roadmap]: Phase 28 (Stacks) depends only on Phase 26, can parallel Phase 27 if needed
- [25-01]: Used sparse updates dict over full-struct returns -- branches return only arrays they modify
- [25-01]: Treated place-on sub-branches (4A, 4B, 4C) as independent top-level branches with ~handled guards
- [25-01]: Module-level _BRANCHES list (not registry) -- matches codebase philosophy, trivially extensible
- [26-01]: DEFAULT_RECIPES uses reward=20.0 matching Overcooked convention; not consumed until Phase 30
- [26-01]: compile_recipes derives legal_pot_ingredients from recipe ingredients rather than hardcoding
- [26-01]: Pot.build_static_tables always compiles DEFAULT_RECIPES; custom config wiring deferred to Phase 27
- [27-01]: Sentinel-aware sort (replace -1 with INT32_MAX before xp.sort) for correct recipe matching
- [27-01]: IS_DELIVERABLE built from recipe_result; backward-compat fallback for missing recipe_tables
- [27-01]: Per-recipe cook_time set in _branch_place_on_pot when pot fills; no tick changes needed
- [27-01]: pot_contents shape stays (n_pots, 3); variable capacity in logic only; shape change deferred
- [28-01]: _BaseStack.pick_up_from uses make_object(self.produces) for runtime-registered type dispatch
- [28-01]: Component scan iterates both [scope, "global"] to catch cross-scope stacks
- [28-01]: Stack type_ids kept in _build_interaction_tables via inline object_to_idx for backward compat
- [29-01]: Deterministic spawn (weighted round-robin) avoids RNG-under-vmap concern
- [29-01]: order_n_expired scalar in extra_state enables Phase 30 reward via prev/curr diff
- [29-01]: Delivery without matching order still succeeds (backward compat; reward handles distinction)
- [29-01]: Order arrays as kwargs + ctx dict, not positional args, to minimize signature disruption
- [30-01]: Backward compat fallback uses coefficient (not coefficient*20) when static_tables absent
- [30-01]: Tip bonus integrated into delivery_reward (not separate class) to avoid duplicating order-consumption detection
- [30-01]: ExpiredOrderPenalty broadcasts to all agents (common penalty)
- [30-01]: Default tip_coefficient=0.0 (disabled) for backward compatibility
- [30-02]: OrderObservation uses fixed defaults (max_active=3, n_recipes=2, time_limit=200) matching _build_order_tables
- [30-02]: OvercookedInventory scans only scope registry (not global) to avoid cross-scope type pollution
- [30-02]: Inventory tests written robust against test-time registry pollution from factory tests
- [30-03]: _INVENTORY_OBS_DIM computed once at import time via _count_pickupable_types() helper
- [30-03]: Import order in __init__.py: overcooked_grid_objects first so registry is populated before features.py runs

### Pending Todos

None yet.

### Blockers/Concerns

- (RESOLVED) Interaction cascade refactored to accumulated-handled pattern -- Phase 25 complete, Phases 27/29 unblocked
- (RESOLVED) RNG under vmap for order spawning -- Phase 29 uses deterministic weighted round-robin, no RNG needed

## Session Continuity

Last session: 2026-02-17
Stopped at: Completed 30-03-PLAN.md (Phase 30 plan 03 complete -- all gaps closed)
Resume file: None
