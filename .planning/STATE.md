# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** Phase 11 -- Composition & Auto-Wiring

## Current Position

Phase: 11 of 14 (Composition & Auto-Wiring) -- COMPLETE
Plan: 2 of 2 complete
Status: Phase complete
Last activity: 2026-02-13 -- Completed 11-02 (Reward Config Auto-Wiring)

Progress: [#############___________________________] 33% (v1.2)

## Performance Metrics

**Velocity:**
- Total plans completed: 34 (18 v1.0 + 12 v1.1 + 4 v1.2)
- Average duration: ~3.5 min/plan (v1.1)
- Total execution time: --

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 7 | -- | -- |
| 1.1 | 3 | -- | -- |
| 2 | 3 | -- | -- |
| 3 | 3 | -- | -- |
| 4 | 2 | -- | -- |
| 5 | 3 | 9min | 3min |
| 6 | 4 | 17min | 4min |
| 7 | 2 | 7min | 3.5min |
| 8 | 2 | 7min | 3.5min |
| 9 | 3 | 12min | 4min |
| 10 | 2 | 7min | 3.5min |
| 11 | 2 | 4min | 2min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.2]: Behavior logic on GridObject classes via classmethods (build_tick_fn, build_interaction_fn, extra_state_schema)
- [v1.2]: Component methods are code generators producing pure functions at init time -- NOT runtime dispatch targets (JAX tracing constraint)
- [v1.2]: Overcooked interaction monolith wrapped initially, NOT split into per-object fragments (pitfall mitigation)
- [v1.2]: Extra state schema is scope-level (not layout-level) for pytree stability
- [v1.2]: Registration is data-only at import time, composition deferred to lazy call
- [10-01]: Lazy import inside decorator body prevents circular import between grid_object and component_registry
- [10-01]: inspect.signature auto-strips cls for classmethods, raw params list catches instance method mistakes
- [10-02]: ArrayReward uses constructor args (not class attributes) for coefficient/common_reward
- [10-02]: register_reward_type re-exported from array_rewards.py for convenient decorator imports
- [11-01]: symbol_table special entries (+/space) added after auto-population to ensure override of Floor.char conflict
- [11-01]: extra_state_schema keys prefixed with scope name at composition time, not in classmethods
- [11-01]: extra_state_builder set to None in auto-wired config; provided via override until Phase 13
- [11-02]: Coefficient/common_reward handling in composition layer, not inside compute() -- compute() returns raw unweighted rewards
- [11-02]: Global-scope rewards merged after scope-specific via get_reward_types("global")
- [11-02]: compute_fn uses cogrid.backend.xp for JAX/numpy compatibility

### Pending Todos

None yet.

### Blockers/Concerns

- Overcooked's 170-line monolithic interaction_body is the complexity risk -- wrapping it (not splitting) is the mitigation
- Handler composition ordering must be deterministic to avoid behavioral differences vs v1.1
- Extra state schema must be scope-level for pytree stability (not layout-level)

## Session Continuity

Last session: 2026-02-13
Stopped at: Completed 11-02-PLAN.md (Phase 11 complete)
Resume file: None
