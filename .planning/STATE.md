# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** v1.3 Composable Array Feature System -- Phase 18 complete, ready for Phase 19

## Current Position

Phase: 18 of 19 (Autowire Integration and Parity)
Plan: 2/2 complete
Status: Phase complete
Last activity: 2026-02-14 -- Phase 18 Plan 02 complete

Progress: [############################..] 39 prior + 8/8 v1.3 plans complete

## Performance Metrics

**Velocity:**
- Total plans completed: 46 (18 v1.0 + 12 v1.1 + 9 v1.2 + 7 v1.3)
- Average duration: ~3 min/plan (v1.2)
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
| 12 | 1 | 2min | 2min |
| 13 | 3 | 7min | 2.3min |
| 14 | 3 | 13min | 4.3min |
| 15 | 2/2 | 7min | 3.5min |
| 16 | 1/1 | 2min | 2min |
| 17 | 2/2 | 3min | 1.5min |
| 18 | 2/2 | 9min | 4.5min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.3]: ArrayFeature uses classmethod builder pattern (like GridObject), not instance compute() (like Reward)
- [v1.3]: Scope-level feature declaration -- scopes declare which features to use, composition is explicit
- [v1.3]: Single ArrayFeature base class with per_agent flag (not separate agent/global types)
- [v1.3]: Old OOP feature system to be removed entirely -- single code path
- [15-01]: ArrayFeature build_feature_fn(cls, scope) validated inline, not via shared _EXPECTED_SIGNATURES, to preserve GridObject backward compat
- [15-02]: compose_feature_fns builds feature functions once at compose time, not per observation call
- [15-02]: Shared _resolve_feature_metas helper for feature name validation used by both compose and obs_dim functions
- [16-01]: ArrayFeature subclasses delegate to bare functions (DRY -- single source of truth for feature computation)
- [16-01]: CanMoveDirection pre-computes can_overlap_table at build_feature_fn time, captured in closure
- [17-01]: ClosestObj uses factory function to create 7 separate registered subclasses with dynamic obs_dim
- [17-01]: DistToOtherPlayers hardcodes n_agents=2 (Overcooked assumption, parameterizable later)
- [17-02]: LayoutID uses class-level _layout_idx attribute (default 0) set externally before build_feature_fn
- [17-02]: EnvironmentLayout pre-computes layout_type_ids from scope at build time, uses class-level _max_layout_shape=(11,7)
- [18-01]: Multi-scope lookup merges features from multiple registry scopes via _resolve_feature_metas scopes param
- [18-01]: _FEATURE_ORDER dict maps scope names to explicit feature lists; unlisted scopes get alphabetical ordering
- [18-01]: build_feature_config_from_components conditionally imports LayoutID only for overcooked scope
- [18-01]: Global ArrayFeature subclasses require import of cogrid.feature_space.array_features to trigger registration
- [18-02]: Feature function always built in reset() via autowire -- no more __init__ fallback
- [18-02]: feature_fn_builder key remains in scope_config as dead weight until Phase 19 removes it

### Pending Todos

None yet.

### Blockers/Concerns

- build_feature_fn currently lives on Pot classmethod -- needs clean migration path (Phase 19 removes it)
- Feature composition must handle ego-centric ordering (focal agent first)
- Pre-existing test ordering issue: JAX backend test pollutes global backend state for numpy tests

## Session Continuity

Last session: 2026-02-14
Stopped at: Completed 18-02-PLAN.md. Phase 18 complete. CoGridEnv uses autowired ArrayFeature composition exclusively. Ready for Phase 19 (remove legacy).
Resume file: None
