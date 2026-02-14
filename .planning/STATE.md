# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-13)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** v1.3 Composable Array Feature System -- Phase 16

## Current Position

Phase: 16 of 19 (Core ArrayFeature Subclasses)
Plan: 1/1 complete
Status: Phase 16 complete
Last activity: 2026-02-14 -- Phase 16 Plan 01 complete

Progress: [#######################.......] 39 prior + 3/3 v1.3 plans complete (Phase 16 done)

## Performance Metrics

**Velocity:**
- Total plans completed: 42 (18 v1.0 + 12 v1.1 + 9 v1.2 + 3 v1.3)
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

### Pending Todos

None yet.

### Blockers/Concerns

- Overcooked feature parity: 677-dim obs for 2 agents must produce identical values
- build_feature_fn currently lives on Pot classmethod -- needs clean migration path (Phase 19 removes it)
- Feature composition must handle ego-centric ordering (focal agent first)

## Session Continuity

Last session: 2026-02-14
Stopped at: Completed 16-01-PLAN.md. Phase 16 complete. Ready to plan Phase 17.
Resume file: None
