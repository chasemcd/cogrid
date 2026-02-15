# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-15)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** Phase 22 -- Function Decomposition

## Current Position

Phase: 22 of 24 (Function Decomposition)
Plan: 1 of 2 in current phase
Status: Executing
Last activity: 2026-02-15 -- Completed 22-01 (decompose move_agents)

Progress: [███████████░░░░░░░░░░░░░░░░░░░] 7/? v1.4 plans complete

## Performance Metrics

**Velocity:**
- Total plans completed: 55 (18 v1.0 + 12 v1.1 + 9 v1.2 + 9 v1.3 + 7 v1.4)
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
| 18.1 | 1/1 | 7min | 7min |
| 19 | 1/1 | 8min | 8min |
| 20 | 3/3 | 15min | 5min |
| 21 | 3/3 | 12min | 4min |
| 22 | 1/2 | 3min | 3min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [22-01]: Staying-agent identification moved into _resolve_collisions; sub-functions at module level not closures
- [21-03]: Pure mechanical extraction of __init__/reset into named helpers -- zero behavioral changes
- [21-02]: EnvRenderer receives numpy image by value (no env reference); get_frame/get_full_render stay on CoGridEnv
- [21-01]: Re-export shim pattern for module splits; lazy imports in method bodies to break circular deps
- [20-03]: Backend conditional helpers (_backend_rng, _maybe_stop_gradient, _maybe_jit) consolidate 11 conditionals to exactly 6
- [20-01]: BackendProxy uses __getattr__ delegation to xp_module -- simplest proxy, xp singleton never reassigned
- [v1.4]: Module-level xp imports -- standard Python pattern; backend __getattr__ handles lazy resolution
- [v1.4]: Rename state_dict to state -- accurate after StateView introduction; shorter, consistent with rewards
- [v1.4]: Full file restructure -- split large files even though import paths change
- [v1.3]: Old OOP feature system removed entirely -- single code path

### Roadmap Evolution

- v1.3 complete (Phases 15-19), shipped 2026-02-14
- v1.4 roadmap created (Phases 20-24), ready for planning

### Pending Todos

None yet.

### Blockers/Concerns

- Pre-existing test ordering issue: JAX backend test pollutes global backend state for numpy tests

## Session Continuity

Last session: 2026-02-15
Stopped at: Completed 22-01-PLAN.md
Resume file: None
