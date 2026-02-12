# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-12)

**Core value:** Minimal code paths, maximal clarity. One functional simulation core that works identically whether xp is numpy or jax.numpy.
**Current focus:** Phase 7 -- Rewards & Scope Config

## Current Position

Phase: 7 of 9 (Rewards & Scope Config) -- COMPLETE
Plan: 2 of 2 in current phase (07-02 complete)
Status: Phase 7 complete, ready for Phase 8
Last activity: 2026-02-12 -- Completed 07-02 (cross-backend reward parity tests)

Progress: [####################################....] 90% (v1.0 complete, v1.1 phases 5-7 complete)

## Performance Metrics

**Velocity:**
- Total plans completed: 27 (18 v1.0 + 9 v1.1)
- Average duration: --
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
| 7 | 2/2 | 7min | 3.5min |

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [v1.1]: Single code path with `xp` dispatch -- no `_jax`/`_numpy` function pairs
- [v1.1]: Pure array ops only -- no Python loops or JAX loop primitives (fori_loop)
- [v1.1]: Generic extra_state dict in EnvState for environment-specific arrays
- [v1.1]: Delete object-based sim code -- rewrite layout parsing without Grid/Agent objects
- [v1.1]: Overcooked scope only for this milestone
- [05-01]: array_ops.set_at uses get_backend() string check (not hasattr) for cleaner dispatch
- [05-01]: extra_state keys use scope-prefix convention (e.g. "overcooked.pot_timer")
- [05-01]: Removed n_pots static field from EnvState (pot count encoded in extra_state array shapes)
- [05-02]: Symbol table in both scope_config and explicit SYMBOL_REGISTRY for dual-access flexibility
- [05-02]: parse_layout always uses numpy for parsing, converts to JAX arrays at the end
- [05-02]: Spawn positions ('+' chars) populate agent_pos but NOT object_type_map
- [05-03]: envstate_to_dict strips scope prefix for backward compat (transitional for Phases 6-7)
- [05-03]: n_pots derived from extra_state array shape[0] instead of removed static field
- [06-01]: move_agents() accepts pre-computed priority array instead of RNG (caller handles backend-specific RNG)
- [06-01]: Cascade blocking pass for propagating blocked-agent positions to lower-priority agents
- [06-01]: Double argsort for vectorized priority rank computation
- [06-02]: Static-range Python loop over n_agents for sequential interaction processing (JIT-compatible, handles any agent count)
- [06-02]: Scope config drops _jax keys -- single function per role (tick_handler, interaction_body)
- [06-03]: full_map_encoding uses xp.pad + xp.stack instead of backend-branching slice assignment
- [06-03]: Agent scatter uses set_at_2d loop over n_agents (static/tiny) instead of fancy indexing
- [06-03]: get_all_agent_obs uses Python loop + xp.stack; vmap deferred to Phase 8
- [06-03]: Backward-compat aliases (build_feature_fn_jax, get_all_agent_obs_jax) for caller migration
- [06-04]: Function-level `from cogrid.backend import xp` imports for correct late-binding backend dispatch
- [06-04]: Priority array pre-computed from RNG in jax_step, not inside move_agents
- [07-01]: Backward-compat alias compute_rewards_jax = compute_rewards for migration period
- [07-01]: Deleted all _array/_jax reward function variants -- unified functions use xp throughout
- [07-02]: importlib.import_module for cross-backend test module loading (avoids package name collision)

### Pending Todos

None yet.

### Blockers/Concerns

- Mutation bugs where `.copy()` + in-place assignment passes numpy but fails under JAX JIT -- mitigated by `set_at()` helper (Phase 5)
- Feature function callers use backward-compat aliases; full migration needed in step pipeline phase
- compute_rewards_jax unified into compute_rewards in 07-01 (backward-compat alias kept)

## Session Continuity

Last session: 2026-02-12
Stopped at: Completed 07-02-PLAN.md (cross-backend reward parity tests, Phase 7 complete)
Resume file: None
