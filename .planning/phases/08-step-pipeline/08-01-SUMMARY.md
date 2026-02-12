---
phase: 08-step-pipeline
plan: 01
subsystem: core
tags: [xp, step-pipeline, backend-agnostic, functional-core]

# Dependency graph
requires:
  - phase: 05-array-state
    provides: EnvState, create_env_state, array_ops (set_at, set_at_2d), layout_parser
  - phase: 06-simulation-functions
    provides: move_agents, process_interactions, get_all_agent_obs, build_feature_fn, scope_config
  - phase: 07-rewards-scope-config
    provides: compute_rewards, build_overcooked_scope_config, overcooked_tick
provides:
  - "Unified step() function composing tick, movement, interactions, observations, rewards via xp"
  - "Unified reset() function building EnvState and initial observations via xp"
  - "envstate_to_dict() bridge function (copied unchanged from jax_step.py)"
affects: [08-02 build_step_fn/build_reset_fn factories, 09 PettingZoo wrapper cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Backend-conditional RNG: get_backend() == 'jax' for jax.random vs np.random"
    - "Backend-conditional stop_gradient: no-op on numpy, lax.stop_gradient on JAX"
    - "Function-level xp import for late-binding backend dispatch"

key-files:
  created:
    - cogrid/core/step_pipeline.py
  modified: []

key-decisions:
  - "step() and reset() use inline get_backend() branching for RNG and stop_gradient rather than caller-provided abstractions"
  - "numpy RNG uses np.random.default_rng() without seed for priority -- reproducible seeding deferred to Phase 9"
  - "envstate_to_dict copied unchanged from jax_step.py -- already fully backend-agnostic"

patterns-established:
  - "Unified pipeline pattern: single step/reset code path with xp, branching only for 2 JAX-specific ops"
  - "Composition order: tick -> movement -> interactions -> observations -> rewards -> dones -> stop_gradient"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 8 Plan 01: Unified Step Pipeline Summary

**Backend-agnostic step() and reset() composing all Phase 5-7 sub-functions via xp with get_backend() branching only for RNG and stop_gradient**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-12T23:52:29Z
- **Completed:** 2026-02-12T23:55:33Z
- **Tasks:** 2
- **Files created:** 1

## Accomplishments

- Created `cogrid/core/step_pipeline.py` with `step()`, `reset()`, and `envstate_to_dict()` -- the unified functional pipeline replacing jax_step.py
- All array operations use `xp`; backend-specific code limited to two conditional blocks (RNG for movement priority / reset directions, stop_gradient for RL training)
- Verified end-to-end on JAX backend: reset + 11 eager steps with correct output shapes (obs, rewards, done) and extra_state persistence
- Verified end-to-end on numpy backend: reset + 6 steps with correct output shapes and no JAX dependency

## Task Commits

Each task was committed atomically:

1. **Task 1: Create step_pipeline.py with unified step(), reset(), and envstate_to_dict()** - `f3ea56a` (feat)
2. **Task 2: Verify step_pipeline works end-to-end on JAX backend** - no code changes (verification-only task, all tests passed as-is)

## Files Created/Modified

- `cogrid/core/step_pipeline.py` - Unified step/reset pipeline: step() composes tick, movement, interactions, observations, rewards, dones; reset() builds initial EnvState with random directions; envstate_to_dict() bridges EnvState to sub-function dict format

## Decisions Made

- **Inline backend branching for RNG:** step() and reset() check `get_backend() == "jax"` directly rather than accepting pre-computed priority/direction arrays from the caller. This keeps the same self-contained pattern as the original jax_step.py.
- **Numpy RNG without seed:** On numpy backend, `np.random.default_rng()` is used without a seed for movement priority. This is adequate for Phase 8 where the numpy functional path is not yet wired into cogrid_env.py. Reproducible seeding is a Phase 9 concern.
- **No modifications to jax_step.py:** The existing jax_step.py is left intact. Backward-compat aliases (jax_step -> step) will be added in Plan 02 when build_step_fn/build_reset_fn are implemented.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- step_pipeline.py is ready for Plan 02 to build `build_step_fn()` and `build_reset_fn()` factories that close over static config and optionally JIT-compile
- jax_step.py backward-compat aliases can be wired in Plan 02
- The numpy path can be tested more thoroughly once build_reset_fn integrates parse_layout

---
*Phase: 08-step-pipeline*
*Completed: 2026-02-12*
