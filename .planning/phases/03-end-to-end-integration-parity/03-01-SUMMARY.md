---
phase: 03-end-to-end-integration-parity
plan: 01
subsystem: core
tags: [jax, jit, pytree, envstate, functional-step, functional-reset, end-to-end]

# Dependency graph
requires:
  - phase: 02-functional-state-model-jit-compatibility
    provides: "EnvState pytree, move_agents_jax, process_interactions_jax, get_all_agent_obs_jax, compute_rewards_jax, overcooked_tick_jax"
provides:
  - "envstate_to_dict: zero-cost EnvState -> dict conversion for sub-functions"
  - "jax_step: end-to-end step composing tick/move/interact/obs/rewards/dones"
  - "jax_reset: initial EnvState from pre-computed layout arrays"
  - "make_jitted_step: factory for JIT-compiled step closure"
  - "make_jitted_reset: factory for JIT-compiled reset closure"
affects: [03-02-PLAN, 03-03-PLAN, phase-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Closure factory pattern: make_jitted_step/reset close over static config, return jax.jit(fn)"
    - "EnvState-to-dict bridge: zero-cost aliasing at function boundary, not deep copy"
    - "Numpy env bootstrap: create numpy env to extract layout, convert to jnp for JAX testing"
    - "Static tables JAX conversion: scope_config numpy arrays must be converted to jnp before JIT tracing"

key-files:
  created:
    - "cogrid/core/jax_step.py"
  modified: []

key-decisions:
  - "JAX imports at function level (not module level) to avoid cogrid/core/typing.py shadowing stdlib typing"
  - "Smoke test uses numpy env to extract layout then converts to JAX arrays, avoiding backend conflict"
  - "Static tables in scope_config must be explicitly converted from numpy to jnp arrays before JIT tracing"
  - "Step ordering verified: prev_state capture -> tick -> move -> interact -> obs -> rewards -> dones"

patterns-established:
  - "Numpy-env-bootstrap: create numpy env, extract arrays, convert to jnp for JAX path"
  - "Static config closure: non-array args closed over in factory, JIT only traces array args"

# Metrics
duration: 5min
completed: 2026-02-11
---

# Phase 3 Plan 01: End-to-End JAX Step/Reset Summary

**Composed all Phase 2 JAX sub-functions into jax_step and jax_reset with JIT-compiled factory closures, verified on Overcooked cramped_room with reset + 11 steps under full JIT**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-12T04:37:19Z
- **Completed:** 2026-02-12T04:42:59Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Created `cogrid/core/jax_step.py` with 5 public functions: `envstate_to_dict`, `jax_step`, `jax_reset`, `make_jitted_step`, `make_jitted_reset`
- End-to-end JIT compilation on Overcooked cramped_room: reset produces (EnvState, obs), step produces (EnvState, obs, rewards, done, infos)
- Step ordering matches existing pipeline: prev_state capture, tick, move, interact, obs, rewards, dones
- Output shapes verified: obs=(2, 443), rewards=(2,), done=scalar bool
- lax.stop_gradient applied to obs, rewards, done following JaxMARL RL pattern

## Task Commits

Each task was committed atomically:

1. **Task 1: Create jax_step, jax_reset, and envstate_to_dict** - `b78f09a` (feat)
2. **Task 2: Create make_jitted_step and make_jitted_reset with smoke test** - `11be6ba` (feat)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `cogrid/core/jax_step.py` - End-to-end JAX step and reset functions with JIT-compiled factory closures and smoke test

## Decisions Made
- **Function-level JAX imports:** Module-level `import jax` causes `ImportError` due to `cogrid/core/typing.py` shadowing stdlib `typing`. Moved all JAX imports inside function bodies.
- **Numpy-env bootstrap for smoke test:** Rather than fighting backend switching, create numpy env to extract layout arrays, then convert to JAX arrays. This avoids the `set_backend()` conflict.
- **Explicit static_tables conversion:** `scope_config["static_tables"]` arrays (built with numpy backend) must be converted to `jnp.array` before use in JIT-traced code. Without this, `CAN_PICKUP[fwd_type]` raises `TracerArrayConversionError` because numpy's `__array__` method is called on traced JAX values.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Moved JAX imports from module-level to function-level**
- **Found during:** Task 2 (smoke test execution)
- **Issue:** Module-level `import jax` triggers circular import via `cogrid/core/typing.py` shadowing stdlib `typing`
- **Fix:** Moved `import jax`, `import jax.numpy as jnp`, `import jax.lax as lax`, and all cogrid imports into each function body
- **Files modified:** cogrid/core/jax_step.py
- **Verification:** `python -c "from cogrid.core.jax_step import ..."` succeeds
- **Committed in:** 11be6ba (Task 2 commit)

**2. [Rule 3 - Blocking] Convert scope_config static_tables from numpy to JAX arrays**
- **Found during:** Task 2 (smoke test execution)
- **Issue:** `CAN_PICKUP[fwd_type]` inside `lax.fori_loop` raises `TracerArrayConversionError` because `CAN_PICKUP` is a numpy array and `fwd_type` is a JAX tracer
- **Fix:** Added explicit numpy-to-jnp conversion for all numpy arrays in `scope_config["static_tables"]` and `scope_config["interaction_tables"]` in the smoke test setup
- **Files modified:** cogrid/core/jax_step.py (smoke test block)
- **Verification:** Smoke test passes: reset + 11 steps complete without error
- **Committed in:** 11be6ba (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary for JIT compilation to work. No scope creep.

## Issues Encountered
- The `cogrid/core/typing.py` module shadows Python's stdlib `typing` when running files directly from the `cogrid/` directory. Workaround: run via `python -m cogrid.core.jax_step` from the parent directory, or use function-level imports.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `jax_step` and `jax_reset` are ready for integration into the PettingZoo wrapper (Plan 03-02)
- The static_tables conversion issue should be handled systematically in Plan 03-02 (inside `CoGridEnv.__init__` when building the JAX path)
- Cross-backend parity testing (Plan 03-03) can now compare numpy step outputs against JAX step outputs

---
*Phase: 03-end-to-end-integration-parity*
*Completed: 2026-02-11*
