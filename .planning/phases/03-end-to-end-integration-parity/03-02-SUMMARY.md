---
phase: 03-end-to-end-integration-parity
plan: 02
subsystem: core
tags: [jax, pettingzoo, backend-dispatch, jit, functional-api, env-wrapper]

# Dependency graph
requires:
  - phase: 03-end-to-end-integration-parity
    plan: 01
    provides: "make_jitted_step, make_jitted_reset, jax_step, jax_reset, envstate_to_dict"
provides:
  - "CoGridEnv(config, backend='jax') JAX backend path in __init__, reset, step"
  - "env.jax_step / env.jax_reset properties for raw functional API access"
  - "_reset_backend_for_testing() for cross-backend test isolation"
  - "PettingZoo-format dict<->array conversion at JAX boundary"
affects: [03-03-PLAN, phase-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Numpy-always layout parsing: layout_to_array_state, create_agent_arrays, _extract_overcooked_state use numpy for mutable construction even under JAX backend"
    - "JAX boundary conversion: dict<->array at PettingZoo wrapper edge, raw arrays inside JIT"
    - "Backend dispatch at step entry: single if check at top of step() routes to _jax_step_wrapper"
    - "Reward name mapping: config names (delivery_reward) stripped to JAX fn names (delivery) via _reward suffix removal"

key-files:
  created: []
  modified:
    - "cogrid/backend/_dispatch.py"
    - "cogrid/cogrid_env.py"
    - "cogrid/core/agent.py"
    - "cogrid/core/grid_utils.py"
    - "cogrid/envs/overcooked/array_config.py"

key-decisions:
  - "Layout parsing always uses numpy: layout_to_array_state, create_agent_arrays, _extract_overcooked_state must use numpy (not xp) because they need mutable in-place assignment. JAX arrays are immutable."
  - "JAX feature names hard-coded: JAX path uses [agent_position, agent_dir, full_map_encoding, can_move_direction, inventory] regardless of config feature space name (e.g. overcooked_features)"
  - "Reward name mapping: strip _reward suffix from config reward names to get JAX function names (delivery_reward -> delivery)"
  - "action_pickup_drop_idx included in reward_config dict as required by compute_rewards_jax"
  - "Scope config static_tables and interaction_tables converted from numpy to JAX arrays at __init__ time for JIT compatibility"

patterns-established:
  - "Numpy-mutable-construction: All functions building arrays from Grid/Agent objects use numpy, callers convert to jnp"
  - "Backend-conditional-init: JAX infrastructure built only when self._backend == 'jax', numpy path unchanged"

# Metrics
duration: 5min
completed: 2026-02-11
---

# Phase 3 Plan 02: PettingZoo ParallelEnv JAX Backend Integration Summary

**Wired JIT-compiled JAX step/reset into PettingZoo ParallelEnv interface with dict<->array boundary conversion and _reset_backend_for_testing for cross-backend test isolation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-12T04:46:35Z
- **Completed:** 2026-02-12T04:51:34Z
- **Tasks:** 1
- **Files modified:** 5

## Accomplishments
- `CoGridEnv(config, backend='jax')` builds JAX infrastructure at init: feature function, reward config, JIT factories, static table conversion
- `env.reset(seed=42)` on JAX backend parses layout with numpy, converts to JAX arrays, runs JIT-compiled reset, returns PettingZoo-format (obs_dict, info_dict)
- `env.step(action_dict)` on JAX backend bypasses all numpy-path logic, calls JIT-compiled step, converts array results to PettingZoo dicts
- `env.jax_step` and `env.jax_reset` properties expose raw JIT-compiled functions for direct JIT/vmap usage
- `_reset_backend_for_testing()` enables switching between numpy and JAX backends in a single test process
- Numpy backend behavior completely unchanged

## Task Commits

Each task was committed atomically:

1. **Task 1: Add _reset_backend_for_testing and JAX init/reset/step paths to CoGridEnv** - `ab1a25c` (feat)

**Plan metadata:** [pending] (docs: complete plan)

## Files Created/Modified
- `cogrid/backend/_dispatch.py` - Added _reset_backend_for_testing() for test isolation
- `cogrid/cogrid_env.py` - JAX backend paths in __init__, reset, step; _jax_step_wrapper; jax_step/jax_reset properties
- `cogrid/core/agent.py` - Fixed create_agent_arrays to always use numpy for mutable construction
- `cogrid/core/grid_utils.py` - Fixed layout_to_array_state to always use numpy for mutable construction
- `cogrid/envs/overcooked/array_config.py` - Fixed _extract_overcooked_state to always use numpy for mutable construction

## Decisions Made
- **Numpy-always for layout parsing:** `layout_to_array_state`, `create_agent_arrays`, and `_extract_overcooked_state` must always use numpy (not the global `xp`) because they build arrays with in-place mutation (`arr[i, j] = val`), which JAX arrays do not support. The JAX path in `reset()` then converts these numpy arrays to `jnp.array` for JIT compilation.
- **Hard-coded JAX feature names:** The config has `"features": ["overcooked_features"]` (a registered FeatureSpace class name), but the JAX path needs raw array feature names. Used the same set as the 03-01 smoke test: `["agent_position", "agent_dir", "full_map_encoding", "can_move_direction", "inventory"]`.
- **Reward name mapping via suffix stripping:** Config reward names like `"delivery_reward"` are mapped to JAX function names like `"delivery"` by removing the `"_reward"` suffix. This bridges the naming convention gap between the reward module registry and the JAX reward function map.
- **Static table conversion at init time:** Scope config `static_tables` and `interaction_tables` (built with numpy by `build_overcooked_scope_config`) are converted to `jnp.array` during `__init__` so they work inside JIT-traced code without `TracerArrayConversionError`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed layout_to_array_state to use numpy instead of xp**
- **Found during:** Task 1 (verification)
- **Issue:** `layout_to_array_state` imports `xp` from `cogrid.backend`, which is `jnp` under JAX backend. In-place assignment (`object_type_map[r, c] = type_id`) raises `TypeError: JAX arrays are immutable`.
- **Fix:** Changed to always use `numpy` (imported as `np` at module top) for array construction. Same fix applied to `create_agent_arrays` in `agent.py` and `_extract_overcooked_state` in `array_config.py`.
- **Files modified:** cogrid/core/grid_utils.py, cogrid/core/agent.py, cogrid/envs/overcooked/array_config.py
- **Verification:** JAX env.reset() completes without error, numpy path unchanged
- **Committed in:** ab1a25c (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential fix for JAX backend -- layout parsing requires mutable arrays. No scope creep; callers convert to JAX arrays when needed.

## Issues Encountered
None beyond the auto-fixed deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PettingZoo API fully works on JAX backend, ready for cross-backend parity testing (Plan 03-03)
- The JAX path uses a different observation feature set (low-level array features) than the numpy path (high-level OvercookedCollectedFeatures). Parity testing in 03-03 should compare the array-level features, not the full Overcooked feature set.
- `_reset_backend_for_testing()` enables running both numpy and JAX envs in a single test process

---
*Phase: 03-end-to-end-integration-parity*
*Completed: 2026-02-11*
