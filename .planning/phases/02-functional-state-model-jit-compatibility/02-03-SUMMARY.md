---
phase: 02-functional-state-model-jit-compatibility
plan: 03
subsystem: backend
tags: [jax, jit, vmap, vectorization, observations, rewards, overcooked]

# Dependency graph
requires:
  - phase: 01-backend-dispatch-array-state
    provides: "Array-based feature extractors and reward functions with Python loops"
  - phase: 02-functional-state-model-jit-compatibility
    plan: 01
    provides: "EnvState pytree, move_agents_jax pattern, JAX 0.4.38 installation"
affects: [phase-3-end-to-end-jit-step, integration, observations-pipeline, rewards-pipeline]

provides:
  - "JAX-path observation functions (5 individual features + vmap composition)"
  - "JAX-path reward functions (3 individual + compute_rewards_jax wrapper)"
  - "get_all_agent_obs_jax() using jax.vmap for vectorized per-agent observation generation"
  - "compute_rewards_jax() composing individual reward functions with sum"
  - "Array-based pot position matching pattern replacing _find_pot_index loop"

# Tech tracking
tech-stack:
  added: [jax.vmap]
  patterns: [vmap-per-agent-vectorization, array-pot-position-matching, closure-based-jit-config, vectorized-reward-computation]

key-files:
  created: []
  modified:
    - cogrid/feature_space/array_features.py
    - cogrid/envs/overcooked/array_rewards.py

key-decisions:
  - "full_map_encoding_feature_jax takes pre-computed agent_type_ids array instead of scope string -- avoids string lookup under JIT"
  - "compute_rewards_jax uses closure pattern for JIT (reward_config captured, not passed as arg) since dicts with strings are not hashable for static_argnames"
  - "Direction vector table created inline in JAX reward helper as jnp.array rather than using shared lazy-init get_dir_vec_table()"
  - "Shared _compute_fwd_positions_jax helper extracts forward position computation used by all three reward functions"

patterns-established:
  - "vmap-per-agent: jax.vmap(lambda i: feature_fn(state_dict, i))(jnp.arange(n_agents)) for vectorized per-agent obs"
  - "Array pot matching: jnp.all(pot_positions[None,:,:] == fwd_pos[:,None,:], axis=2) for (n_agents, n_pots) match matrix"
  - "Closure JIT config: @jax.jit wrapping function that closes over static config dict rather than passing as argument"
  - "Vectorized reward: broadcast condition masks (is_interact & holds_item & faces_target & in_bounds) then multiply by coefficient"

# Metrics
duration: 5min
completed: 2026-02-12
---

# Phase 2 Plan 3: Observations & Rewards JAX Path Summary

**JAX-path observation extraction with jax.vmap per-agent vectorization and fully vectorized reward computation with array-based pot position matching**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-12T03:58:41Z
- **Completed:** 2026-02-12T04:04:15Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- 5 individual JAX-path feature extractors (agent_pos, agent_dir, full_map_encoding, can_move_direction, inventory) all JIT-compile without error
- get_all_agent_obs_jax uses jax.vmap to vectorize observation generation across agents, producing correct (n_agents, obs_dim) output
- 3 JAX-path reward functions (delivery, onion_in_pot, soup_in_dish) fully vectorized across agents with no Python loops
- compute_rewards_jax composes individual reward functions and JIT-compiles as a single unit via closure pattern
- Array-based pot position matching replaces _find_pot_index Python loop for JIT compatibility
- All existing numpy-path functions completely unchanged -- 100% backward compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement JAX-path observation functions with vmap** - `44402a1` (feat)
2. **Task 2: Implement JAX-path reward functions with vectorized agent processing** - `3179b53` (feat)

## Files Created/Modified
- `cogrid/feature_space/array_features.py` - Added 5 JAX feature extractors, build_feature_fn_jax, get_all_agent_obs_jax with vmap
- `cogrid/envs/overcooked/array_rewards.py` - Added 3 JAX reward functions, _compute_fwd_positions_jax helper, compute_rewards_jax wrapper

## Decisions Made
- **Pre-computed agent_type_ids:** full_map_encoding_feature_jax takes a (4,) int32 array mapping direction to type_id instead of a scope string, avoiding string-based object_to_idx lookups under JIT. The array is pre-computed at init time by build_feature_fn_jax.
- **Closure pattern for compute_rewards_jax:** reward_config (containing Python strings and nested dicts) is captured in a closure rather than passed as a JIT argument, since JAX requires static args to be hashable. Usage: `@jax.jit def f(prev, state, acts): return compute_rewards_jax(prev, state, acts, reward_config)`.
- **Inline dir_vec_table in JAX rewards:** Created direction vector table as `jnp.array` inline in `_compute_fwd_positions_jax` rather than using the shared lazy-init `get_dir_vec_table()`, matching the pattern established in 02-01 for avoiding cross-backend array type issues.
- **Shared forward position helper:** Extracted `_compute_fwd_positions_jax()` to compute forward positions, clipped coordinates, in-bounds masks, and forward type IDs once -- shared across all three reward functions.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- With Plans 02-01 (EnvState + movement) and 02-03 (observations + rewards), 4 of 5 Phase 2 success criteria are met:
  1. EnvState frozen dataclass with JAX pytree registration (02-01)
  2. move_agents_jax JIT-compiles (02-01)
  3. get_all_agent_obs_jax with vmap JIT-compiles (02-03)
  4. compute_rewards_jax JIT-compiles (02-03)
- Remaining: Plan 02-02 (process_interactions_jax with lax.fori_loop) completes the phase
- All JAX-path functions can be composed into an end-to-end jitted step() function in Phase 3

## Self-Check: PASSED

- [x] cogrid/feature_space/array_features.py exists (modified)
- [x] cogrid/envs/overcooked/array_rewards.py exists (modified)
- [x] 02-03-SUMMARY.md exists
- [x] Commit 44402a1 found (Task 1)
- [x] Commit 3179b53 found (Task 2)

---
*Phase: 02-functional-state-model-jit-compatibility*
*Completed: 2026-02-12*
