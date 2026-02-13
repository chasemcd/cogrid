# Roadmap: CoGrid

## Milestones

- v1.0 Vectorization & Dual Backend - Phases 1-4 (shipped 2026-02-12)
- v1.1 Unified Functional Architecture - Phases 5-9 (shipped 2026-02-12)

## Phases

<details>
<summary>v1.0 Vectorization & Dual Backend (Phases 1-4) - SHIPPED 2026-02-12</summary>

### Phase 1: Dual Backend & Vectorized Core Rewrite
**Goal**: All simulation logic operates on arrays via parallel operations instead of Python loops, with a backend dispatch module enabling numpy/jax.numpy swap
**Plans**: 7/7 complete

### Phase 1.1: Fix Environment Separation of Concerns (INSERTED)
**Goal**: Remove all Overcooked-specific logic from core modules via scope config registry pattern
**Plans**: 3/3 complete

### Phase 2: Functional State Model & JIT Compatibility
**Goal**: Immutable EnvState pytree with PRNG key threading, JAX-compatible control flow, all sub-functions JIT-compilable
**Plans**: 3/3 complete

### Phase 3: End-to-End Integration & Parity
**Goal**: Complete step(state, actions) function that JIT-compiles end-to-end, PettingZoo interface, cross-backend parity
**Plans**: 3/3 complete

### Phase 4: vmap Batching & Benchmarks
**Goal**: Batched parallel rollouts via jax.vmap verified at scale with benchmark suite
**Plans**: 2/2 complete

</details>

### v1.1 Unified Functional Architecture (SHIPPED 2026-02-12)

**Milestone Goal:** Rewrite CoGrid to have a single, purely functional code path using `xp` -- no code duplication between backends, no object-based simulation loop, no environment-specific logic in core.

**Phase Numbering:**
- Integer phases (5, 6, 7, 8, 9): Planned milestone work
- Decimal phases (e.g. 5.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 5: Foundation -- State Model & Backend Helpers** - Generic EnvState with extra_state, array mutation helpers, array-based layout parser
- [x] **Phase 6: Core Algorithms** - Unified movement, collision, interaction, and feature functions using xp
- [x] **Phase 7: Rewards & Scope Config** - Unified reward functions and Overcooked-specific handlers using xp
- [x] **Phase 8: Step Pipeline** - Functional step/reset composition with init-time function wiring
- [x] **Phase 9: Integration & Cleanup** - PettingZoo wrapper, functional API exposure, old code deletion, final verification

## Phase Details

### Phase 5: Foundation -- State Model & Backend Helpers
**Goal**: The data layer is ready for unified functional code -- EnvState holds environment-specific arrays generically, a tiny helper module abstracts the only operations where numpy and JAX genuinely differ, and layouts parse directly into arrays without intermediate objects
**Depends on**: Phase 4 (v1.0 complete)
**Requirements**: STATE-01, STATE-02, STATE-03, ARCH-01, ARCH-05
**Success Criteria** (what must be TRUE):
  1. EnvState contains an `extra_state` dict instead of hardcoded pot fields, and the Overcooked scope populates it with `pot_contents`, `pot_timer`, `pot_positions` arrays at init time
  2. `set_at()` and `set_at_2d()` in `backend/array_ops.py` correctly mutate arrays on both numpy and JAX backends -- numpy uses `.copy()` + direct assignment, JAX uses `.at[].set()`, with no `hasattr(arr, 'at')` checks anywhere else in the codebase
  3. EnvState with `extra_state` dict round-trips through `jax.jit(lambda s: s)` without error or shape change -- pytree registration handles the dict correctly
  4. An Overcooked layout (e.g., cramped_room) parses from ASCII string directly into array-based EnvState without creating Grid or Agent objects in the parsing path
**Plans:** 3 plans

Plans:
- [x] 05-01-PLAN.md -- array_ops.py, EnvState rewrite with extra_state, hasattr cleanup
- [x] 05-02-PLAN.md -- Layout parser, scope_config extensions, Overcooked symbol/schema registration
- [x] 05-03-PLAN.md -- jax_step.py migration to extra_state, JIT round-trip verification

### Phase 6: Core Algorithms
**Goal**: The core simulation algorithms -- movement, collision resolution, interactions, and feature extraction -- each exist as a single function using `xp` array operations, replacing all duplicate `_jax`/`_array` function pairs
**Depends on**: Phase 5
**Requirements**: UNIFY-01, UNIFY-02, UNIFY-03, UNIFY-04, CLEAN-05, TEST-02
**Success Criteria** (what must be TRUE):
  1. `move_agents()` resolves all agent movements simultaneously via vectorized propose-filter-resolve pattern with pairwise conflict detection and priority masking -- no Python loops, no `lax.fori_loop` -- and produces identical results on numpy and JAX backends
  2. Collision resolution handles all edge cases correctly: head-on collisions, T-intersection conflicts, 3-way conflicts, and agent swap detection, verified by dedicated unit tests
  3. `process_interactions()` handles pickup, drop, and object interactions via priority-masked vectorization using `xp` -- a single function, no `_jax` variant
  4. Feature extractors (agent_pos, agent_dir, full_map_encoding, can_move_direction, inventory) are consolidated into a single set of functions using `xp`, with no duplicate implementations
  5. Zero `int()` casts on array values exist anywhere in the movement, interaction, or feature extraction code paths
**Plans:** 4 plans

Plans:
- [x] 06-01-PLAN.md -- Unified move_agents() with vectorized collision resolution
- [x] 06-02-PLAN.md -- Unified process_interactions(), overcooked_tick(), overcooked_interaction_body()
- [x] 06-03-PLAN.md -- Unified feature extractors, build_feature_fn(), get_all_agent_obs()
- [x] 06-04-PLAN.md -- Collision resolution tests, int() cast elimination, jax_step.py wiring

### Phase 7: Rewards & Scope Config
**Goal**: Reward functions and Overcooked-specific tick/interaction handlers each exist as a single function using `xp`, with scope config registration using the new `extra_state` schema
**Depends on**: Phase 6
**Requirements**: UNIFY-05, UNIFY-06, UNIFY-07, TEST-01
**Success Criteria** (what must be TRUE):
  1. Reward functions (delivery, onion_in_pot, soup_in_dish) are consolidated from 6 function pairs into 3 single functions using `xp`, producing identical reward values on both backends for the same state transitions
  2. Overcooked tick handler is a single function using `xp` that updates pot state via `extra_state` dict -- no `_jax` variant exists
  3. Overcooked interaction handler is a single function using `xp` that handles pot/counter/delivery interactions -- no `_jax` variant exists
  4. Cross-backend parity test verifies that unified functions produce identical outputs on numpy and JAX backends for scripted state transitions
**Plans:** 2 plans

Plans:
- [x] 07-01-PLAN.md -- Unify reward functions, compute_rewards, jax_step.py lax.fori_loop elimination
- [x] 07-02-PLAN.md -- Cross-backend reward parity tests, update existing test imports

### Phase 8: Step Pipeline
**Goal**: A complete functional `step()` and `reset()` compose all unified sub-functions into a pure pipeline, with init-time function wiring that eliminates per-step dispatch overhead
**Depends on**: Phase 7
**Requirements**: ARCH-02, ARCH-03, ARCH-04
**Success Criteria** (what must be TRUE):
  1. `step(state, actions)` returns `(state, obs, rewards, dones, infos)` as a pure function using `xp` -- composing tick, movement, interaction, observation, and reward sub-functions from Phases 6-7
  2. `reset(rng)` returns `(state, obs)` as a pure function using `xp` -- creating initial EnvState from layout and computing initial observations
  3. Init-time function composition (`build_step_fn`, `build_feature_fn`, `build_reward_fn`) wires scope-specific handlers and feature/reward selections at environment creation time, so the step function has zero dispatch overhead at runtime
  4. `jax.jit(step)(state, actions)` compiles and executes without error on the JAX backend
**Plans:** 2 plans

Plans:
- [x] 08-01-PLAN.md -- Unified step_pipeline.py with step(), reset(), envstate_to_dict()
- [x] 08-02-PLAN.md -- Build factories, backward-compat shim, end-to-end tests

### Phase 9: Integration & Cleanup
**Goal**: The PettingZoo wrapper delegates to the functional core, the functional API is directly accessible for JIT/vmap usage, all old duplicate code is deleted, and the full test suite verifies correctness
**Depends on**: Phase 8
**Requirements**: ARCH-06, ARCH-07, CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, TEST-03, TEST-04
**Success Criteria** (what must be TRUE):
  1. PettingZoo ParallelEnv wrapper holds EnvState internally and delegates `step()`/`reset()` to the functional core -- it is a thin stateful shell that translates dict-based API to array-based functional calls
  2. `env.jax_step` and `env.jax_reset` are exposed for direct JIT/vmap usage, bypassing PettingZoo overhead
  3. All 29 duplicate `_jax` functions are deleted, `core/jax_step.py` is deleted, the object-based simulation loop is removed from `cogrid_env.py`, and no environment-specific logic (pot, Overcooked) exists in any `cogrid/core/` module
  4. `jax.vmap(env.jax_step)(batched_states, batched_actions)` executes correctly at 1024 parallel environments with the unified step function
  5. PettingZoo wrapper passes standard env API checks (reset returns observations, step accepts action dict, done agents are handled correctly)
**Plans:** 3 plans

Plans:
- [x] 09-01-PLAN.md -- Make step_pipeline.py scope-generic (remove Overcooked hardcoding)
- [x] 09-02-PLAN.md -- Rewrite cogrid_env.py as thin wrapper delegating to step_pipeline
- [x] 09-03-PLAN.md -- Delete dead code, update imports, fix tests, verify vmap

## Progress

**Execution Order:**
Phases execute in numeric order: 5 -> 6 -> 7 -> 8 -> 9

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Dual Backend & Vectorized Core Rewrite | v1.0 | 7/7 | Complete | 2026-02-11 |
| 1.1. Fix Environment Separation of Concerns | v1.0 | 3/3 | Complete | 2026-02-11 |
| 2. Functional State Model & JIT Compatibility | v1.0 | 3/3 | Complete | 2026-02-11 |
| 3. End-to-End Integration & Parity | v1.0 | 3/3 | Complete | 2026-02-11 |
| 4. vmap Batching & Benchmarks | v1.0 | 2/2 | Complete | 2026-02-12 |
| 5. Foundation -- State Model & Backend Helpers | v1.1 | 3/3 | Complete | 2026-02-12 |
| 6. Core Algorithms | v1.1 | 4/4 | Complete | 2026-02-12 |
| 7. Rewards & Scope Config | v1.1 | 2/2 | Complete | 2026-02-12 |
| 8. Step Pipeline | v1.1 | 2/2 | Complete | 2026-02-12 |
| 9. Integration & Cleanup | v1.1 | 3/3 | Complete | 2026-02-12 |
