# Roadmap: CoGrid Vectorization & Dual Backend

## Overview

This roadmap transforms CoGrid from a pure-Python grid-world environment into a dual JAX/numpy backend system with vectorized operations and a functional state model. The work is front-loaded: Phase 1 conducts the bulk rewrite -- converting all simulation logic (movement, collision, interactions, observations, rewards) from Python loops to parallel array operations while establishing the dual-backend dispatch. Subsequent phases layer on the functional state model and JIT compatibility, wire up end-to-end integration with parity verification, and deliver vmap batching for the 100x+ throughput payoff.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Dual Backend & Vectorized Core Rewrite** - Backend dispatch module, array-based state representation, and rewrite of all simulation logic (movement, collision, interactions, observations, rewards) from Python loops to parallel array operations
- [x] **Phase 1.1: Fix environment separation of concerns** - No environment-specific logic in core methods (INSERTED)
- [x] **Phase 2: Functional State Model & JIT Compatibility** - Immutable EnvState pytree, PRNG key threading, JAX-specific control flow primitives, JIT-compilability of all sub-functions
- [x] **Phase 3: End-to-End Integration & Parity** - Full step() JIT compilation, PettingZoo wrapper, functional API, cross-backend parity verification
- [ ] **Phase 4: vmap Batching & Benchmarks** - Batched parallel rollouts and performance verification

## Phase Details

### Phase 1: Dual Backend & Vectorized Core Rewrite
**Goal**: All simulation logic operates on arrays via parallel operations instead of Python loops, with a backend dispatch module enabling numpy/jax.numpy swap -- the core rewrite that constitutes the bulk of the project's work
**Depends on**: Nothing (first phase)
**Requirements**: BACK-01, BACK-02, BACK-03, STATE-02, STATE-03, STATE-05, MOVE-01, MOVE-02, MOVE-03, MOVE-04, INTR-01, INTR-02, INTR-04, OBS-01, OBS-02, OBS-03, OBS-04, REW-01, REW-02, REW-03, TEST-05
**Success Criteria** (what must be TRUE):
  1. `CoGridEnv(config, backend='numpy')` instantiates and behaves identically to current `CoGridEnv(config)` -- all existing tests pass without modification
  2. Importing cogrid and running with `backend='numpy'` succeeds in an environment where JAX is not installed
  3. A backend dispatch module exists that provides array operations (`xp.zeros`, `xp.array`, `xp.where`, etc.) resolved at init time based on backend selection
  4. Object property lookup tables (CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON) exist as integer-indexed arrays, replacing GridObj polymorphism for property queries
  5. An existing ASCII layout (e.g., Overcooked cramped_room) can be parsed into array-based state with correct wall_map, object positions, and agent positions
  6. All variable-length containers (inventory, pot contents) are represented as fixed-size arrays with count fields and sentinel values
  7. Movement resolution computes all agent attempted positions simultaneously via array operations -- no Python loop over agents -- with conflict detection, priority-based resolution (random permutation), swap detection/reversion, and wall/bounds checking all operating on arrays
  8. Pickup, drop, and pot interaction logic operates on inventory arrays and grid state arrays via array operations -- objects move between grid arrays and inventory arrays without Python object manipulation
  9. Core feature extractors (agent_pos, agent_dir, full_map_encoding, can_move_direction) are implemented as functions operating on state arrays, with feature composition resolved at init time
  10. Reward functions are implemented as functions with signature calculate_reward(prev_state, state, actions) -> reward_array, with reward composition resolved at init time
  11. For a given grid state and action sequence, observation arrays and reward values from the refactored code match the current implementation on the numpy backend
**Plans**: 7 plans

Plans:
- [ ] 01-01-PLAN.md -- Backend dispatch module + object type registry with lookup tables
- [ ] 01-02-PLAN.md -- Array state representation (layout parser + agent arrays)
- [ ] 01-03-PLAN.md -- Vectorized movement resolution with collision handling
- [ ] 01-04-PLAN.md -- Interaction processing with lookup tables (pickup/drop/pot)
- [ ] 01-05-PLAN.md -- Array-based feature extractors with init-time composition
- [ ] 01-06-PLAN.md -- Array-based reward functions with init-time composition
- [ ] 01-07-PLAN.md -- Integration into CoGridEnv + regression verification

### Phase 01.1: Fix environment separation of concerns (no environment-specific logic in core methods) (INSERTED)

**Goal:** Remove all Overcooked-specific logic from core modules (cogrid_env.py, core/interactions.py, core/grid_utils.py, core/array_rewards.py) by introducing a scope config registry pattern where environments register their array-based configuration and handlers, consumed by generic core infrastructure
**Depends on:** Phase 1
**Plans:** 3 plans

Plans:
- [ ] 01.1-01-PLAN.md -- Scope config registry + Overcooked array config module
- [ ] 01.1-02-PLAN.md -- Move Overcooked array rewards and interaction test to envs/overcooked/
- [ ] 01.1-03-PLAN.md -- Wire scope config into cogrid_env.py, core/interactions.py, core/grid_utils.py (cleanup)

### Phase 2: Functional State Model & JIT Compatibility
**Goal**: All vectorized code from Phase 1 is wrapped in an immutable EnvState pytree with PRNG key threading, all control flow uses JAX-compatible primitives, and every sub-function JIT-compiles without error
**Depends on**: Phase 1
**Requirements**: STATE-01, STATE-04, MOVE-05, INTR-03, INTR-05
**Success Criteria** (what must be TRUE):
  1. An EnvState dataclass exists with all fields as fixed-shape arrays (agent_pos, agent_dir, agent_inv, wall_map, object_type_map, object_state_map, pot_contents, rng_key) and is registered as a JAX pytree -- `jax.jit(lambda s: s)` round-trips an EnvState without error or shape change
  2. Every stochastic operation consumes a subkey via `jax.random.split` pattern -- no stateful RNG, PRNG key threaded through state
  3. Toggle action dispatches to type-specific behavior via `jax.lax.switch` on object type ID -- no Python isinstance or virtual method dispatch on traced values
  4. All interaction logic uses `jnp.where` or `jax.lax.cond/switch` instead of Python if/else on traced values -- no ConcretizationTypeError under JIT
  5. `jax.jit(move_agents)(state, actions, key)`, `jax.jit(process_interactions)(state, actions)`, `jax.jit(get_obs)(state)`, and `jax.jit(compute_rewards)(prev_state, state, actions)` all execute without error
**Plans**: 3 plans

Plans:
- [ ] 02-01-PLAN.md -- EnvState frozen dataclass + JAX-path movement with lax.fori_loop
- [ ] 02-02-PLAN.md -- JAX-path interactions (core + Overcooked handlers)
- [ ] 02-03-PLAN.md -- JAX-path observations (vmap) + rewards (vectorized)

### Phase 3: End-to-End Integration & Parity
**Goal**: A complete step(state, actions) -> (new_state, obs, rewards, dones, infos) function that JIT-compiles end-to-end, wrapped in both PettingZoo interface and raw functional API, with verified cross-backend parity
**Depends on**: Phase 2
**Requirements**: BACK-04, BACK-06, BACK-07, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. `jax.jit(env.jax_step)(state, actions)` compiles and executes end-to-end without error, producing (new_state, obs, rewards, dones, infos) with correct types and shapes
  2. The PettingZoo ParallelEnv interface (env.step(action_dict)) works on the JAX backend: it holds EnvState internally, translates dict-based API to array-based functional core, and returns standard PettingZoo outputs
  3. `env.jax_step` and `env.jax_reset` are exposed as the raw functional API for direct JIT/vmap usage, bypassing PettingZoo overhead
  4. Cross-backend parity: running the same seed and action sequence on both backends produces array-equal observations (or allclose with documented tolerance for floats) and identical reward values, verified across at least 100 steps on 3 layouts
  5. Every test runs both eagerly and under jax.jit to catch silent mutation bugs -- at least the core step/reset/obs/reward tests have JIT variants
**Plans**: 3 plans

Plans:
- [ ] 03-01-PLAN.md -- End-to-end jax_step and jax_reset functional API on EnvState
- [ ] 03-02-PLAN.md -- PettingZoo wrapper for JAX backend + functional API exposure
- [ ] 03-03-PLAN.md -- Cross-backend parity test suite (scripted + random + eager vs JIT)

### Phase 4: vmap Batching & Benchmarks
**Goal**: Batched parallel environment rollouts via jax.vmap verified at scale, with benchmark suite quantifying the speedup
**Depends on**: Phase 3
**Requirements**: BACK-05, TEST-04
**Success Criteria** (what must be TRUE):
  1. `jax.vmap(env.jax_step)(batched_states, batched_actions)` executes correctly with 1024 parallel environments, producing batched outputs with correct shapes (batch_dim, ...)
  2. Each environment in the vmap batch produces identical results to running that environment individually (verified by comparing single-env output to corresponding slice of batched output)
  3. A benchmark suite measures and reports: numpy single-env steps/sec, JAX single-env JIT steps/sec, JAX vmap at 1024 parallel envs steps/sec -- with measurable speedup documented
  4. Benchmark results are reproducible: running the suite twice produces consistent measurements (within 10% variance)
**Plans**: 2 plans

Plans:
- [ ] 04-01-PLAN.md -- vmap correctness tests (reset/step at 1024 envs + single-env vs batched parity)
- [ ] 04-02-PLAN.md -- Benchmark suite (numpy single, JAX single JIT, JAX vmap@1024 + reproducibility)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 1.1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Dual Backend & Vectorized Core Rewrite | 7/7 | Complete | 2026-02-11 |
| 1.1. Fix environment separation of concerns | 3/3 | Complete | 2026-02-11 |
| 2. Functional State Model & JIT Compatibility | 3/3 | Complete | 2026-02-11 |
| 3. End-to-End Integration & Parity | 3/3 | Complete | 2026-02-11 |
| 4. vmap Batching & Benchmarks | 0/2 | Not started | - |
