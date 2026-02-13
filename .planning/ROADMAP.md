# Roadmap: CoGrid

## Milestones

- v1.0 Vectorization & Dual Backend - Phases 1-4 (shipped 2026-02-12)
- v1.1 Unified Functional Architecture - Phases 5-9 (shipped 2026-02-12)
- v1.2 Component-Based Environment API - Phases 10-14 (in progress)

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

<details>
<summary>v1.1 Unified Functional Architecture (Phases 5-9) - SHIPPED 2026-02-12</summary>

### Phase 5: Foundation -- State Model & Backend Helpers
**Goal**: The data layer is ready for unified functional code -- EnvState holds environment-specific arrays generically, a tiny helper module abstracts the only operations where numpy and JAX genuinely differ, and layouts parse directly into arrays without intermediate objects
**Plans**: 3/3 complete

### Phase 6: Core Algorithms
**Goal**: The core simulation algorithms -- movement, collision resolution, interactions, and feature extraction -- each exist as a single function using `xp` array operations, replacing all duplicate `_jax`/`_array` function pairs
**Plans**: 4/4 complete

### Phase 7: Rewards & Scope Config
**Goal**: Reward functions and Overcooked-specific tick/interaction handlers each exist as a single function using `xp`, with scope config registration using the new `extra_state` schema
**Plans**: 2/2 complete

### Phase 8: Step Pipeline
**Goal**: A complete functional `step()` and `reset()` compose all unified sub-functions into a pure pipeline, with init-time function wiring that eliminates per-step dispatch overhead
**Plans**: 2/2 complete

### Phase 9: Integration & Cleanup
**Goal**: The PettingZoo wrapper delegates to the functional core, the functional API is directly accessible for JIT/vmap usage, all old duplicate code is deleted, and the full test suite verifies correctness
**Plans**: 3/3 complete

</details>

### v1.2 Component-Based Environment API (In Progress)

**Milestone Goal:** Make creating new grid-world environments trivial by putting behavior on composable component classes (GridObject, Reward) with clear abstract interfaces. The base env auto-wires everything -- no manual scope_config, lookup_table, or reward_config assembly.

**Phase Numbering:**
- Integer phases (10, 11, 12, 13, 14): Planned milestone work
- Decimal phases (e.g. 10.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 10: Component Registration Infrastructure** - Extended decorator auto-discovers and stores component classmethods in registry
- [ ] **Phase 11: Composition & Auto-Wiring** - Auto-wiring module composes registered component metadata into complete scope_config and reward_config
- [ ] **Phase 12: Generic Interaction Signature** - process_interactions() generalized to pass/return extra_state dict instead of scope-specific arrays
- [ ] **Phase 13: Overcooked Migration** - Overcooked domain refactored to define all behavior through component interfaces exclusively
- [ ] **Phase 14: Auto-Wired CoGridEnv & Validation** - CoGridEnv uses auto-wiring by default, new envs need only component classes + layout, full parity verified

## Phase Details

### Phase 10: Component Registration Infrastructure
**Goal**: GridObject and Reward subclasses can declare array-level behavior (tick, interact, extra_state, compute) via classmethods, and the registration decorator auto-discovers and stores this metadata for downstream composition
**Depends on**: Phase 9 (v1.1 complete)
**Requirements**: COMP-01, COMP-02, COMP-03, COMP-04, COMP-05, COMP-06
**Success Criteria** (what must be TRUE):
  1. A GridObject subclass with `build_tick_fn()` classmethod has that method discoverable via the component registry after import -- calling `get_component_methods("Pot")` returns the tick function builder
  2. A GridObject subclass with `extra_state_schema()` and `extra_state_builder()` classmethods has its schema retrievable from the registry -- the schema declares array shapes/dtypes and the builder produces matching initial arrays
  3. An ArrayReward subclass with `compute()` method, `coefficient`, and `common_reward` class attributes is auto-registered and retrievable by scope -- listing rewards for "overcooked" scope returns the registered subclass
  4. Registration is data-only at import time -- no composition or function building happens during registration, only metadata storage
**Plans:** 2 plans
Plans:
- [x] 10-01-PLAN.md -- Component registry module + extended @register_object_type decorator
- [x] 10-02-PLAN.md -- ArrayReward base class + comprehensive test suite

### Phase 11: Composition & Auto-Wiring
**Goal**: A new `autowire` module collects registered component metadata for a scope and composes it into complete scope_config and reward_config dicts that the existing step pipeline consumes unchanged
**Depends on**: Phase 10
**Requirements**: WIRE-01, WIRE-02, WIRE-03, WIRE-04
**Success Criteria** (what must be TRUE):
  1. `build_scope_config_from_components(scope)` returns a complete scope_config dict (type_ids, tick_handler, interaction_body, symbol_table, static_tables, interaction_tables) built entirely from registered component metadata -- no manual dict assembly required
  2. `build_reward_config_from_components(scope)` returns a reward_config dict with a composed `compute_fn` that calls each registered ArrayReward.compute() and sums weighted results -- producing `(n_agents,)` float32 output
  3. GridObject.char values auto-populate the symbol_table for a scope -- adding a new GridObject subclass with `char = "P"` makes "P" parseable in layouts without any manual symbol registration
  4. Extra state schemas from all GridObject subclasses in a scope merge into a single scope-level schema -- the merged schema is deterministic (sorted by key) so pytree structure is stable across runs
**Plans:** 2 plans
Plans:
- [ ] 11-01-PLAN.md -- Auto-wire scope_config from component registry (TDD)
- [ ] 11-02-PLAN.md -- Auto-wire reward_config with composed compute_fn (TDD)

### Phase 12: Generic Interaction Signature
**Goal**: The interaction pipeline passes and returns extra_state as a generic dict, so interaction handlers from any scope work through the same interface without scope-specific parameter unpacking
**Depends on**: Phase 11
**Requirements**: INTG-01
**Success Criteria** (what must be TRUE):
  1. `process_interactions()` accepts `extra_state` as a dict parameter and returns updated `extra_state` as a dict -- the function signature is scope-agnostic
  2. The existing Overcooked interaction_body continues to work through a backward-compatible wrapper that unpacks/repacks the extra_state dict -- all existing Overcooked tests pass without modification
**Plans**: TBD

### Phase 13: Overcooked Migration
**Goal**: The Overcooked domain defines all its behavior (pot ticking, interactions, extra state, rewards) through component interfaces exclusively, proving the component API handles the most complex existing scope
**Depends on**: Phase 12
**Requirements**: INTG-02, COMP-01, COMP-02, COMP-03, COMP-04, COMP-05
**Success Criteria** (what must be TRUE):
  1. Pot, DeliveryZone, and other Overcooked GridObject subclasses define `build_tick_fn()`, `build_interaction_fn()`, `extra_state_schema()`, and `extra_state_builder()` classmethods -- their behavior is declared on the classes, not in a separate array_config module
  2. Overcooked reward functions (delivery, onion_in_pot, soup_in_dish) are ArrayReward subclasses with `compute()`, `coefficient`, and `common_reward` -- no manual reward_config dict construction
  3. `build_scope_config_from_components("overcooked")` produces a scope_config that, when fed to the step pipeline, yields identical step outputs (state, obs, rewards, terminateds, truncateds) as v1.1 manual-wired Overcooked
  4. The manual `build_overcooked_scope_config()` function in array_config.py is no longer called -- auto-wiring replaces it
**Plans**: TBD

### Phase 14: Auto-Wired CoGridEnv & Validation
**Goal**: CoGridEnv uses auto-wiring as its default path, new environments can be created with only component subclasses and a layout, and the full system is validated for parity, JIT, vmap, and cross-backend correctness
**Depends on**: Phase 13
**Requirements**: INTG-03, INTG-04, TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. CoGridEnv.__init__() calls auto-wiring functions by default -- no `if self.scope == "overcooked"` branches, no manual scope_config or reward_config assembly code remains
  2. A minimal goal-finding environment works through the component API with zero manual scope_config -- only GridObject subclasses, an ArrayReward subclass, a layout string, and a config dict
  3. Component-based Overcooked produces identical step outputs to v1.1 manual-wired Overcooked across both numpy and JAX backends -- verified by automated parity test
  4. Component-based Overcooked works with `jax.jit` compilation and `jax.vmap` at 1024 parallel environments without error
  5. Dead manual-wiring code (scope-specific builders, manual scope_config dicts, `if scope ==` branches) is deleted from the codebase

## Progress

**Execution Order:**
Phases execute in numeric order: 10 -> 11 -> 12 -> 13 -> 14

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
| 10. Component Registration Infrastructure | v1.2 | 2/2 | Complete | 2026-02-13 |
| 11. Composition & Auto-Wiring | v1.2 | 0/2 | Not started | - |
| 12. Generic Interaction Signature | v1.2 | 0/TBD | Not started | - |
| 13. Overcooked Migration | v1.2 | 0/TBD | Not started | - |
| 14. Auto-Wired CoGridEnv & Validation | v1.2 | 0/TBD | Not started | - |
