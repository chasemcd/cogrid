# Roadmap: CoGrid

## Milestones

- v1.0 Vectorization & Dual Backend - Phases 1-4 (shipped 2026-02-12)
- v1.1 Unified Functional Architecture - Phases 5-9 (shipped 2026-02-12)
- v1.2 Component-Based Environment API - Phases 10-14 (shipped 2026-02-13)
- v1.3 Composable Array Feature System - Phases 15-19 (in progress)

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

<details>
<summary>v1.2 Component-Based Environment API (Phases 10-14) - SHIPPED 2026-02-13</summary>

### Phase 10: Component Registration Infrastructure
**Goal**: GridObject and Reward subclasses can declare array-level behavior (tick, interact, extra_state, compute) via classmethods, and the registration decorator auto-discovers and stores this metadata for downstream composition
**Plans**: 2/2 complete

### Phase 11: Composition & Auto-Wiring
**Goal**: A new `autowire` module collects registered component metadata for a scope and composes it into complete scope_config and reward_config dicts that the existing step pipeline consumes unchanged
**Plans**: 2/2 complete

### Phase 12: Generic Interaction Signature
**Goal**: The interaction pipeline passes and returns extra_state as a generic dict, so interaction handlers from any scope work through the same interface without scope-specific parameter unpacking
**Plans**: 1/1 complete

### Phase 13: Overcooked Migration
**Goal**: The Overcooked domain defines all its behavior (pot ticking, interactions, extra state, rewards) through component interfaces exclusively, proving the component API handles the most complex existing scope
**Plans**: 3/3 complete

### Phase 14: Auto-Wired CoGridEnv & Validation
**Goal**: CoGridEnv uses auto-wiring as its default path, new environments can be created with only component subclasses and a layout, and the full system is validated for parity, JIT, vmap, and cross-backend correctness
**Plans**: 3/3 complete

</details>

### v1.3 Composable Array Feature System (In Progress)

**Milestone Goal:** Extend the component API pattern to features. Each feature is an ArrayFeature subclass with a classmethod builder that returns a pure function. Scopes declare which features to use. The old OOP feature system is removed -- single code path.

**Phase Numbering:**
- Integer phases (15, 16, 17, 18, 19): Planned milestone work
- Decimal phases (e.g. 15.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 15: ArrayFeature Infrastructure** - Base class, registration decorator, and composition layer that assembles per-agent + global features into ego-centric obs
- [x] **Phase 16: Core ArrayFeature Subclasses** - Generic features (direction, position, movement, inventory) as ArrayFeature subclasses validating the pattern
- [x] **Phase 17: Overcooked ArrayFeature Subclasses** - All Overcooked-specific features wrapped as individual ArrayFeature subclasses
- [x] **Phase 18: Autowire Integration & Parity** - Autowire discovers registered features, CoGridEnv uses composed feature function, Overcooked 677-dim obs matches exactly
- [ ] **Phase 19: Legacy Feature System Removal** - Old OOP feature system deleted, build_feature_fn removed from GridObject convention, single code path

## Phase Details

### Phase 15: ArrayFeature Infrastructure
**Goal**: A standardized ArrayFeature base class with `@register_feature_type` decorator and composition layer exists, enabling features to be declared as subclasses and auto-composed into ego-centric observation vectors
**Depends on**: Phase 14 (v1.2 complete)
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04
**Success Criteria** (what must be TRUE):
  1. An ArrayFeature subclass with `per_agent = True` and a `build_feature_fn` classmethod can be registered via `@register_feature_type("agent_dir", scope="global")` and retrieved from the feature registry by scope
  2. The composition layer discovers all registered features for a scope and returns a single `fn(state_dict, agent_idx) -> (obs_dim,) float32` that concatenates all feature outputs
  3. Composed observations are ego-centric -- focal agent's per-agent features appear first, then other agents' per-agent features in index order, then global features
  4. A minimal test with 2 dummy features (one per-agent, one global) produces the expected concatenated output with correct ego-centric ordering
**Plans:** 2 plans
- [x] 15-01-PLAN.md -- ArrayFeature base class, FeatureMetadata, @register_feature_type decorator
- [x] 15-02-PLAN.md -- Composition layer (compose_feature_fns, obs_dim_for_features) with TDD test suite

### Phase 16: Core ArrayFeature Subclasses
**Goal**: The generic feature extractors (agent direction, position, movement, inventory) are wrapped as ArrayFeature subclasses registered to the global scope, proving the pattern works end-to-end with real feature functions
**Depends on**: Phase 15
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04
**Success Criteria** (what must be TRUE):
  1. `AgentDir` ArrayFeature produces a `(4,)` one-hot direction vector identical to the existing `agent_dir_feature()` function output
  2. `AgentPosition` ArrayFeature produces a `(2,)` position vector identical to `agent_pos_feature()`
  3. `CanMoveDirection` ArrayFeature produces a `(4,)` movement vector identical to `can_move_direction_feature()`
  4. `Inventory` ArrayFeature produces a one-hot inventory vector identical to `inventory_feature()`
  5. All four features are registered to the global scope and discoverable via the feature registry
**Plans:** 1 plan
- [x] 16-01-PLAN.md -- TDD: four ArrayFeature subclasses with parity tests

### Phase 17: Overcooked ArrayFeature Subclasses
**Goal**: Every Overcooked-specific feature function is wrapped as an individual ArrayFeature subclass registered to the "overcooked" scope, producing numerically identical outputs to the existing loose functions
**Depends on**: Phase 16
**Requirements**: OVCK-01, OVCK-02, OVCK-03, OVCK-04, OVCK-05, OVCK-06, OVCK-07, OVCK-08
**Success Criteria** (what must be TRUE):
  1. `OvercookedInventory` produces `(5,)` one-hot, `NextToCounter` produces `(4,)`, `NextToPot` produces `(16,)`, and `ClosestObj` (parameterized) produces `(2n,)` -- all matching existing function outputs
  2. `OrderedPotFeatures` produces `(12 * max_pots,)` and `DistToOtherPlayers` produces `(2 * (n_agents-1),)` -- matching existing function outputs
  3. `LayoutID` and `EnvironmentLayout` are registered as global features (`per_agent = False`) producing `(5,)` and binary masks respectively
  4. All 8 Overcooked features are registered to the "overcooked" scope (or global for layout features) and discoverable via the feature registry
**Plans:** 2 plans
- [x] 17-01-PLAN.md -- TDD: 6 per-agent Overcooked ArrayFeature subclasses (OvercookedInventory, NextToCounter, NextToPot, ClosestObj x7, OrderedPotFeatures, DistToOtherPlayers)
- [x] 17-02-PLAN.md -- TDD: 2 global Overcooked ArrayFeature subclasses (LayoutID, EnvironmentLayout) + full registry validation

### Phase 18: Autowire Integration & Parity
**Goal**: The autowire module discovers registered features for a scope, composes them into the observation function, CoGridEnv uses this composed function, and Overcooked produces the exact same 677-dim observation as before
**Depends on**: Phase 17
**Requirements**: OVCK-09, INTG-01, INTG-02, INTG-03
**Success Criteria** (what must be TRUE):
  1. `build_feature_config_from_components(scope)` discovers registered ArrayFeature subclasses for a scope and returns a composed feature function -- no manual feature list assembly
  2. CoGridEnv uses the autowired feature composition instead of the current `feature_fn_builder` / `build_feature_fn` fallback paths -- no manual feature function construction in `cogrid_env.py`
  3. The step pipeline receives the composed feature function without interface changes -- the `get_all_agent_obs` call site works unchanged
  4. Overcooked composed output matches the existing 677-dim observation exactly -- verified by automated parity test comparing element-by-element across multiple states
**Plans:** 2 plans
- [ ] 18-01-PLAN.md -- Extend compose_feature_fns (multi-scope, preserve_order) + build_feature_config_from_components in autowire
- [ ] 18-02-PLAN.md -- Wire CoGridEnv to autowired feature composition + element-by-element 677-dim parity test

### Phase 18.1: Remove environment-specific logic from core files (INSERTED)

**Goal:** Zero environment-specific logic in cogrid_env.py or any file under cogrid/core/ -- all domain knowledge (feature ordering, layout index mapping, conditional imports) pushed to domain modules via registration APIs
**Depends on:** Phase 18
**Plans:** 1 plan

Plans:
- [ ] 18.1-01-PLAN.md -- Add feature order / layout index registration APIs, move Overcooked-specific logic to domain module, clean up core docstrings

### Phase 19: Legacy Feature System Removal
**Goal**: The old OOP feature system is deleted, `build_feature_fn` is removed from the GridObject component classmethod convention, and the codebase has a single code path for features
**Depends on**: Phase 18
**Requirements**: CLNP-01, CLNP-02, CLNP-03, CLNP-04
**Success Criteria** (what must be TRUE):
  1. `feature.py` and `feature_space.py` no longer exist in `cogrid/feature_space/` -- the old OOP Feature base class and FeatureSpace registry are deleted
  2. `features.py` and `overcooked_features.py` no longer exist -- old OOP feature implementations are deleted
  3. `build_feature_fn` is no longer a recognized classmethod in the component registry -- `_EXPECTED_SIGNATURES` and `ComponentMetadata.has_feature_fn` no longer reference it, and Pot no longer defines it
  4. All tests pass and training produces identical results to pre-v1.3 -- no behavioral regression
**Plans:** TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 15 -> 16 -> 17 -> 18 -> 19

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
| 11. Composition & Auto-Wiring | v1.2 | 2/2 | Complete | 2026-02-13 |
| 12. Generic Interaction Signature | v1.2 | 1/1 | Complete | 2026-02-13 |
| 13. Overcooked Migration | v1.2 | 3/3 | Complete | 2026-02-13 |
| 14. Auto-Wired CoGridEnv & Validation | v1.2 | 3/3 | Complete | 2026-02-13 |
| 15. ArrayFeature Infrastructure | v1.3 | 2/2 | Complete | 2026-02-13 |
| 16. Core ArrayFeature Subclasses | v1.3 | 1/1 | Complete | 2026-02-14 |
| 17. Overcooked ArrayFeature Subclasses | v1.3 | 2/2 | Complete | 2026-02-14 |
| 18. Autowire Integration & Parity | v1.3 | 2/2 | Complete | 2026-02-14 |
| 19. Legacy Feature System Removal | v1.3 | 0/TBD | Not started | - |
