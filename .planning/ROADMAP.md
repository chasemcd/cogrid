# Roadmap: CoGrid

## Milestones

- v1.0 Vectorization & Dual Backend - Phases 1-4 (shipped 2026-02-12)
- v1.1 Unified Functional Architecture - Phases 5-9 (shipped 2026-02-12)
- v1.2 Component-Based Environment API - Phases 10-14 (shipped 2026-02-13)
- v1.3 Composable Array Feature System - Phases 15-19 (shipped 2026-02-14)
- v1.4 Developer Experience & Code Clarity - Phases 20-24 (in progress)

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

<details>
<summary>v1.3 Composable Array Feature System (Phases 15-19) - SHIPPED 2026-02-14</summary>

### Phase 15: ArrayFeature Infrastructure
**Goal**: A standardized ArrayFeature base class with `@register_feature_type` decorator and composition layer exists, enabling features to be declared as subclasses and auto-composed into ego-centric observation vectors
**Depends on**: Phase 14 (v1.2 complete)
**Requirements**: FEAT-01, FEAT-02, FEAT-03, FEAT-04
**Plans:** 2/2 complete

### Phase 16: Core ArrayFeature Subclasses
**Goal**: The generic feature extractors (agent direction, position, movement, inventory) are wrapped as ArrayFeature subclasses registered to the global scope, proving the pattern works end-to-end with real feature functions
**Depends on**: Phase 15
**Requirements**: CORE-01, CORE-02, CORE-03, CORE-04
**Plans:** 1/1 complete

### Phase 17: Overcooked ArrayFeature Subclasses
**Goal**: Every Overcooked-specific feature function is wrapped as an individual ArrayFeature subclass registered to the "overcooked" scope, producing numerically identical outputs to the existing loose functions
**Depends on**: Phase 16
**Requirements**: OVCK-01, OVCK-02, OVCK-03, OVCK-04, OVCK-05, OVCK-06, OVCK-07, OVCK-08
**Plans:** 2/2 complete

### Phase 18: Autowire Integration & Parity
**Goal**: The autowire module discovers registered features for a scope, composes them into the observation function, CoGridEnv uses this composed function, and Overcooked produces the exact same 677-dim observation as before
**Depends on**: Phase 17
**Requirements**: OVCK-09, INTG-01, INTG-02, INTG-03
**Plans:** 2/2 complete

### Phase 18.1: Remove environment-specific logic from core files (INSERTED)
**Goal:** Zero environment-specific logic in cogrid_env.py or any file under cogrid/core/ -- all domain knowledge pushed to domain modules via registration APIs
**Depends on:** Phase 18
**Plans:** 1/1 complete

### Phase 19: Legacy Feature System Removal
**Goal**: The old OOP feature system is deleted, `build_feature_fn` is removed from the GridObject component classmethod convention, and the codebase has a single code path for features
**Depends on**: Phase 18
**Requirements**: CLNP-01, CLNP-02, CLNP-03, CLNP-04
**Plans:** 1/1 complete

</details>

### v1.4 Developer Experience & Code Clarity (In Progress)

**Milestone Goal:** Make the codebase instantly readable to someone with cursory experience -- minimal complexity, clean imports, well-structured files, consistent naming. Zero behavioral changes -- all existing tests must pass.

**Key Constraints:**
- Zero behavioral changes -- all tests must pass after every phase
- Single code path -- `xp` operations everywhere
- JAX optional -- numpy backend works without JAX installed
- JIT/vmap compatible -- component methods produce JIT-compilable code
- Import paths may change (full restructure approved)

**Phase Numbering:**
- Integer phases (20, 21, 22, 23, 24): Planned milestone work
- Decimal phases (e.g. 20.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 20: Imports & Backend Cleanup** - Module-level xp/dispatch imports everywhere, backend conditionals reduced to structural minimum
- [ ] **Phase 21: File Restructuring** - Rendering extracted, grid_object.py split, cogrid_env.py init/reset decomposed
- [ ] **Phase 22: Function Decomposition** - move_agents() and overcooked_interaction_body() broken into named sub-functions
- [ ] **Phase 23: Naming & Consistency** - state_dict renamed to state, no abbreviations, consistent terminology
- [ ] **Phase 24: Cleanup Pass** - Stale TODOs resolved, dead code removed, docstrings trimmed, full test parity verified

## Phase Details

### Phase 20: Imports & Backend Cleanup
**Goal**: Every file follows standard Python import conventions -- `xp` and `get_backend` imported at module level, backend-conditional blocks are the only place raw `jax`/`numpy` imports appear, and those blocks are reduced to the structural minimum where the two backends genuinely differ
**Depends on**: Phase 19 (v1.3 complete)
**Requirements**: IMP-01, IMP-02, IMP-03, IMP-04
**Success Criteria** (what must be TRUE):
  1. Running `grep -r "from cogrid.backend import xp" --include="*.py"` inside any function body returns zero hits -- all `xp` imports are at module level
  2. Running `grep -r "from cogrid.backend._dispatch import get_backend" --include="*.py"` inside any function body returns zero hits -- all dispatch imports are at module level
  3. Raw `import jax` or `import numpy` statements appear only inside `if get_backend() == "jax"` conditional blocks (RNG splitting, stop_gradient, pytree registration, immutable array workarounds) -- nowhere else
  4. Backend conditional checks (`get_backend() == "jax"`) exist in at most 6 locations, each corresponding to a genuine behavioral difference between backends (not a convenience shortcut)
  5. All existing tests pass without modification
**Plans:** 3 plans
Plans:
- [ ] 20-01-PLAN.md -- BackendProxy and module-level xp imports in core/ files
- [ ] 20-02-PLAN.md -- Module-level xp imports in feature and env-specific files
- [ ] 20-03-PLAN.md -- Consolidate backend conditionals to 6 locations

### Phase 21: File Restructuring
**Goal**: Large files are split along clear responsibility boundaries -- rendering lives in its own module, grid object concerns are separated, and cogrid_env.py is decomposed into focused methods that each do one thing
**Depends on**: Phase 20
**Requirements**: STRC-01, STRC-02, STRC-03
**Success Criteria** (what must be TRUE):
  1. `cogrid_env.py` contains zero PyGame/rendering logic -- all rendering code lives in a dedicated module (e.g. `cogrid/rendering/`) that cogrid_env.py imports and delegates to
  2. `grid_object.py` no longer exists as a monolith -- registration machinery lives in one file, the GridObject base class in another, and concrete object definitions (Wall, Floor, etc.) in a third
  3. `CoGridEnv.__init__` and `CoGridEnv.reset` each read as a sequence of clearly named method calls -- no method exceeds ~50 lines
  4. All existing imports that referenced the old locations still resolve (either via re-exports or the files genuinely moved)
  5. All existing tests pass without modification
**Plans**: TBD

### Phase 22: Function Decomposition
**Goal**: The two longest monolithic functions are broken into named sub-functions that each handle one concern, making the logic scannable and each piece independently testable
**Depends on**: Phase 21
**Requirements**: FUNC-01, FUNC-02
**Success Criteria** (what must be TRUE):
  1. `move_agents()` is a short orchestrator (~20-30 lines) that calls named sub-functions for direction update, position computation, collision resolution, and state mutation -- each sub-function has a clear name describing what it does
  2. `overcooked_interaction_body()` is a short dispatcher that calls named per-object-type handlers (e.g. `_interact_with_pot`, `_interact_with_counter`, `_interact_with_serving_loc`) -- each handler encapsulates the logic for one object type
  3. No individual sub-function exceeds ~50 lines
  4. All existing tests pass without modification
**Plans**: TBD

### Phase 23: Naming & Consistency
**Goal**: A developer reading any function signature or variable name immediately understands what it refers to -- no ambiguity from abbreviations, no confusion from inconsistent terminology, no legacy naming that no longer matches the architecture
**Depends on**: Phase 22
**Requirements**: NAME-01, NAME-02, NAME-03
**Success Criteria** (what must be TRUE):
  1. The parameter name `state_dict` does not appear anywhere in the codebase -- all feature, reward, and termination functions use `state` as the parameter name for the environment state
  2. No function signature contains single-letter parameter names (except standard loop variables `i`, `j`, `k` and established conventions like `n` for count in local scope) -- grep of `def .*\b[a-z]\b:` in signatures returns zero hits outside loop vars
  3. Terminology is consistent: one convention used throughout (e.g. `n_agents` everywhere, not sometimes `num_agents`; `obs` everywhere, not sometimes `observation`) -- a grep for the deprecated variant returns zero hits
  4. All existing tests pass without modification
**Plans**: TBD

### Phase 24: Cleanup Pass
**Goal**: The codebase contains zero stale artifacts -- every TODO is actionable or gone, every line of code is reachable and serves a purpose, every docstring adds information the code does not already convey, and the full test suite confirms zero behavioral regression
**Depends on**: Phase 23
**Requirements**: CLNP-01, CLNP-02, CLNP-03, CLNP-04
**Success Criteria** (what must be TRUE):
  1. `grep -r "TODO" --include="*.py"` returns zero stale TODOs -- each remaining TODO (if any) has an associated issue or is tagged with a specific phase/milestone
  2. No dead code remains: unused imports (flagged by a linter or manual audit), unreachable branches, and commented-out code blocks are all removed
  3. Docstrings are concise and informative -- no docstring restates what the function signature already says (e.g. no "Args: x: the x parameter"), no boilerplate template docstrings
  4. The full test suite (125+ tests + 5 overcooked env tests) passes with zero modifications to test assertions -- confirming zero behavioral changes across the entire v1.4 milestone
  5. A clean `git diff --stat` against the v1.3 tag shows only refactoring changes (no new features, no changed test assertions)
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 20 -> 21 -> 22 -> 23 -> 24

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
| 18.1. Remove environment-specific logic from core files | v1.3 | 1/1 | Complete | 2026-02-14 |
| 19. Legacy Feature System Removal | v1.3 | 1/1 | Complete | 2026-02-14 |
| 20. Imports & Backend Cleanup | v1.4 | 0/3 | Not started | - |
| 21. File Restructuring | v1.4 | 0/? | Not started | - |
| 22. Function Decomposition | v1.4 | 0/? | Not started | - |
| 23. Naming & Consistency | v1.4 | 0/? | Not started | - |
| 24. Cleanup Pass | v1.4 | 0/? | Not started | - |
