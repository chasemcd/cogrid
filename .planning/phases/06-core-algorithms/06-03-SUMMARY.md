---
phase: 06-core-algorithms
plan: 03
subsystem: features
tags: [xp, array-features, backend-agnostic, feature-extractors]

# Dependency graph
requires:
  - phase: 05-foundation-state-model-backend-helpers
    provides: "xp dispatch, array_ops.set_at_2d, build_lookup_tables"
provides:
  - "5 unified feature extractors using xp (agent_pos, agent_dir, full_map_encoding, can_move_direction, inventory)"
  - "Unified build_feature_fn() with pre-computed agent_type_ids and can_overlap_table"
  - "Unified get_all_agent_obs() using xp.stack"
  - "Backward-compat aliases (build_feature_fn_jax, get_all_agent_obs_jax)"
affects: [08-step-pipeline, jax_step, cogrid_env]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "xp.pad for backend-agnostic slice assignment avoidance"
    - "set_at_2d loop over static n_agents for agent scatter"
    - "xp.where instead of int() + Python if/else"
    - "Broadcast comparison for one-hot encoding (xp.arange == val)"

key-files:
  created: []
  modified:
    - "cogrid/feature_space/array_features.py"

key-decisions:
  - "full_map_encoding uses xp.pad + xp.stack instead of slice assignment to avoid backend branching"
  - "Agent scatter uses set_at_2d loop over n_agents (static/tiny) instead of fancy indexing"
  - "get_all_agent_obs uses Python loop with xp.stack; vmap deferred to Phase 8"
  - "Backward-compat aliases added for build_feature_fn_jax and get_all_agent_obs_jax"

patterns-established:
  - "xp.pad for building padded arrays without backend-specific slice assignment"
  - "Pre-computed lookup tables (agent_type_ids, can_overlap_table) closed over at init time"

# Metrics
duration: 5min
completed: 2026-02-12
---

# Phase 6 Plan 3: Unified Feature Extractors Summary

**5 unified xp-based feature extractors replacing 10 numpy+JAX duplicates, with zero int() casts and pre-computed lookup tables**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-12T22:01:41Z
- **Completed:** 2026-02-12T22:07:13Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Consolidated 10 feature functions (5 numpy + 5 JAX) into 5 unified implementations using xp
- Consolidated 4 composition functions (build_feature_fn, build_feature_fn_jax, get_all_agent_obs, get_all_agent_obs_jax) into 2
- Eliminated all int() casts from feature function bodies (9 casts removed)
- full_map_encoding uses xp.pad + set_at_2d for backend-agnostic encoding construction
- Removed ~580 lines of duplicate JAX code

## Task Commits

Each task was committed atomically:

1. **Task 1: Unify core feature extractors to use xp** - `7f9b800` (feat)
2. **Task 2: Unify build_feature_fn() and get_all_agent_obs()** - `87d94c2` (feat)

## Files Created/Modified
- `cogrid/feature_space/array_features.py` - 5 unified feature extractors, unified build_feature_fn/get_all_agent_obs, backward-compat aliases

## Decisions Made
- **xp.pad for slice assignment:** Used `xp.pad` to construct padded channel arrays instead of backend-branching slice assignment (`arr[:H, :W] = ...` vs `arr.at[:H, :W].set(...)`). This avoids the anti-pattern of backend branching inside algorithm functions.
- **set_at_2d loop for agent scatter:** Loop over `n_agents` (static, typically 2) using `set_at_2d` for each agent. This uses the established `array_ops` helper and avoids fancy indexing which would require a different backend abstraction.
- **vmap deferred:** `get_all_agent_obs` uses Python loop with `xp.stack`. vmap is JAX-only and deferred to Phase 8 step pipeline optimization.
- **Backward-compat aliases:** Added `build_feature_fn_jax = build_feature_fn` and `get_all_agent_obs_jax = get_all_agent_obs` so callers in `jax_step.py` and `cogrid_env.py` continue working without modification.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated build_feature_fn in Task 1 instead of Task 2**
- **Found during:** Task 1 (feature extractor unification)
- **Issue:** full_map_encoding_feature signature changed from `scope` param to `agent_type_ids` param, but build_feature_fn (Task 2 scope) still passed `scope=s`. Code would not be importable between Task 1 and Task 2 commits.
- **Fix:** Updated build_feature_fn in Task 1 to pass agent_type_ids and pre-compute them with xp. Task 2 then only needed to add aliases and finalize docstrings.
- **Files modified:** cogrid/feature_space/array_features.py
- **Verification:** Import test passed after Task 1 commit
- **Committed in:** 7f9b800 (Task 1 commit)

**2. [Rule 3 - Blocking] Added backward-compat aliases for callers**
- **Found during:** Task 2 (composition function unification)
- **Issue:** `jax_step.py` and `cogrid_env.py` import `build_feature_fn_jax` and `get_all_agent_obs_jax` which no longer exist as separate functions
- **Fix:** Added module-level aliases `build_feature_fn_jax = build_feature_fn` and `get_all_agent_obs_jax = get_all_agent_obs`
- **Files modified:** cogrid/feature_space/array_features.py
- **Verification:** `from cogrid.feature_space.array_features import build_feature_fn_jax` succeeds
- **Committed in:** 87d94c2 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Both fixes necessary to keep codebase importable between task commits. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Feature extractors unified; ready for Phase 6 Plan 4 (int() cast elimination and edge-case tests)
- Callers (jax_step.py, cogrid_env.py) use backward-compat aliases; will be migrated in their respective phases
- xp.pad pattern established for avoiding slice assignment -- can be reused in other encoding functions

## Self-Check: PASSED

- FOUND: cogrid/feature_space/array_features.py
- FOUND: .planning/phases/06-core-algorithms/06-03-SUMMARY.md
- FOUND: commit 7f9b800
- FOUND: commit 87d94c2

---
*Phase: 06-core-algorithms*
*Completed: 2026-02-12*
