---
phase: 19-legacy-feature-system-removal
plan: 01
subsystem: feature-system
tags: [array-features, autowire, cleanup, dead-code-removal]

# Dependency graph
requires:
  - phase: 18-autowire-integration-parity
    provides: "ArrayFeature subclasses + autowire composition as sole feature path"
  - phase: 18.1-remove-environment-specific-logic-from-core-files
    provides: "Feature order and pre-compose hooks registered in component_registry"
provides:
  - "Single feature code path: ArrayFeature subclasses composed by autowire"
  - "No legacy OOP Feature/FeatureSpace/features classes"
  - "No build_feature_fn in GridObject component convention"
  - "No feature_fn_builder in scope_config"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ArrayFeature subclasses + autowire is the sole feature composition path"
    - "GridObject classmethods: build_tick_fn, build_interaction_fn, extra_state_schema, extra_state_builder, build_static_tables, build_render_sync_fn (no build_feature_fn)"

key-files:
  created: []
  modified:
    - cogrid/feature_space/__init__.py
    - cogrid/feature_space/array_features.py
    - cogrid/core/grid_object.py
    - cogrid/core/component_registry.py
    - cogrid/core/autowire.py
    - cogrid/envs/overcooked/overcooked_grid_objects.py
    - cogrid/envs/overcooked/overcooked_array_features.py
    - cogrid/envs/overcooked/__init__.py
    - cogrid/cogrid_env.py
    - cogrid/tests/test_step_pipeline.py
    - cogrid/tests/test_overcooked_array_features.py
    - cogrid/tests/test_autowire.py
    - cogrid/test_overcooked_env.py
    - docs/content/cogrid_env.rst

key-decisions:
  - "Deleted 4 OOP feature files entirely: feature.py, feature_space.py, features.py, overcooked_features.py"
  - "Removed build_feature_fn from _COMPONENT_METHODS (6 methods remain) and _EXPECTED_SIGNATURES"
  - "Removed legacy FeatureSpace/observation_spaces/get_obs from cogrid_env.py -- vectorized path is sole path"
  - "Removed build_overcooked_feature_fn monolithic builder and compose_features/build_feature_fn from array_features.py"

patterns-established:
  - "ArrayFeature subclasses are the only way to define features -- no legacy Feature base class"
  - "Tests that inspect Grid objects after step() use _sync_objects_from_state() helper"
  - "Test actions use integer indices via _ACTION_IDX mapping, not string Actions"

# Metrics
duration: 8min
completed: 2026-02-14
---

# Phase 19 Plan 01: Legacy Feature System Removal Summary

**Deleted the old OOP Feature/FeatureSpace system, removed build_feature_fn from GridObject convention, leaving ArrayFeature subclasses + autowire as the sole feature composition path with 1500 lines of dead code eliminated**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-14T16:32:50Z
- **Completed:** 2026-02-14T16:41:02Z
- **Tasks:** 2
- **Files modified:** 14 (4 deleted, 10 edited)

## Accomplishments
- Deleted 4 old OOP feature files (feature.py, feature_space.py, features.py, overcooked_features.py) -- 1500+ lines of dead code
- Removed build_feature_fn from _COMPONENT_METHODS, _EXPECTED_SIGNATURES, ComponentMetadata, and scope_config
- Removed Pot.build_feature_fn classmethod and build_overcooked_feature_fn monolithic builder
- Removed legacy FeatureSpace/observation_spaces/get_obs from cogrid_env.py
- Updated all tests to use autowire path -- 112 cogrid/tests/ + 5 test_overcooked_env.py pass

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete old OOP feature files and remove build_feature_fn from component convention** - `0f3b62b` (feat)
2. **Task 2: Update tests, fix remaining references, and verify all tests pass** - `d4ca570` (fix)

## Files Created/Modified
- `cogrid/feature_space/feature.py` - DELETED (old OOP Feature base class)
- `cogrid/feature_space/feature_space.py` - DELETED (old FeatureSpace registry)
- `cogrid/feature_space/features.py` - DELETED (old OOP feature implementations)
- `cogrid/envs/overcooked/overcooked_features.py` - DELETED (old OOP Overcooked features)
- `cogrid/feature_space/__init__.py` - Now only imports array_features
- `cogrid/feature_space/array_features.py` - Removed compose_features() and build_feature_fn()
- `cogrid/core/grid_object.py` - Removed build_feature_fn from _COMPONENT_METHODS
- `cogrid/core/component_registry.py` - Removed build_feature_fn from _EXPECTED_SIGNATURES, removed has_feature_fn property
- `cogrid/core/autowire.py` - Removed feature_fn_builder composition block and key from scope_config
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Removed Pot.build_feature_fn classmethod
- `cogrid/envs/overcooked/overcooked_array_features.py` - Removed build_overcooked_feature_fn monolithic builder
- `cogrid/envs/overcooked/__init__.py` - Removed overcooked_features import
- `cogrid/cogrid_env.py` - Removed feature_space import, FeatureSpace/observation_spaces/get_obs/observation_space
- `cogrid/tests/test_step_pipeline.py` - Replaced build_feature_fn with autowire path
- `cogrid/tests/test_overcooked_array_features.py` - Removed parity/monolithic/autowire feature_fn_builder tests
- `cogrid/tests/test_autowire.py` - Removed feature_fn_builder from required_keys
- `cogrid/test_overcooked_env.py` - Removed old OOP feature imports/class, fixed action indices, added grid sync
- `docs/content/cogrid_env.rst` - Removed feature module autodoc directive

## Decisions Made
- Deleted 4 files entirely rather than gutting them -- cleaner than leaving empty modules
- Removed compose_features() and build_feature_fn() from array_features.py since they were only used by the old path -- autowire uses compose_feature_fns() in cogrid/core/array_features.py
- Removed build_overcooked_feature_fn entirely since it was the monolithic builder replaced by autowire composition in Phase 18
- Kept all test_overcooked_env.py test methods (they test grid object mechanics, not features)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_overcooked_env.py render_mode="human" requiring pygame**
- **Found during:** Task 2 (test updates)
- **Issue:** Tests used render_mode="human" which requires pygame, not available in test environment
- **Fix:** Changed to render_mode=None for tests
- **Files modified:** cogrid/test_overcooked_env.py
- **Verification:** Tests run without pygame dependency
- **Committed in:** d4ca570

**2. [Rule 1 - Bug] Fixed test_overcooked_env.py passing string Actions to vectorized step()**
- **Found during:** Task 2 (test updates)
- **Issue:** Tests passed Actions.MoveLeft (string "move_left") but step() expects integer indices
- **Fix:** Added _ACTION_IDX mapping and used integer indices throughout
- **Files modified:** cogrid/test_overcooked_env.py
- **Verification:** All 5 tests pass with integer action indices
- **Committed in:** d4ca570

**3. [Rule 1 - Bug] Fixed test_overcooked_env.py Grid object sync after vectorized step()**
- **Found during:** Task 2 (test updates)
- **Issue:** Tests inspect self.env.grid.grid_agents after step() but vectorized pipeline only syncs Grid objects on render path (render_mode != None)
- **Fix:** Added _step() helper that calls _sync_objects_from_state() after env.step()
- **Files modified:** cogrid/test_overcooked_env.py
- **Verification:** All 5 tests pass with grid objects correctly synced
- **Committed in:** d4ca570

---

**Total deviations:** 3 auto-fixed (3 Rule 1 bugs in pre-existing test file)
**Impact on plan:** All auto-fixes necessary for test correctness. The test file had pre-existing incompatibilities with the vectorized pipeline that only surfaced now because the old OOP feature system was masking them.

## Issues Encountered
- Pre-existing JAX backend state pollution: running all cogrid/tests/ together causes test_overcooked_array_features to fail after JAX backend tests set the global backend. This is documented in STATE.md and not caused by our changes. Running tests individually or without the cross-backend tests works correctly.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 19 complete -- the legacy feature system is entirely removed
- The codebase has exactly one feature code path: ArrayFeature subclasses composed by autowire
- All v1.3 milestone work is complete

---
*Phase: 19-legacy-feature-system-removal*
*Completed: 2026-02-14*
