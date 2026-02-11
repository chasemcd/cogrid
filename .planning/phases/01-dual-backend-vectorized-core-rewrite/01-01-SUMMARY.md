---
phase: 01-dual-backend-vectorized-core-rewrite
plan: 01
subsystem: core
tags: [numpy, jax, backend-dispatch, type-registry, lookup-tables, decorator]

# Dependency graph
requires: []
provides:
  - "cogrid.backend module with xp, set_backend, get_backend for array backend dispatch"
  - "@register_object_type decorator for static property metadata on GridObj subclasses"
  - "build_lookup_tables(scope) function returning integer-indexed property arrays"
  - "_OBJECT_TYPE_PROPERTIES registry mapping (scope, object_id) to boolean properties"
affects: [01-02, 01-03, 01-04, 01-05, 01-06, 01-07]

# Tech tracking
tech-stack:
  added: [cogrid.backend]
  patterns: [backend-dispatch-via-xp, lazy-getattr-module-resolution, decorator-based-type-registry, integer-indexed-lookup-tables]

key-files:
  created:
    - cogrid/backend/__init__.py
    - cogrid/backend/_dispatch.py
  modified:
    - cogrid/core/grid_object.py
    - cogrid/envs/overcooked/overcooked_grid_objects.py

key-decisions:
  - "Used __getattr__ lazy resolution in cogrid/backend/__init__.py so 'from cogrid.backend import xp' always returns current backend even after set_backend() call"
  - "Separate 1D int32 arrays per property (CAN_PICKUP, CAN_OVERLAP, etc.) rather than a single properties matrix"
  - "build_lookup_tables handles numpy and JAX arrays uniformly via .at[].set() detection"
  - "free_space (idx 1) handled as hardcoded overlappable entry since it is not in OBJECT_REGISTRY"

patterns-established:
  - "Backend access: from cogrid.backend import xp -- lazy resolved via __getattr__"
  - "Object registration: @register_object_type(id, scope, **props) decorator replaces register_object() calls"
  - "Lookup table creation: build_lookup_tables(scope) at environment init time, not at import time"
  - "Backward compatibility: old register_object() still works for undecorated scopes (search_rescue)"

# Metrics
duration: 25min
completed: 2026-02-11
---

# Phase 01 Plan 01: Backend Dispatch & Type Registry Summary

**numpy/jax backend dispatch module with lazy xp resolution, @register_object_type decorator on 16 GridObj subclasses, and build_lookup_tables() producing integer-indexed property arrays**

## Performance

- **Duration:** 25 min
- **Started:** 2026-02-11T15:00:59Z
- **Completed:** 2026-02-11T15:26:40Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created cogrid.backend package with xp/set_backend/get_backend -- array backend dispatch that defaults to numpy and supports jax.numpy
- Added @register_object_type decorator that replaces manual register_object() calls while maintaining full backward compatibility
- Built build_lookup_tables(scope) returning CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON, CAN_PICKUP_FROM, IS_WALL as int32 arrays matching existing object_to_idx encoding
- Annotated all 5 global + 11 Overcooked GridObj subclasses with the decorator

## Task Commits

Each task was committed atomically:

1. **Task 1: Create backend dispatch module** - `64e7978` (feat)
2. **Task 2: Add @register_object_type decorator and build_lookup_tables** - `3ac2b39` (feat)

## Files Created/Modified
- `cogrid/backend/__init__.py` - Package init with lazy __getattr__ for xp resolution, re-exports set_backend/get_backend
- `cogrid/backend/_dispatch.py` - Backend dispatch logic: xp module reference, set_backend(), get_backend()
- `cogrid/core/grid_object.py` - Added _OBJECT_TYPE_PROPERTIES, register_object_type(), build_lookup_tables(), _np_set(); decorated Wall, Floor, Counter, Key, Door
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Decorated all 11 Overcooked GridObj subclasses, removed manual register_object() calls

## Decisions Made
- Used __getattr__ lazy resolution pattern for xp in __init__.py rather than requiring set_backend() before all imports -- this is the cleanest solution to Python's from-import binding semantics
- Chose separate 1D arrays per property (not a single matrix) -- aligns with research recommendation and is more cache-friendly for individual property lookups
- Handled free_space as hardcoded overlappable entry in build_lookup_tables since it exists in get_object_names() but is not in OBJECT_REGISTRY
- No FreeSpace class exists (plan referenced one) -- free_space is a special hardcoded name handled directly in build_lookup_tables

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] No FreeSpace class exists**
- **Found during:** Task 2 (annotating GridObj subclasses)
- **Issue:** Plan asked to annotate FreeSpace class with @register_object_type("free_space", can_overlap=True), but no FreeSpace class exists. "free_space" is a hardcoded entry in get_object_names() at index 1.
- **Fix:** Handle free_space directly in build_lookup_tables() by setting CAN_OVERLAP[1] = 1, same as None at index 0.
- **Files modified:** cogrid/core/grid_object.py
- **Verification:** tables['CAN_OVERLAP'][1] == 1 confirmed
- **Committed in:** 3ac2b39 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor -- hardcoded handling achieves the same result as the planned decorator approach.

## Issues Encountered
- Pre-existing test failures in test_gridworld_env.py (missing FIXED_GRIDS import) and test_overcooked_env.py (requires pygame for render_mode="human") -- neither related to our changes. The one passing test (cogrid/tests/) continues to pass.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Backend dispatch module ready for all subsequent plans to use `from cogrid.backend import xp`
- Type registry and lookup tables ready for vectorized state representation (Plan 02) and vectorized operations (Plans 03-06)
- search_rescue scope objects remain on old register_object() pattern -- will need decorator migration if that scope gets vectorized

---
*Phase: 01-dual-backend-vectorized-core-rewrite*
*Completed: 2026-02-11*
