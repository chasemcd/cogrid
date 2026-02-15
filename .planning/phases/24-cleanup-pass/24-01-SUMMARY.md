---
phase: 24-cleanup-pass
plan: 01
subsystem: codebase
tags: [cleanup, imports, dead-code, technical-debt]

# Dependency graph
requires:
  - phase: 23-naming-conventions
    provides: standardized parameter names across codebase
provides:
  - clean codebase with zero stale TODOs, no dead code, no unused imports
affects: [24-02]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Keep from __future__ import annotations only in files with forward references"

key-files:
  created: []
  modified:
    - cogrid/cogrid_env.py
    - cogrid/core/agent.py
    - cogrid/core/grid.py
    - cogrid/core/grid_object_base.py
    - cogrid/core/grid_objects.py
    - cogrid/envs/overcooked/overcooked_grid_objects.py
    - cogrid/run_interactive.py

key-decisions:
  - "Retained from __future__ import annotations in 3 files that need forward references (grid.py, component_registry.py, grid_object_base.py)"
  - "Removed annotations import from 20 files where Python 3.11 native syntax suffices"

patterns-established:
  - "Forward reference rule: only use from __future__ import annotations when class forward references require it"

# Metrics
duration: 25min
completed: 2026-02-15
---

# Phase 24 Plan 01: Stale Code Cleanup Summary

**Removed 8 TODO comments, 3 PHASE2 markers, 4 commented-out blocks, and 68 unused import lines across 35 files**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-02-15
- **Completed:** 2026-02-15
- **Tasks:** 2
- **Files modified:** 35

## Accomplishments
- Eliminated all stale TODO/PHASE2 comments and dead commented-out code from 8 files
- Removed unused imports from 27 production and test files (68 import lines removed)
- Verified all 131 tests pass after cleanup (125 core + 6 overcooked)

## Task Commits

Each task was committed atomically:

1. **Task 1: Resolve stale TODOs, PHASE2 comments, and commented-out code** - `3a1783e` (chore)
2. **Task 2: Remove unused imports** - `576c200` (chore)

## Files Created/Modified

### Task 1 (stale comments and dead code)
- `cogrid/test_overcooked_env.py` - Removed TODO comment
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Removed TODO comment
- `cogrid/core/grid.py` - Removed 2 TODO comments
- `cogrid/core/grid_objects.py` - Removed TODO comment
- `cogrid/core/grid_object_base.py` - Removed 2 TODO comments
- `cogrid/core/agent.py` - Removed TODO, 2 PHASE2 comments, commented-out dir_to_vec block
- `cogrid/core/grid_utils.py` - Removed PHASE2 comment
- `cogrid/run_interactive.py` - Removed commented-out onnxruntime/scipy imports, onnx inference call, print statement

### Task 2 (unused imports)
- `cogrid/cogrid_env.py` - Removed collections, combinations, duplicate directions, sync_arrays_to_agents, get_dir_vec_table, move_agents
- `cogrid/constants.py` - Removed numpy import
- `cogrid/core/grid.py` - Removed Callable from typing, grid_object module import
- `cogrid/core/grid_object_base.py` - Removed Colors, constants, point_in_circle
- `cogrid/core/grid_object_registry.py` - Removed annotations import, GridConstants
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Removed annotations, deepcopy, math, GridConstants, COLORS, point_in_rect, point_in_triangle, rotate_fn
- `cogrid/envs/search_rescue/search_rescue_grid_objects.py` - Removed GridConstants, ObjectColors, COLORS
- `cogrid/test_overcooked_env.py` - Removed time, numpy, Directions, grid_object
- `cogrid/envs/overcooked/test_interactions.py` - Removed annotations, copy, registry, get_object_names, layout_to_array_state, create_agent_arrays, grid_actions
- `cogrid/tests/test_autowire.py` - Removed pytest
- 17 additional files - Removed `from __future__ import annotations` where not needed

## Decisions Made
- Kept `from __future__ import annotations` in 3 files that require it for forward references: `grid.py` (Grid class self-reference), `component_registry.py` (`callable | None` syntax), `grid_object_base.py` (GridAgent forward reference)
- Removed it from 20 files where Python 3.11 native type union syntax suffices

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Re-added annotations import to component_registry.py**
- **Found during:** Task 2 (unused import removal)
- **Issue:** Removing `from __future__ import annotations` caused `TypeError: unsupported operand type(s) for |: 'builtin_function_or_method' and 'NoneType'` because `callable` (lowercase) is a builtin function, not a type, and `callable | None` union syntax requires deferred annotation evaluation
- **Fix:** Re-added `from __future__ import annotations` to component_registry.py
- **Files modified:** cogrid/core/component_registry.py
- **Verification:** Module imports successfully, all tests pass
- **Committed in:** 576c200 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor -- single import needed to be kept rather than removed. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Codebase is clean and ready for plan 02 (lint and type annotation cleanup)
- All tests passing, no regressions

---
*Phase: 24-cleanup-pass*
*Completed: 2026-02-15*
