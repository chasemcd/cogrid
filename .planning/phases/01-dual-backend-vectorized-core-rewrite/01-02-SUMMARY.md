---
phase: 01-dual-backend-vectorized-core-rewrite
plan: 02
subsystem: core
tags: [numpy, array-state, vectorized-representation, grid-conversion, agent-arrays, direction-vectors]

# Dependency graph
requires:
  - "01-01: cogrid.backend module with xp dispatch, object_to_idx / get_object_names type registry"
provides:
  - "layout_to_array_state(grid, scope) producing object_type_map, object_state_map, wall_map, pot_contents, pot_timer arrays"
  - "create_agent_arrays(env_agents, scope) producing agent_pos, agent_dir, agent_inv arrays with -1 sentinel"
  - "grid_to_array_state(grid, env_agents, scope) convenience wrapper combining grid and agent arrays"
  - "get_dir_vec_table() returning (4,2) int32 direction vector lookup indexed by direction enum"
  - "sync_arrays_to_agents(arrays, env_agents) for writing pos/dir back to Agent objects during Phase 1 transition"
affects: [01-03, 01-04, 01-05, 01-06, 01-07]

# Tech tracking
tech-stack:
  added: []
  patterns: [grid-to-array-conversion, agent-array-representation, sentinel-values, lazy-direction-table]

key-files:
  created: []
  modified:
    - cogrid/core/grid_utils.py
    - cogrid/core/agent.py

key-decisions:
  - "Empty grid cells use 0 in object_type_map (matching object_to_idx(None)==0), NOT -1"
  - "pot_contents and agent_inv use -1 sentinel for empty slots per locked encoding conventions"
  - "DIR_VEC_TABLE initialized lazily via get_dir_vec_table() to avoid import-time backend dependency"
  - "Agent arrays sorted by agent_id for deterministic ordering -- agent_ids list maps array index to AgentID"
  - "Spawn points not extractable from Grid (removed before decode) -- callers use env.spawn_points"

patterns-established:
  - "Array state from Grid: layout_to_array_state(grid, scope) at any point to snapshot grid into arrays"
  - "Agent arrays: create_agent_arrays(env_agents, scope) with sorted key ordering"
  - "Direction lookup: get_dir_vec_table()[dir_int] returns [delta_row, delta_col]"
  - "Sync back: sync_arrays_to_agents(arrays, env_agents) for pos/dir writeback during dual-representation period"

# Metrics
duration: 3min
completed: 2026-02-11
---

# Phase 01 Plan 02: Array State Representation Summary

**layout_to_array_state and create_agent_arrays converting Grid/Agent objects to parallel int32 array representation with pot_contents/inventory -1 sentinels and (4,2) direction vector lookup table**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-11T15:29:38Z
- **Completed:** 2026-02-11T15:32:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created layout_to_array_state() converting Grid objects to object_type_map, object_state_map, wall_map, pot_contents, pot_timer arrays
- Created create_agent_arrays() converting Agent objects to agent_pos (n,2), agent_dir (n,), agent_inv (n,1) arrays
- Added grid_to_array_state() convenience wrapper combining grid and agent array conversion
- Added get_dir_vec_table() returning lazy-initialized (4,2) direction vector lookup table matching existing Agent.dir_vec
- Added sync_arrays_to_agents() for writing array state back to Agent objects during Phase 1 transition period

## Task Commits

Each task was committed atomically:

1. **Task 1: Create layout_to_array_state function** - `a7c00fa` (feat)
2. **Task 2: Create agent array state functions** - `5b18677` (feat)

## Files Created/Modified
- `cogrid/core/grid_utils.py` - Added layout_to_array_state() and grid_to_array_state() functions for converting Grid to array state
- `cogrid/core/agent.py` - Added create_agent_arrays(), sync_arrays_to_agents(), get_dir_vec_table(), DIR_VEC_TABLE

## Decisions Made
- Empty grid cells use type_id=0 (matching object_to_idx(None)==0) per research Pitfall 4 -- NOT -1
- pot_contents uses -1 sentinel for empty ingredient slots; agent_inv uses -1 sentinel for empty inventory
- DIR_VEC_TABLE is lazily initialized via get_dir_vec_table() to avoid requiring backend to be set at import time
- Agent arrays are sorted by agent_id key for deterministic array index ordering
- Spawn points are not extractable from the Grid object (they are removed and replaced with FreeSpace before Grid.decode) -- the spawn_points field returns empty list; callers should use env.spawn_points from CoGridEnv

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Plan verification script referenced `get_layout('cramped_room')` but actual registered name is `overcooked_cramped_room_v0` -- used correct name in verification
- Pre-existing test failures persist (FIXED_GRIDS import in test_gridworld_env.py, pygame requirement in test_overcooked_env.py) -- neither related to our changes

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Array state representation ready for all subsequent vectorized operation plans (03-07)
- layout_to_array_state can be called on any Grid to produce the array snapshot needed by vectorized movement, interactions, observations, and rewards
- Direction vector lookup table ready for vectorized movement in Plan 03
- sync_arrays_to_agents available for writing back during Phase 1 dual-representation period

## Self-Check: PASSED

- All files exist (grid_utils.py, agent.py, SUMMARY.md)
- All commits found (a7c00fa, 5b18677)
- All functions importable (layout_to_array_state, grid_to_array_state, create_agent_arrays, sync_arrays_to_agents, get_dir_vec_table)

---
*Phase: 01-dual-backend-vectorized-core-rewrite*
*Completed: 2026-02-11*
