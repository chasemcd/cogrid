---
phase: 05-foundation-state-model-backend-helpers
plan: 02
subsystem: core
tags: [layout-parser, scope-config, symbol-registry, extra-state, overcooked]

# Dependency graph
requires:
  - phase: 05-foundation-state-model-backend-helpers
    plan: 01
    provides: "EnvState with generic extra_state dict, create_env_state(), validate_extra_state()"
provides:
  - "cogrid/core/layout_parser.py with register_symbols, get_symbols, parse_layout"
  - "scope_config extended with symbol_table, extra_state_schema, extra_state_builder keys"
  - "Overcooked symbol table, extra_state schema, and build_overcooked_extra_state builder"
  - "ASCII layout to EnvState parsing without Grid/Agent objects"
affects: [05-03, phase-06, phase-07, phase-08, phase-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Symbol registry for character-to-meaning mappings in scope configs"
    - "extra_state_builder callable pattern for environment-specific array init from parsed layout"
    - "parse_layout uses numpy always (init-time), converts to JAX arrays at the end"

key-files:
  created:
    - cogrid/core/layout_parser.py
  modified:
    - cogrid/core/scope_config.py
    - cogrid/envs/overcooked/array_config.py
    - cogrid/envs/overcooked/__init__.py

key-decisions:
  - "Symbol table lives in both scope_config and explicit SYMBOL_REGISTRY for flexibility"
  - "Dual registration: symbols registered in both scope_config and layout_parser registry"
  - "parse_layout always uses numpy for parsing, converts to JAX arrays only at the end"

patterns-established:
  - "Layout parsing via symbol_table in scope_config -- no Grid/Agent object creation"
  - "extra_state_builder callable produces scope-prefixed extra_state from parsed arrays"
  - "get_symbols checks explicit registry first, falls back to scope_config symbol_table"

# Metrics
duration: 3min
completed: 2026-02-12
---

# Phase 5 Plan 02: Layout Parser & Scope Config Extensions Summary

**Array-based layout parser that converts ASCII grids directly into EnvState with Overcooked symbol table and pot extra_state builder -- no Grid/Agent objects involved**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-12T21:05:14Z
- **Completed:** 2026-02-12T21:08:08Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Created layout_parser.py that parses ASCII layout strings into fully initialized EnvState
- Extended scope_config with symbol_table, extra_state_schema, and extra_state_builder keys
- Built Overcooked-specific extra_state builder that detects pots from object_type_map and creates pot arrays
- Registered Overcooked symbol table in both scope_config and explicit layout_parser registry
- Verified end-to-end: cramped_room layout parses into correct 6x7 grid with 1 pot, correct wall_map, and populated extra_state

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend scope_config and create layout_parser.py** - `3a14978` (feat)
2. **Task 2: Register Overcooked symbols, schema, and extra_state builder** - `a27967b` (feat)

## Files Created/Modified
- `cogrid/core/layout_parser.py` - NEW: Array-based layout parser with symbol registry, parse_layout returns EnvState
- `cogrid/core/scope_config.py` - Added symbol_table, extra_state_schema, extra_state_builder to default_scope_config()
- `cogrid/envs/overcooked/array_config.py` - Added build_overcooked_extra_state() and v1.1 keys to scope config
- `cogrid/envs/overcooked/__init__.py` - Added register_symbols() call for Overcooked symbol table

## Decisions Made
- Dual registration approach: symbols in both scope_config (for parse_layout to find via fallback) and explicit SYMBOL_REGISTRY (for direct access before scope_config is loaded)
- parse_layout always uses numpy for grid iteration (Python loops are fine at init-time), then bulk-converts to JAX arrays at the end
- Spawn positions ('+' chars) are NOT placed in object_type_map -- they only populate agent_pos

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- layout_parser.py ready for use by reset/init paths that need to convert ASCII layouts to EnvState
- Overcooked scope config now has all v1.1 keys needed by the parser and future step function
- Plan 03 can now build on top of parse_layout for the full env reset flow

## Self-Check: PASSED

All 4 created/modified files verified present. Both task commits (3a14978, a27967b) verified in git log.

---
*Phase: 05-foundation-state-model-backend-helpers*
*Completed: 2026-02-12*
