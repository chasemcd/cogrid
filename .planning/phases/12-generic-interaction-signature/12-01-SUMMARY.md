---
phase: 12-generic-interaction-signature
plan: 01
subsystem: core
tags: [interactions, extra_state, scope-agnostic, wrapper-pattern]

# Dependency graph
requires:
  - phase: 11-composition-auto-wiring
    provides: "Scope config with interaction_body, extra_state_schema, and extra_state_builder"
provides:
  - "Generic extra_state dict pass-through in process_interactions()"
  - "Wrapper translating dict protocol to positional-arg overcooked_interaction_body"
  - "Explicit extra_state= parameter in step_pipeline process_interactions call"
affects: [13-component-interaction-handlers, 14-cleanup]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "interaction_body generic protocol: (i, inv, otm, osm, fwd_r, fwd_c, fwd_type, inv_item, base_ok, extra_state, static_tables) -> (inv, otm, osm, extra_state)"
    - "_wrap_overcooked_interaction_body adapter translates dict <-> positional args"

key-files:
  created: []
  modified:
    - cogrid/core/interactions.py
    - cogrid/envs/overcooked/array_config.py
    - cogrid/core/step_pipeline.py

key-decisions:
  - "Backward-compat **kwargs preserved in process_interactions() so existing test callers using **es pattern work without modification"
  - "Wrapper lives in array_config.py (Overcooked-specific) not in core"

patterns-established:
  - "Generic interaction_body protocol: extra_state as opaque dict, not positional scope-specific arrays"
  - "Adapter wrapper pattern: scope-specific functions wrapped at config-build time, not called through if/elif dispatch"

# Metrics
duration: 2min
completed: 2026-02-13
---

# Phase 12 Plan 01: Generic Interaction Signature Summary

**Generic extra_state dict pass-through in process_interactions() with backward-compatible Overcooked wrapper adapter**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-13T11:40:10Z
- **Completed:** 2026-02-13T11:42:23Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Removed all Overcooked-specific key names (pot_contents, pot_timer, pot_positions) from cogrid/core/interactions.py
- process_interactions() now accepts/returns extra_state as an opaque dict -- any scope's interaction_body works through the same interface
- _wrap_overcooked_interaction_body() cleanly translates between generic dict protocol and Overcooked's positional-arg protocol
- All 8 Overcooked parity tests and 3 step pipeline tests pass without any test file modifications

## Task Commits

Each task was committed atomically:

1. **Task 1: Generalize process_interactions and create Overcooked wrapper** - `0b8eddc` (feat)
2. **Task 2: Update step_pipeline caller to use explicit extra_state parameter** - `e1c45f2` (feat)

## Files Created/Modified
- `cogrid/core/interactions.py` - Generic extra_state dict signature with **kwargs backward compat; interaction_body call passes dict instead of positional args
- `cogrid/envs/overcooked/array_config.py` - Added _wrap_overcooked_interaction_body() adapter; build_overcooked_scope_config() uses wrapped version
- `cogrid/core/step_pipeline.py` - Renamed extra_kwargs to extra_state; passes extra_state=extra_state instead of **extra_kwargs splat

## Decisions Made
- Preserved **extra_state_kwargs backward compatibility in process_interactions() so all existing test callers using **es pattern work without modification -- zero changes to test files
- Wrapper function placed in array_config.py (next to the function it wraps) rather than in core, keeping scope-specific knowledge out of core modules

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Generic interaction_body protocol established; Phase 13 can implement component-based interaction handlers using the dict protocol directly
- Overcooked interaction_body remains wrapped (not rewritten) -- Phase 13/14 can migrate it to component interface

## Self-Check: PASSED

All artifacts verified:
- 12-01-SUMMARY.md exists
- Commit 0b8eddc (Task 1) exists
- Commit e1c45f2 (Task 2) exists
- extra_state=None in interactions.py signature
- _wrap_overcooked_interaction_body in array_config.py
- extra_state=extra_state in step_pipeline.py
- Zero Overcooked-specific keys in cogrid/core/interactions.py

---
*Phase: 12-generic-interaction-signature*
*Completed: 2026-02-13*
