---
phase: 06-testing
plan: 02
subsystem: testing
tags:
  - integration-tests
  - determinism
  - serialization

dependency-graph:
  requires:
    - 05-01 (environment serialization tests)
    - 01-02 (RedVictim serialization)
  provides:
    - Centralized integration test file for serialization
    - Extended determinism tests (50+ step sequences)
    - Cross-environment validation tests
  affects:
    - Future serialization changes must pass these integration tests

tech-stack:
  added: []
  patterns:
    - Parametrized pytest tests for cross-environment validation
    - Extended action sequence determinism verification

file-tracking:
  key-files:
    created:
      - cogrid/tests/__init__.py
      - cogrid/tests/test_serialization_integration.py
    modified:
      - cogrid/core/grid_object.py (Door.encode scope parameter bug fix)

decisions:
  - key: door-encode-scope-fix
    choice: Fixed Door.encode() to accept scope parameter
    reason: Bug blocking S&R serialization tests
    context: Task 1

metrics:
  duration: ~2m
  completed: 2026-01-19
---

# Phase 6 Plan 02: Integration Tests Summary

Integration tests for extended determinism and cross-environment serialization validation, with 7 new tests covering 50+ step sequences and boundary conditions.

## What Was Done

### Task 1: Create tests directory and integration test file
Created `cogrid/tests/` directory with centralized integration tests:

- **TestDeterminismExtended**: Two tests running 50-step sequences for Overcooked and S&R environments, verifying rewards and termination flags match exactly after checkpoint/restore
- **TestObservationSpaceMatch**: Verifies observation array values (not just shapes) match after restore

**Bug Fix Applied (Rule 1):** Fixed `Door.encode()` missing `scope` parameter - this was blocking S&R serialization tests. The base `GridObject.encode()` accepts `scope` but `Door.encode()` override did not pass it through.

### Task 2: Add cross-environment and edge case tests
Added validation and edge case coverage:

- **TestCrossEnvironmentValidation**: Parametrized test verifying both `Overcooked-CrampedRoom-V0` and `SearchRescue-Test-V0` support get_state/set_state roundtrip
- **TestEdgeCaseSerialization**:
  - `test_max_steps_boundary`: Verifies truncation triggers correctly after restoring at t=max_steps-1
  - `test_grid_state_preserved`: Verifies full grid encoding matches after restore

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed Door.encode() missing scope parameter**
- **Found during:** Task 1 test execution
- **Issue:** `Door.encode()` did not accept `scope` parameter, but base class `GridObject.encode()` does. When grid serialization called `encode(scope=...)` on Door objects in S&R environment, it failed with TypeError.
- **Fix:** Added `scope: str = "global"` parameter to `Door.encode()` and passed through to `super().encode()`
- **Files modified:** cogrid/core/grid_object.py
- **Commit:** 6d71646

## Verification Results

Full serialization test suite: **69 tests passed**

```
cogrid/tests/test_serialization_integration.py: 7 tests
cogrid/envs/overcooked/test_state_serialization.py: 40 tests
cogrid/envs/search_rescue/test_sr_objects_serialization.py: 22 tests
```

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| 6d71646 | feat | Add extended determinism and observation matching tests |
| 5ab11af | feat | Add cross-environment and edge case serialization tests |

## Key Files

**Created:**
- `cogrid/tests/__init__.py` - Package init
- `cogrid/tests/test_serialization_integration.py` (275 lines) - Centralized integration tests

**Modified:**
- `cogrid/core/grid_object.py` - Door.encode() scope parameter fix

## Test Classes Created

| Class | Tests | Purpose |
|-------|-------|---------|
| TestDeterminismExtended | 2 | 50+ step determinism for Overcooked and S&R |
| TestObservationSpaceMatch | 1 | Observation array value verification |
| TestCrossEnvironmentValidation | 2 (parametrized) | Roundtrip validation for all registered envs |
| TestEdgeCaseSerialization | 2 | max_steps boundary, grid state preservation |

## Next Phase Readiness

Phase 6 testing complete. All serialization tests pass (69 total). The integration test file provides a single entry point for verifying serialization works correctly across environments and extended action sequences.
