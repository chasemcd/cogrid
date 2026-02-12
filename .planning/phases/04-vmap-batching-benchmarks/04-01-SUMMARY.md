---
phase: 04-vmap-batching-benchmarks
plan: 01
subsystem: testing
tags: [jax, vmap, pytree, batching, correctness, parity]

# Dependency graph
requires:
  - phase: 03-end-to-end-integration-parity
    provides: "JIT-compiled jax_step and jax_reset with EnvState pytree, cross-backend parity"
provides:
  - "vmap correctness tests verifying batched reset and step at 1024 parallel environments"
  - "Single-env vs batched parity verification for reset and multi-step trajectories"
  - "jit(vmap(fn)) composition verification"
affects: [04-02 benchmark suite]

# Tech tracking
tech-stack:
  added: []
  patterns: ["jax.vmap(fn) for batched env rollouts", "spot-check parity via N_SAMPLE_INDICES"]

key-files:
  created:
    - cogrid/tests/test_vmap_correctness.py
  modified: []

key-decisions:
  - "Spot-check 8 sample indices [0,1,2,3,512,1021,1022,1023] for parity instead of all 1024 for test speed"
  - "rng_key comparison uses jax.random.key_data() to extract underlying integer data from opaque JAX key types"

patterns-established:
  - "vmap parity testing: run vmapped fn, then single-env fn on same inputs, assert_array_equal on sampled slices"

# Metrics
duration: 2min
completed: 2026-02-12
---

# Phase 4 Plan 01: vmap Correctness Test Suite Summary

**13 pytest tests verifying jax.vmap correctness at 1024 parallel environments across 3 Overcooked layouts with single-env vs batched parity**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-12T14:29:23Z
- **Completed:** 2026-02-12T14:31:09Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- vmap reset and step produce correct shapes at 1024 environments (batch dim on obs, rewards, done, all state fields)
- Single-env outputs match corresponding batched slices exactly for both reset (8 indices) and step (5 steps with varied actions, 8 indices)
- jit(vmap(fn)) composition executes without error for reset + 3 steps
- Static meta fields (n_agents, height, width, n_pots, action_set) are NOT batched, confirming proper pytree registration
- All 13 tests pass across all 3 layouts in 8.5 seconds

## Task Commits

Each task was committed atomically:

1. **Task 1: Create vmap correctness test suite at 1024 environments** - `25bc6fa` (feat)

## Files Created/Modified
- `cogrid/tests/test_vmap_correctness.py` - 5 test functions covering vmap reset shapes, step shapes, reset parity, step parity, and jit+vmap composition

## Decisions Made
- Spot-check 8 sample indices (first 4, middle, last 3) for parity instead of exhaustive comparison of all 1024 environments -- balances correctness confidence with test execution time
- Used `jax.random.key_data()` for rng_key field comparison since JAX key types are opaque and cannot be compared directly with `assert_array_equal`

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- vmap correctness verified, ready for Plan 02 (benchmark suite)
- All 3 layouts work with vmap at 1024 environments
- jit(vmap(fn)) pattern confirmed working for performance benchmarking

---
*Phase: 04-vmap-batching-benchmarks*
*Completed: 2026-02-12*
