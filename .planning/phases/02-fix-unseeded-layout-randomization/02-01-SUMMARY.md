# Phase 2 Plan 1: Thread np_random to Layout Functions Summary

## Frontmatter

```yaml
phase: 02
plan: 01
subsystem: environment-initialization
tags: [determinism, rng, layout]
dependency-graph:
  requires: [01-01]  # Step dynamics fixed first
  provides: [seeded-layout-selection]
  affects: []
tech-stack:
  added: []
  patterns: [rng-threading]
key-files:
  created: []
  modified:
    - cogrid/cogrid_env.py
    - cogrid/envs/__init__.py
    - cogrid/tests/test_determinism.py
decisions:
  - id: backwards-compat-fallback
    choice: Fallback to stdlib random when np_random not passed
    rationale: Maintains API compatibility for external layout_fn implementations
metrics:
  duration: ~2 min
  completed: 2026-01-20
```

## One-liner

Fixed unseeded layout randomization by threading np_random through layout_fn, ensuring deterministic layout selection when seed is set.

## What Was Done

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Pass np_random to layout_fn | 886389d | cogrid/cogrid_env.py |
| 2 | Update randomized_layout_fn to use np_random | 22be6bb | cogrid/envs/__init__.py |
| 3 | Add determinism test for RandomizedLayout | 5ecf14a | cogrid/tests/test_determinism.py |
| 4 | Run verification tests | - | 3 tests pass |

## Technical Details

### Problem

`randomized_layout_fn` used Python's `random.choice()` which is not connected to the environment seed:

```python
def randomized_layout_fn(**kwargs):
    layout_name = random.choice([...])  # BUG: unseeded
```

### Solution

1. **Thread np_random through**: cogrid_env.py now passes `np_random=self.np_random` when calling `layout_fn`

2. **Use seeded RNG**: `randomized_layout_fn` now accepts and uses the passed np_random:

```python
def randomized_layout_fn(np_random=None, **kwargs):
    if np_random is None:
        import random as stdlib_random
        layout_name = stdlib_random.choice(...)  # Backwards compat
    else:
        layout_name = np_random.choice(...)  # Deterministic
```

3. **Removed module-level `import random`**: No longer needed at module level

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Backwards compatibility | Fallback to stdlib random | External layout_fn implementations may not pass np_random |

## Verification Results

```
3 passed in 0.57s
- TestRandomizedLayoutDeterminism::test_randomized_layout_deterministic PASSED
- TestStepDeterminism::test_collision_resolution_deterministic PASSED
- TestStepDeterminism::test_identical_actions_produce_identical_states PASSED
```

Serialization tests: 33 passed

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Phase 02 complete. The unseeded layout randomization bug is fixed.

Remaining randomness audit items:
- envs/search_rescue/sr_utils.py:21 - legacy RandomState fallback (lower priority)
