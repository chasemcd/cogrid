---
phase: 05-foundation-state-model-backend-helpers
verified: 2026-02-12T21:17:03Z
status: passed
score: 4/4 success criteria verified
re_verification: false
---

# Phase 05: Foundation State Model & Backend Helpers Verification Report

**Phase Goal:** The data layer is ready for unified functional code -- EnvState holds environment-specific arrays generically, a tiny helper module abstracts the only operations where numpy and JAX genuinely differ, and layouts parse directly into arrays without intermediate objects

**Verified:** 2026-02-12T21:17:03Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Based on the phase goal and user-provided success criteria, the following truths must hold:

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | EnvState contains an `extra_state` dict instead of hardcoded pot fields, and the Overcooked scope populates it with `pot_contents`, `pot_timer`, `pot_positions` arrays at init time | ✓ VERIFIED | EnvState dataclass has `extra_state` field. Old fields `pot_contents`, `pot_timer`, `pot_positions`, `n_pots` removed. Overcooked scope config includes `extra_state_builder` that creates scope-prefixed arrays. Layout parser test shows `extra_state` populated with 1 pot at init. |
| 2 | `set_at()` and `set_at_2d()` in `backend/array_ops.py` correctly mutate arrays on both numpy and JAX backends -- numpy uses `.copy()` + direct assignment, JAX uses `.at[].set()`, with no `hasattr(arr, 'at')` checks anywhere else in the codebase | ✓ VERIFIED | `array_ops.py` exists with both functions using `get_backend()` check. Tested on numpy backend (arrays mutated correctly). Tested on JAX backend (arrays mutated correctly). Grep confirms zero `hasattr.*'at'` matches in entire codebase. |
| 3 | EnvState with `extra_state` dict round-trips through `jax.jit(lambda s: s)` without error or shape change -- pytree registration handles the dict correctly | ✓ VERIFIED | JIT round-trip test passes: EnvState with extra_state containing pot arrays survives `jax.jit` preserving all shapes. vmap round-trip test passes: batched EnvState (batch_size=4) has correct shapes `(4, 1, 3)` for pot_contents. |
| 4 | An Overcooked layout (e.g., cramped_room) parses from ASCII string directly into array-based EnvState without creating Grid or Agent objects in the parsing path | ✓ VERIFIED | `layout_parser.py` exists with `parse_layout()` that returns EnvState directly. Cramped_room layout parses correctly: 6x7 grid, 1 pot detected at position [1,3], correct wall_map, populated extra_state. No Grid/Agent imports or instantiation in parser. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `cogrid/backend/array_ops.py` | Backend-aware mutation helpers with `set_at` and `set_at_2d` | ✓ VERIFIED | File exists, 31 lines. Exports `set_at` and `set_at_2d`. Uses `get_backend()` string check (not `hasattr`). Numpy path uses `.copy()` + direct assignment. JAX path uses `.at[].set()`. |
| `cogrid/backend/env_state.py` | Rewritten EnvState with `extra_state` dict field and access helpers | ✓ VERIFIED | File exists, 213 lines. EnvState dataclass has `extra_state: object` field. Old pot fields removed. Exports `get_extra()`, `replace_extra()`, `validate_extra_state()`. Smoke test uses extra_state dict. |
| `cogrid/core/grid_object.py` | Updated `build_lookup_tables` using `array_ops.set_at` instead of hasattr checks | ✓ VERIFIED | File imports `set_at` from array_ops. All 3 hasattr checks replaced with `set_at()` calls. `_np_set()` helper deleted. |
| `cogrid/envs/overcooked/array_config.py` | Updated `_build_interaction_tables` using `array_ops.set_at` instead of hasattr checks, plus `build_overcooked_extra_state` builder | ✓ VERIFIED | File imports `set_at` from array_ops. 2 hasattr checks replaced with `set_at()` calls. `build_overcooked_extra_state()` function exists (lines 27-63), creates scope-prefixed pot arrays from parsed layout. Scope config includes `extra_state_schema` and `extra_state_builder` keys. |
| `cogrid/core/layout_parser.py` | Array-based layout parser with symbol registry | ✓ VERIFIED | File exists, 225 lines. Exports `register_symbols`, `get_symbols`, `parse_layout`. Parses ASCII strings into EnvState without Grid/Agent objects. Uses numpy at parse time, converts to JAX arrays when backend active. Calls `extra_state_builder` from scope config. |
| `cogrid/core/scope_config.py` | Updated with `symbol_table`, `extra_state_schema`, `extra_state_builder` keys | ✓ VERIFIED | `default_scope_config()` includes all three keys in returned dict. |
| `cogrid/envs/overcooked/__init__.py` | Overcooked symbol registration | ✓ VERIFIED | File calls `register_symbols("overcooked", {...})` with complete symbol table mapping. |
| `cogrid/core/jax_step.py` | Updated to read/write pot arrays via `extra_state` instead of direct state fields | ✓ VERIFIED | `envstate_to_dict()` flattens extra_state with scope prefix stripping. `jax_step()` reads pot arrays via `state.extra_state["overcooked.pot_contents"]` etc. Updates extra_state with new pot arrays in tick and interaction sections. `jax_reset()` builds extra_state from layout_arrays. |

### Key Link Verification

All critical wiring verified:

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `array_ops.py` | `_dispatch.get_backend` | import from _dispatch | ✓ WIRED | Import found: `from cogrid.backend._dispatch import get_backend` |
| `grid_object.py` | `array_ops.py` | build_lookup_tables calls set_at | ✓ WIRED | Import found, 3 call sites verified |
| `array_config.py` | `array_ops.py` | _build_interaction_tables calls set_at | ✓ WIRED | Import found, 5 call sites verified |
| `layout_parser.py` | `scope_config` | parse_layout reads symbol_table and extra_state_builder | ✓ WIRED | `scope_config.get("symbol_table")` and `scope_config.get("extra_state_builder")` found |
| `overcooked/__init__.py` | `layout_parser.py` | Overcooked registers symbols | ✓ WIRED | `register_symbols` import and call found |
| `layout_parser.py` | `env_state.py` | parse_layout creates EnvState | ✓ WIRED | `create_env_state()` call found, returns EnvState |
| `jax_step.py` | `env_state.py` | imports get_extra, replace_extra | ✓ WIRED | Uses extra_state dict directly, `envstate_to_dict` flattens it |

### Requirements Coverage

No REQUIREMENTS.md entries mapped to Phase 05.

### Anti-Patterns Found

No blockers or warnings detected. All code follows the established patterns.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| - | - | - | - | - |

**Clean codebase:** Zero `hasattr(arr, 'at')` checks remain outside `array_ops.py`. All backend branching properly isolated.

### Human Verification Required

None. All success criteria are programmatically verifiable and have been verified.

### Gaps Summary

No gaps. All 4 success criteria verified:

1. ✓ EnvState has `extra_state` dict instead of pot fields -- Overcooked populates it at init
2. ✓ `set_at()` and `set_at_2d()` work on both numpy and JAX -- no `hasattr` checks elsewhere
3. ✓ EnvState with `extra_state` round-trips through `jax.jit` and `jax.vmap` without error
4. ✓ Overcooked layout parses from ASCII to EnvState without Grid/Agent objects

## Detailed Verification Evidence

### Success Criterion 1: EnvState with extra_state

**Evidence:**
```python
# EnvState fields verification
fields = ['agent_pos', 'agent_dir', 'agent_inv', 'wall_map', 
          'object_type_map', 'object_state_map', 'extra_state', 
          'rng_key', 'time', 'n_agents', 'height', 'width', 'action_set']

# Old fields absent: pot_contents, pot_timer, pot_positions, n_pots ✓
# New field present: extra_state ✓
```

**Overcooked scope config:**
```python
# From array_config.py lines 111-116
"extra_state_schema": {
    "overcooked.pot_contents": {"shape": ("n_pots", 3), "dtype": "int32"},
    "overcooked.pot_timer": {"shape": ("n_pots",), "dtype": "int32"},
    "overcooked.pot_positions": {"shape": ("n_pots", 2), "dtype": "int32"},
},
"extra_state_builder": build_overcooked_extra_state,
```

**Layout parser test output:**
```
Layout parser end-to-end OK
  Grid: 6x7
  Pots: 1 at positions [[1 3]]
  Agent pos: [[0 0] [0 0]]
```

### Success Criterion 2: array_ops.py mutation helpers

**array_ops.py implementation:**
- Lines 9-18: `set_at()` with `get_backend()` check
- Lines 21-30: `set_at_2d()` with `get_backend()` check
- No `hasattr` checks anywhere

**Verification tests:**
```
numpy array_ops OK  # set_at and set_at_2d tested on numpy backend
JAX array_ops OK    # set_at and set_at_2d tested on JAX backend
```

**Codebase scan:**
```bash
grep -rn "hasattr.*'at'" cogrid/ --include="*.py"
# Result: No files found ✓
```

**Replacement evidence:**
- `grid_object.py`: 3 hasattr blocks → 3 `set_at()` calls (commit 0c28afd)
- `array_config.py`: 2 hasattr blocks → 5 `set_at()` calls (commit 0c28afd)

### Success Criterion 3: JIT and vmap round-trip

**Test output:**
```
JIT + vmap round-trip PASSED
```

**Test details:**
- Created EnvState with extra_state containing 3 pot arrays
- Round-tripped through `jax.jit(lambda s: s)` → all shapes preserved ✓
- Batched with `jax.vmap` at batch_size=4 → shapes `(4, 1, 3)` correct ✓
- Verified `get_extra()` works on JIT/vmap outputs ✓

**JAX pytree registration:**
- `env_state.py` lines 92-105: `register_envstate_pytree()` uses `jax.tree_util.register_dataclass`
- JAX natively handles dict fields in pytree nodes ✓

### Success Criterion 4: Layout parser without Grid/Agent objects

**Layout parser implementation:**
- `layout_parser.py` lines 79-224: `parse_layout()` function
- Uses numpy loops over ASCII chars
- Builds `wall_map`, `object_type_map`, `object_state_map` arrays directly
- Calls `extra_state_builder` from scope config
- Returns `create_env_state()` with all arrays
- No imports of Grid, GridObj, or Agent classes ✓

**Cramped room parse test:**
```python
# Input: 6x7 ASCII layout with 1 pot ('U' char)
# Output: EnvState with:
#   - wall_map.shape = (6, 7) ✓
#   - object_type_map.shape = (6, 7) ✓
#   - extra_state["overcooked.pot_positions"] = [[1, 3]] ✓
#   - extra_state["overcooked.pot_contents"].shape = (1, 3) ✓
#   - extra_state["overcooked.pot_timer"].shape = (1,) ✓
```

## Commit Verification

All commits from summaries verified in git log:

| Commit | Plan | Description | Status |
|--------|------|-------------|--------|
| 9edcc6e | 05-01 | Create array_ops.py and rewrite EnvState | ✓ VERIFIED |
| 0c28afd | 05-01 | Replace hasattr checks with array_ops | ✓ VERIFIED |
| 3a14978 | 05-02 | Add layout parser and extend scope config | ✓ VERIFIED |
| a27967b | 05-02 | Register Overcooked symbols and extra_state builder | ✓ VERIFIED |
| 952834c | 05-03 | Update jax_step.py to use extra_state dict | ✓ VERIFIED |
| 092661f | 05-03 | Update smoke test for extra_state verification | ✓ VERIFIED |

## Phase Completion Summary

Phase 05 successfully completed all objectives:

**Plan 01 (Foundation):**
- ✓ Created `array_ops.py` as single backend mutation dispatch point
- ✓ Rewrote EnvState with generic `extra_state` dict
- ✓ Eliminated all 5 `hasattr(arr, 'at')` checks from codebase
- ✓ Added `get_extra()`, `replace_extra()`, `validate_extra_state()` helpers

**Plan 02 (Layout Parser):**
- ✓ Created `layout_parser.py` with symbol registry and array-based parsing
- ✓ Extended scope config with `symbol_table`, `extra_state_schema`, `extra_state_builder`
- ✓ Registered Overcooked symbol table and pot extra_state builder
- ✓ Verified end-to-end: cramped_room parses without Grid/Agent objects

**Plan 03 (JAX Integration):**
- ✓ Updated `jax_step.py` to read/write pot arrays via `extra_state`
- ✓ Updated `envstate_to_dict` to flatten extra_state for backward compatibility
- ✓ Updated `jax_reset` to build extra_state from layout_arrays
- ✓ Verified JIT and vmap round-trip with extra_state dict

**Impact:**
- The data layer is now ready for unified functional code
- All environment-specific state uses the generic `extra_state` pattern
- Backend mutation properly abstracted in `array_ops.py`
- Layouts parse directly to arrays without intermediate objects
- JAX pytree handling works correctly with the new data model

---

_Verified: 2026-02-12T21:17:03Z_
_Verifier: Claude (gsd-verifier)_
