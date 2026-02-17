# Phase 25: Interaction Cascade Refactor - Research

**Researched:** 2026-02-17
**Domain:** JAX-compatible branchless interaction dispatch refactoring
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Uniform interface: every branch function gets a standardized signature (handled_mask, shared_context) and returns (condition, updates, new_handled_mask)
- This replaces the current per-branch unique signatures where individual `~b1_cond`, `~b2_cond` guards are threaded through as separate parameters
- Adding a new branch in Phase 27/29 should be trivial with this uniform contract
- Moderate test depth: ~100-500 random interaction states per test run
- Seeded random generation for deterministic, reproducible failures
- Test across multiple layouts (2-3 different pot/stack/counter configurations) to cover layout-sensitive edge cases
- Existing parity tests are the sole verification -- if they pass, the refactor is correct
- Failing parity test = bug in the refactor, always fix the refactor, never modify test assertions
- Old N-1 guard code removed completely -- clean break, no commented-out remnants
- Update the interaction cascade documentation (docstring/comment block at top of file) to reflect the new accumulated-handled pattern

### Claude's Discretion
- Registry pattern vs. fixed-sequence orchestration
- Full update struct vs. sparse update returns
- Branch priority reordering (must preserve identical behavior)
- Whether invariant test also checks correct-branch-fires (beyond at-most-one)
- Exact number of random states within the 100-500 range

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

## Summary

This phase refactors `overcooked_interaction_body` in `cogrid/envs/overcooked/config.py` to replace explicit N-1 guard chains (`~b1_cond & ~b2_cond & ...`) with an accumulated `handled` boolean mask. Currently, each branch function receives and checks against all earlier branch conditions individually, threading unique variable names through the cascade. This pattern is fragile: adding a new branch in Phases 27 or 29 requires updating every subsequent branch's guard clause, creating a high risk of double-fire bugs.

The refactor is purely structural -- no behavioral changes. The seven existing branches (pickup, pickup-from-pot, pickup-from-stack, drop-on-empty, place-on-pot, place-on-delivery, place-on-counter) must produce identical results. The existing 8 deterministic parity tests in `cogrid/envs/overcooked/test_interactions.py` are the verification standard. A new invariant test will fuzz 100-500 random states across multiple layouts to verify the at-most-one-branch-fires property.

The codebase uses a `cogrid.backend.xp` abstraction that resolves to either `numpy` or `jax.numpy`. All interaction code must remain branchless (no Python `if/else` on array values) for JAX traceability. The `xp.where(cond, new, old)` pattern and `set_at`/`set_at_2d` helpers from `cogrid.backend.array_ops` are the mutation primitives.

**Primary recommendation:** Use a fixed-sequence list of branch functions with a uniform interface, passing an accumulated `handled` scalar through the sequence. Each branch receives `(handled, ctx)` and returns `(branch_cond, updates_dict, new_handled)` where `new_handled = handled | branch_cond`. The orchestrator merges updates via `xp.where` after each branch. This is simpler than a registry pattern and matches the codebase's existing code-over-configuration philosophy.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | >=1.26,<3.0 | Default array backend | Already the codebase default via `cogrid.backend.xp` |
| jax | >=0.4.20,<1.0 (optional) | Alternative array backend for JIT/vmap | Already supported, interactions must stay trace-compatible |
| pytest | >=7.0 | Test framework | Already in dev dependencies, used for all tests |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy.random | (bundled) | Seeded random state generation for invariant tests | Use `np.random.default_rng(seed)` for reproducible test state generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| numpy.random | jax.random | Would add JAX dependency to tests; numpy.random is sufficient since tests run on numpy backend |

## Architecture Patterns

### Current File Structure (files touched by this phase)
```
cogrid/envs/overcooked/
    config.py              # overcooked_interaction_body + 7 branch functions + _apply_interaction_updates
    test_interactions.py   # 8 existing parity tests (DO NOT MODIFY assertions)
```

### Pattern 1: Accumulated-Handled with Shared Context
**What:** Replace per-branch unique parameter lists with a uniform `(handled, ctx)` -> `(cond, updates, handled)` interface. A `ctx` dict bundles all shared arrays. An `updates` dict returns only the arrays a branch modifies.

**When to use:** When branches are mutually exclusive, evaluated in priority order, and the set of branches may grow.

**Example (target architecture):**
```python
# Shared context assembled once at top of overcooked_interaction_body
ctx = {
    "base_ok": base_ok,
    "fwd_type": fwd_type,
    "fwd_r": fwd_r,
    "fwd_c": fwd_c,
    "inv_item": inv_item,
    "agent_idx": agent_idx,
    "agent_inv": agent_inv,
    "object_type_map": object_type_map,
    "object_state_map": object_state_map,
    "pot_contents": pot_contents,
    "pot_timer": pot_timer,
    "pot_idx": pot_idx,
    "has_pot_match": has_pot_match,
    # static tables unpacked as individual keys
    "CAN_PICKUP": CAN_PICKUP,
    "CAN_PICKUP_FROM": CAN_PICKUP_FROM,
    "CAN_PLACE_ON": CAN_PLACE_ON,
    "pickup_from_produces": pickup_from_produces,
    "legal_pot_ingredients": legal_pot_ingredients,
    "pot_id": pot_id,
    "plate_id": plate_id,
    "tomato_id": tomato_id,
    "onion_soup_id": onion_soup_id,
    "tomato_soup_id": tomato_soup_id,
    "delivery_zone_id": delivery_zone_id,
    "cooking_time": cooking_time,
}

# Uniform branch signature
def _branch_pickup(handled, ctx):
    """Branch 1: Pick up loose object."""
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] > 0) & ...
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), ctx["fwd_type"])
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    updates = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return cond, updates, handled | cond

# Branch list in priority order
BRANCHES = [
    _branch_pickup,
    _branch_pickup_from_pot,
    _branch_pickup_from_stack,
    _branch_drop_on_empty,
    _branch_place_on_pot,
    _branch_place_on_delivery,
    _branch_place_on_counter,
]

# Orchestrator
def overcooked_interaction_body(...):
    ctx = { ... }  # assembled once
    handled = xp.bool_(False)
    results = {}  # cond -> updates mapping

    for branch_fn in BRANCHES:
        cond, updates, handled = branch_fn(handled, ctx)
        results[branch_fn] = (cond, updates)

    # Merge: apply updates in priority order using xp.where
    final_inv = ctx["agent_inv"]
    final_otm = ctx["object_type_map"]
    ...
    for branch_fn in BRANCHES:
        cond, updates = results[branch_fn]
        if "agent_inv" in updates:
            final_inv = xp.where(cond, updates["agent_inv"], final_inv)
        if "object_type_map" in updates:
            final_otm = xp.where(cond, updates["object_type_map"], final_otm)
        ...

    return final_inv, final_otm, final_osm, final_pc, final_pt
```

### Pattern 2: Sparse Updates Dict (Recommended)
**What:** Each branch returns only the arrays it modifies in a dict, rather than returning all 5 arrays.

**Why preferred:** Current branches modify different subsets of arrays (see the table in config.py docstring). Branch 2B (pickup from stack) only touches `agent_inv`. Branch 4B (delivery) only touches `agent_inv`. Requiring every branch to return all 5 arrays would add unnecessary boilerplate and obscure which arrays each branch actually modifies.

**Tradeoff:** The merge loop must check `if key in updates` for each array. This is pure Python dict lookup at trace time, not a runtime cost, and adds negligible complexity.

### Pattern 3: Fixed-Sequence List (Recommended over Registry)
**What:** Branches are a plain Python list in the function body, not a global registry.

**Why preferred over registry:**
1. The codebase already uses explicit function composition (see `autowire.py` tick handler composition). A registry pattern would be an unfamiliar paradigm.
2. Branch order is semantically meaningful (priority order). A registry requires explicit priority keys, adding complexity.
3. JAX tracing requires the branch list to be static at trace time. A mutable registry risks trace-time mutation bugs.
4. Adding a branch in Phase 27/29 means adding one function and appending to the list -- trivially safe.
5. Total branch count is small (7 now, ~9-10 at most) -- not enough to justify registry infrastructure.

### Anti-Patterns to Avoid
- **Passing `handled` as a mutable state through deeply nested calls:** The handled mask should be a scalar bool passed to each branch and OR'd with the branch condition. Do NOT use a mutable container.
- **Encoding handled as a multi-bit integer (branch ID):** A bool scalar is simpler and matches the existing `base_ok` pattern. An integer would require `handled != 0` checks, and the benefit (knowing which branch fired) can be tested separately.
- **Using Python if/else on `handled` value:** This breaks JAX tracing. The `~handled` guard must be part of the boolean condition expression, computed unconditionally.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Array mutation | In-place assignment (`arr[i] = v`) | `set_at(arr, idx, val)` / `set_at_2d(arr, r, c, val)` | Backend-aware; returns new array on JAX |
| Conditional updates | Python if/else on array values | `xp.where(cond, new, old)` | Required for JAX tracing |
| Random state generation | Manual state construction | `np.random.default_rng(seed)` with structured generation | Reproducible, seed-controlled |

**Key insight:** The `set_at` / `set_at_2d` / `xp.where` trio is the ONLY mutation vocabulary in this codebase. The refactor must not introduce any new mutation patterns.

## Common Pitfalls

### Pitfall 1: Breaking JAX Traceability with Python Branching
**What goes wrong:** Wrapping branch evaluation in `if handled:` or `if cond:` Python statements. JAX cannot trace through Python-level branches -- both paths must execute unconditionally, with `xp.where` selecting the result.
**Why it happens:** Natural instinct to short-circuit when `handled` is True.
**How to avoid:** Every branch function must compute its condition as `~handled & base_ok & ...` and return the would-be updates unconditionally. The `~handled` term is a boolean AND, not a Python guard.
**Warning signs:** Tests pass on numpy but fail on JAX backend; `jax.jit` raises `ConcretizationTypeError`.

### Pitfall 2: Forgetting to OR Handled After Branch
**What goes wrong:** A branch fires and updates `handled` locally but the updated value is not propagated to subsequent branches, allowing double-fire.
**Why it happens:** In the old code, each branch receives `~b1_cond` etc. explicitly. With the new pattern, `handled` must be updated after each branch: `handled = handled | cond`.
**How to avoid:** The branch function itself returns the new handled value: `return cond, updates, handled | cond`. The orchestrator MUST use the returned handled value for the next branch.
**Warning signs:** Invariant test finds two branches firing for the same interaction state.

### Pitfall 3: Dict Key Typos in Sparse Updates
**What goes wrong:** A branch returns `{"agent_inventory": inv}` instead of `{"agent_inv": inv}`. The merge loop silently ignores the key, and the branch has no effect.
**Why it happens:** Sparse dicts use string keys. No compile-time or type checking.
**How to avoid:** Define the canonical key names as module-level constants (e.g., `K_INV = "agent_inv"`) and use them consistently in branches and the merge loop. Alternatively, add an assertion in the merge loop that validates returned keys against a known set.
**Warning signs:** A branch's test scenario suddenly fails after refactor, but the condition-fires correctly.

### Pitfall 4: Place-On Branches Share a Base Condition
**What goes wrong:** The three place-on branches (4A, 4B, 4C) currently share a `b4_base` precondition that includes guards against branches 1-3. When converting to accumulated-handled, it's tempting to compute `b4_base` once and pass it to sub-branches. But if any of 4A/4B fire, 4C must not fire.
**Why it happens:** The current code computes `b4_base` once and passes it to 4A, 4B, 4C. With accumulated-handled, 4A updating `handled` before 4B/4C evaluation is sufficient.
**How to avoid:** Treat 4A, 4B, 4C as three separate branches in the list (not sub-branches of a single branch). Each checks `~handled` independently. The orchestrator's handled propagation ensures mutual exclusion automatically.
**Warning signs:** Both 4A and 4C fire when a pot is at full capacity and the agent holds an ingredient.

### Pitfall 5: ctx Dict Creates Stale References After Updates
**What goes wrong:** A branch modifies `agent_inv` but the next branch reads the old `agent_inv` from `ctx`, not the updated one.
**Why it happens:** In the current code, each branch computes would-be results independently and `_apply_interaction_updates` merges them at the end. The branches don't chain mutations.
**How to avoid:** This is actually the correct behavior! Since at most one branch fires, only one set of updates is applied. The branches should all read from the ORIGINAL arrays in `ctx`, not from each other's would-be results. The merge loop applies updates from the single firing branch. Do NOT update `ctx` between branches.
**Warning signs:** None -- this is the correct pattern. Just don't try to "optimize" by mutating ctx.

### Pitfall 6: Invariant Test Doesn't Cover All Object Types
**What goes wrong:** Random state generation places only a few object types on the grid, missing edge cases where multiple branch conditions could be nearly-true simultaneously.
**Why it happens:** Generating truly representative random states requires all object types (pot, counter, delivery_zone, onion, tomato, plate, stacks, soups) and various inventory states.
**How to avoid:** Generate states from the actual registered type IDs (available from `scope_config["type_ids"]`). Include: empty inventory vs. each holdable item type, various forward cell types, pot states (empty, partial, full, cooked), counter states (empty vs. occupied).
**Warning signs:** Invariant test runs 300 states but never generates a "place on full pot" scenario.

## Code Examples

### Example 1: Current N-1 Guard Chain (to be replaced)
```python
# Current: each branch explicitly guards against ALL prior branches
b1_cond = base_ok & (fwd_type > 0) & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1)
# ...
b2_pot_cond = base_ok & ~b1_cond & is_pot & has_pot_match & has_contents & is_ready & (inv_item == plate_id)
# ...
b2_stack_cond = base_ok & ~b1_cond & is_stack & (inv_item == -1) & (produced > 0)
# ...
b3_cond = base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & (fwd_type == 0) & (inv_item != -1)
# ...
b4_base = base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1) & (inv_item != -1)
```

### Example 2: New Accumulated-Handled Pattern (target)
```python
# New: each branch receives and propagates a single `handled` scalar
def _branch_pickup(handled, ctx):
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] > 0) & (ctx["CAN_PICKUP"][ctx["fwd_type"]] == 1) & (ctx["inv_item"] == -1)
    # compute updates unconditionally...
    return cond, {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}, handled | cond

def _branch_pickup_from_pot(handled, ctx):
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] == ctx["pot_id"]) & ctx["has_pot_match"] & ...
    # compute updates unconditionally...
    return cond, {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}, handled | cond
```

### Example 3: Orchestrator Merge Loop
```python
def overcooked_interaction_body(...):
    ctx = { ... }
    handled = xp.bool_(False)

    # Collect all branch results
    branch_results = []
    for branch_fn in BRANCHES:
        cond, updates, handled = branch_fn(handled, ctx)
        branch_results.append((cond, updates))

    # Merge using xp.where in priority order
    inv = ctx["agent_inv"]
    otm = ctx["object_type_map"]
    osm = ctx["object_state_map"]
    pc = ctx["pot_contents"]
    pt = ctx["pot_timer"]

    for cond, updates in branch_results:
        if "agent_inv" in updates:
            inv = xp.where(cond, updates["agent_inv"], inv)
        if "object_type_map" in updates:
            otm = xp.where(cond, updates["object_type_map"], otm)
        if "object_state_map" in updates:
            osm = xp.where(cond, updates["object_state_map"], osm)
        if "pot_contents" in updates:
            pc = xp.where(cond, updates["pot_contents"], pc)
        if "pot_timer" in updates:
            pt = xp.where(cond, updates["pot_timer"], pt)

    return inv, otm, osm, pc, pt
```

### Example 4: Invariant Test Structure
```python
def test_at_most_one_branch_fires():
    """Verify that at most one branch fires per agent-interaction pair across random states."""
    import numpy as np
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()
    import cogrid.envs  # noqa: F401

    rng = np.random.default_rng(seed=42)

    # Get type IDs and static tables
    scope_cfg = build_scope_config_from_components("overcooked")
    scope_cfg["interaction_fn"] = overcooked_interaction_fn
    type_ids = scope_cfg["type_ids"]

    holdable = [type_ids[n] for n in ["onion", "tomato", "plate", "onion_soup", "tomato_soup"]]
    fwd_cell_types = [0] + [type_ids[n] for n in type_ids]  # 0 = empty, plus all types

    layouts_to_test = [
        "overcooked_cramped_room_v0",
        "overcooked_asymmetric_advantages_v0",
        "overcooked_coordination_ring_v0",
    ]

    n_states = 200
    for layout_name in layouts_to_test:
        for _ in range(n_states):
            # Generate random state: random inv, random fwd cell, random pot state
            inv_item = rng.choice([-1] + holdable)
            fwd_type = rng.choice(fwd_cell_types)
            # ... build state, call interaction, count how many branches fire
            # Assert at most 1
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| N-1 guard chains (`~b1 & ~b2 & ...`) | Accumulated handled mask (`~handled & ...`) | This phase (Phase 25) | New branches can be added without editing existing guards |

**Deprecated/outdated:**
- The explicit `b1_cond`, `b2_pot_cond`, `b2_stack_cond`, `b3_cond` variable threading will be fully removed. No backward-compat shim needed since these are internal to `config.py`.

## Key Implementation Details

### Files Modified
1. **`cogrid/envs/overcooked/config.py`** -- Primary target:
   - Rewrite `_interact_pickup`, `_interact_pickup_from_pot`, `_interact_pickup_from_stack`, `_interact_drop_on_empty`, `_interact_place_on_pot`, `_interact_place_on_delivery`, `_interact_place_on_counter` to uniform `(handled, ctx) -> (cond, updates, handled)` signature
   - Rewrite `overcooked_interaction_body` to use branch list + merge loop
   - Remove `_apply_interaction_updates` (its logic absorbed into the merge loop)
   - Update module-level docstring to describe new pattern

2. **`cogrid/envs/overcooked/test_interactions.py`** -- Add invariant test (DO NOT modify existing test assertions)

### Signature of `overcooked_interaction_fn` (UNCHANGED)
```python
def overcooked_interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config) -> state
```
This is the public API consumed by `process_interactions` in `cogrid/core/interactions.py`. It must NOT change.

### Signature of `overcooked_interaction_body` (CHANGED)
```python
# Current: 13 positional parameters
def overcooked_interaction_body(agent_idx, agent_inv, object_type_map, object_state_map, fwd_r, fwd_c, fwd_type, inv_item, base_ok, pot_contents, pot_timer, pot_positions, static_tables)

# New: same 13 parameters (backward compatible)
# Internal changes only: ctx dict assembly, branch list, merge loop
```

### Branch Count and Which Arrays Each Modifies
| Branch | Name | Modifies |
|--------|------|----------|
| 1 | pickup | agent_inv, object_type_map, object_state_map |
| 2A | pickup_from_pot | agent_inv, pot_contents, pot_timer |
| 2B | pickup_from_stack | agent_inv |
| 3 | drop_on_empty | agent_inv, object_type_map, object_state_map |
| 4A | place_on_pot | agent_inv, pot_contents |
| 4B | place_on_delivery | agent_inv |
| 4C | place_on_counter | agent_inv, object_state_map |

### Merge Order Keys
The five arrays merged in the orchestrator:
1. `agent_inv` -- modified by all 7 branches
2. `object_type_map` -- modified by branches 1, 3
3. `object_state_map` -- modified by branches 1, 3, 4C
4. `pot_contents` -- modified by branches 2A, 4A
5. `pot_timer` -- modified by branch 2A only

### Existing Test Coverage (8 parity tests, MUST ALL PASS)
1. `overcooked_tick` vs `Pot.tick()` (timer/state machine parity)
2. Pickup from stack (onion_stack -> onion in inventory)
3. Full pot workflow (3 onions -> cook -> pickup with plate -> onion_soup)
4. Delivery zone accepts soup only (soup delivered, non-soup rejected)
5. Counter place and pickup (place on empty counter, reject on occupied)
6. Tomato soup type detection (all-tomato pot -> tomato_soup)
7. Priority order (pickup beats place-on when ambiguous)
8. Agent-ahead blocking (agent in front blocks interaction)

## Open Questions

1. **xp.bool_(False) initialization**
   - What we know: `xp.where` requires array-like operands. A Python `False` may work with numpy but could cause issues with JAX tracing.
   - What's unclear: Whether `xp.bool_(False)` is the correct initializer or if it should be `xp.array(False)`.
   - Recommendation: Use `xp.bool_(False)` which matches how the codebase creates scalar booleans elsewhere (e.g., `base_ok` is a scalar bool from `is_interact & ~agent_ahead`). Verify with JAX backend test if available.

2. **Should invariant test also verify correct-branch-fires (not just at-most-one)?**
   - What we know: At-most-one is the primary invariant (CASC-02). Verifying correct-branch-fires would mean encoding expected behavior for each random state.
   - What's unclear: Whether this adds sufficient value vs. complexity.
   - Recommendation: Focus on at-most-one for the invariant test. Correct-branch behavior is already covered by the 8 deterministic parity tests. Adding correct-branch verification to random tests would essentially re-implement the interaction logic in the test, which defeats the purpose.

3. **Performance impact of dict-based ctx and sparse updates**
   - What we know: The dicts are pure Python and only accessed at trace time (JAX) or at negligible cost (numpy). No runtime array allocation is added.
   - What's unclear: Whether JAX tracing time increases measurably.
   - Recommendation: Proceed with dict pattern. Profile only if JAX tracing time regresses measurably (unlikely given the small number of branches).

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis: `cogrid/envs/overcooked/config.py` (1184 lines) -- full reading of all 7 branch functions, orchestrator, merge function, and docstrings
- Direct codebase analysis: `cogrid/core/interactions.py` (125 lines) -- process_interactions API and generic branch pattern
- Direct codebase analysis: `cogrid/envs/overcooked/test_interactions.py` (419 lines) -- all 8 parity tests verified passing
- Direct codebase analysis: `cogrid/backend/array_ops.py` (26 lines) -- set_at, set_at_2d mutation helpers
- Direct codebase analysis: `cogrid/backend/env_state.py` -- EnvState dataclass, extra_state dict pattern
- Direct codebase analysis: `cogrid/core/autowire.py` -- composition patterns (tick handler, render sync)
- Direct test execution: All 8 parity tests pass on current code (verified 2026-02-17)

### Secondary (MEDIUM confidence)
- JAX tracing constraints: Based on established JAX documentation patterns -- no Python-level branching on traced values, all paths must execute unconditionally
- numpy/JAX `xp.where` behavior: Standard numpy/JAX semantics, well-established

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- No new libraries needed; pure refactor within existing numpy/JAX abstractions
- Architecture: HIGH -- Pattern directly derived from existing codebase patterns (autowire composition) and the user's locked design decisions
- Pitfalls: HIGH -- All pitfalls identified from direct code analysis of current guard chain mechanics and JAX tracing requirements
- Test strategy: HIGH -- Invariant test requirements specified by user; random state generation is straightforward with numpy.random

**Research date:** 2026-02-17
**Valid until:** Indefinite (internal refactor of stable codebase; no external dependency changes)
