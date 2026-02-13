# Phase 14: Auto-Wired CoGridEnv & Validation - Research

**Researched:** 2026-02-13
**Domain:** Internal codebase refactor -- switching CoGridEnv from manual scope_config/reward_config wiring to auto-wiring from component registries, validating parity with v1.1, and deleting dead manual-wiring code
**Confidence:** HIGH

## Summary

Phase 14 is the final phase of v1.2. The auto-wiring infrastructure built in Phases 10-13 is proven functional (8 parity tests pass, 85 total tests pass), but CoGridEnv.__init__() still uses the old manual path: `get_scope_config(self.scope)` which calls `build_overcooked_scope_config()` via the scope config registry, and the reward_config is assembled manually with an `if self.scope == "overcooked"` branch. Phase 14 replaces this with `build_scope_config_from_components(scope)` and `build_reward_config_from_components(scope, ...)` as the default path, proves full end-to-end parity across both backends, proves JIT+vmap at 1024 envs, demonstrates the component API with a goal-finding example environment, and deletes all dead manual-wiring code.

The work divides into four concrete streams: (1) Rewire CoGridEnv.__init__() to use auto-wiring functions for scope_config and reward_config, removing the `if self.scope == "overcooked"` branch and the `get_scope_config` call. (2) Convert the existing goal_finding.py example to use the component API (ArrayReward subclass + zero manual scope_config). (3) Write end-to-end parity tests proving component-based Overcooked matches v1.1 manual-wired Overcooked across numpy and JAX backends, plus JIT+vmap at 1024 envs. (4) Delete dead code: `build_overcooked_scope_config()`, `compute_rewards()` dispatcher, `register_scope_config()` call, `register_symbols()` call, and the scope_config registry module itself (if no other scopes use it).

The critical insight is that most infrastructure is already built and tested. Phase 14 is a "flip the switch" phase -- changing the wiring point in CoGridEnv, running existing tests to confirm nothing breaks, adding targeted parity and performance tests, and cleaning up.

**Primary recommendation:** Execute in three plans: (A) Rewire CoGridEnv + update overcooked __init__.py + run existing test suite, (B) Goal-finding example via component API + JIT/vmap parity tests, (C) Dead code deletion + final verification.

## Standard Stack

### Core

No new external libraries. This phase modifies only existing internal modules.

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| cogrid.core.autowire | Existing | `build_scope_config_from_components()`, `build_reward_config_from_components()` | Phase 11+13 infrastructure -- proven by 8 parity tests |
| cogrid.core.component_registry | Existing | Stores component metadata at import time | Phase 10 infrastructure |
| cogrid.core.array_rewards | Existing | `ArrayReward` base class, `@register_reward_type` decorator | Phase 10 infrastructure |
| cogrid.core.scope_config | Existing (will be deprecated) | `get_scope_config()`, `register_scope_config()` | Current manual path -- being replaced |
| cogrid.core.layout_parser | Existing | `register_symbols()`, `get_symbols()`, `parse_layout()` | Symbol resolution -- needs to work with autowired symbol_table |
| cogrid.cogrid_env | Existing (primary modification target) | `CoGridEnv.__init__()` scope_config and reward_config wiring | The main switch point for auto-wiring |

### Supporting

None. Pure internal refactor.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Replacing get_scope_config entirely | Making get_scope_config delegate to auto-wiring internally | Indirection layer adds confusion; direct autowire call is simpler and more readable |
| Deleting scope_config.py module | Keeping it as a thin wrapper around autowire | Dead code is dead code; keeping it creates maintenance burden and misleads future developers |
| Moving goal-finding into cogrid/envs/ | Keeping as examples/goal_finding.py | Example file better demonstrates the user-facing API; moving into envs/ makes it look like a core feature |

## Architecture Patterns

### Current Wiring (What Exists -- Phase 13 Complete)

```
cogrid/cogrid_env.py __init__():
    1. self._scope_config = get_scope_config(self.scope)
       -> calls SCOPE_CONFIG_REGISTRY["overcooked"]()
       -> calls build_overcooked_scope_config()
       -> returns manual dict with tick_handler, interaction_body, etc.

    2. self._reward_config = {type_ids, n_agents, rewards, action_pickup_drop_idx}
       if self.scope == "overcooked":
           self._reward_config["compute_fn"] = compute_rewards  # manual import

cogrid/envs/overcooked/__init__.py:
    register_scope_config("overcooked", build_overcooked_scope_config)
    register_symbols("overcooked", {...})
```

### Target Wiring (After Phase 14)

```
cogrid/cogrid_env.py __init__():
    1. self._scope_config = build_scope_config_from_components(self.scope)
       -> queries component registry for scope's GridObject subclasses
       -> auto-composes tick_handler, interaction_body, extra_state_builder,
          static_tables, type_ids, symbol_table, extra_state_schema

    2. self._reward_config = build_reward_config_from_components(
           self.scope, n_agents, type_ids, action_pickup_drop_idx)
       -> queries reward registry for scope's ArrayReward subclasses
       -> composes compute_fn from registered rewards

    NO scope-specific branches. NO manual imports.

cogrid/envs/overcooked/__init__.py:
    # Registration happens at import time via decorators:
    # @register_object_type on GridObject subclasses
    # @register_reward_type on ArrayReward subclasses
    # Symbols auto-populated from component chars
    import cogrid.envs.overcooked.overcooked_grid_objects  # trigger registration
    import cogrid.envs.overcooked.array_rewards  # trigger registration
```

### Pattern 1: CoGridEnv Auto-Wiring

**What:** Replace `get_scope_config()` + manual reward assembly with two autowire calls.

**When to use:** All scopes. The autowire gracefully handles scopes with zero components (returns empty/None fields).

**Example:**
```python
# In cogrid/cogrid_env.py __init__():

from cogrid.core.autowire import (
    build_scope_config_from_components,
    build_reward_config_from_components,
)

# Auto-wire scope config from registered components
self._scope_config = build_scope_config_from_components(self.scope)
self._type_ids = self._scope_config["type_ids"]
self._interaction_tables = self._scope_config.get("interaction_tables")

# Auto-wire reward config from registered ArrayReward subclasses
self._reward_config = build_reward_config_from_components(
    self.scope,
    n_agents=self.config["num_agents"],
    type_ids=self._type_ids,
    action_pickup_drop_idx=self._action_pickup_drop_idx,
)
```

**Critical constraint:** The `action_pickup_drop_idx` must be computed BEFORE `build_reward_config_from_components` is called, because the reward config needs it. Currently in `cogrid_env.py`, the action indices are computed at lines 230-235, which is BEFORE the scope_config and reward_config blocks. This ordering must be preserved.

### Pattern 2: Symbol Table Fallback in Layout Parser

**What:** The layout parser's `get_symbols()` checks SYMBOL_REGISTRY first, then falls back to `scope_config["symbol_table"]`. After Phase 14, the SYMBOL_REGISTRY registration in `overcooked/__init__.py` will be removed. The autowired scope_config contains a `symbol_table` key, so the fallback path will activate.

**When to use:** When `parse_layout()` or `layout_to_array_state()` is called with an autowired scope_config.

**Important detail:** The current `cogrid_env.py` does NOT use `parse_layout()` -- it uses the Grid/GridObj path via `_gen_grid() -> Grid.decode()`. The Grid.decode path uses its own symbol resolution. So removing `register_symbols("overcooked", ...)` should have no effect on the current environment initialization path. The `parse_layout()` function exists for the direct functional API.

However, there is a subtlety: `Grid.decode()` uses the scope's registered object types to resolve object IDs. The `register_symbols()` call is separate from `register_object_type()` -- the former feeds the layout parser, the latter feeds Grid.decode and the component registry. Since Phase 14 removes the `register_symbols()` call but keeps `register_object_type()` decorators, Grid.decode will continue to work.

### Pattern 3: Goal-Finding Via Component API

**What:** Demonstrate INTG-04: new environment creation with zero manual scope_config. Requires only GridObject subclasses, ArrayReward subclass, layout, and config dict.

**Example:**
```python
from cogrid.core.grid_object import GridObj, register_object_type
from cogrid.core.array_rewards import ArrayReward
from cogrid.core.component_registry import register_reward_type
from cogrid.core import layouts

# 1. Register goal object type
@register_object_type("goal", can_overlap=True)
class Goal(GridObj):
    object_id = "goal"
    char = "g"
    def __init__(self, **kwargs):
        super().__init__(state=0)

# 2. Register layout
layouts.register_layout("goal_simple_v0", [...])

# 3. Register reward
@register_reward_type("goal", scope="global", coefficient=1.0, common_reward=True)
class GoalReward(ArrayReward):
    def compute(self, prev_state, state, actions, reward_config):
        from cogrid.backend import xp
        goal_id = reward_config["type_ids"].get("goal", -1)
        otm = state["object_type_map"]
        rows = state["agent_pos"][:, 0]
        cols = state["agent_pos"][:, 1]
        return (otm[rows, cols] == goal_id).astype(xp.float32)

# 4. Create env with config -- NO env subclass needed
config = {
    "name": "goal_finding",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["agent_position", "full_map_encoding"],
    "rewards": [],
    "grid": {"layout": "goal_simple_v0"},
    "max_steps": 50,
    "scope": "global",
}
env = CoGridEnv(config=config, backend="numpy")
```

**Critical question:** Does CoGridEnv.__init__() work when there are no registered rewards for a scope? Looking at `build_reward_config_from_components()`: if `reward_metas` is empty, `instances` is empty, and `compute_fn` returns `xp.zeros(n_agents)` every step. This is correct behavior for a scope with no rewards. But for the goal-finding example, the GoalReward is registered with `scope="global"`, and `build_reward_config_from_components("global", ...)` will pick it up via `get_reward_types("global")`. This works.

**Another question:** Does CoGridEnv.__init__() currently compute `type_ids` before it's passed to `build_reward_config_from_components()`? Yes -- `build_scope_config_from_components()` returns a dict with `type_ids`, so `self._type_ids = self._scope_config["type_ids"]` is available before the reward config call.

### Anti-Patterns to Avoid

- **Keeping the scope_config registry as a "fallback":** This creates two code paths and confusion about which one is active. Delete it.
- **Making CoGridEnv.__init__() call BOTH get_scope_config() and build_scope_config_from_components():** Defeats the purpose. One path only.
- **Leaving `register_symbols()` calls that duplicate autowired symbol_table:** Redundant data that can go stale if components change.
- **Deleting `build_overcooked_scope_config()` before parity is fully verified:** The Phase 13 parity tests already compare auto-wired vs manual configs. But the end-to-end parity test (running actual environments through step()) must pass BEFORE deleting the manual path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Scope config assembly in CoGridEnv | Manual dict building per scope | `build_scope_config_from_components(scope)` | Already proven by 8 parity tests in test_overcooked_migration.py |
| Reward config assembly in CoGridEnv | Manual import + if-branch per scope | `build_reward_config_from_components(scope, ...)` | Already composes compute_fn from registered ArrayReward subclasses |
| Symbol table registration | Manual register_symbols() calls | Component char auto-population in autowire | Symbols are already auto-populated from `meta.char` in _build_symbol_table() |
| End-to-end parity tests | New bespoke test infrastructure | Extend existing test_cross_backend_parity.py patterns | The scripted parity, vmap, and JIT test patterns are proven and can be adapted |

**Key insight:** Phase 14 is a wiring change, not an algorithmic change. The auto-wiring produces identical output to manual wiring (proven by Phase 13 parity tests). The work is: change the call site in CoGridEnv, verify existing tests pass, add end-to-end verification, delete dead code.

## Common Pitfalls

### Pitfall 1: Import Order / Registration Timing

**What goes wrong:** `build_scope_config_from_components("overcooked")` returns empty results because the Overcooked GridObject and ArrayReward subclasses haven't been imported yet when CoGridEnv.__init__() runs.

**Why it happens:** Component registration happens at import time via decorators. If `cogrid.envs.overcooked.overcooked_grid_objects` hasn't been imported before `build_scope_config_from_components("overcooked")` is called, no components are registered for scope "overcooked".

**How to avoid:** The current Overcooked environment is created via `registry.make("Overcooked-CrampedRoom-V0")`, which calls `functools.partial(overcooked.Overcooked, config=...)`. The `overcooked.Overcooked` import triggers `cogrid.envs.overcooked.overcooked` -> `cogrid.envs.overcooked` -> `cogrid.envs.overcooked.__init__` which currently imports `overcooked_grid_objects` indirectly. After Phase 14, `overcooked/__init__.py` must still ensure `overcooked_grid_objects` and `array_rewards` are imported so that components and rewards are registered. Explicit `import cogrid.envs.overcooked.overcooked_grid_objects` and `import cogrid.envs.overcooked.array_rewards` in `__init__.py` suffices.

**Warning signs:** `build_scope_config_from_components("overcooked")` returns empty type_ids, no tick_handler, no interaction_body.

### Pitfall 2: build_reward_config Needs action_pickup_drop_idx Before It's Used

**What goes wrong:** The reward config's `compute_fn` closure captures `action_pickup_drop_idx` from the reward_config dict. If `action_pickup_drop_idx` is not set in the reward_config when `build_reward_config_from_components()` is called, reward computation may fail.

**Why it happens:** In the current code, `self._action_pickup_drop_idx` is computed at lines 230-232. The manual reward_config construction at lines 248-253 includes it. `build_reward_config_from_components()` accepts it as a parameter (default 4). So the action index must be computed before calling the autowire function.

**How to avoid:** Ensure the action set resolution and `action_pickup_drop_idx` computation happens BEFORE the `build_reward_config_from_components()` call. This is already the case in the existing code structure -- just preserve the ordering when refactoring.

**Warning signs:** `KeyError: 'action_pickup_drop_idx'` in reward compute, or rewards always being 0 because the action index doesn't match.

### Pitfall 3: state_extractor Function Used in layout_to_array_state

**What goes wrong:** The `layout_to_array_state()` function (called in `cogrid_env.py` reset) uses `scope_config.get("state_extractor")` to extract pot state from the Grid object. The autowired config does NOT have a `state_extractor` key (it uses `extra_state_builder` which works on parsed arrays, not Grid objects).

**Why it happens:** There are two different paths for building initial state:
1. **Grid-object path** (current CoGridEnv): `_gen_grid() -> Grid.decode() -> layout_to_array_state(grid, scope_config)` -- uses `state_extractor` to extract pot contents from Grid.Pot objects.
2. **Array-only path** (parse_layout): `parse_layout(layout_strings, scope, scope_config)` -- uses `extra_state_builder` to build pot arrays from parsed_arrays.

The current `cogrid_env.py` uses path 1. The `state_extractor` in the manual scope_config points to `_extract_overcooked_state()` which iterates Grid cells to find Pot objects and build pot arrays. The autowired config has `state_extractor=None` (not a recognized component classmethod).

**How to avoid:** Two options:
- (a) Set `state_extractor` in the autowired config. Add it as a recognized classmethod or pass-through. But this means the Grid-object path is still used.
- (b) Switch CoGridEnv to use the array-only path (parse_layout + extra_state_builder). This is cleaner but a larger change -- the current reset() builds a Grid, then converts to arrays. parse_layout() would skip the Grid entirely.
- (c) Keep using the Grid-object path but compute extra_state separately: after `layout_to_array_state()`, call the autowired `extra_state_builder` on the parsed arrays. This requires the parsed arrays (object_type_map) which `layout_to_array_state()` already produces.

**Recommendation:** Option (c) is safest. After `layout_to_array_state()` returns `self._array_state`, call `scope_config["extra_state_builder"](parsed_arrays, scope)` and merge the result into `self._array_state`. This is analogous to what the current code does with `state_extractor` but uses the component-based builder.

But wait -- checking the current code more carefully: `layout_to_array_state()` at line 89-92 calls `state_extractor(grid, scope)` which extracts pot state from Grid objects. The autowired `extra_state_builder(parsed_arrays, scope)` takes parsed_arrays (dict with object_type_map), not a Grid object. So after `layout_to_array_state()` returns, we can call `extra_state_builder(self._array_state, scope)` separately.

Actually, the simplest approach: pass `state_extractor=None` (which is the autowired default), then after `layout_to_array_state()`, use the extra_state_builder:

```python
self._array_state = layout_to_array_state(self.grid, scope=self.scope, scope_config=self._scope_config)
# Build extra state via autowired builder
if self._scope_config.get("extra_state_builder"):
    extra = self._scope_config["extra_state_builder"](self._array_state, self.scope)
    self._array_state.update(extra)
```

**Warning signs:** Missing `pot_contents`, `pot_timer`, `pot_positions` in `self._array_state` after reset, causing errors when building layout_arrays for the step pipeline.

### Pitfall 4: Interaction Tables in JAX Conversion Block

**What goes wrong:** The JAX setup block in `cogrid_env.py` (lines 291-297) converts `self._scope_config["interaction_tables"]` to JAX arrays. The autowired config has `interaction_tables=None`, so the conversion is a no-op. But if any code downstream reads from `interaction_tables`, it will get None instead of a dict.

**Why it happens:** The manual config sets `interaction_tables` to a dict with `pickup_from_produces` and `legal_pot_ingredients` arrays. The autowired config sets it to None. These arrays are now in `static_tables` (merged by the autowire from Pot.build_static_tables()). The interaction body reads from `static_tables`, NOT `interaction_tables`.

**How to avoid:** The JAX conversion block already has an `if ... is not None` guard. The `self._interaction_tables = self._scope_config["interaction_tables"]` assignment at line 211 will be None, which is fine as long as nothing reads from it. Verify no downstream code uses `self._interaction_tables`.

**Warning signs:** `AttributeError: 'NoneType' object has no attribute 'items'` in the JAX conversion block.

### Pitfall 5: Overcooked Subclass vs Base CoGridEnv

**What goes wrong:** The `Overcooked(CoGridEnv)` subclass doesn't override anything except the agent class. But the environment registry creates Overcooked instances (not CoGridEnv). After Phase 14, the Overcooked subclass can be eliminated OR kept as a thin wrapper for backward compatibility.

**Why it happens:** The Overcooked subclass exists historically to provide scope-specific customization. With auto-wiring, the base CoGridEnv handles everything generically.

**How to avoid:** Keep the Overcooked subclass for now (it sets `agent_class=OvercookedAgent`). The goal-finding example (INTG-04) proves that the base CoGridEnv works without a subclass. The Overcooked subclass is not "dead code" because it sets the agent class.

**Warning signs:** None -- the subclass is harmless.

### Pitfall 6: Tests That Import build_overcooked_scope_config Directly

**What goes wrong:** After deleting `build_overcooked_scope_config()`, tests that import it directly will fail with ImportError.

**Why it happens:** These tests exist:
- `test_overcooked_migration.py` -- imports `build_overcooked_scope_config` for parity comparison
- `test_step_pipeline.py` -- imports it for scope_config setup
- `test_reward_parity.py` -- imports it for type_ids
- `test_cross_backend_parity.py` -- creates envs via registry (does NOT import it directly)
- `test_interactions.py` -- imports it for scope_config setup

**How to avoid:** Before deleting `build_overcooked_scope_config()`, update all tests to use `build_scope_config_from_components("overcooked")` instead. The parity test (test_overcooked_migration.py) becomes a regression test (verify auto-wired config still produces correct results) rather than a comparison test.

**Warning signs:** `ImportError: cannot import name 'build_overcooked_scope_config'`.

## Code Examples

### Example 1: Rewired CoGridEnv.__init__() -- Scope Config Section

```python
# In cogrid/cogrid_env.py __init__() -- REPLACE lines 204-211:

# Build scope config via auto-wiring from registered components
from cogrid.core.autowire import (
    build_scope_config_from_components,
    build_reward_config_from_components,
)
self._scope_config = build_scope_config_from_components(self.scope)
self._type_ids = self._scope_config["type_ids"]
self._interaction_tables = self._scope_config.get("interaction_tables")
```

### Example 2: Rewired CoGridEnv.__init__() -- Reward Config Section

```python
# In cogrid/cogrid_env.py __init__() -- REPLACE lines 237-257:

# Auto-wire reward config from registered ArrayReward subclasses
self._reward_config = build_reward_config_from_components(
    self.scope,
    n_agents=self.config["num_agents"],
    type_ids=self._type_ids,
    action_pickup_drop_idx=self._action_pickup_drop_idx,
)
```

### Example 3: Updated reset() Extra State Handling

```python
# In cogrid/cogrid_env.py reset() -- REPLACE or AUGMENT the array_state building:

self._array_state = layout_to_array_state(
    self.grid, scope=self.scope, scope_config=self._scope_config
)

# Build extra state from autowired builder (if any)
extra_state_builder = self._scope_config.get("extra_state_builder")
if extra_state_builder is not None:
    extra = extra_state_builder(self._array_state, self.scope)
    self._array_state.update(extra)

agent_arrays = create_agent_arrays(self.env_agents, scope=self.scope)
self._array_state.update(agent_arrays)
```

### Example 4: Updated overcooked/__init__.py

```python
# cogrid/envs/overcooked/__init__.py -- AFTER Phase 14

# Trigger registration of component classmethods and reward subclasses
from cogrid.envs.overcooked import overcooked_features
from cogrid.envs.overcooked import overcooked_grid_objects  # noqa: F401
from cogrid.envs.overcooked import array_rewards  # noqa: F401

# No more register_scope_config() or register_symbols() calls.
# Components and rewards are auto-discovered via decorators.
```

### Example 5: End-to-End Parity Test

```python
# In cogrid/tests/test_phase14_parity.py

def test_autowired_overcooked_matches_manual_step_outputs():
    """Auto-wired Overcooked produces identical step outputs to v1.1 manual path.

    Creates two environments: one with auto-wired scope_config (Phase 14)
    and one with manual scope_config (v1.1). Runs 50 scripted steps on
    both and compares state, obs, rewards at every step.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    # Create auto-wired env (the new default)
    from cogrid.core.autowire import build_scope_config_from_components
    auto_scope_config = build_scope_config_from_components("overcooked")

    # Create env using auto-wired config
    auto_env = _create_env_with_config(auto_scope_config, backend="numpy")
    auto_env.reset(seed=42)

    # Create env using manual config (v1.1)
    from cogrid.envs.overcooked.array_config import build_overcooked_scope_config
    manual_scope_config = build_overcooked_scope_config()

    manual_env = _create_env_with_config(manual_scope_config, backend="numpy")
    manual_env.reset(seed=42)

    # Run 50 steps with identical scripted actions
    for step_idx in range(50):
        actions = {0: step_idx % 7, 1: (step_idx + 3) % 7}
        auto_obs, auto_rew, _, _, _ = auto_env.step(actions)
        manual_obs, manual_rew, _, _, _ = manual_env.step(actions)

        # Compare rewards
        for aid in [0, 1]:
            np.testing.assert_allclose(auto_rew[aid], manual_rew[aid], atol=1e-7)
```

## Files That Change

| File | What Changes | Estimated Lines | Risk |
|------|-------------|-----------------|------|
| `cogrid/cogrid_env.py` | Replace get_scope_config + manual reward assembly with autowire calls; handle extra_state_builder in reset | ~20 lines changed | MEDIUM -- central wiring change |
| `cogrid/envs/overcooked/__init__.py` | Remove register_scope_config and register_symbols calls; add explicit grid_objects/rewards imports | ~10 lines changed | LOW -- cleanup |
| `examples/goal_finding.py` | Convert to component API (ArrayReward subclass, no env subclass) | ~30 lines changed | LOW -- example file |
| `cogrid/tests/test_overcooked_migration.py` | Update to test auto-wired env end-to-end (not just config comparison) | ~50 lines changed | LOW -- test update |
| `cogrid/tests/test_step_pipeline.py` | Replace build_overcooked_scope_config import with autowire | ~5 lines changed | LOW -- test update |
| `cogrid/tests/test_reward_parity.py` | Replace build_overcooked_scope_config import with autowire | ~5 lines changed | LOW -- test update |
| `cogrid/tests/test_cross_backend_parity.py` | May need update if compute_rewards import changes | ~2 lines changed | LOW -- test update |
| `cogrid/envs/overcooked/test_interactions.py` | Replace build_overcooked_scope_config import with autowire | ~5 lines changed | LOW -- test update |
| `cogrid/envs/overcooked/array_config.py` | DELETE build_overcooked_scope_config, _build_type_ids. Keep overcooked_tick_state, overcooked_interaction_body, build_overcooked_extra_state (referenced by component classmethods) | -50 lines | MEDIUM -- must verify no callers remain |
| `cogrid/envs/overcooked/array_rewards.py` | DELETE compute_rewards dispatcher. Keep standalone reward functions + ArrayReward subclasses | -50 lines | MEDIUM -- must verify no callers remain |
| `cogrid/core/scope_config.py` | Evaluate for deletion or simplification | TBD | LOW -- downstream analysis needed |
| `cogrid/tests/test_phase14_parity.py` | NEW: End-to-end parity + JIT + vmap tests for auto-wired Overcooked | +100-150 lines | N/A -- new test file |

## Key Technical Details

### What build_scope_config_from_components Returns vs What CoGridEnv Needs

The autowired config returns:
```python
{
    "scope": scope,
    "interaction_tables": None,  # autowire default
    "type_ids": {...},
    "state_extractor": None,  # autowire default
    "tick_handler": <fn>,
    "interaction_body": <fn>,
    "static_tables": {...},  # includes base + Overcooked-specific tables
    "symbol_table": {...},
    "extra_state_schema": {...},
    "extra_state_builder": <fn>,
}
```

CoGridEnv.__init__() reads:
- `self._scope_config["type_ids"]` -- CHECK: available
- `self._scope_config["interaction_tables"]` -- CHECK: None (acceptable, guard exists)
- `scope_config.get("static_tables")` -- CHECK: available (used in JAX conversion block)
- `scope_config.get("tick_handler")` -- CHECK: available (consumed by step_pipeline)
- `scope_config.get("interaction_body")` -- CHECK: available (consumed by step_pipeline)
- `scope_config.get("extra_state_builder")` -- CHECK: available (consumed in reset)
- `scope_config.get("state_extractor")` -- CHECK: None (need to handle separately)

The `state_extractor` gap is the main technical risk. See Pitfall 3.

### scope_config.py Module -- Can It Be Deleted?

The `scope_config.py` module provides:
- `SCOPE_CONFIG_REGISTRY` dict
- `register_scope_config()` function
- `get_scope_config()` function
- `default_scope_config()` function

After Phase 14, no code should call `get_scope_config()` or `register_scope_config()`. The module can be deleted. But check: does `default_scope_config()` get called anywhere else? Searching the codebase: only by `get_scope_config()` when no registered builder exists for a scope. After Phase 14, this path is never taken.

**Recommendation:** Delete scope_config.py in the dead-code cleanup plan. It serves no purpose once auto-wiring is the default path.

### layout_parser.py register_symbols -- Impact of Removal

After removing `register_symbols("overcooked", {...})` from `overcooked/__init__.py`:
- `SYMBOL_REGISTRY` will NOT have an "overcooked" entry.
- `get_symbols("overcooked", scope_config)` will fall through to `scope_config["symbol_table"]`.
- This works because `build_scope_config_from_components("overcooked")` populates `symbol_table`.
- But `parse_layout()` is only used by the functional API (not by CoGridEnv's Grid-object path).
- The Grid-object path (`Grid.decode()`) uses the object registry directly.

**Conclusion:** Removing `register_symbols()` is safe. The `parse_layout()` fallback to `scope_config["symbol_table"]` works. And CoGridEnv doesn't use `parse_layout()`.

### compute_rewards Dispatcher -- Who Still Calls It?

The manual `compute_rewards()` function in `cogrid/envs/overcooked/array_rewards.py` (line 263) is called from:
1. `cogrid/cogrid_env.py` line 257: `self._reward_config["compute_fn"] = compute_rewards` -- WILL BE REPLACED by autowire
2. `cogrid/tests/test_step_pipeline.py` line 55: `"compute_fn": compute_rewards` -- MUST UPDATE
3. `cogrid/tests/test_cross_backend_parity.py` line 547: `from cogrid.envs.overcooked.array_rewards import compute_rewards` -- MUST UPDATE
4. `cogrid/tests/test_reward_parity.py` line 332: `ar_mod.compute_rewards(...)` -- MUST UPDATE

After updating all callers, the function can be deleted.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `get_scope_config()` -> manual builder per scope | `build_scope_config_from_components(scope)` auto-wires from registry | Phase 14 (this phase) | Adding new environments requires only GridObject subclasses + layout, no manual scope_config |
| `if self.scope == "overcooked"` branch for reward_config | `build_reward_config_from_components(scope, ...)` composes from registered ArrayReward subclasses | Phase 14 (this phase) | Zero scope-specific branches in CoGridEnv |
| Manual `register_symbols()` per scope | Component char auto-population in autowire's `_build_symbol_table()` | Phase 14 (this phase) | No manual symbol registration needed |
| `register_scope_config()` + SCOPE_CONFIG_REGISTRY | Auto-wiring from component registry | Phase 14 (this phase) | scope_config.py module can be deleted |
| GoalFindingEnv subclass with manual reward_config | Direct CoGridEnv usage with ArrayReward registration | Phase 14 (this phase) | New environments need zero CoGridEnv subclassing |

**Deprecated/outdated after Phase 14:**
- `cogrid/core/scope_config.py` -- entire module replaced by autowire
- `build_overcooked_scope_config()` in array_config.py -- replaced by auto-wiring
- `_build_type_ids()` in array_config.py -- replaced by autowire type_ids composition
- `compute_rewards()` dispatcher in array_rewards.py -- replaced by autowire's composed compute_fn
- `register_scope_config("overcooked", ...)` call in `__init__.py`
- `register_symbols("overcooked", ...)` call in `__init__.py`
- `if self.scope == "overcooked"` branch in cogrid_env.py
- `compose_rewards()` in core/array_rewards.py -- superseded by autowire (check if used)

## Open Questions

1. **Should scope_config.py be deleted or kept as a stub?**
   - What we know: After Phase 14, no production code calls `get_scope_config()` or `register_scope_config()`. The module is dead.
   - What's unclear: Whether any third-party code or examples reference it.
   - Recommendation: Delete it. If third-party code breaks, they should migrate to the auto-wiring API. Keep a deprecation comment in the commit message.

2. **Should CoGridEnv.reset() switch to parse_layout() instead of Grid.decode()?**
   - What we know: The current reset uses the Grid-object path (_gen_grid -> Grid.decode -> layout_to_array_state). parse_layout() is the array-only alternative. Both produce equivalent results.
   - What's unclear: Whether switching to parse_layout() would break rendering (which needs Grid objects).
   - Recommendation: Do NOT switch in Phase 14. The Grid-object path is needed for rendering. Switching is a future simplification task.

3. **Should the Overcooked env subclass be deleted?**
   - What we know: `Overcooked(CoGridEnv)` only sets `agent_class=OvercookedAgent`. After Phase 14, everything else is handled by auto-wiring.
   - What's unclear: Whether OvercookedAgent has special behavior that would be needed even with auto-wiring.
   - Recommendation: Keep the Overcooked subclass. It serves a real purpose (setting the agent class). The goal-finding example proves the base CoGridEnv works without a subclass. Deleting Overcooked would break the existing environment registry entries.

4. **How should the state_extractor gap be handled?**
   - What we know: The autowired config has `state_extractor=None`. The current CoGridEnv uses `state_extractor` in `layout_to_array_state()` to extract pot arrays from Grid objects. The autowired `extra_state_builder` works on parsed arrays (dict with object_type_map).
   - What's unclear: Whether `layout_to_array_state()` should be modified to use `extra_state_builder` instead of `state_extractor`, or whether CoGridEnv should call them separately.
   - Recommendation: Call `extra_state_builder` separately in CoGridEnv.reset() after `layout_to_array_state()`. This is a 3-line change and avoids modifying `layout_to_array_state()`.

5. **Should existing vmap tests be rerun with auto-wired config or should new tests be written?**
   - What we know: Existing `test_vmap_correctness.py` creates environments via `registry.make()`, which uses the manual wiring path. After Phase 14, `registry.make()` will use auto-wiring automatically (since CoGridEnv.__init__() will call autowire).
   - What's unclear: Whether we need BOTH the existing tests (now running through auto-wiring) AND a new explicit auto-wiring vmap test.
   - Recommendation: The existing vmap tests will implicitly validate auto-wiring after the switch. Add one explicit test that creates an auto-wired env and runs jit(vmap(step)) at 1024 to satisfy TEST-02 explicitly.

## Sources

### Primary (HIGH confidence)

- **cogrid/cogrid_env.py** -- Complete read (1192 lines). Contains CoGridEnv.__init__() with scope_config wiring at lines 204-211, reward_config at lines 237-257, reset() at lines 399-523, state_extractor usage in layout_to_array_state.
- **cogrid/core/autowire.py** -- Complete read (261 lines). Contains `build_scope_config_from_components()` (lines 17-143) and `build_reward_config_from_components()` (lines 203-261). Fully composed with tick/interaction/extra_state_builder/static_tables from Phase 13.
- **cogrid/core/scope_config.py** -- Complete read (80 lines). Contains `SCOPE_CONFIG_REGISTRY`, `register_scope_config()`, `get_scope_config()`, `default_scope_config()`.
- **cogrid/core/layout_parser.py** -- Complete read (225 lines). Contains `SYMBOL_REGISTRY`, `register_symbols()`, `get_symbols()` fallback logic, `parse_layout()`.
- **cogrid/core/step_pipeline.py** -- Complete read (456 lines). Contains `step()`, `reset()`, `build_step_fn()`, `build_reset_fn()`. Consumes scope_config and reward_config.
- **cogrid/core/grid_utils.py** -- Complete read (120 lines). Contains `layout_to_array_state()` with `state_extractor` usage at lines 89-92.
- **cogrid/envs/overcooked/__init__.py** -- Complete read (18 lines). Contains `register_scope_config` and `register_symbols` calls to be removed.
- **cogrid/envs/overcooked/overcooked.py** -- Complete read (33 lines). Contains Overcooked subclass (only sets agent_class).
- **cogrid/envs/overcooked/array_config.py** -- Complete read (591 lines). Contains `build_overcooked_scope_config()`, `_build_type_ids()`, `compute_rewards()` dispatcher -- dead code candidates.
- **cogrid/envs/overcooked/array_rewards.py** -- Complete read (362 lines). Contains standalone reward functions (KEEP), `compute_rewards()` dispatcher (DELETE), ArrayReward subclasses (KEEP).
- **cogrid/tests/test_overcooked_migration.py** -- Complete read (196 lines). 8 parity tests comparing auto-wired vs manual scope_config.
- **cogrid/tests/test_cross_backend_parity.py** -- Complete read (604 lines). Cross-backend parity tests -- pattern for end-to-end parity.
- **cogrid/tests/test_vmap_correctness.py** -- Complete read (480 lines). vmap tests at 1024 envs -- pattern for JIT+vmap validation.
- **cogrid/tests/test_step_pipeline.py** -- Read (355 lines). Uses build_overcooked_scope_config -- needs update.
- **cogrid/tests/test_reward_parity.py** -- Read. Uses build_overcooked_scope_config -- needs update.
- **cogrid/envs/overcooked/test_interactions.py** -- Read (345 lines). Uses build_overcooked_scope_config -- needs update.
- **examples/goal_finding.py** -- Complete read (260 lines). Current goal-finding example with manual reward_config -- migration target.
- **cogrid/envs/__init__.py** -- Complete read (195 lines). Environment registry entries.
- **cogrid/envs/registry.py** -- Complete read (23 lines). Environment registry.
- **cogrid/core/component_registry.py** -- Complete read (247 lines). Component and reward metadata registries.
- **cogrid/core/array_rewards.py** -- Complete read (80 lines). ArrayReward base class, compose_rewards.
- **cogrid/backend/env_state.py** -- Read (60 lines). EnvState dataclass.
- **.planning/phases/13-overcooked-migration/13-VERIFICATION.md** -- Phase 13 verification report.
- **.planning/phases/13-overcooked-migration/13-03-SUMMARY.md** -- Phase 13 plan 03 summary.

### Secondary (MEDIUM confidence)

- **.planning/phases/13-overcooked-migration/13-RESEARCH.md** -- Phase 13 research documenting autowire patterns and composition rules.
- **.planning/ROADMAP.md** -- Phase 14 definition, success criteria, requirements.
- **.planning/STATE.md** -- Current project state and accumulated decisions.

### Tertiary (LOW confidence)

None -- this is entirely internal codebase work with no external dependencies.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- no external deps, pure internal refactor using Phase 10-13 infrastructure
- Architecture: HIGH -- all source files read in full, all function signatures and data flows mapped, wiring changes documented with specific line references
- Pitfalls: HIGH -- all pitfalls identified from actual code reading (import ordering, state_extractor gap, dead code callers, action_pickup_drop_idx ordering)

**Research date:** 2026-02-13
**Valid until:** Indefinite (internal codebase, no external dependency changes)
