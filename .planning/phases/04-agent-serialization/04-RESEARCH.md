# Phase 4: Agent Serialization - Research

**Researched:** 2026-01-19
**Domain:** Agent state serialization (GridAgent, Agent, OvercookedAgent)
**Confidence:** HIGH

## Summary

Phase 4 research confirms that agent serialization is **already fully implemented** and working. The base `Agent` class (in `cogrid/core/agent.py`) has complete `get_state()` and `from_state()` methods that handle all agent state including inventory with full object state preservation. The `OvercookedAgent` class inherits from `Agent` without adding any additional stateful attributes that require serialization.

Existing tests in `cogrid/envs/overcooked/test_state_serialization.py` already verify that agent inventory serialization works correctly. The `test_agent_inventory_serialization` test passes, demonstrating that agents holding `Plate` objects roundtrip correctly.

**Primary recommendation:** This phase is verification-only. Add roundtrip tests for edge cases (agents holding complex objects like OnionSoup, agents holding Plates with soup) to ensure complete coverage, but no new serialization code is needed.

## Standard Stack

### Core Classes Analyzed

| Class | Location | Serialization Status |
|-------|----------|---------------------|
| `Agent` | `cogrid/core/agent.py` | COMPLETE - has `get_state()` and `from_state()` |
| `GridAgent` | `cogrid/core/grid_object.py` | N/A - ephemeral render object, regenerated each step |
| `OvercookedAgent` | `cogrid/envs/overcooked/agent.py` | INHERITS from Agent - no additional state |

### Key Insight: GridAgent vs Agent Distinction

There are two distinct agent concepts in the codebase:

1. **`Agent`** (in `cogrid/core/agent.py`): The actual agent with persistent state (position, direction, inventory, terminated, etc.). This IS serialized.

2. **`GridAgent`** (in `cogrid/core/grid_object.py`): A transient `GridObj` subclass used for rendering agents on the grid. Created fresh every step via `update_grid_agents()`. NOT directly serialized - regenerated from `Agent` state.

This distinction is critical: the audit script in Phase 1 flagged `GridAgent` as needing serialization, but `GridAgent` is actually ephemeral. The real agent state lives in `Agent`, which already serializes.

## Architecture Patterns

### Existing Agent Serialization Pattern

**Location:** `cogrid/core/agent.py`, lines 109-176

The `Agent.get_state()` method already serializes:
- `id`: Agent identifier
- `pos`: Position tuple
- `dir`: Direction integer
- `role` and `role_idx`: Role information
- `terminated`: Termination flag
- `collision`: Collision flag
- `orientation`: Orientation string
- `inventory_capacity`: Inventory limit
- `inventory`: List of held objects with FULL recursive state

```python
# Existing implementation (verified in codebase)
def get_state(self, scope: str = "global") -> dict:
    return {
        "id": self.id,
        "pos": tuple(self.pos),
        "dir": int(self.dir),
        "role": self.role,
        "role_idx": self.role_idx,
        "terminated": self.terminated,
        "collision": self.collision,
        "orientation": self.orientation,
        "inventory_capacity": self.inventory_capacity,
        "inventory": [
            {
                "object_id": obj.object_id,
                "state": obj.state,
                "extra_state": obj.get_extra_state(scope),
            }
            for obj in self.inventory
        ],
    }
```

The `from_state()` classmethod reconstructs agents including inventory objects with their extra state.

### OvercookedAgent Analysis

**Location:** `cogrid/envs/overcooked/agent.py`

The `OvercookedAgent` class extends `Agent` with ONE method override:
- `can_pickup()`: Modified logic for pot/plate interaction

**Key finding:** `OvercookedAgent` adds NO additional stateful attributes. It only modifies behavior through method override, not through new state. Therefore, the base `Agent` serialization handles `OvercookedAgent` completely.

### Environment Integration

**Location:** `cogrid/cogrid_env.py`, lines 1141-1316

The `CoGridEnv.get_state()` and `set_state()` methods already:
1. Serialize all agents via `agent.get_state(scope=self.scope)`
2. Restore agents via `self.agent_class.from_state(agent_data, scope=self.scope)`
3. Call `update_grid_agents()` after restoration to regenerate `GridAgent` objects

This pattern correctly handles domain-specific agents (like `OvercookedAgent`) because:
- The environment stores `agent_class` (e.g., `OvercookedAgent`)
- `from_state()` is a classmethod called on `self.agent_class`
- The restored agent is the correct subclass type

## Don't Hand-Roll

| Problem | Don't Build | Already Exists |
|---------|-------------|----------------|
| Agent state serialization | Custom serialization | `Agent.get_state()` / `from_state()` |
| Inventory object serialization | Manual inventory handling | Recursive pattern in `get_state()` |
| GridAgent serialization | Direct GridAgent save/restore | Regenerated via `update_grid_agents()` |

## Common Pitfalls

### Pitfall 1: Confusing GridAgent with Agent

**What goes wrong:** Attempting to serialize `GridAgent` objects directly.
**Why it happens:** The audit script flags `GridAgent` as a class without serialization.
**How to avoid:** Understand that `GridAgent` is ephemeral - it's recreated from `Agent` state each step.
**Warning signs:** Looking for `get_extra_state()` on `GridAgent` class.

### Pitfall 2: Missing extra_state for Complex Inventory Items

**What goes wrong:** Agent holding an object with extra state (like a Pot with contents) loses nested state.
**Why it happens:** Forgetting that inventory objects need their `extra_state` serialized too.
**How to avoid:** The existing implementation already handles this - it calls `obj.get_extra_state(scope)` for each inventory item.
**Warning signs:** N/A - already handled correctly.

### Pitfall 3: Domain-Specific Agent State

**What goes wrong:** Assuming domain agents (like `OvercookedAgent`) need special serialization.
**Why it happens:** Not checking if subclass adds stateful attributes.
**How to avoid:** Verify that subclasses only override methods, not add state.
**Warning signs:** Looking for additional attributes in domain agent classes.

## Code Examples

### Existing Agent Serialization (Verified Working)

```python
# From cogrid/core/agent.py - already implemented
def get_state(self, scope: str = "global") -> dict:
    return {
        "id": self.id,
        "pos": tuple(self.pos),
        "dir": int(self.dir),
        "role": self.role,
        "role_idx": self.role_idx,
        "terminated": self.terminated,
        "collision": self.collision,
        "orientation": self.orientation,
        "inventory_capacity": self.inventory_capacity,
        "inventory": [
            {
                "object_id": obj.object_id,
                "state": obj.state,
                "extra_state": obj.get_extra_state(scope),
            }
            for obj in self.inventory
        ],
    }

@classmethod
def from_state(cls, state_dict: dict, scope: str = "global"):
    from cogrid.core.grid_object import make_object
    from cogrid.core.directions import Directions

    agent = cls(
        agent_id=state_dict["id"],
        start_position=state_dict["pos"],
        start_direction=Directions(state_dict["dir"]),
        inventory_capacity=state_dict.get("inventory_capacity", 1),
    )

    agent.role = state_dict.get("role")
    agent.role_idx = state_dict.get("role_idx")
    agent.terminated = state_dict["terminated"]
    agent.collision = state_dict["collision"]
    agent.orientation = state_dict["orientation"]

    # Restore inventory with full object state
    agent.inventory = []
    for obj_data in state_dict["inventory"]:
        obj = make_object(
            obj_data["object_id"], state=obj_data["state"], scope=scope
        )
        if obj_data["extra_state"]:
            obj.set_extra_state(obj_data["extra_state"], scope)
        agent.inventory.append(obj)

    return agent
```

### Existing Test (Verified Passing)

```python
# From cogrid/envs/overcooked/test_state_serialization.py
def test_agent_inventory_serialization(self):
    """Test that agent inventory items are properly serialized."""
    self.env.reset(seed=42)

    # Give agent an item
    agent = self.env.env_agents[0]
    agent.inventory = [overcooked_grid_objects.Plate()]

    # Save and restore
    state = self.env.get_state()
    env2 = registry.make("OvercookedStateTest")
    env2.set_state(state)

    # Verify inventory restored
    restored_agent = env2.env_agents[0]
    self.assertEqual(len(restored_agent.inventory), 1)
    self.assertIsInstance(
        restored_agent.inventory[0], overcooked_grid_objects.Plate
    )
```

## State of the Art

| Previous Understanding | Actual State | Impact |
|----------------------|--------------|--------|
| GridAgent needs serialization | GridAgent is ephemeral, regenerated from Agent | No work needed on GridAgent |
| Agent serialization incomplete | Agent.get_state()/from_state() fully implemented | Phase is verification-only |
| OvercookedAgent has domain state | OvercookedAgent has no additional state attributes | Base Agent serialization sufficient |

## Requirements Analysis

### AGNT-01: Agent inventory serializes held objects with full state (not just type)

**Status:** ALREADY IMPLEMENTED

**Evidence:**
- `Agent.get_state()` serializes inventory items with `object_id`, `state`, AND `extra_state`
- `Agent.from_state()` reconstructs inventory items using `make_object()` and `set_extra_state()`
- Test `test_agent_inventory_serialization` verifies this works

**Recommendation:** Add additional tests for edge cases:
- Agent holding OnionSoup/TomatoSoup (objects with different object_ids)
- Agent holding objects that have extra_state (though stateless objects like Plate/Onion return None)

### AGNT-02: Domain-specific agent state (OvercookedAgent) serializes completely

**Status:** ALREADY IMPLEMENTED (no additional state to serialize)

**Evidence:**
- `OvercookedAgent` extends `Agent` with only a method override (`can_pickup`)
- No additional instance attributes are added in `OvercookedAgent.__init__` (it doesn't override `__init__`)
- The base `Agent` serialization handles all state

**Recommendation:** Add explicit test verifying `OvercookedAgent` roundtrip preserves type and behavior.

## Open Questions

None - all questions resolved through code analysis.

1. **Q: Does GridAgent need serialization?**
   - **A:** No. GridAgent is ephemeral, recreated each step from Agent state via `update_grid_agents()`.

2. **Q: Does OvercookedAgent have additional state?**
   - **A:** No. It only overrides `can_pickup()` method, no additional instance attributes.

3. **Q: Is inventory serialization recursive?**
   - **A:** Yes. `get_state()` captures `extra_state` for each inventory item, enabling nested state.

## Testing Approach

Since serialization is already implemented, testing should focus on:

1. **Edge case coverage** - Agents holding various object types:
   - Simple objects (Onion, Tomato, Plate)
   - Soup objects (OnionSoup, TomatoSoup)
   - Objects with extra_state (if any become valid inventory items)

2. **Type preservation** - Verify `OvercookedAgent` restored as `OvercookedAgent`, not base `Agent`

3. **Behavior preservation** - Verify `can_pickup()` behavior works correctly after roundtrip

## Sources

### Primary (HIGH confidence)
- `/Users/chasemcd/Repositories/cogrid/cogrid/core/agent.py` - Agent class with get_state/from_state
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/agent.py` - OvercookedAgent class
- `/Users/chasemcd/Repositories/cogrid/cogrid/cogrid_env.py` - Environment serialization methods
- `/Users/chasemcd/Repositories/cogrid/cogrid/core/grid_object.py` - GridAgent class (lines 422-557)
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/test_state_serialization.py` - Existing tests

### Secondary (MEDIUM confidence)
- Test execution confirming `test_agent_inventory_serialization` passes

## Metadata

**Confidence breakdown:**
- Agent serialization status: HIGH - verified in source code
- OvercookedAgent analysis: HIGH - verified class has no additional state
- GridAgent analysis: HIGH - verified regeneration pattern in `update_grid_agents()`
- Testing recommendations: HIGH - based on existing test patterns

**Research date:** 2026-01-19
**Valid until:** Indefinitely (core architecture unlikely to change)

## Conclusion

**Phase 4 is verification-only.** The serialization infrastructure for agents is already complete:

1. `Agent.get_state()` and `Agent.from_state()` handle all state including recursive inventory serialization
2. `OvercookedAgent` has no additional state requiring serialization
3. `GridAgent` is ephemeral and correctly regenerated from `Agent` state

**Recommended actions:**
1. Add roundtrip tests for various inventory object types
2. Add test verifying `OvercookedAgent` type preservation
3. No implementation work required
