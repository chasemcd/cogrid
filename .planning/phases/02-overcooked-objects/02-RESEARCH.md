# Phase 2: Overcooked Objects - Research

**Researched:** 2026-01-19
**Domain:** Overcooked domain object serialization for state checkpointing
**Confidence:** HIGH

## Summary

Phase 2 focuses on implementing serialization for Overcooked domain objects. After detailed analysis of the codebase, the findings show that **most Overcooked objects are already properly handled** and require minimal new implementation.

The audit script reveals:
- **Pot**: Already implements `get_extra_state()`/`set_extra_state()` (IMPLEMENTED)
- **Counter**: Already implements `get_extra_state()`/`set_extra_state()` in core (IMPLEMENTED)
- **OnionStack, TomatoStack, PlateStack**: STATELESS (infinite sources, no internal state)
- **Onion, Tomato, Plate, OnionSoup, TomatoSoup, DeliveryZone**: STATELESS (no attributes beyond base class)

The **only gap** is that the existing Pot serialization doesn't store the full state of objects_in_pot contents (it only stores `object_id`). This works because Onion/Tomato are stateless, but for completeness and future-proofing, this should be documented or enhanced.

**Primary recommendation:** Verify existing implementations via roundtrip tests for all requirements (OVER-01 through OVER-06). Add tests for edge cases. No new `get_extra_state()`/`set_extra_state()` implementations are needed.

## Object Analysis

### Objects with Existing Serialization

| Object | Status | Internal State | Serialized? | Notes |
|--------|--------|---------------|-------------|-------|
| `Pot` | IMPLEMENTED | `objects_in_pot`, `cooking_timer`, `capacity` | Yes | Lines 251-280 |
| `Counter` | IMPLEMENTED | `obj_placed_on` | Yes | In core/grid_object.py |

### Stateless Objects (No Serialization Needed)

| Object | Why Stateless | Verification |
|--------|---------------|--------------|
| `Onion` | Only uses base class `state=0`, no custom attributes | Lines 22-45 |
| `Tomato` | Only uses base class `state=0`, no custom attributes | Lines 49-73 |
| `OnionStack` | Infinite source, no count tracking | Lines 76-103 |
| `TomatoStack` | Infinite source, no count tracking | Lines 106-133 |
| `PlateStack` | Infinite source, no count tracking | Lines 286-323 |
| `Plate` | Only uses base class `state=0`, no custom attributes | Lines 326-352 |
| `OnionSoup` | Only uses base class `state=0`, no custom attributes | Lines 394-426 |
| `TomatoSoup` | Only uses base class `state=0`, no custom attributes | Lines 429-463 |
| `DeliveryZone` | Only uses base class `state=0`, no custom attributes | Lines 355-391 |

### Requirement Mapping

| Requirement | Object(s) | Current Status | Action Needed |
|-------------|-----------|---------------|---------------|
| OVER-01 | Pot | IMPLEMENTED | Verify roundtrip test |
| OVER-02 | Counter | IMPLEMENTED | Verify roundtrip test |
| OVER-03 | Plate | STATELESS | No implementation needed (Plate is stateless) |
| OVER-04 | OnionStack | STATELESS | No implementation needed (infinite source) |
| OVER-05 | TomatoStack | STATELESS | No implementation needed (infinite source) |
| OVER-06 | All | Mixed | Add comprehensive roundtrip tests |

## Standard Stack

The serialization infrastructure is already complete. Phase 1 established all necessary patterns.

### Core (Already Exists)
| Component | Location | Purpose |
|-----------|----------|---------|
| `GridObj.get_extra_state()` | `grid_object.py:252-299` | Base template method |
| `GridObj.set_extra_state()` | `grid_object.py:301-345` | Base restore method |
| `make_object()` | `grid_object.py:50-69` | Factory with scope support |
| `Pot.get_extra_state()` | `overcooked_grid_objects.py:251-263` | Pot serialization |
| `Pot.set_extra_state()` | `overcooked_grid_objects.py:265-280` | Pot restoration |
| `Counter.get_extra_state()` | `grid_object.py:612-629` | Counter serialization |
| `Counter.set_extra_state()` | `grid_object.py:631-645` | Counter restoration |

### Testing (Already Exists)
| Component | Location | Purpose |
|-----------|----------|---------|
| `test_state_serialization.py` | `envs/overcooked/` | Pot and Counter roundtrip tests |
| pytest framework | Standard | Test execution |

## Architecture Patterns

### Pattern 1: Recursive Serialization (Established in Phase 1)

**What:** Objects containing other objects serialize them recursively.

**Counter example (current implementation):**
```python
# Source: cogrid/core/grid_object.py lines 612-645
def get_extra_state(self, scope: str = "global") -> dict | None:
    if self.obj_placed_on is None:
        return None

    return {
        "obj_placed_on": {
            "object_id": self.obj_placed_on.object_id,
            "state": self.obj_placed_on.state,
            "extra_state": self.obj_placed_on.get_extra_state(scope),
        }
    }

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    if state_dict and "obj_placed_on" in state_dict:
        obj_data = state_dict["obj_placed_on"]
        self.obj_placed_on = make_object(
            obj_data["object_id"], state=obj_data["state"], scope=scope
        )
        if obj_data["extra_state"]:
            self.obj_placed_on.set_extra_state(obj_data["extra_state"], scope)
```

### Pattern 2: Simplified List Serialization (Pot)

**What:** When contained objects are stateless, store only `object_id`.

**Pot example (current implementation):**
```python
# Source: cogrid/envs/overcooked/overcooked_grid_objects.py lines 251-280
def get_extra_state(self, scope: str = "global") -> dict:
    return {
        "objects_in_pot": [obj.object_id for obj in self.objects_in_pot],
        "cooking_timer": self.cooking_timer,
        "capacity": self.capacity,
    }

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    from cogrid.core.grid_object import make_object

    self.objects_in_pot = [
        make_object(obj_id, scope=scope)
        for obj_id in state_dict["objects_in_pot"]
    ]
    self.cooking_timer = state_dict["cooking_timer"]
    self.capacity = state_dict.get("capacity", 3)
```

**Note:** This works because Onion/Tomato are stateless. If future ingredients have state, this pattern would need enhancement to full recursive serialization.

### Pattern 3: Stateless Objects (Default)

**What:** Objects with no attributes beyond base class inherit default `get_extra_state()` which returns `None`.

**Example:** All of Onion, Tomato, Plate, OnionSoup, TomatoSoup, DeliveryZone, OnionStack, TomatoStack, PlateStack.

**When it works:** Object has no `__init__` attributes beyond what's passed to `super().__init__()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Object factory | Custom instantiation | `make_object()` with scope | Already handles scope, registry |
| Nested object restoration | Manual construction | Recursive `set_extra_state()` | Pattern established in Counter |
| Environment-level serialization | Custom grid iteration | `env.get_state()`/`env.set_state()` | Already works with object_metadata |

**Key insight:** All infrastructure exists. This phase is about **verification**, not implementation.

## Common Pitfalls

### Pitfall 1: Assuming Stacks Track Count

**What goes wrong:** Developer assumes OnionStack/TomatoStack/PlateStack track remaining items.

**Reality:** These are infinite sources. The `pick_up_from()` method creates a new object each time:
```python
# Source: overcooked_grid_objects.py line 86-87
def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
    return Onion()  # Creates new Onion, no count decrement
```

**How to verify:** Check that requirements OVER-04 and OVER-05 are satisfied by the stateless nature.

### Pitfall 2: Plate Holding Soup Confusion

**What goes wrong:** Developer thinks Plate needs special serialization when holding soup.

**Reality:** In this implementation, soup is a separate object (OnionSoup/TomatoSoup), not a Plate with contents. When a player picks up soup from a pot, they get a soup object, not a Plate holding soup:
```python
# Source: overcooked_grid_objects.py lines 165-176
def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
    soup = OnionSoup()
    if all([isinstance(grid_obj, Tomato) for grid_obj in self.objects_in_pot]):
        soup = TomatoSoup()
    # ...
    return soup  # Returns soup, not Plate
```

**How to verify:** OVER-03 requirement may need clarification - in current implementation, Plate is stateless and soup is a separate object type.

### Pitfall 3: Missing Scope in Tests

**What goes wrong:** Tests use `scope="global"` when objects are registered in `scope="overcooked"`.

**How to avoid:** Always use `scope="overcooked"` for Overcooked object serialization:
```python
pot.get_extra_state(scope="overcooked")
make_object("onion", scope="overcooked")
```

### Pitfall 4: Pot Cooking State Confusion

**What goes wrong:** Developer confuses `cooking_timer` with `is_cooking`/`is_ready`.

**Clarification:**
- `cooking_timer`: The actual counter (decrements from `cooking_time` to 0)
- `is_cooking`: Computed property: `len(objects_in_pot) == capacity and cooking_timer > 0`
- `dish_ready` (alias `is_ready`): Computed property: `cooking_timer == 0`

**The serialization only needs to store `cooking_timer`** - the properties are derived.

## Code Examples

### Example 1: Testing Pot Roundtrip (Existing)

```python
# Source: cogrid/envs/overcooked/test_state_serialization.py lines 245-290
def test_pot_state_serialization(self):
    """Test that Pot objects with ingredients are properly serialized."""
    self.env.reset(seed=42)

    # Find a pot in the grid
    pot = self.env.grid.get(*pot_pos)

    # Manually add ingredients to the pot
    pot.objects_in_pot = [
        overcooked_grid_objects.Tomato(),
        overcooked_grid_objects.Tomato(),
    ]
    pot.cooking_timer = 25

    # Save state
    state = self.env.get_state()

    # Restore state
    env2 = registry.make("OvercookedStateTest")
    env2.set_state(state)

    # Verify pot state was restored
    restored_pot = env2.grid.get(*pot_pos)
    self.assertEqual(len(restored_pot.objects_in_pot), 2)
    self.assertEqual(restored_pot.cooking_timer, 25)
```

### Example 2: Testing Counter with Object (Existing)

```python
# Source: cogrid/envs/overcooked/test_state_serialization.py lines 363-393
def test_counter_with_object_serialization(self):
    """Test that Counter objects with items placed on them are serialized."""
    counter = self.env.grid.get(*counter_pos)
    counter.obj_placed_on = overcooked_grid_objects.Onion()

    # Save and restore
    state = self.env.get_state()
    env2 = registry.make("OvercookedStateTest")
    env2.set_state(state)

    # Verify object on counter restored
    restored_counter = env2.grid.get(*counter_pos)
    self.assertIsNotNone(restored_counter.obj_placed_on)
    self.assertIsInstance(
        restored_counter.obj_placed_on, overcooked_grid_objects.Onion
    )
```

### Example 3: Standalone Object Roundtrip Test Pattern

```python
# Pattern for testing individual object roundtrip
def test_standalone_pot_roundtrip():
    """Test Pot roundtrip without full environment."""
    # Create pot with state
    pot = Pot(state=0)
    pot.objects_in_pot = [Tomato(), Tomato(), Tomato()]
    pot.cooking_timer = 15

    # Serialize
    extra_state = pot.get_extra_state(scope="overcooked")

    # Create new pot and restore
    new_pot = Pot(state=0)
    new_pot.set_extra_state(extra_state, scope="overcooked")

    # Verify
    assert len(new_pot.objects_in_pot) == 3
    assert new_pot.cooking_timer == 15
    assert all(isinstance(obj, Tomato) for obj in new_pot.objects_in_pot)
```

## State of the Art

| Component | Status | Notes |
|-----------|--------|-------|
| Pot serialization | Complete | Implemented in Phase 1 or earlier |
| Counter serialization | Complete | In core grid_object.py |
| Test coverage | Partial | Environment-level tests exist, need standalone object tests |
| Stateless objects | N/A | No implementation needed |

**Key finding:** Most of this phase's requirements are already satisfied by existing implementations.

## Open Questions

1. **OVER-03: Plate holding soup**
   - What we know: Current implementation uses separate OnionSoup/TomatoSoup objects, not Plate+contents
   - What's unclear: Does the requirement expect Plate to have a `contents` attribute?
   - Recommendation: Verify with requirements author. Current implementation: Plate is stateless, soup is separate object type.

2. **OVER-04/OVER-05: Stack remaining count**
   - What we know: OnionStack/TomatoStack are infinite sources with no count
   - What's unclear: Do requirements expect finite stacks?
   - Recommendation: Verify with requirements author. Current implementation: Stacks are infinite, stateless.

3. **Pot simplified serialization**
   - What we know: Pot only stores `object_id` of ingredients, not full state
   - What's unclear: Should this be enhanced to full recursive serialization?
   - Recommendation: Works for current stateless ingredients. Document assumption.

## Sources

### Primary (HIGH confidence)
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/overcooked_grid_objects.py` - All Overcooked objects, lines 1-464
- `/Users/chasemcd/Repositories/cogrid/cogrid/core/grid_object.py` - Counter implementation, lines 591-648
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/overcooked/test_state_serialization.py` - Existing tests, lines 1-443
- `/Users/chasemcd/Repositories/cogrid/.planning/phases/01-framework-foundation/01-RESEARCH.md` - Phase 1 patterns

### Secondary (MEDIUM confidence)
- Audit script output - Current serialization status

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Direct code examination
- Architecture: HIGH - Patterns verified in codebase
- Pitfalls: HIGH - Identified from actual implementation

**Research date:** 2026-01-19
**Valid until:** 90+ days (codebase-specific, stable domain)

## Implementation Recommendations

### What Needs to Be Done

1. **Add standalone roundtrip tests** for each Overcooked object type
2. **Verify existing tests pass** for Pot and Counter
3. **Document the stateless nature** of OnionStack/TomatoStack/PlateStack
4. **Clarify requirements** OVER-03, OVER-04, OVER-05 based on actual implementation

### What Does NOT Need to Be Done

1. No new `get_extra_state()` implementations - all needed ones exist
2. No new `set_extra_state()` implementations - all needed ones exist
3. No changes to OnionStack/TomatoStack (they are infinite sources by design)
4. No changes to Plate (it is stateless, soup is separate object)

### Test Coverage Gaps

| Requirement | Existing Test | Gap |
|-------------|---------------|-----|
| OVER-01 | `test_pot_state_serialization` | Add test for `is_cooking`/`is_ready` properties |
| OVER-02 | `test_counter_with_object_serialization` | Complete |
| OVER-03 | None | Need test (but Plate is stateless) |
| OVER-04 | None | Need test (but OnionStack is stateless) |
| OVER-05 | None | Need test (but TomatoStack is stateless) |
| OVER-06 | Partial | Need comprehensive roundtrip test for all objects |
