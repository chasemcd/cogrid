# Phase 3: Search & Rescue Objects - Research

**Researched:** 2026-01-19
**Domain:** Search & Rescue GridObj serialization for state checkpointing
**Confidence:** HIGH

## Summary

The Search & Rescue domain has 7 GridObj subclasses total: MedKit, Pickaxe, Rubble, GreenVictim, PurpleVictim, YellowVictim, and RedVictim. Analysis of the codebase and the audit script output shows that **6 of these 7 objects are stateless** and require no serialization implementation beyond the base `state` integer. RedVictim is the only object with extra state (`toggle_countdown`, `first_toggle_agent`) and was **already implemented in Phase 1** (task 01-02).

This means Phase 3 has minimal implementation work. The primary tasks are:
1. Verify the audit script correctly identifies all S&R objects as STATELESS or IMPLEMENTED
2. Create roundtrip tests to confirm stateless objects work correctly with the existing serialization infrastructure
3. Document the analysis for future maintainers

**Primary recommendation:** Create verification tests for all S&R objects and confirm RedVictim's existing implementation handles all edge cases. No new `get_extra_state()`/`set_extra_state()` implementations needed.

## Object Analysis

### Complete Inventory of Search & Rescue Objects

| Object | Line | Extra Attributes | Audit Status | Serialization Needed? |
|--------|------|------------------|--------------|----------------------|
| MedKit | 17 | None | STATELESS | No - pickupable item, no internal state |
| Pickaxe | 46 | None | STATELESS | No - pickupable item, no internal state |
| Rubble | 86 | None | STATELESS | No - removed when toggled, no persistent state |
| GreenVictim | 128 | None | STATELESS | No - removed when rescued |
| PurpleVictim | 158 | None | STATELESS | No - removed when rescued |
| YellowVictim | 188 | None | STATELESS | No - removed when rescued |
| RedVictim | 227 | `toggle_countdown`, `first_toggle_agent` | IMPLEMENTED | Yes - done in Phase 1 |

### Detailed Object State Analysis

#### MedKit (lines 17-43)
```python
def __init__(self, state=0):
    super().__init__(state=state)
```
**State analysis:** No attributes beyond base GridObj. Used as a tool that agents pick up. Behavior is fully stateless - the MedKit itself doesn't track who picked it up or usage history.
**Verdict:** STATELESS - no serialization needed

#### Pickaxe (lines 46-83)
```python
def __init__(self, state=0):
    super().__init__(state=state)
```
**State analysis:** Identical to MedKit. A pickupable tool with no internal state.
**Verdict:** STATELESS - no serialization needed

#### Rubble (lines 86-125)
```python
def __init__(self, state=0):
    super().__init__(
        state=state,
        toggle_value=0.05,  # reward for clearing rubble
    )
```
**State analysis:** Rubble is a blocking object that gets removed from the grid when cleared by an Engineer/agent with Pickaxe. The `toggle_value` is a constant set in `__init__`. There's no "partially cleared" state - it's either present or removed. Once toggled successfully, it calls `self._remove_from_grid(env.grid)`.
**Verdict:** STATELESS - no serialization needed. Presence/absence on grid is the only state.

#### GreenVictim (lines 128-155)
```python
def __init__(self, state=0):
    super().__init__(
        state=state,
        toggle_value=0.1,  # 0.1 reward for rescuing
    )
```
**State analysis:** Simplest victim type. Can be rescued by any adjacent agent. Once toggled, removed from grid. No rescue countdown or multi-step rescue process.
**Verdict:** STATELESS - no serialization needed

#### PurpleVictim (lines 158-185)
```python
def __init__(self, state=0):
    super().__init__(
        state=state,
        toggle_value=0.2,
    )
```
**State analysis:** Functionally identical to GreenVictim with different reward value.
**Verdict:** STATELESS - no serialization needed

#### YellowVictim (lines 188-224)
```python
def __init__(self, state=0):
    super().__init__(
        state=state,
        toggle_value=0.2,
    )
```
**State analysis:** Requires a Medic role or agent with MedKit to rescue, but this is checked at toggle time, not stored as state. Once successfully toggled, removed from grid.
**Verdict:** STATELESS - no serialization needed

#### RedVictim (lines 227-304) - ALREADY IMPLEMENTED
```python
def __init__(self, state=0):
    super().__init__(state=state)
    self.toggle_countdown = 0
    self.first_toggle_agent_id: typing.AgentID = None
```
**State analysis:** Most complex victim. Requires two-step rescue:
1. First, a Medic/agent with MedKit toggles, starting a 30-step countdown
2. Then, a different agent must toggle within the countdown window

The `toggle_countdown` decrements each tick and `first_toggle_agent` (note: code uses `first_toggle_agent` not `first_toggle_agent_id` in toggle method) tracks which agent started the rescue.

**Current implementation (Phase 1):**
```python
def get_extra_state(self, scope: str = "global") -> dict | None:
    if self.toggle_countdown == 0 and not hasattr(self, "first_toggle_agent"):
        return None
    first_toggle = getattr(self, "first_toggle_agent", None)
    if self.toggle_countdown == 0 and first_toggle is None:
        return None
    return {
        "toggle_countdown": self.toggle_countdown,
        "first_toggle_agent": first_toggle,
    }

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    if state_dict:
        self.toggle_countdown = state_dict.get("toggle_countdown", 0)
        self.first_toggle_agent = state_dict.get("first_toggle_agent")
```
**Verdict:** IMPLEMENTED - Phase 1 completed this. Tests exist in `test_redvictim_serialization.py`

## Standard Stack

No new libraries or infrastructure needed. Phase 3 uses what Phase 1 established.

### Core
| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `GridObj.get_extra_state()` | `grid_object.py:252` | Base serialization method | Established |
| `GridObj.set_extra_state()` | `grid_object.py:301` | Base deserialization method | Established |
| `make_object()` | `grid_object.py:50` | Factory for object instantiation | Established |
| Audit script | `scripts/audit_serialization.py` | Status tracking | Created in Phase 1 |

### Testing
| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `pytest` | Test framework | All roundtrip verification |
| Existing test patterns | `test_redvictim_serialization.py` | Template for new tests |

## Architecture Patterns

### Pattern 1: Stateless Object Verification
**What:** Objects without extra state should roundtrip correctly using only the base `state` integer.

**When to use:** All STATELESS objects (MedKit, Pickaxe, Rubble, GreenVictim, PurpleVictim, YellowVictim).

**Example:**
```python
# Source: Pattern from test_redvictim_serialization.py
def test_stateless_object_roundtrip():
    """Stateless objects roundtrip via state integer alone."""
    medkit = MedKit(state=0)

    # Encode captures state
    encoded = medkit.encode()
    state_value = encoded[2]  # (char, 0, state)

    # Create new object with same state
    new_medkit = MedKit(state=state_value)

    # No extra state needed
    assert medkit.get_extra_state() is None
    assert new_medkit.object_id == medkit.object_id
```

### Pattern 2: Presence-Based State
**What:** Some objects' primary "state" is whether they exist on the grid at all.

**When to use:** Objects like Rubble and Victims that get removed when interacted with.

**Key insight:** Once Rubble is cleared or a Victim is rescued, they are removed from the grid entirely. The grid serialization handles this automatically - if an object isn't in the grid, it won't be serialized.

### Anti-Patterns to Avoid
- **Over-engineering stateless objects:** Don't add `get_extra_state()` to objects that don't need it. The base class already returns None.
- **Testing non-existent complexity:** Don't test "cleared Rubble state" - cleared Rubble doesn't exist.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Object instantiation | Custom constructors | `make_object(object_id, state=state, scope="search_rescue")` | Registry handles scoping |
| Finding objects needing serialization | Manual inspection | `python -m cogrid.scripts.audit_serialization` | Automated, consistent |
| Test structure | Custom test patterns | Existing `test_redvictim_serialization.py` patterns | Proven structure |

**Key insight:** Most of Phase 3 work is verification, not implementation. The audit script and existing tests provide the patterns.

## Common Pitfalls

### Pitfall 1: Assuming Victims Need Rescue State
**What goes wrong:** Implementing serialization for "rescue progress" that doesn't exist in simpler victims.

**Why it happens:** RedVictim's multi-step rescue creates expectation that all victims have progress state.

**How to avoid:** Read the actual `toggle()` method for each victim type. Green, Purple, and Yellow victims have single-step rescue - they're either present or removed.

**Warning signs:** Adding `rescue_progress` or similar attributes where none exist.

### Pitfall 2: Missing the Attribute Name Inconsistency
**What goes wrong:** Serializing `first_toggle_agent_id` but toggle logic uses `first_toggle_agent`.

**Why it happens:** `__init__` declares `first_toggle_agent_id` but toggle method assigns to `first_toggle_agent`.

**How to avoid:** Phase 1 already handled this correctly - use `first_toggle_agent` (the actual runtime attribute).

**Warning signs:** Serialization tests pass but full environment tests fail.

### Pitfall 3: Testing Objects After Removal
**What goes wrong:** Trying to serialize/restore a Rubble or Victim that has been cleared/rescued.

**Why it happens:** Misunderstanding that these objects cease to exist after interaction.

**How to avoid:** Test objects before they're toggled. After toggle, they're removed from the grid and don't need individual serialization.

## Code Examples

### Example 1: Stateless Victim Test Pattern
```python
# Recommended test structure for stateless S&R objects
class TestGreenVictimSerialization:
    """Tests for GreenVictim serialization roundtrip."""

    def test_greenvictim_default_state_roundtrip(self):
        """GreenVictim in default state should roundtrip correctly."""
        victim = GreenVictim(state=0)

        # No extra state needed
        assert victim.get_extra_state() is None

        # Encode/decode via state integer
        encoded = victim.encode()
        state_value = encoded[2]

        new_victim = GreenVictim(state=state_value)
        assert new_victim.object_id == victim.object_id
        assert new_victim.state == victim.state
```

### Example 2: Tool Object Test Pattern
```python
# MedKit and Pickaxe follow same pattern
class TestMedKitSerialization:
    """Tests for MedKit serialization roundtrip."""

    def test_medkit_roundtrip(self):
        """MedKit should roundtrip correctly with no extra state."""
        medkit = MedKit(state=0)

        # Stateless
        assert medkit.get_extra_state() is None

        # Can pickup is a method, not state
        assert medkit.can_pickup(agent=None) is True  # Always true

        # Encode/decode
        encoded = medkit.encode()
        new_medkit = MedKit(state=encoded[2])

        assert new_medkit.object_id == medkit.object_id
```

### Example 3: Full Coverage Test File Structure
```python
"""Roundtrip serialization tests for Search & Rescue objects.

Tests verify that all S&R objects survive get_state/set_state roundtrip:
- Stateless objects (MedKit, Pickaxe, Rubble, Green/Purple/Yellow Victim)
- Stateful objects (RedVictim - already tested in Phase 1)
"""

import pytest
from cogrid.envs.search_rescue.search_rescue_grid_objects import (
    MedKit, Pickaxe, Rubble,
    GreenVictim, PurpleVictim, YellowVictim, RedVictim
)

class TestStatelessSRObjects:
    """Verify stateless S&R objects need no extra serialization."""

    @pytest.mark.parametrize("obj_class,obj_id", [
        (MedKit, "medkit"),
        (Pickaxe, "pickaxe"),
        (Rubble, "rubble"),
        (GreenVictim, "green_victim"),
        (PurpleVictim, "purple_victim"),
        (YellowVictim, "yellow_victim"),
    ])
    def test_stateless_object_no_extra_state(self, obj_class, obj_id):
        """Stateless objects return None from get_extra_state."""
        obj = obj_class(state=0)
        assert obj.get_extra_state() is None
        assert obj.object_id == obj_id

    @pytest.mark.parametrize("obj_class", [
        MedKit, Pickaxe, Rubble,
        GreenVictim, PurpleVictim, YellowVictim,
    ])
    def test_stateless_object_roundtrip(self, obj_class):
        """Stateless objects roundtrip via state integer."""
        obj = obj_class(state=0)
        encoded = obj.encode()
        new_obj = obj_class(state=encoded[2])

        assert new_obj.object_id == obj.object_id
        assert new_obj.state == obj.state
        assert new_obj.get_extra_state() is None
```

## Requirements Mapping

| Requirement | Status | Analysis |
|-------------|--------|----------|
| SRCH-01: Victim objects serialize rescue state | MOSTLY DONE | RedVictim done in Phase 1. Green/Purple/Yellow are stateless - no rescue state to serialize. |
| SRCH-02: Rubble serializes cleared state | NO WORK NEEDED | Cleared Rubble is removed from grid. Uncleared Rubble has no state. |
| SRCH-03: Tools serialize ownership/usage state | NO WORK NEEDED | MedKit/Pickaxe are stateless pickupable items. Ownership is tracked by Agent inventory, not the tool itself. |
| SRCH-04: All S&R objects audited and serialized | AUDIT COMPLETE | All 7 objects analyzed. 6 stateless, 1 already implemented. |

## Implementation Recommendation

Given that RedVictim was implemented in Phase 1 and all other S&R objects are stateless, Phase 3 should focus on:

### Task 1: Verification Testing
Create a test file that:
- Confirms all stateless S&R objects return `None` from `get_extra_state()`
- Confirms roundtrip via `encode()`/constructor works for each object
- Confirms audit script correctly classifies each object

### Task 2: Documentation Update
Update any phase-specific documentation to note that:
- SRCH-01 through SRCH-03 required minimal implementation due to stateless design
- RedVictim serialization was completed in Phase 1 as part of demonstrating the pattern
- The S&R domain is now fully serialization-ready

### Task 3: Audit Verification
Run `python -m cogrid.scripts.audit_serialization` and verify:
- RedVictim shows as IMPLEMENTED
- All other S&R objects show as STATELESS

## Open Questions

1. **PurpleVictim in Audit but Not in Requirements**
   - What we know: The audit shows PurpleVictim exists but REQUIREMENTS.md only mentions Green/Yellow/Red
   - What's unclear: Is PurpleVictim a new addition or oversight in requirements?
   - Recommendation: Include it in testing. It's stateless so no additional work.

2. **Tool Ownership Tracking**
   - What we know: SRCH-03 mentions "ownership/usage state" but MedKit/Pickaxe have no such attributes
   - What's unclear: Was this a misunderstanding of how tools work?
   - Recommendation: Verify that Agent inventory serialization (Phase 4) handles the tools correctly when picked up. The tools themselves are stateless.

## Sources

### Primary (HIGH confidence)
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/search_rescue/search_rescue_grid_objects.py` - Direct code examination, all 305 lines
- `/Users/chasemcd/Repositories/cogrid/cogrid/envs/search_rescue/test_redvictim_serialization.py` - Existing test patterns from Phase 1
- `/Users/chasemcd/Repositories/cogrid/cogrid/scripts/audit_serialization.py` - Audit script, 330 lines
- Audit script output (run during research) - Current classification of all objects

### Secondary (MEDIUM confidence)
- Phase 1 research and summary documents - Established patterns and decisions

## Metadata

**Confidence breakdown:**
- Object analysis: HIGH - Direct code examination, complete inventory
- Implementation status: HIGH - Audit script confirms, code review verifies
- Testing approach: HIGH - Existing patterns from Phase 1 to follow

**Research date:** 2026-01-19
**Valid until:** 90+ days (codebase-specific, minimal external dependencies)
