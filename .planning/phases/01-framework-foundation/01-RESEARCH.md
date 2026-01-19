# Phase 1: Framework Foundation - Research

**Researched:** 2026-01-19
**Domain:** Python object serialization patterns for hierarchical GridObj class system
**Confidence:** HIGH

## Summary

The cogrid codebase has an existing serialization infrastructure with `get_extra_state()`/`set_extra_state()` methods defined on the base `GridObj` class (lines 236-275 of `grid_object.py`). This infrastructure is partially implemented: the `Counter` and `Pot` classes have working implementations, but 18 other GridObj subclasses are missing these methods.

The current pattern follows a recursive approach where objects that contain other objects (Counter, Pot, Agent inventory) call `get_extra_state()` on their contained objects and reconstruct them via `make_object()` during `set_extra_state()`. This pattern is sound and should be extended to all objects with internal state.

**Primary recommendation:** Create an audit script/command that programmatically identifies all GridObj subclasses and their serialization status, then implement `get_extra_state()`/`set_extra_state()` for objects with internal state beyond the basic `state` integer field.

## Current State Analysis

### GridObj Subclasses Inventory

| Class | Location | Has Extra State? | Implements Methods? | Notes |
|-------|----------|------------------|---------------------|-------|
| `GridObj` (base) | `grid_object.py:110` | Base class | Yes (defaults) | Returns None, pass |
| `GridAgent` | `grid_object.py:352` | Yes (inventory, dir) | No | Handled via Agent serialization |
| `Wall` | `grid_object.py:489` | No | No | Stateless |
| `Floor` | `grid_object.py:504` | No | No | Stateless |
| `Counter` | `grid_object.py:521` | Yes (`obj_placed_on`) | **Yes** | Recursive serialization |
| `Key` | `grid_object.py:581` | No | No | Stateless |
| `Door` | `grid_object.py:612` | Yes (`is_open`, `is_locked`) | **No** | Needs implementation |
| `Onion` | `overcooked_grid_objects.py:22` | No | No | Stateless |
| `Tomato` | `overcooked_grid_objects.py:49` | No | No | Stateless |
| `OnionStack` | `overcooked_grid_objects.py:76` | No | No | Stateless (infinite source) |
| `TomatoStack` | `overcooked_grid_objects.py:106` | No | No | Stateless (infinite source) |
| `Pot` | `overcooked_grid_objects.py:136` | Yes (`objects_in_pot`, `cooking_timer`) | **Yes** | Recursive serialization |
| `PlateStack` | `overcooked_grid_objects.py:286` | No | No | Stateless (infinite source) |
| `Plate` | `overcooked_grid_objects.py:326` | No | No | Stateless |
| `DeliveryZone` | `overcooked_grid_objects.py:355` | No | No | Stateless |
| `OnionSoup` | `overcooked_grid_objects.py:394` | No | No | Stateless |
| `TomatoSoup` | `overcooked_grid_objects.py:429` | No | No | Stateless |
| `MedKit` | `search_rescue_grid_objects.py:17` | No | No | Stateless |
| `Pickaxe` | `search_rescue_grid_objects.py:46` | No | No | Stateless |
| `Rubble` | `search_rescue_grid_objects.py:86` | No | No | Stateless |
| `GreenVictim` | `search_rescue_grid_objects.py:128` | No | No | Stateless |
| `PurpleVictim` | `search_rescue_grid_objects.py:158` | No | No | Stateless |
| `YellowVictim` | `search_rescue_grid_objects.py:188` | No | No | Stateless |
| `RedVictim` | `search_rescue_grid_objects.py:227` | Yes (`toggle_countdown`, `first_toggle_agent_id`) | **No** | Needs implementation |

### Summary Statistics
- **Total GridObj subclasses:** 23 (including base GridObj)
- **Classes with extra state:** 5 (Counter, Door, Pot, GridAgent, RedVictim)
- **Classes that implement serialization:** 2 (Counter, Pot)
- **Classes that need serialization added:** 2 (Door, RedVictim)
- **Stateless classes (no action needed):** 16

## Standard Stack

The serialization infrastructure already exists in the codebase. No external libraries needed.

### Core
| Component | Location | Purpose | Why Standard |
|-----------|----------|---------|--------------|
| `GridObj.get_extra_state()` | `grid_object.py:236` | Serialize complex state | Base class template method |
| `GridObj.set_extra_state()` | `grid_object.py:256` | Restore complex state | Base class template method |
| `make_object()` | `grid_object.py:34` | Factory for object instantiation | Registry-based factory |
| `Grid.get_state_dict()` | `grid.py:532` | Full grid serialization | Iterates all objects |
| `Grid.set_state_dict()` | `grid.py:589` | Full grid restoration | Uses object_metadata |
| `Agent.get_state()` | `agent.py:109` | Agent serialization | Includes inventory |
| `Agent.from_state()` | `agent.py:137` | Agent restoration | Factory method |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| Python `ast` module | Code analysis | Audit script for finding subclasses |
| Python `inspect` module | Runtime introspection | Alternative audit approach |
| `unittest` | Testing | Verify serialization correctness |

## Architecture Patterns

### Recommended Project Structure
```
cogrid/
├── core/
│   ├── grid_object.py        # Base GridObj with serialization methods
│   ├── grid.py               # Grid serialization via get_state_dict/set_state_dict
│   ├── agent.py              # Agent serialization via get_state/from_state
│   └── serialization_utils.py  # NEW: Audit utilities (optional)
├── envs/
│   ├── overcooked/
│   │   └── overcooked_grid_objects.py  # Pot already implemented
│   └── search_rescue/
│       └── search_rescue_grid_objects.py  # RedVictim needs implementation
└── scripts/                  # NEW: Audit script location (alternative)
    └── audit_serialization.py
```

### Pattern 1: Template Method Pattern (Existing)

**What:** Base class defines `get_extra_state()`/`set_extra_state()` with default behavior (return None/pass). Subclasses override only when they have extra state.

**When to use:** All GridObj subclasses with internal state beyond `self.state`.

**Example (existing implementation in Counter):**
```python
# Source: cogrid/core/grid_object.py lines 542-575
def get_extra_state(self, scope: str = "global") -> dict | None:
    """Serialize counter's obj_placed_on state."""
    if self.obj_placed_on is None:
        return None

    return {
        "obj_placed_on": {
            "object_id": self.obj_placed_on.object_id,
            "state": self.obj_placed_on.state,
            "extra_state": self.obj_placed_on.get_extra_state(scope),  # Recursive!
        }
    }

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    """Restore counter's obj_placed_on state from serialization."""
    if state_dict and "obj_placed_on" in state_dict:
        obj_data = state_dict["obj_placed_on"]
        self.obj_placed_on = make_object(
            obj_data["object_id"], state=obj_data["state"], scope=scope
        )
        if obj_data["extra_state"]:
            self.obj_placed_on.set_extra_state(obj_data["extra_state"], scope)  # Recursive!
```

### Pattern 2: Recursive Object Serialization (Existing)

**What:** When an object contains other objects (composition), it serializes those objects recursively by calling their `get_extra_state()` method.

**When to use:** Any object that holds references to other GridObj instances (Counter with `obj_placed_on`, Pot with `objects_in_pot`, Agent with `inventory`).

**Example (existing implementation in Pot):**
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

### Pattern 3: Audit via AST Analysis

**What:** Use Python's `ast` module to parse source files and find all classes that inherit from `GridObj`.

**When to use:** For the audit script/command required by FRAM-01.

**Example:**
```python
import ast
import os
from pathlib import Path

def find_gridobj_subclasses(directory: str) -> list[dict]:
    """Find all GridObj subclasses in Python files."""
    results = []

    for py_file in Path(directory).rglob("*.py"):
        with open(py_file) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if inherits from GridObj
                for base in node.bases:
                    base_name = None
                    if isinstance(base, ast.Name):
                        base_name = base.name
                    elif isinstance(base, ast.Attribute):
                        base_name = base.attr

                    if base_name in ("GridObj", "grid_object.GridObj"):
                        # Check for get_extra_state method
                        has_get = any(
                            isinstance(item, ast.FunctionDef)
                            and item.name == "get_extra_state"
                            for item in node.body
                        )
                        has_set = any(
                            isinstance(item, ast.FunctionDef)
                            and item.name == "set_extra_state"
                            for item in node.body
                        )

                        results.append({
                            "class_name": node.name,
                            "file": str(py_file),
                            "line": node.lineno,
                            "has_get_extra_state": has_get,
                            "has_set_extra_state": has_set,
                        })

    return results
```

### Anti-Patterns to Avoid

- **Incomplete recursion:** When serializing an object that holds another object, always call `get_extra_state()` on the contained object. The Pot implementation only stores `object_id` (not full state) which works because Onion/Tomato are stateless, but this would break for objects with state.

- **Forgetting scope parameter:** Always pass `scope` through recursive calls to ensure proper object registry lookup.

- **Modifying state during serialization:** `get_extra_state()` should be read-only; never modify object state in this method.

- **Not handling None:** Always check if `state_dict` is not None before accessing its contents in `set_extra_state()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Object instantiation from type name | Custom factory | `make_object()` | Already handles scope, registry lookup |
| Finding subclasses at runtime | Manual list | Python `ast` or registry introspection | More maintainable, auto-discovers |
| Deep copying objects | Manual copying | `get_extra_state()`/`set_extra_state()` | Explicit serialization is more reliable |

**Key insight:** The serialization infrastructure is already built. The work is implementing the pattern consistently across all objects with internal state.

## Common Pitfalls

### Pitfall 1: Objects with State in `encode()` but not in `get_extra_state()`

**What goes wrong:** Some objects encode state in the `encode()` method (like Door with `is_open`/`is_locked`) but don't implement `get_extra_state()`. The basic `state` integer can be recovered from `encode()`, but derived attributes (like Door's `is_open` property derived from `state`) need explicit handling.

**Why it happens:** `encode()` returns a tuple `(char, color_idx, state)` which captures `self.state`, but objects may have additional attributes not reflected in `self.state`.

**How to avoid:** For objects where `__init__` derives attributes from `state` (like Door), verify that:
1. The constructor properly restores derived attributes from `state`, OR
2. Implement `get_extra_state()` to capture those attributes

**Warning signs:** Object has properties or attributes set in `__init__` based on `state` parameter.

### Pitfall 2: Missing Import in `set_extra_state()`

**What goes wrong:** `set_extra_state()` may need to import `make_object` to reconstruct nested objects, but circular imports can occur.

**Why it happens:** `make_object` is in `grid_object.py`, which may import the subclass module.

**How to avoid:** Use local imports inside `set_extra_state()` method (see Pot implementation at line 273).

**Warning signs:** ImportError at runtime when restoring state.

### Pitfall 3: Incomplete Recursive Serialization

**What goes wrong:** Nested objects are serialized by `object_id` only, losing their internal state.

**Why it happens:** The Pot implementation serializes `objects_in_pot` as just `[obj.object_id for obj in self.objects_in_pot]`. This works for stateless Onion/Tomato but would lose state for more complex objects.

**How to avoid:** Always serialize nested objects fully:
```python
"nested_obj": {
    "object_id": obj.object_id,
    "state": obj.state,
    "extra_state": obj.get_extra_state(scope),
}
```

**Warning signs:** Nested objects that have their own `get_extra_state()` implementations.

### Pitfall 4: Scope Not Propagated

**What goes wrong:** Objects in "overcooked" scope are deserialized with "global" scope, causing `make_object()` to fail.

**Why it happens:** Forgetting to pass `scope` parameter through recursive calls.

**How to avoid:** Always pass `scope` to `make_object()` and recursive `set_extra_state()` calls.

**Warning signs:** `ValueError: Object with object_id 'X' not registered in scope 'global'`.

## Code Examples

### Example 1: Door Serialization Implementation (Needed)

```python
# Door has is_open and is_locked derived from state in __init__
# The current implementation at grid_object.py:612-704 does NOT have get_extra_state

# Current Door.__init__ (line 617-619):
def __init__(self, state):
    super().__init__(state=state)
    self.is_open = state == 2
    self.is_locked = state == 0

# Door's encode() already updates self.state based on is_open/is_locked (line 640-656)
# This means Door can be fully restored from state alone via encode()/decode()
# NO extra_state implementation needed - state integer captures full object state

# VERIFICATION: Door state encoding:
# state=0: locked (is_locked=True, is_open=False)
# state=1: closed, unlocked (is_locked=False, is_open=False)
# state=2: open (is_locked=False, is_open=True)
```

### Example 2: RedVictim Serialization Implementation (Needed)

```python
# RedVictim has toggle_countdown and first_toggle_agent_id that need serialization
# Source: search_rescue_grid_objects.py:227-268

def get_extra_state(self, scope: str = "global") -> dict | None:
    """Serialize RedVictim's toggle countdown state."""
    if self.toggle_countdown == 0:
        return None  # Default state, no extra serialization needed

    return {
        "toggle_countdown": self.toggle_countdown,
        "first_toggle_agent_id": self.first_toggle_agent_id,
    }

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    """Restore RedVictim's toggle countdown state."""
    if state_dict:
        self.toggle_countdown = state_dict.get("toggle_countdown", 0)
        self.first_toggle_agent_id = state_dict.get("first_toggle_agent_id")
```

### Example 3: Audit Script Structure

```python
#!/usr/bin/env python
"""Audit GridObj subclasses for serialization implementation status."""

import ast
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SerializationStatus:
    class_name: str
    file_path: str
    line_number: int
    has_get_extra_state: bool
    has_set_extra_state: bool
    has_extra_attributes: bool  # Heuristic: attributes set in __init__ beyond state

    @property
    def status(self) -> str:
        if not self.has_extra_attributes:
            return "STATELESS"
        elif self.has_get_extra_state and self.has_set_extra_state:
            return "IMPLEMENTED"
        elif self.has_get_extra_state or self.has_set_extra_state:
            return "PARTIAL"
        else:
            return "MISSING"

def audit_serialization(cogrid_path: str) -> list[SerializationStatus]:
    """Audit all GridObj subclasses."""
    # Implementation uses ast module as shown in Pattern 3
    pass

def print_audit_report(results: list[SerializationStatus]) -> None:
    """Print formatted audit report."""
    print("\n=== GridObj Serialization Audit ===\n")

    by_status = {}
    for r in results:
        by_status.setdefault(r.status, []).append(r)

    for status, items in sorted(by_status.items()):
        print(f"\n{status} ({len(items)}):")
        for item in items:
            print(f"  - {item.class_name} ({item.file_path}:{item.line_number})")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| No serialization | `get_state()`/`set_state()` on CoGridEnv | Recent (commit 6deb188) | Full environment checkpointing possible |
| Object state via `encode()` only | `encode()` + `get_extra_state()` | Recent | Complex objects can be fully serialized |

**Deprecated/outdated:**
- None - this is a new feature being implemented

## Implementation Requirements Mapping

### FRAM-01: Audit GridObj Subclasses

**Approach:** Create a Python script using `ast` module that:
1. Recursively finds all `.py` files in `cogrid/`
2. Parses each file and finds classes inheriting from `GridObj`
3. Checks for presence of `get_extra_state()` and `set_extra_state()` methods
4. Outputs a report showing status for each class

**Output format:**
```
=== GridObj Serialization Audit ===

STATELESS (16):
  - Wall (cogrid/core/grid_object.py:489)
  - Floor (cogrid/core/grid_object.py:504)
  ...

IMPLEMENTED (2):
  - Counter (cogrid/core/grid_object.py:521)
  - Pot (cogrid/envs/overcooked/overcooked_grid_objects.py:136)

MISSING (1):
  - RedVictim (cogrid/envs/search_rescue/search_rescue_grid_objects.py:227)
```

### FRAM-02: Recursive Object Serialization

**Already implemented for:**
- `Counter.obj_placed_on` - correctly serializes nested object with full state
- `Pot.objects_in_pot` - serializes by object_id only (works for stateless Onion/Tomato)
- `Agent.inventory` - correctly serializes with full state including extra_state

**Needs verification:**
- Ensure Pot serialization handles potential future stateful ingredients

### FRAM-03: Documentation

**Recommended documentation location:** `cogrid/core/grid_object.py` docstrings (already have good examples in lines 236-275)

**Documentation should include:**
1. When to implement `get_extra_state()` (has state beyond `self.state` integer)
2. Pattern for recursive serialization of nested objects
3. Importance of `scope` parameter
4. Example implementation (reference Counter or Pot)

## Open Questions

1. **Door State Completeness**
   - What we know: Door encodes `is_open`/`is_locked` into `self.state` via `encode()`, and `__init__` restores from `state` parameter
   - What's unclear: Is this bidirectional encoding complete? Are there edge cases?
   - Recommendation: Add test case to verify Door round-trips correctly

2. **Pot's Simplified Serialization**
   - What we know: Pot only stores `object_id` of ingredients, not full state
   - What's unclear: Would this break if ingredients had state?
   - Recommendation: Current approach is fine since Onion/Tomato are stateless; document this assumption

## Sources

### Primary (HIGH confidence)
- `cogrid/core/grid_object.py` - Direct code examination, all 705 lines
- `cogrid/core/grid.py` - Direct code examination, get_state_dict/set_state_dict
- `cogrid/core/agent.py` - Direct code examination, get_state/from_state
- `cogrid/envs/overcooked/overcooked_grid_objects.py` - Direct code examination, Pot implementation
- `cogrid/envs/search_rescue/search_rescue_grid_objects.py` - Direct code examination, RedVictim
- `cogrid/envs/overcooked/test_state_serialization.py` - Existing test patterns
- `cogrid/cogrid_env.py` - Environment-level serialization

### Secondary (MEDIUM confidence)
- Python `ast` module documentation - Standard library, well-documented

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Direct code examination, existing implementation
- Architecture: HIGH - Patterns exist in codebase, well-documented
- Pitfalls: HIGH - Identified from actual code patterns and potential issues

**Research date:** 2026-01-19
**Valid until:** 60+ days (codebase-specific, not dependent on external libraries)
