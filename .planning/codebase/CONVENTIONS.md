# Coding Conventions

**Analysis Date:** 2026-01-19

## Naming Patterns

**Files:**
- snake_case for module names: `cogrid_env.py`, `grid_object.py`, `overcooked_grid_objects.py`
- Test files: prefix with `test_`: `test_gridworld_env.py`, `test_state_serialization.py`
- `__init__.py` files for package exports

**Functions:**
- snake_case for all functions and methods: `calculate_reward()`, `get_state()`, `can_pickup()`
- Private methods prefixed with underscore: `_gen_grid()`, `_setup_agents()`, `_action_idx_to_str()`
- Properties use snake_case: `@property def front_pos(self)`, `@property def agent_ids(self)`

**Variables:**
- snake_case for variables: `agent_id`, `start_position`, `cooking_timer`
- UPPER_CASE for module-level constants and registry dicts: `OBJECT_REGISTRY`, `REWARD_REGISTRY`, `COLOR_TO_IDX`

**Types/Classes:**
- PascalCase for classes: `CoGridEnv`, `GridObj`, `GridAgent`, `Onion`, `DeliveryZone`
- Dataclasses use PascalCase: `Actions`, `ActionSets`, `Directions`, `GridConstants`, `Colors`
- Type aliases follow PascalCase: `AgentID`, `ActionType`, `ObsType`

## Code Style

**Formatting:**
- No explicit formatter configured (no `.prettierrc`, `black`, `ruff` config found)
- Implicit 4-space indentation throughout
- Line length appears informal, generally under 100 characters

**Linting:**
- No explicit linter config files (no `.flake8`, `.pylintrc`, `ruff.toml`)
- Code follows PEP 8 conventions implicitly

**String Quotes:**
- Double quotes for strings: `"overcooked"`, `"agent_"`, `"pick_up_or_drop"`
- Both single and double quotes used inconsistently in some places

## Import Organization

**Order:**
1. Standard library imports (`from __future__ import`, `import collections`, `import copy`, `import math`, `import uuid`)
2. Third-party imports (`import numpy as np`, `import pygame`, `from gymnasium import spaces`, `import pettingzoo`)
3. Local/project imports (`from cogrid.core import ...`, `from cogrid.envs import ...`)

**Patterns:**
- Grouped by category with blank lines between
- Relative imports within packages: `from cogrid.core import grid_object`
- Explicit module imports preferred over wildcard: `from cogrid.core import actions as grid_actions`

**Path Aliases:**
- None configured. All imports use full paths from `cogrid` root

## Error Handling

**Patterns:**
- Use `assert` for internal invariant checks:
```python
assert len(self.agent_pos) == len(set(self.agent_pos)), "Agents do not have unique positions!"
assert 0 <= col < self.width, f"column index {col} outside of grid of width {self.width}"
```

- Use `raise ValueError` for invalid inputs:
```python
if action_str not in valid_actions:
    raise ValueError(f"Invalid or None action set string: {action_str}.")
```

- Use `raise NotImplementedError` for abstract methods:
```python
def calculate_reward(self, ...):
    raise NotImplementedError
```

- Optional imports with try/except:
```python
try:
    import pygame
except ImportError:
    pygame = None
```

## Logging

**Framework:** No dedicated logging framework. Uses `print()` for warnings.

**Patterns:**
```python
# Warning pattern in reward.py:
print(f"A reward is already registered with the ID {reward_id}. Overwriting it.")
```

## Comments

**When to Comment:**
- Docstrings for all public classes and methods (reStructuredText format)
- Inline comments for non-obvious logic
- TODO comments for future work: `# TODO(chase): Move PyGame/rendering logic outside of this class.`

**Docstring Format (reStructuredText):**
```python
def calculate_reward(
    self,
    state: grid.Grid,
    agent_actions: dict[int | str, int | float],
    state_transition: grid.Grid,
) -> dict[str | int, float]:
    """Calculates the reward based on the state, actions, and state transition.

    :param state: Previous CoGrid environment state.
    :type state: grid.Grid
    :param agent_actions: Actions taken in the previous state.
    :type agent_actions: dict[int  |  str, int  |  float]
    :param state_transition: Current CoGrid environment state.
    :type state_transition: grid.Grid
    :raises NotImplementedError: This method must be implemented in the subclass.
    :return: Rewards keyed by agent ID.
    :rtype: dict[str | int, float]
    """
```

## Function Design

**Size:**
- Methods typically 10-50 lines
- Larger methods (100+ lines) exist in `cogrid_env.py` for core environment logic

**Parameters:**
- Type hints on all parameters and return types
- Default arguments for optional params: `def __init__(self, config: dict, render_mode: str | None = None)`
- Use `**kwargs` for extensibility

**Return Values:**
- Type hints on returns: `-> tuple[dict[typing.AgentID, typing.ObsType], dict[str, typing.Any]]`
- Return `None` explicitly where needed
- Multi-value returns use tuples

## Module Design

**Exports:**
- `__init__.py` files for explicit package exports
- Registry pattern for extensibility:
```python
OBJECT_REGISTRY: dict[str, dict[str, GridObj]] = {}
def register_object(object_id: str, obj_class: GridObj, scope: str = "global") -> None:
    ...
def make_object(object_id: str | None, scope: str = "global", **kwargs) -> GridObj:
    ...
```

**Barrel Files:**
- `cogrid/envs/__init__.py` imports submodules: `from cogrid.envs.overcooked import overcooked`
- `cogrid/feature_space/__init__.py` imports core classes

## Type Annotations

**Pattern:** Modern Python 3.10+ type hints throughout:
```python
def __init__(
    self,
    config: dict,
    render_mode: str | None = None,
    agent_class: agent.Agent | None = None,
    **kwargs,
):
```

**Union types:** Use `|` operator: `str | int`, `GridObj | None`

**Generic types:** `dict[typing.AgentID, float]`, `list[tuple[int, int]]`

**Type aliases in `cogrid/core/typing.py`:**
```python
ActionType = env.ActionType
AgentID = env.AgentID
ObsType = env.ObsType
```

## Class Patterns

**Dataclasses for constants:**
```python
@dataclasses.dataclass
class Actions:
    PickupDrop = "pick_up_or_drop"
    Toggle = "toggle"
    Noop = "no-op"
```

**IntEnum for enumerated values:**
```python
class Directions(IntEnum):
    Right = 0
    Down = 1
    Left = 2
    Up = 3
```

**Class-level attributes for GridObj subclasses:**
```python
class Onion(grid_object.GridObj):
    object_id = "onion"
    color = constants.Colors.Yellow
    char = "o"
```

**Hook methods for extensibility:**
```python
def on_reset(self) -> None:
    """Hook for subclasses to implement custom logic after the environment is reset."""
    pass

def on_step(self) -> None:
    """Hook for subclasses to implement custom logic after each step."""
    pass
```

## Serialization Pattern

**State serialization follows get_state/set_state pattern:**
```python
def get_state(self, scope: str = "global") -> dict:
    """Serialize agent state to a dictionary."""
    return {...}

@classmethod
def from_state(cls, state_dict: dict, scope: str = "global"):
    """Reconstruct an agent from serialized state."""
    ...
```

**Extra state for complex objects:**
```python
def get_extra_state(self, scope: str = "global") -> dict | None:
    """Override to serialize complex internal state."""
    ...

def set_extra_state(self, state_dict: dict, scope: str = "global") -> None:
    """Override to restore complex internal state."""
    ...
```

---

*Convention analysis: 2026-01-19*
