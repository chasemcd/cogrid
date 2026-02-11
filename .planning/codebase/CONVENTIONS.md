# Coding Conventions

**Analysis Date:** 2026-02-10

## Naming Patterns

**Files:**
- Snake case: `grid_object.py`, `grid_utils.py`, `overcooked_features.py`
- Test files: `test_*.py` pattern (e.g., `test_gridworld_env.py`, `test_overcooked_env.py`)
- Specialized prefixes: `sr_utils.py` (search_rescue), env files named by feature

**Classes:**
- PascalCase: `CoGridEnv`, `Agent`, `Grid`, `GridObj`, `Wall`, `Actions`, `ActionSets`, etc.
- Abstract/base classes prefixed where appropriate: e.g., `CoreConstants`, `ObjectColors`

**Functions/Methods:**
- Snake case: `get_grid_agent_at_position()`, `_generate_encoded_grid_states()`, `make_object()`, `register_object()`
- Private methods: Single leading underscore `_setup_agents()`, `_generate_encoded_grid_states()`
- Properties: `@property` decorator used for computed attributes like `front_pos`, `dir_vec`, `right_vec`, `np_random`, `observation_space`, `action_space`

**Variables:**
- Snake case: `grid_config`, `layout_name`, `layout_fn`, `start_position`, `start_direction`
- Class attributes: `metadata` (dict for environment metadata), `tile_cache` (module-level cache)
- Type hints used extensively with full path imports: `tuple[int, int]`, `GridObj | None`, `dict[str, Any]`

**Constants:**
- UPPER_CASE: `OBJECT_REGISTRY`, `COLOR_TO_IDX`, `IDX_TO_COLOR`, `COLORS`, `COLOR_NAMES`, `CHANNEL_FIRST`
- Dataclass constants: `CoreConstants`, `Colors`, `ObjectColors`

## Code Style

**Formatting:**
- 4-space indentation (Python standard)
- Line breaks at logical points, multiline function calls are indented
- No apparent formatter (Black, isort) enforced based on imports and spacing

**Type Hints:**
- Modern Python 3.10+ syntax: `str | None` instead of `Optional[str]`
- Comprehensive type hints on function signatures and docstrings
- Union types used: `GridAgent | None`, `tuple[np.ndarray, np.ndarray] | str`

**Docstrings:**
- Google-style docstrings with parameter descriptions
- Format: `:param name: description` and `:type name: type`
- Return documentation: `:return:` and `:rtype:`
- Raises documentation: `:raises ExceptionType: description`
- Examples: `"""Test that we can get tomato from the stack and put it in the pot. Tests Pot.can_place_on() for Tomato objects"""`

**Comments:**
- Implementation comments on complex logic (direction mappings, coordinate transforms)
- TODO comments with owner attribution: `# TODO(chase): Move PyGame/rendering logic outside of this class.`
- Descriptive block comments for major sections
- Inline comments for non-obvious logic

## Import Organization

**Order:**
1. Standard library: `import sys`, `import time`, `import unittest`, `import functools`, `import math`, `import uuid`, `from copy import deepcopy`
2. Third-party packages: `import numpy as np`, `import pygame`, `from gymnasium import spaces`, `import pettingzoo`
3. Local imports: `from cogrid.core import agent`, `from cogrid.constants import GridConstants`
4. Relative imports within package: Rare, mostly avoided

**Path Aliases:**
- No apparent aliases (no `from cogrid.core as core`)
- Imports use full qualified paths: `from cogrid.core.grid_object import GridObj, Wall`
- Common import pattern: `from cogrid.core import actions as grid_actions`

**Import Style:**
- Specific imports preferred over wildcard imports
- Imports grouped by module/functionality
- Exception: conditional imports for optional dependencies (pygame)

```python
try:
    import pygame
except ImportError:
    pygame = None
```

## Error Handling

**Patterns:**
- Explicit assertions: `assert height >= 3 and width >= 3, "Both dimensions must be >= 3."`
- Raises with descriptive messages: `raise ValueError("Must provide either a `layout` name or layout-generating function...")`
- Type checking in validation logic:
  ```python
  if isinstance(seed, int) is False:
      raise ValueError(f"Seed must be a python integer, actual type: {type(seed)}")
  ```
- Assertions for internal consistency checks: `assert np.array_equal(self.shape, encoding.shape)`

**No try-except blocks observed** for routine error handling in main code (only conditional imports)

## Logging

**Framework:** `print()` and Python standard logging (no explicit logger setup observed)

**Patterns:**
- `print()` used in utility functions: `print("Press any key to continue...")`
- No structured logging or custom logger instances detected in core files
- Debug output commented out in development code

## Comments

**When to Comment:**
- Complex algorithmic logic (direction vector mappings)
- Non-obvious coordinate system choices (0 is top, positive Y goes down)
- Registry patterns and metadata about how systems work
- Implementation notes referencing external sources: `# Grid representation derived from Minigrid:`

**JSDoc/TSDoc:**
- Not used (Python project, uses docstrings instead)
- Google-style docstrings provide all documentation

## Function Design

**Size:** No strict limits observed; largest functions are 100+ lines but typically focused on single responsibilities

**Parameters:**
- Use of `**kwargs` for optional configuration: `def __init__(self, config: dict, render_mode: str | None = None, agent_class: agent.Agent | None = None, **kwargs)`
- Clear separation of required vs. optional parameters
- Type hints for all parameters

**Return Values:**
- Explicit return type hints: `:rtype: tuple[np.ndarray, np.ndarray]`
- Tuples for multiple return values: `(rng, np_seed)`, `(height, width)`
- Dictionary returns for complex data: `observation_spaces[agent_id]`
- Single return values preferred where possible

## Module Design

**Exports:**
- Module-level constants and registries: `OBJECT_REGISTRY`, `COLORS`
- Factory functions: `make_object()`, `get_object_class()`
- Class definitions as primary exports
- Dataclasses for constants: `@dataclasses.dataclass`

**Barrel Files:**
- Not used extensively; imports are specific to needed classes/functions
- `__init__.py` files present but checked in separate exploration

**Class Organization:**
- Inheritance used: `class DummyMapEnv(CoGridEnv)`, `class DummyAgent(Agent)`
- Methods ordered: `__init__`, properties, public methods, private methods
- `@staticmethod` used for utility functions: `_set_np_random()`
- `@property` decorators for computed attributes

## Naming Edge Cases

**Abbreviations:**
- Full words preferred: `grid_agents`, `environment`
- Limited abbreviations: `FoV` (field of view), `POV` (point of view), `obs` (observations - common in RL)

**Acronyms in types:**
- `RNG` and `RandomNumberGenerator` both defined as aliases for `np.random.Generator`
- API methods from upstream libs preserved: `parallel_api_test()` from PettingZoo

---

*Convention analysis: 2026-02-10*
