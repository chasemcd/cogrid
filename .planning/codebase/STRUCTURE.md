# Codebase Structure

**Analysis Date:** 2026-02-10

## Directory Layout

```
cogrid/
├── __init__.py                       # Package initialization (empty)
├── cogrid_env.py                     # Base CoGridEnv class (42KB)
├── constants.py                      # Global grid constants
├── run_interactive.py                # Interactive gameplay entry point
├── test_*.py                         # Unit/integration tests
├── core/                             # Core simulation engine
│   ├── __init__.py
│   ├── agent.py                      # Agent class (position, direction, inventory)
│   ├── actions.py                    # Action enums (CardinalActions, RotationActions)
│   ├── constants.py                  # Tile pixels, colors, object codes
│   ├── directions.py                 # Direction enum (Up, Down, Left, Right)
│   ├── grid.py                       # Grid class (2D world representation)
│   ├── grid_object.py                # GridObj hierarchy (all world objects)
│   ├── grid_utils.py                 # ASCII to numpy conversion utilities
│   ├── layouts.py                    # Layout registry and getter
│   ├── reward.py                     # Reward base class and registry
│   ├── roles.py                      # Agent role constants
│   └── typing.py                     # Type aliases (AgentID, ActionType, etc.)
├── envs/                             # Environment-specific implementations
│   ├── __init__.py                   # Environment registration and layout setup
│   ├── registry.py                   # Environment factory registry
│   ├── overcooked/                   # Overcooked cooking task
│   │   ├── __init__.py
│   │   ├── agent.py                  # OvercookedAgent subclass
│   │   ├── overcooked.py             # Overcooked environment class
│   │   ├── overcooked_features.py    # Task-specific observations
│   │   ├── overcooked_grid_objects.py # Stove, Plate, Sink, etc.
│   │   ├── overcooked_utils.py       # Utility functions (empty)
│   │   ├── rewards.py                # Delivery reward implementation
│   │   └── README.md                 # Overcooked documentation
│   ├── search_rescue/                # Search & Rescue task
│   │   ├── __init__.py
│   │   ├── search_rescue.py          # SearchRescueEnv subclass
│   │   ├── search_rescue_grid_objects.py # Victim, Obstacle, Landmark objects
│   │   ├── sr_utils.py               # Search rescue utilities
│   │   ├── rewards.py                # Rescue reward implementation
│   │   ├── search_rescue_tests.py    # Tests (empty)
│   │   ├── grid_configurations/      # Predefined map layouts
│   │   └── README.md                 # Search rescue documentation
│   └── goal_seeking/                 # Goal reaching task
│       ├── __init__.py
│       ├── agent.py                  # GoalSeekingAgent subclass
│       ├── goal_seeking.py           # GoalSeekingEnv subclass
│       └── grid_configurations/      # Predefined map layouts
├── feature_space/                    # Observation generation system
│   ├── __init__.py
│   ├── feature.py                    # Feature base class
│   ├── feature_space.py              # FeatureSpace class and registry
│   └── features.py                   # Concrete feature implementations
├── visualization/                    # Rendering and display
│   ├── __init__.py
│   └── rendering.py                  # Pygame rendering utilities
├── testing/                          # Test utilities
│   └── pettingzoo_test.py           # PettingZoo compatibility tests
├── scripts/                          # Utility scripts (empty)
└── tests/                            # Test suite directory (empty, tests colocated)
```

## Directory Purposes

**Core Module (`cogrid/core/`):**
- Purpose: Core simulation primitives and data structures
- Contains: Agent, Grid, Actions, Rewards, layout system
- Key files: `grid.py` (2D world), `grid_object.py` (object hierarchy), `reward.py` (reward registry)

**Environments Module (`cogrid/envs/`):**
- Purpose: Environment-specific implementations and object definitions
- Contains: Three domain-specific subclasses (Overcooked, SearchRescue, GoalSeeking)
- Key files: `__init__.py` (registers all environments), `registry.py` (environment factory)

**Feature Space Module (`cogrid/feature_space/`):**
- Purpose: Observation generation for RL agents
- Contains: Feature base class, registry, concrete feature implementations
- Key files: `feature_space.py` (FeatureSpace class), `features.py` (implementations)

**Visualization Module (`cogrid/visualization/`):**
- Purpose: Pygame rendering for human visualization
- Contains: Rendering utilities, tile generation, color management
- Key files: `rendering.py` (fill_coords, point_in_rect, highlight_img functions)

## Key File Locations

**Entry Points:**
- `cogrid/cogrid_env.py`: Main CoGridEnv class (42KB, base environment loop)
- `cogrid/run_interactive.py`: Interactive human play entry point
- `cogrid/envs/__init__.py`: Environment registration and layout setup
- `cogrid/__init__.py`: Package root (empty)

**Configuration:**
- `cogrid/constants.py`: Global constants (GridConstants.Spawn, FreeSpace, etc.)
- `cogrid/core/constants.py`: Tile pixel size, color definitions
- `cogrid/core/layouts.py`: Layout registry for grid definitions
- `cogrid/envs/__init__.py`: Pre-defined environment configs as dicts

**Core Logic:**
- `cogrid/core/grid.py`: 2D Grid class with get/set operations
- `cogrid/core/grid_object.py`: GridObj base class and object registry
- `cogrid/core/agent.py`: Agent class (position, direction, inventory management)
- `cogrid/core/reward.py`: Reward base class and registry

**Testing:**
- `cogrid/test_gridworld_env.py`: CoGridEnv unit tests (23KB)
- `cogrid/test_overcooked_env.py`: Overcooked environment tests (14KB)
- `cogrid/test_gridworld.py`: Grid utility tests
- `cogrid/testing/pettingzoo_test.py`: PettingZoo API compliance

## Naming Conventions

**Files:**
- Test files: `test_*.py` (colocated with module, not in separate tests/ directory)
- Environment classes: `*_env.py` or `*.py` in envs subdirectories
- Grid objects: `*_grid_objects.py` (e.g., `overcooked_grid_objects.py`)
- Features: `*_features.py` (e.g., `overcooked_features.py`)
- Utilities: `*_utils.py` (e.g., `sr_utils.py`)

**Directories:**
- Domain-specific: lowercase with underscores (`goal_seeking`, `search_rescue`, `overcooked`)
- Feature-based: lowercase plural (`core`, `envs`, `feature_space`, `visualization`)

**Classes:**
- PascalCase: `CoGridEnv`, `GridObj`, `OvercookedAgent`, `SearchRescueEnv`
- Enum/Constants: PascalCase or UPPER_SNAKE_CASE: `Directions`, `Actions`, `GridConstants`

**Functions/Methods:**
- snake_case: `get_obs()`, `compute_rewards()`, `move_agents()`, `setup_agents()`
- Prefixed with action: `on_reset()`, `on_step()`, `on_interact()` (hooks)
- Utility helpers: `make_object()`, `register_reward()`, `get_layout()`

**Variables:**
- snake_case: `agent_id`, `grid_config`, `spawn_points`, `action_set`
- Private/protected: `_setup_agents()`, `_gen_grid()`, `_np_random`
- Instance state: `self.env_agents`, `self.per_agent_reward`, `self.observation_spaces`

## Where to Add New Code

**New Feature (Observation Type):**
- Implementation: `cogrid/feature_space/features.py` → subclass Feature
- Registration: Add `register_feature()` call (likely in the class file or __init__.py)
- Tests: Colocate as `test_gridworld_env.py` (existing pattern)
- Config: Reference in environment config dict under "features" key

**New Environment/Domain:**
- Base class: `cogrid/envs/{domain_name}/{domain_name}.py` → subclass CoGridEnv
- Grid objects: `cogrid/envs/{domain_name}/{domain_name}_grid_objects.py`
- Features: `cogrid/envs/{domain_name}/{domain_name}_features.py` (if specialized)
- Rewards: `cogrid/envs/{domain_name}/rewards.py`
- Registration: Add to `cogrid/envs/__init__.py` → register_layout(), register_reward(), registry.register()
- Tests: `cogrid/test_{domain_name}_env.py`

**New Reward Function:**
- Implementation: Subclass `cogrid/core/reward.py:Reward`
- Location: Either in `cogrid/envs/{domain}/rewards.py` or `cogrid/core/reward.py`
- Registration: Call `register_reward(reward_id, RewardClass)` in module or envs/__init__.py
- Config: Reference in environment config dict under "rewards" key

**New Grid Object Type:**
- Implementation: Subclass `cogrid/core/grid_object.py:GridObj`
- Location: `cogrid/envs/{domain}/{domain}_grid_objects.py` (scope-specific) or `cogrid/core/grid_object.py` (global)
- Registration: Call `register_object(object_id, ObjectClass, scope="{domain_name}")` in environment module
- Layout chars: Define char attribute; use in layout strings
- Behavior: Override `toggle()`, `can_overlap()`, `can_pickup()`, `can_place_on()` as needed

**New Agent Subclass:**
- Implementation: Subclass `cogrid/core/agent.py:Agent`
- Location: `cogrid/envs/{domain}/agent.py`
- Registration: Pass via `agent_class=YourAgent` parameter to CoGridEnv.__init__()
- Custom behavior: Override inventory management, consume logic (if applicable)

**Utility/Helper Functions:**
- Shared helpers: `cogrid/core/grid_utils.py` (grid operations)
- Domain-specific: `cogrid/envs/{domain}/{domain}_utils.py`
- Rendering: `cogrid/visualization/rendering.py`

## Special Directories

**`cogrid/envs/{domain}/grid_configurations/`:**
- Purpose: Pre-defined map layouts for easy environment instantiation
- Generated: Manually created (not auto-generated)
- Committed: Yes, tracked in git
- Usage: Referenced by layout_fn in environment config

**`build/`, `dist/`, `cogrid.egg-info/`:**
- Purpose: Python package distribution artifacts
- Generated: By `python setup.py build` / `pip install -e .`
- Committed: No, in .gitignore
- Ignored: Use .gitignore exclusions

**`docs/`:**
- Purpose: Sphinx documentation source
- Generated: No (source files)
- Committed: Yes
- Ignored: `docs/_build/` is generated and .gitignored

## Configuration Pattern

All environments use configuration dictionaries with this structure:

```python
config = {
    "name": "environment_name",
    "num_agents": 2,
    "action_set": "cardinal_actions" | "rotation_actions",
    "features": ["feature_id1", "feature_id2"] | str | dict[agent_id, list],
    "rewards": ["reward_id1", "reward_id2"],
    "grid": {
        "layout": "layout_name"  # Registered layout
        # OR
        "layout_fn": callable    # Function that returns (name, layout_strings, state_array)
    },
    "max_steps": 1000,
    "agent_view_size": 7,       # Optional, defaults to 7
    "scope": "domain_name",     # Optional, for object registry scope
    # ... other domain-specific keys
}
```

Standard configs pre-defined in `cogrid/envs/__init__.py` and registered via `registry.register()`.

---

*Structure analysis: 2026-02-10*
