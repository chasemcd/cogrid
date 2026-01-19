# Codebase Structure

**Analysis Date:** 2026-01-19

## Directory Layout

```
cogrid/
├── cogrid/                    # Main package
│   ├── __init__.py           # Package init (empty)
│   ├── cogrid_env.py         # Core CoGridEnv base class
│   ├── constants.py          # Grid-level constants (FreeSpace, Spawn, etc.)
│   ├── run_interactive.py    # PyGame interactive mode runner
│   ├── test_gridworld.py     # Basic gridworld tests
│   ├── test_gridworld_env.py # Environment-level tests
│   ├── test_overcooked_env.py# Overcooked-specific tests
│   ├── core/                 # Core abstractions
│   │   ├── __init__.py       # Empty init
│   │   ├── actions.py        # Action definitions and ActionSets
│   │   ├── agent.py          # Base Agent class
│   │   ├── constants.py      # Core constants (Colors, TilePixels)
│   │   ├── directions.py     # Direction IntEnum
│   │   ├── grid.py           # Grid class
│   │   ├── grid_object.py    # GridObj base + global objects (Wall, Floor, etc.)
│   │   ├── grid_utils.py     # Grid utility functions
│   │   ├── layouts.py        # Layout registry
│   │   ├── reward.py         # Reward base class + registry
│   │   ├── roles.py          # Agent roles (for search_rescue)
│   │   └── typing.py         # Type aliases
│   ├── envs/                 # Domain-specific environments
│   │   ├── __init__.py       # Registers all envs and layouts
│   │   ├── registry.py       # Environment registry (make/register)
│   │   ├── overcooked/       # Overcooked environment
│   │   │   ├── __init__.py   # Empty init
│   │   │   ├── agent.py      # OvercookedAgent subclass
│   │   │   ├── overcooked.py # Overcooked env class
│   │   │   ├── overcooked_features.py  # Overcooked-specific features
│   │   │   ├── overcooked_grid_objects.py  # Pot, Plate, Onion, etc.
│   │   │   ├── overcooked_utils.py     # Utility functions
│   │   │   ├── rewards.py    # Delivery/shaping rewards
│   │   │   └── test_state_serialization.py  # State tests
│   │   ├── search_rescue/    # Search & Rescue environment
│   │   │   ├── __init__.py   # Empty init
│   │   │   ├── search_rescue.py          # SearchRescueEnv class
│   │   │   ├── search_rescue_grid_objects.py  # Victim, Rubble, etc.
│   │   │   ├── search_rescue_tests.py    # SR-specific tests
│   │   │   ├── sr_utils.py   # Utility functions
│   │   │   ├── rewards.py    # SR reward functions
│   │   │   └── grid_configurations/      # Layout configs
│   │   └── goal_seeking/     # Goal-seeking environment
│   │       ├── __init__.py   # Empty init
│   │       ├── agent.py      # Goal-seeking agent
│   │       ├── goal_seeking.py  # GoalSeeking env class
│   │       └── grid_configurations/  # Layout configs
│   ├── feature_space/        # Observation features
│   │   ├── __init__.py       # Empty init
│   │   ├── feature.py        # Feature base class
│   │   ├── feature_space.py  # FeatureSpace manager + registry
│   │   └── features.py       # Concrete feature implementations
│   ├── visualization/        # Rendering utilities
│   │   ├── __init__.py       # Empty init
│   │   └── rendering.py      # Drawing primitives
│   └── testing/              # Testing utilities
│       └── pettingzoo_test.py  # PettingZoo compatibility tests
├── docs/                     # Sphinx documentation
│   ├── conf.py              # Sphinx config
│   ├── content/             # Doc content pages
│   └── _static/             # Static assets
├── setup.py                 # Package setup
├── setup.cfg                # Setup config
├── pyproject.toml           # Build config
├── requirements.txt         # Dependencies
├── README.rst               # Project readme
└── LICENSE                  # Apache 2.0 license
```

## Directory Purposes

**`cogrid/core/`:**
- Purpose: Framework fundamentals shared across all environments
- Contains: Grid, GridObj, Agent, Actions, Directions, Layouts, Rewards, Constants
- Key files: `grid.py`, `grid_object.py`, `agent.py`, `reward.py`

**`cogrid/envs/`:**
- Purpose: Domain-specific environments and their components
- Contains: Environment subclasses, custom GridObjects, Agents, Rewards per domain
- Key files: `registry.py` (environment factory), `__init__.py` (registration)

**`cogrid/envs/overcooked/`:**
- Purpose: Overcooked-AI reproduction for cooperative cooking tasks
- Contains: Overcooked environment, kitchen objects (Pot, Plate, Onion, etc.), delivery rewards
- Key files: `overcooked.py`, `overcooked_grid_objects.py`, `rewards.py`

**`cogrid/envs/search_rescue/`:**
- Purpose: Search & Rescue task with role-based agent specialization
- Contains: Victims (colored by rescue requirements), tools (MedKit, Pickaxe), obstacles (Rubble)
- Key files: `search_rescue.py`, `search_rescue_grid_objects.py`

**`cogrid/feature_space/`:**
- Purpose: Pluggable observation generation system
- Contains: Feature base class, FeatureSpace manager, built-in features
- Key files: `feature.py` (base), `feature_space.py` (manager), `features.py` (implementations)

**`cogrid/visualization/`:**
- Purpose: Rendering grid worlds to RGB images
- Contains: Drawing primitives for tiles (circles, rectangles, triangles)
- Key files: `rendering.py`

## Key File Locations

**Entry Points:**
- `cogrid/envs/registry.py`: `make()` function to create environments
- `cogrid/envs/__init__.py`: Imports trigger environment registration
- `cogrid/run_interactive.py`: Interactive keyboard-controlled testing

**Configuration:**
- `cogrid/core/layouts.py`: Layout registry, `register_layout()` function
- `cogrid/envs/__init__.py`: Built-in layout definitions (Overcooked, SearchRescue)
- `cogrid/core/constants.py`: Colors, tile pixel size

**Core Logic:**
- `cogrid/cogrid_env.py`: Main `CoGridEnv` class (1300+ lines)
- `cogrid/core/grid.py`: `Grid` class with encode/decode/render
- `cogrid/core/grid_object.py`: `GridObj` base class + OBJECT_REGISTRY

**Testing:**
- `cogrid/test_*.py`: Top-level test files
- `cogrid/envs/*/test_*.py`: Domain-specific tests
- `cogrid/testing/pettingzoo_test.py`: PettingZoo API compliance

## Naming Conventions

**Files:**
- `snake_case.py`: All Python files use snake_case
- Domain files prefixed: `overcooked_grid_objects.py`, `search_rescue_grid_objects.py`
- Test files prefixed: `test_*.py`

**Directories:**
- `snake_case/`: All directories use snake_case
- Domain directories: `overcooked/`, `search_rescue/`, `goal_seeking/`

**Classes:**
- `PascalCase`: `CoGridEnv`, `GridObj`, `OvercookedAgent`
- Objects: `OnionStack`, `DeliveryZone`, `GreenVictim`

**Functions/Methods:**
- `snake_case`: `can_pickup()`, `get_state()`, `calculate_reward()`
- Private methods: `_gen_grid()`, `_setup_agents()`

**Constants:**
- Classes with class attributes: `Colors.Yellow`, `GridConstants.FreeSpace`
- Module-level dicts: `OBJECT_REGISTRY`, `LAYOUT_REGISTRY`

## Where to Add New Code

**New Environment:**
1. Create directory: `cogrid/envs/<env_name>/`
2. Create `<env_name>.py` with class extending `CoGridEnv`
3. Create `<env_name>_grid_objects.py` for domain objects
4. Register objects with scope: `register_object(..., scope="<env_name>")`
5. Create optional: `agent.py`, `rewards.py`, `<env_name>_features.py`
6. Register environment in `cogrid/envs/__init__.py`

**New GridObject:**
- Global objects: Add to `cogrid/core/grid_object.py`, call `register_object(..., scope="global")`
- Domain objects: Add to `cogrid/envs/<domain>/<domain>_grid_objects.py`, use domain scope

**New Feature:**
- Add class to `cogrid/feature_space/features.py`
- Extend `Feature` base class from `cogrid/feature_space/feature.py`
- Call `feature_space.register_feature("<name>", <FeatureClass>)`

**New Reward:**
- Add class to `cogrid/core/reward.py` (global) or `cogrid/envs/<domain>/rewards.py` (domain)
- Extend `Reward` base class
- Call `reward.register_reward("<name>", <RewardClass>)`

**New Layout:**
- Register in `cogrid/envs/__init__.py` or domain `__init__.py`
- Call `layouts.register_layout("<name>", [<ascii_rows>])`

**Utilities:**
- Grid utilities: `cogrid/core/grid_utils.py`
- Domain utilities: `cogrid/envs/<domain>/<domain>_utils.py`
- Visualization: `cogrid/visualization/rendering.py`

## Special Directories

**`docs/`:**
- Purpose: Sphinx documentation source
- Generated: `docs/_build/` contains generated HTML
- Committed: Source files committed, `_build/` should be gitignored

**`build/` and `dist/`:**
- Purpose: Python package build artifacts
- Generated: Yes, by `python setup.py build/sdist/bdist_wheel`
- Committed: No, should be gitignored

**`*.egg-info/`:**
- Purpose: Package metadata for editable installs
- Generated: Yes, by `pip install -e .`
- Committed: No, should be gitignored

**`.planning/codebase/`:**
- Purpose: Architecture and planning documentation
- Generated: No, manually created
- Committed: Yes

---

*Structure analysis: 2026-01-19*
