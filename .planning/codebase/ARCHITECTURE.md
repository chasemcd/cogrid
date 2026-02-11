# Architecture

**Analysis Date:** 2026-02-10

## Pattern Overview

**Overall:** Multi-agent grid-world environment framework using PettingZoo's ParallelEnv pattern with pluggable reward, feature, and environment-specific logic.

**Key Characteristics:**
- Event-driven simulation loop (reset → step → observe → reward)
- Registry-based object instantiation (objects, rewards, features, environments)
- PettingZoo compatible for multi-agent RL training
- Modular environment subclasses for domain-specific tasks
- Configuration-driven setup via dictionaries

## Layers

**Environment Base Layer:**
- Purpose: Core multi-agent environment loop and orchestration
- Location: `cogrid/cogrid_env.py`
- Contains: CoGridEnv class (inherits from pettingzoo.ParallelEnv)
- Depends on: Grid, Agent, Features, Rewards, Actions, Directions
- Used by: Overcooked, SearchRescueEnv, GoalSeeking environments

**Grid & World Model:**
- Purpose: 2D spatial representation and object management
- Location: `cogrid/core/grid.py`, `cogrid/core/grid_object.py`
- Contains: Grid class, GridObj hierarchy, GridAgent wrapper
- Depends on: Directions, Colors, Rendering utilities
- Used by: CoGridEnv for state management and rendering

**Agent Model:**
- Purpose: Individual agent state and physics (position, direction, inventory)
- Location: `cogrid/core/agent.py`
- Contains: Agent base class with inventory, direction vectors, rotation
- Depends on: Directions enum
- Used by: CoGridEnv for agent instantiation and movement

**Feature Generation (Observation):**
- Purpose: Convert grid state to agent observations for RL training
- Location: `cogrid/feature_space/`
- Contains: FeatureSpace class, Feature base class, feature registry
- Depends on: Grid, Agent state
- Used by: CoGridEnv to generate observations in get_obs()

**Reward Computation:**
- Purpose: Define reward functions based on state transitions
- Location: `cogrid/core/reward.py`
- Contains: Reward base class, reward registry
- Depends on: Grid (state and state_transition)
- Used by: CoGridEnv.compute_rewards()

**Environment-Specific Implementations:**
- Purpose: Domain-specific rules and objects
- Locations:
  - `cogrid/envs/overcooked/` - Cooking task
  - `cogrid/envs/search_rescue/` - Search and rescue task
  - `cogrid/envs/goal_seeking/` - Goal reaching task
- Contains: Environment subclass, grid objects, features, rewards
- Depends on: CoGridEnv base layer
- Used by: Registry for environment instantiation

## Data Flow

**Initialization Flow:**

1. Configuration loaded (dict with environment name, agents, action_set, features, rewards, grid)
2. `CoGridEnv.__init__()` calls:
   - `_gen_grid()` → loads layout from registry → creates Grid instance
   - Initializes action/observation spaces
   - Creates FeatureSpace instances (one per agent)
   - Instantiates Reward modules
3. Spawn points identified from grid
4. Environment ready for reset()

**Step Execution Flow:**

```
step(actions: dict[AgentID, Action])
  ├─ Copy previous grid state (for reward computation)
  ├─ Grid.tick() → update timestep-dependent objects
  ├─ _action_idx_to_str() → convert action indices to string names
  ├─ move_agents(actions)
  │  ├─ Validate moves (no collision, passable terrain)
  │  ├─ Randomize move order for fairness
  │  └─ Update agent.pos
  ├─ interact(actions)
  │  ├─ RotateLeft/RotateRight → agent.dir ± 1
  │  ├─ PickupDrop → manage agent.inventory
  │  ├─ Toggle → trigger object.toggle()
  │  └─ Hooks: on_pickup_drop(), on_toggle(), on_interact()
  ├─ update_grid_agents() → sync GridAgent wrapper with Agent
  ├─ get_obs() → FeatureSpace.generate_features() for each agent
  ├─ compute_rewards()
  │  └─ For each reward module:
  │     └─ reward.calculate_reward(prev_grid, actions, new_grid)
  ├─ get_terminateds_truncateds() → episode end conditions
  ├─ render() → optional visualization
  └─ Return (obs, rewards, terminateds, truncateds, infos)
```

**State Management:**

- `self.grid: Grid` - Current world state (objects and agents)
- `self.prev_grid: Grid` - Previous state (for reward delta calculation)
- `self.env_agents: dict[AgentID, Agent]` - Agent instances with position/inventory
- `self.per_agent_reward: dict[AgentID, float]` - Reward accumulator for current step
- `self.per_component_reward: dict[RewardName, dict[AgentID, float]]` - Per-module reward breakdown

## Key Abstractions

**Grid Representation:**
- Purpose: Sparse 2D grid storing GridObj instances
- Examples: `cogrid/core/grid.py`
- Pattern: Flat list with (height, width) indexing → 1D position = row * width + col
- Access: `grid.get(row, col)` → GridObj | None
- Update: `grid.set(row, col, obj)` → stores reference

**GridObj Hierarchy:**
- Purpose: All world objects (agents, walls, items, interactive objects)
- Examples: `cogrid/core/grid_object.py`, `cogrid/envs/overcooked/overcooked_grid_objects.py`
- Base methods: `can_overlap(agent)`, `toggle(env, agent)`, `can_pickup()`, `can_place_on()`
- Subclasses: Wall, Stove, Plate (Overcooked), Victim, Obstacle (SearchRescue)

**Registry Pattern:**
- Used for: Objects, Rewards, Features, Environments
- Pattern: Dict[scope, Dict[id, class]]
- Examples:
  - `OBJECT_REGISTRY` in `cogrid/core/grid_object.py` (scopes: "global", "overcooked", "search_rescue")
  - `REWARD_REGISTRY` in `cogrid/core/reward.py`
  - `FEATURE_SPACE_REGISTRY` in `cogrid/feature_space/feature_space.py`
  - `ENVS_REGISTRY` in `cogrid/envs/registry.py`
- Factory functions: `make_object()`, `make_reward()`, `make_feature_generator()`

**Action/Direction Enums:**
- Actions: `cogrid/core/actions.py` → ActionSets.CardinalActions or ActionSets.RotationActions
- Directions: `cogrid/core/directions.py` → integer enum (Right=0, Down=1, Left=2, Up=3)
- Conversion: Direction to 2D vector via `agent.dir_vec` property

**Layout System:**
- Purpose: ASCII-based grid definition with state encoding
- Examples: `cogrid/core/layouts.py` and environment-specific registrations
- Format: List of strings (e.g., "#" = wall, " " = free space, custom chars for objects)
- State: Parallel numpy array encoding object state (e.g., pot heat level)

## Entry Points

**Primary Entry (Training/Testing):**
- Location: `cogrid/envs/__init__.py`
- Triggers: Import cogrid.envs to auto-register environments
- Responsibilities:
  - Register all standard environment configs via `registry.register()`
  - Register layouts via `layouts.register_layout()`
  - Provide factory functions for environment instantiation

**Secondary Entry (Interactive Play):**
- Location: `cogrid/run_interactive.py`
- Triggers: Direct execution for human gameplay
- Responsibilities: Pygame rendering, keyboard input handling

**Direct Instantiation:**
- Location: Any script importing CoGridEnv or subclasses
- Pattern: `env = Overcooked(config=config_dict, render_mode="human")`
- Example: See `cogrid/test_gridworld_env.py`, `cogrid/test_overcooked_env.py`

## Error Handling

**Strategy:** Exception-based with descriptive error messages.

**Patterns:**

1. **Config Validation:**
   - `_gen_grid()` raises ValueError if layout not found
   - `__init__()` raises ValueError if action_set invalid
   - Feature/Reward/Object setup raises ValueError if ID not in registry

2. **Runtime Assertions:**
   - Grid size validation: `assert height >= 3 and width >= 3`
   - Agent direction bounds: modulo 4 wrapping in `rotate_left()`, `rotate_right()`
   - Position bounds: Grid operations assume valid (row, col) inputs

3. **Graceful Degradation:**
   - Rendering optional: `import pygame` wrapped in try/except
   - Render mode: `render_mode=None` disables visualization

## Cross-Cutting Concerns

**Logging:**
- No centralized logging; uses print() for registry overwrites
- Warning on duplicate feature registration in `FEATURE_SPACE_REGISTRY`
- Example: `cogrid/core/reward.py` line 58-61

**Validation:**
- Config dict validation on initialization (action_set, num_agents, features, rewards, grid)
- Bounds checking: spawn points, grid dimensions, agent positions
- No validation on intermediate state transitions (trusted after CoGridEnv control)

**Authentication:**
- Not applicable (single-process simulation)

**State Consistency:**
- Grid and env_agents kept in sync via `update_grid_agents()` after every interaction
- prev_grid snapshot taken before each step for reward computation
- No optimistic locking; sequential execution enforced by PettingZoo ParallelEnv

---

*Architecture analysis: 2026-02-10*
