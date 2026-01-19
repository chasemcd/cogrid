# Architecture

**Analysis Date:** 2026-01-19

## Pattern Overview

**Overall:** Multi-Agent Grid World Framework with PettingZoo ParallelEnv Pattern

**Key Characteristics:**
- Inheritance-based environment extension (base `CoGridEnv` class extended by domain-specific envs)
- Registry pattern for objects, environments, features, rewards, and layouts
- Component-based observation generation via pluggable Feature classes
- Modular reward calculation through composable Reward classes
- Scoped object registries enabling character/ID reuse across environments (e.g., "overcooked", "search_rescue")

## Layers

**Environment Layer (`cogrid/cogrid_env.py`):**
- Purpose: Core environment logic implementing PettingZoo ParallelEnv interface
- Location: `cogrid/cogrid_env.py`
- Contains: `CoGridEnv` base class with step/reset/render logic, agent management, action handling
- Depends on: Grid, Agent, Feature Space, Reward modules
- Used by: Domain-specific environments (Overcooked, SearchRescue)

**Core Layer (`cogrid/core/`):**
- Purpose: Fundamental abstractions for grid-based environments
- Location: `cogrid/core/`
- Contains: Grid, GridObj, Agent, Actions, Directions, Layouts, Rewards, Constants
- Depends on: NumPy, visualization utilities
- Used by: Environment layer, domain-specific extensions

**Domain Environment Layer (`cogrid/envs/`):**
- Purpose: Specific multi-agent environments with custom objects and behaviors
- Location: `cogrid/envs/overcooked/`, `cogrid/envs/search_rescue/`, `cogrid/envs/goal_seeking/`
- Contains: Environment subclasses, domain-specific GridObjects, Agents, Rewards
- Depends on: Core layer, CoGridEnv base class
- Used by: End users via registry

**Feature Space Layer (`cogrid/feature_space/`):**
- Purpose: Observation generation and space definition
- Location: `cogrid/feature_space/`
- Contains: Feature base class, FeatureSpace manager, concrete feature implementations
- Depends on: Core layer, environment state
- Used by: CoGridEnv for observation generation

**Visualization Layer (`cogrid/visualization/`):**
- Purpose: Rendering grid states to RGB images
- Location: `cogrid/visualization/`
- Contains: Rendering primitives (fill_coords, point_in_circle, etc.)
- Depends on: NumPy, optional OpenCV
- Used by: GridObj.render(), Grid.render(), CoGridEnv.render()

## Data Flow

**Environment Step Flow:**

1. `step(actions)` receives dict of agent_id -> action_idx
2. Actions converted to strings via `_action_idx_to_str()`
3. `move_agents()` resolves movement conflicts and updates positions
4. `interact()` processes pickup/drop/toggle actions per agent
5. `update_grid_agents()` syncs GridAgent representations
6. `compute_rewards()` iterates Reward modules on state transition
7. `get_obs()` generates features via FeatureSpace for each agent
8. Returns (observations, rewards, terminateds, truncateds, infos)

**Grid Encoding/Decoding Flow:**

1. ASCII layout strings registered via `layouts.register_layout()`
2. `_gen_grid()` fetches layout, converts to numpy via `grid_utils.ascii_to_numpy()`
3. Spawn points extracted ('+' chars), grid encoded as (2, H, W) array [object_idx, state]
4. `Grid.decode()` reconstructs Grid with GridObj instances at positions
5. `Grid.encode()` serializes back to numpy for features/serialization

**State Management:**
- Grid state: `self.grid` (Grid object with list of GridObj)
- Agent state: `self.env_agents` dict mapping agent_id -> Agent instance
- Grid maintains `grid_agents` dict for GridAgent visual representations
- State serialization via `get_state()`/`set_state()` for checkpointing

## Key Abstractions

**GridObj (`cogrid/core/grid_object.py`):**
- Purpose: Base class for all objects in the grid world
- Examples: `Wall`, `Floor`, `Counter`, `Door`, `Key` (global); `Pot`, `Plate`, `Onion` (overcooked); `Rubble`, `GreenVictim` (search_rescue)
- Pattern: Template method pattern with overridable methods: `can_overlap()`, `can_pickup()`, `toggle()`, `render()`, `encode()`

**Agent (`cogrid/core/agent.py`):**
- Purpose: Represents an agent with position, direction, inventory, and state
- Examples: Base `Agent`, `OvercookedAgent` (domain-specific)
- Pattern: Subclassing for domain-specific behaviors (e.g., `can_pickup()` logic)

**Feature (`cogrid/feature_space/feature.py`):**
- Purpose: Generates a single observation component from environment state
- Examples: `FullMapEncoding`, `FoVImage`, `AgentPosition`, `Inventory`
- Pattern: Strategy pattern - each Feature implements `generate(env, agent_id)`

**Reward (`cogrid/core/reward.py`):**
- Purpose: Calculates reward based on state transition R(s, a, s')
- Examples: `SoupDeliveryReward`, `OnionInPotReward`
- Pattern: Strategy pattern - each Reward implements `calculate_reward(state, actions, new_state)`

**Grid (`cogrid/core/grid.py`):**
- Purpose: 2D container for GridObj instances with spatial operations
- Examples: Single Grid class used throughout
- Pattern: Value object with encode/decode for serialization, slice/rotate for views

## Entry Points

**Environment Creation (`cogrid/envs/registry.py`):**
- Location: `cogrid/envs/registry.py`
- Triggers: `registry.make("Overcooked-CrampedRoom-V0")`
- Responsibilities: Instantiates registered environment class with config

**Interactive Mode (`cogrid/run_interactive.py`):**
- Location: `cogrid/run_interactive.py`
- Triggers: Script execution or import
- Responsibilities: PyGame-based keyboard control for environment testing

**Environment Registration (`cogrid/envs/__init__.py`):**
- Location: `cogrid/envs/__init__.py`
- Triggers: Import of cogrid.envs
- Responsibilities: Registers all built-in layouts and environment configs

## Error Handling

**Strategy:** Assertions and ValueError exceptions for invalid states

**Patterns:**
- Invalid action set: `ValueError` in `CoGridEnv.__init__()`
- Invalid layout: `ValueError` in `layouts.get_layout()` if unregistered
- Invalid object decode: `ValueError` in `GridObj.decode()` for unknown chars
- Position bounds checking: `assert` statements in `Grid.get()`/`Grid.set()`
- Agent collision detection: Position uniqueness assertion in `move_agents()`

## Cross-Cutting Concerns

**Logging:** Not implemented - uses print statements for warnings (e.g., feature registry overwrites)

**Validation:**
- Config validation happens at environment init time
- Grid bounds checking via assertions
- Inventory capacity enforced in Agent/GridObj interactions

**Serialization:**
- `get_state()`/`set_state()` methods on `CoGridEnv` for full checkpoint/restore
- `Grid.get_state_dict()`/`set_state_dict()` for grid serialization
- `Agent.get_state()`/`from_state()` for agent serialization
- `GridObj.get_extra_state()`/`set_extra_state()` for complex objects (Pot, Counter)
- Version field ("1.0") for future compatibility

**Scoping:**
- Object registry uses "scope" parameter (e.g., "global", "overcooked", "search_rescue")
- Same character can map to different objects in different scopes
- Encoding/decoding respects scope for correct object instantiation

---

*Architecture analysis: 2026-01-19*
