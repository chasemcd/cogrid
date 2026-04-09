# Goal Seeking

A navigation task where agents move through a grid to reach goal cells. Based on the dynamic decision-making paradigm from [Nguyen & Gonzalez (2020)](https://www.cmu.edu/dietrich/sds/ddmlab/papers/NguyenGonzalez2020.pdf).

## Description

Agents start at spawn points and navigate to goal targets placed on the grid. Each goal has an associated value. The agent receives a reward upon reaching a goal. Supports both single-agent and multi-agent configurations.

## Environment Class

`GoalSeeking` extends `CoGridEnv` and adds:

- **Target values** — each goal character maps to a reward value.
- **Optimal path length** — provided per grid configuration for evaluation.
- **Multi-agent spawns** — separate spawn positions for multi-agent setups.

## Objects

| Char | Name | Description |
|------|------|-------------|
| `+` | Spawn | Agent start position |
| `#` | Wall | Impassable boundary |
| ` ` | Floor | Walkable cell |
| (custom) | Goal | Target cells with assigned values |

Goal characters are defined per grid configuration. The grid data maps each character to its reward value.

## Rewards

The `GoalSeekingAgent` defines two penalties:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `step_penalty` | 0.01 | Per-step cost to encourage efficient paths |
| `collision_penalty` | 0.05 | Penalty when agents collide (multi-agent) |

Goal rewards are defined by the `target_values` mapping in the grid configuration.

## Configuration

Goal-seeking environments are configured via a grid data file that specifies:

```python
grid_data = {
    "layout": [...],                    # ASCII grid rows
    "values": {"G": 1.0, "X": -0.5},   # character -> reward value
    "optimal_path_length": 8,           # for evaluation metrics
    "ma_spawns": [(2, 1), (3, 1)],      # multi-agent spawn positions
}
```

The environment is instantiated with a path to this grid data:

```python
from cogrid.envs.goal_seeking.goal_seeking import GoalSeeking

env = GoalSeeking(grid_path="path/to/grid.json", config=config)
```

## Agent API

`GoalSeekingAgent` provides:

- `create_inventory_ob()` — returns a binary vector of collected target objects.
- `inventory_capacity` — number of distinct target types.
- `step_penalty` / `collision_penalty` — per-step and collision costs.

## Spawn Selection

The environment supports multiple spawn modes:

- **Random spawn** — shuffles available spawn points each reset.
- **Pre-defined spawn** — uses `ma_spawns` positions for multi-agent setups.
- **Generated random spawn** — samples from all free spaces (excluding map-specified spawns), controlled by `config["env"]["gen_random_spawn"]`.
