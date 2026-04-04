# Observations

Observations are composed from modular **Feature** extractors. Each feature produces a fixed-size array segment. The engine concatenates all features into a single flat observation vector per agent.

## Feature Base Class

All features inherit from `Feature`:

```python
from cogrid.core.features import Feature, register_feature_type

@register_feature_type("my_feature", scope="my_env")
class MyFeature(Feature):
    per_agent = True   # True: one value per agent; False: global
    obs_dim = 8        # output dimension after ravel()

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state, agent_idx):
            # state is a StateView with dot access to core arrays
            return ...  # (obs_dim,) array
        return fn
```

??? api "`Feature`"

    ::: cogrid.core.features.Feature
        options:
          heading_level: 4
          members:
            - per_agent
            - obs_dim
            - build_feature_fn
            - compute_obs_dim

## Per-Agent vs Global

**Per-agent features** are computed once for each agent and arranged in ego-centric order:

1. Focal agent's features
2. Other agents' features (ascending index, skipping focal)

**Global features** are appended once, identically for all agents.

For 2 agents with per-agent `obs_dim=4` and global `obs_dim=3`, the total observation is `4 + 4 + 3 = 11`.

## Composition

Features are listed by name in the config and composed at init time:

```python
config = {
    "features": [
        "agent_dir",           # per-agent, dim 4
        "agent_position",      # per-agent, dim 2
        "can_move_direction",  # per-agent, dim 4
    ],
    ...
}
```

The engine calls `compose_feature_fns()` which:

1. Looks up each name in the feature registry for the environment's scope.
2. Calls `build_feature_fn()` once per feature at init time.
3. Returns a single function `fn(state, agent_idx) -> (obs_dim,) float32` that concatenates all features in ego-centric order.

??? api "`compose_feature_fns`"

    ::: cogrid.core.features.compose_feature_fns
        options:
          heading_level: 4

## Built-in Features (Global Scope)

Available in all environments:

| Name | Per-Agent | Dim | Description |
|------|-----------|-----|-------------|
| `agent_dir` | Yes | 4 | One-hot facing direction |
| `agent_position` | Yes | 2 | Grid `(row, col)` coordinates |
| `can_move_direction` | Yes | 4 | Binary mask of passable cardinal neighbors |
| `inventory` | Yes | 1 | Held item type ID (0 if empty) |

## Overcooked Features

Registered in the `"overcooked"` scope:

| Name | Per-Agent | Dim | Description |
|------|-----------|-----|-------------|
| `overcooked_inventory` | Yes | 5 | One-hot encoding over pickupable types |
| `next_to_counter` | Yes | 4 | Cardinal adjacency to counters |
| `next_to_pot` | Yes | 16 | Pot adjacency with contents/timer encoding |
| `ordered_pot_features` | Yes | 24 | Per-pot features in grid-scan order (12 per pot) |
| `dist_to_other_players` | Yes | 2 | Delta vector to partner agent |
| `closest_objects` | Yes | 44 | Distances to 7 object types |
| `object_type_masks` | No | 770 | Binary spatial masks for 10 object types (7x11 grid, zero-padded) |
| `order_observation` | No | 9 | Active orders: recipe one-hot + normalized time (3 orders x 3 dims) |
| `layout_id` | No | 7 | One-hot layout identifier |
| `environment_layout` | No | 462 | Binary spatial masks for 6 layout types |

## Writing a Custom Feature

```python
from cogrid.backend import xp
from cogrid.core.features import Feature, register_feature_type

@register_feature_type("agent_health", scope="my_env")
class AgentHealth(Feature):
    per_agent = True
    obs_dim = 1

    @classmethod
    def build_feature_fn(cls, scope):
        def fn(state, agent_idx):
            # Access extra state via StateView attribute fallthrough
            return xp.array([state.agent_health[agent_idx]], dtype=xp.float32)
        return fn
```

Add `"agent_health"` to the config's `"features"` list. The engine handles composition automatically.

If the dimension depends on config values, override `compute_obs_dim`:

```python
@classmethod
def compute_obs_dim(cls, scope, env_config=None):
    if env_config is not None and "n_items" in env_config:
        return env_config["n_items"]
    return cls.obs_dim
```

Pass `env_config` by accepting it in `build_feature_fn`:

```python
@classmethod
def build_feature_fn(cls, scope, env_config=None):
    n_items = env_config["n_items"] if env_config else 5
    def fn(state, agent_idx):
        ...
    return fn
```
