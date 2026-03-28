# Rewards

Rewards are modular components that compute per-agent reward values each step. Multiple reward instances are listed in the config and summed by the step pipeline.

## Reward Base Class

```python
from cogrid.core.rewards import Reward

class MyReward(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        coefficient = self.config.get("coefficient", 1.0)
        # prev_state, state: StateView objects (pre- and post-step)
        # actions: (n_agents,) int32 array
        # reward_config: dict with type_ids, action indices, static_tables, etc.
        rewards = ...
        return rewards  # (n_agents,) float32
```

**`compute()` parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prev_state` | `StateView` | Grid state before the step. |
| `state` | `StateView` | Grid state after the step. |
| `actions` | `ndarray` | `(n_agents,)` int32 action indices. |
| `reward_config` | `dict` | Type ID mappings, action indices, static tables, `n_agents`. |

`reward_config` keys include:

- `type_ids` — dict mapping object names to integer type IDs.
- `action_pickup_drop_idx` — integer index of the PickupDrop action.
- `action_toggle_idx` — integer index of the Toggle action.
- `n_agents` — number of agents.
- `static_tables` — compiled recipe tables, deliverability tables, etc.

## InteractionReward

For rewards triggered by agent-object interactions, use the declarative `InteractionReward` base:

```python
from cogrid.core.rewards import InteractionReward

class OnionInPotReward(InteractionReward):
    action = "pickup_drop"   # "pickup_drop", "toggle", or None
    holds = "onion"          # agent must hold this type
    faces = "pot"            # forward cell must contain this type
```

**Class attributes (declare trigger conditions):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `action` | `str | None` | Required. `"pickup_drop"`, `"toggle"`, or `None` (no action filter). |
| `holds` | `str | None` | Object type the agent must hold. |
| `faces` | `str | None` | Object type in the agent's forward cell. |
| `overlaps` | `str | None` | Object type at the agent's current position. |
| `direction` | `int | None` | Direction the agent must face (0=Right, 1=Down, 2=Left, 3=Up). |

**Instance config (passed via `__init__` kwargs):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coefficient` | `1.0` | Scalar multiplier on the reward. |
| `common_reward` | `False` | If `True`, all agents receive the reward when any agent triggers it. |

The engine checks each condition in sequence, building a boolean mask over agents. Agents matching all conditions receive `coefficient`. When `common_reward=True`, the reward is broadcast to all agents.

### extra_condition()

Override `extra_condition()` for domain-specific checks beyond the declarative attributes:

```python
class OnionInPotReward(InteractionReward):
    action = "pickup_drop"
    holds = "onion"
    faces = "pot"

    def extra_condition(self, mask, prev_state, fwd_r, fwd_c, reward_config):
        # Only reward if the pot has remaining capacity
        pot_contents = prev_state.pot_contents
        ...
        return mask & has_capacity
```

## Composition

Rewards are listed in the config and computed each step:

```python
from cogrid.envs.overcooked.rewards import DeliveryReward, OnionInPotReward, SoupInDishReward

config = {
    "rewards": [
        DeliveryReward(coefficient=1.0, common_reward=True),
        OnionInPotReward(coefficient=0.1, common_reward=False),
        SoupInDishReward(coefficient=0.3, common_reward=False),
    ],
    ...
}
```

Each step, the engine calls `compute()` on every reward instance and sums the results into a single `(n_agents,)` float32 array.

## Writing a Custom Reward

```python
from cogrid.core.rewards import InteractionReward

class GoalReward(InteractionReward):
    action = None          # no specific action required
    overlaps = "goal"      # agent stands on a goal cell

# Usage in config:
config = {
    "rewards": [GoalReward(coefficient=10.0, common_reward=False)],
    ...
}
```

For rewards that do not fit the interaction pattern, subclass `Reward` directly and implement `compute()`.
