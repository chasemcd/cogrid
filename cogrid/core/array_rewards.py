"""Generic array-based reward composition utility.

Provides the ``ArrayReward`` base class for array-based reward components and
the ``compose_rewards`` function for building composed reward callables.

Environment-specific reward functions live in their respective envs/ modules:
- Overcooked: ``cogrid.envs.overcooked.array_rewards``
"""

# Re-export for convenience (decorator lives in component_registry)
from cogrid.core.component_registry import register_reward_type  # noqa: F401


class ArrayReward:
    """Base class for array-based reward functions.

    Subclasses define compute() which receives state dicts and returns
    (n_agents,) float32 reward arrays. coefficient and common_reward
    are constructor args so they can be overridden by config at
    instantiation time.

    Usage::

        @register_reward_type("delivery", scope="overcooked",
                              coefficient=1.0, common_reward=True)
        class DeliveryReward(ArrayReward):
            def compute(self, prev_state, state, actions, reward_config):
                ...
                return rewards  # (n_agents,) float32
    """

    def __init__(self, coefficient: float = 1.0, common_reward: bool = False):
        self.coefficient = coefficient
        self.common_reward = common_reward

    def compute(self, prev_state, state, actions, reward_config):
        """Compute reward for this component.

        Args:
            prev_state: Dict of state arrays before step.
            state: Dict of state arrays after step.
            actions: (n_agents,) int32 action indices.
            reward_config: Dict with type_ids, n_agents, etc.

        Returns:
            (n_agents,) float32 reward array.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.compute() is not implemented. "
            f"Subclasses must override compute()."
        )


def compose_rewards(reward_configs: list) -> callable:
    """Build a composed reward function from reward configs.

    Called at init time. Each config dict has:
        - 'fn': one of the reward functions above
        - 'coefficient': float scaling factor
        - 'common_reward': bool (optional, default False)

    Returns a closure ``(prev_state, state, actions, n_agents, type_ids, action_pickup_drop_idx) -> reward_array``
    that sums all configured rewards.
    """
    def composed_reward(prev_state, state, actions, n_agents, type_ids, action_pickup_drop_idx=4):
        from cogrid.backend import xp

        total = xp.zeros(n_agents, dtype=xp.float32)
        for config in reward_configs:
            r = config['fn'](
                prev_state, state, actions, type_ids, n_agents,
                coefficient=config['coefficient'],
                common_reward=config.get('common_reward', False),
                action_pickup_drop_idx=action_pickup_drop_idx,
            )
            total = total + r
        return total

    return composed_reward
