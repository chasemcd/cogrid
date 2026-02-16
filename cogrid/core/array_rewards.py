"""Generic array-based reward composition utility.

Provides the ``ArrayReward`` base class for array-based reward components.
Reward composition is handled automatically by the auto-wiring layer in
``cogrid.core.autowire.build_reward_config_from_components()``.

Environment-specific reward functions live in their respective envs/ modules:
- Overcooked: ``cogrid.envs.overcooked.array_rewards``
"""

# Re-export for convenience (decorator lives in component_registry)
from cogrid.core.component_registry import register_reward_type  # noqa: F401


class ArrayReward:
    """Base class for array-based reward functions.

    Subclasses define compute() which receives StateView objects and returns
    (n_agents,) float32 reward arrays. The returned values are the final
    rewards -- apply any scaling or broadcasting inside compute().

    Usage::

        @register_reward_type("delivery", scope="overcooked")
        class DeliveryReward(ArrayReward):
            def compute(self, prev_state, state, actions, reward_config):
                ...
                return rewards  # (n_agents,) float32
    """

    def compute(self, prev_state, state, actions, reward_config):
        """Compute and return (n_agents,) float32 reward array.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.compute() is not implemented. "
            f"Subclasses must override compute()."
        )
