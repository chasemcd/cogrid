"""Generic reward composition utility.

Provides the ``Reward`` base class for reward components.
Reward instances are passed explicitly in the env config ``"rewards"`` list.
Composition is handled by ``cogrid.core.autowire.build_reward_config()``.

Environment-specific reward functions live in their respective envs/ modules:
- Overcooked: ``cogrid.envs.overcooked.rewards``
"""


class Reward:
    """Base class for reward functions.

    Subclasses define compute() which receives StateView objects and returns
    (n_agents,) float32 reward arrays. The returned values are the final
    rewards -- apply any scaling or broadcasting inside compute().

    Parameters are passed via __init__ kwargs and stored in self.config.

    Usage::

        class DeliveryReward(Reward):
            def compute(self, prev_state, state, actions, reward_config):
                coefficient = self.config.get("coefficient", 1.0)
                ...
                return rewards  # (n_agents,) float32

        config = {
            "rewards": [DeliveryReward(coefficient=1.0, common_reward=True)],
        }
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    def compute(self, prev_state, state, actions, reward_config):
        """Compute and return (n_agents,) float32 reward array.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.compute() is not implemented. "
            f"Subclasses must override compute()."
        )
