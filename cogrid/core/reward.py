from cogrid.core import grid


class Reward:
    """Base class for rewards in CoGrid. Rewards are used to define the environment's reward function
    based on actions and state transitions, e.g., R(s, a, s').
    """

    def __init__(
        self,
        name: str,
        agent_ids: list[str | int],
        coefficient: float = 1.0,
        common_reward: bool = False,
        **kwargs,
    ):
        """_summary_

        :param name: The reward name to show up in metrics.
        :type name: str
        :param agent_ids: List of agent IDs that this reward applies to.
        :type agent_ids: list[str | int]
        :param coefficient: Scaling coefficient of this reward, defaults to 1.0
        :type coefficient: float, optional
        :param common_reward: Whether or not the reward is given to all agents, defaults to False
        :type common_reward: bool, optional
        """
        self.name = name
        self.agent_ids = agent_ids
        self.coefficient = coefficient
        self.common_reward = common_reward

    def calculate_reward(
        self,
        state: grid.Grid,
        agent_actions: dict[int | str, int | float],
        state_transition: grid.Grid,
    ) -> dict[str | int, float]:
        """Calculates the reward based on the state, actions, and state transition.

        :param state: Previous CoGrid environment state.
        :type state: grid.Grid
        :param agent_actions: Actions taken in the previous state.
        :type agent_actions: dict[int  |  str, int  |  float]
        :param state_transition: Current CoGrid environment state after taking the actions in the previous state.
        :type state_transition: grid.Grid
        :raises NotImplementedError: This method must be implemented in the subclass.
        """
        raise NotImplementedError

    @property
    def is_common_reward(self) -> bool:
        """Returns whether or not the reward is common to all agents.

        :return: Whether or not the reward is common to all agents.
        :rtype: bool
        """
        return self.common_reward