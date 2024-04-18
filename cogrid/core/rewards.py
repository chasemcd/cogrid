from cogrid import cogrid_env


class Reward:
    """Base class for rewards in CoGrid. Rewards are used to define the environment's reward function
    based on actions and state transitions, e.g., R(s, a, s').
    """

    def __init__(
        self,
        name: str,
        coefficient: float = 1.0,
        **kwargs,
    ):
        self.name = name
        self.coefficient = coefficient

    def calculate_reward(
        self,
        state: cogrid_env.CoGridEnv,
        actions: dict[int | str, int | float],
        state_transition: cogrid_env.CoGridEnv,
    ):
        """Calculates the reward based on the state, actions, and state transition.

        :param state: Previous CoGrid environment state.
        :type state: cogrid_env.CoGridEnv
        :param actions: Actions taken in the previous state.
        :type actions: dict[int  |  str, int  |  float]
        :param state_transition: Current CoGrid environment state after taking the actions in the previous state.
        :type state_transition: cogrid_env.CoGridEnv
        :raises NotImplementedError: This method must be implemented in the subclass.
        """
        raise NotImplementedError
