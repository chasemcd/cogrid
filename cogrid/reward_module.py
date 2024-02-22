from cogrid.core import grid_object


class RewardModule:
    """
    A reward module is a class that defines a reward function for an environment. The environment
    maintains a list of RewardModules and at each step uses them to calculate the reward.
    """

    # Used for identifying the reward source
    reward_id: str = None

    # If true, all agents get the reward. Otherwise, just the focal agent.
    common_reward = False

    def compute_reward(
        self,
        focal_agent_id: str | int,
        current_grid_state: list[grid_object.GridObj],
        previous_grid_state: list[grid_object.GridObj],
        actions: dict[str | int, str | int],
    ) -> float:
        raise NotImplementedError
