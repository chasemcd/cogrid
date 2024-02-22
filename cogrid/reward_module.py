from cogrid import cogrid_env


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
        env: cogrid_env.CoGridEnv,
    ) -> float:
        raise NotImplementedError
