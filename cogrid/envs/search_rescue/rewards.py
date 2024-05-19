from cogrid.core import reward
from cogrid.core import actions
from cogrid.core.grid import Grid
from cogrid.envs.search_rescue import search_rescue_grid_objects
from cogrid.core import typing


class RescueReward(reward.Reward):
    """Provide a reward for rescuing one of the victims."""

    def __init__(self, agent_ids: list[str | int], **kwargs):
        super().__init__(
            name="rescue_reward", agent_ids=agent_ids, coefficient=1.0, **kwargs
        )

    def calculate_reward(
        self,
        state: Grid,
        agent_actions: dict[typing.AgentID, typing.ActionType],
        new_state: Grid,
    ) -> dict[typing.AgentID, float]:
        """Calcaute the reward for delivering a soup dish.

        :param state: The previous state of the grid.
        :type state: Grid
        :param actions: Actions taken by each agent in the previous state of the grid.
        :type actions: dict[int  |  str, int  |  float]
        :param new_state: The new state of the grid.
        :type new_state: Grid
        """
        prev_num_green = state.get_obj_count(
            search_rescue_grid_objects.GreenVictim
        )
        prev_num_yellow = state.get_obj_count(
            search_rescue_grid_objects.YellowVictim
        )
        prev_num_red = state.get_obj_count(search_rescue_grid_objects.RedVictim)

        new_num_green = new_state.get_obj_count(
            search_rescue_grid_objects.GreenVictim
        )
        new_num_yellow = new_state.get_obj_count(
            search_rescue_grid_objects.YellowVictim
        )
        new_num_red = new_state.get_obj_count(
            search_rescue_grid_objects.RedVictim
        )

        green_reward = prev_num_green - new_num_green
        yellow_reward = (prev_num_yellow - new_num_yellow) * 2
        red_reward = (prev_num_red - new_num_red) * 3

        reward_dict = {
            agent_id: green_reward + yellow_reward + red_reward
            for agent_id in self.agent_ids
        }

        return reward_dict
