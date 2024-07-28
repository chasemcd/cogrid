import copy

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core import grid_utils
from cogrid.core.actions import Actions
from cogrid.core.directions import Directions
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs.search_rescue import search_rescue_grid_objects
from cogrid.envs import registry


class SearchRescueEnv(CoGridEnv):
    """
    The Search & Rescue task is a reproduction of the __Minimap__ game.
    """

    # def _generate_encoded_grid_states(self) -> tuple[np.ndarray, np.ndarray]:
    #     grid, states = super()._generate_encoded_grid_states()

    #     if grid in ()

    #     grid_gen_kwargs = self.config["grid_gen_kwargs"]
    #     num_green = grid_gen_kwargs.get("num_green", 0)
    #     num_yellow = grid_gen_kwargs.get("num_yellow", 0)
    #     num_red = grid_gen_kwargs.get("num_red", 0)
    #     populate_rubble = grid_gen_kwargs.get("populate_rubble", False)
    #     num_agents = self.config["num_agents"]

    #     # Get free space indices
    #     free_spaces = list(np.argwhere(grid[:, :] == GridConstants.FreeSpace))

    #     objs_to_place = num_green + num_yellow + num_red + num_agents
    #     assert (
    #         len(free_spaces) >= objs_to_place
    #     ), "Not enough free spaces for specified number of objects!"

    #     self.np_random.shuffle(free_spaces)
    #     for _ in range(num_green):
    #         r, c = free_spaces.pop()
    #         grid[r, c] = GridConstants.GreenVictim

    #     for _ in range(num_yellow):
    #         r, c = free_spaces.pop()
    #         grid[r, c] = GridConstants.YellowVictim

    #     for _ in range(num_red):
    #         r, c = free_spaces.pop()
    #         grid[r, c] = GridConstants.RedVictim

    #     for _ in range(num_agents):
    #         r, c = free_spaces.pop()
    #         grid[r, c] = GridConstants.Spawn

    #     if populate_rubble:

    #         def surround_by_rubble(grid, row, col):
    #             for r, c in grid_utils.adjacent_positions(row, col):
    #                 if grid[r, c] == GridConstants.FreeSpace:
    #                     grid[r, c] = GridConstants.Rubble
    #             return grid

    #         # populate rubble around yellow and red
    #         for r, c in np.argwhere(grid == GridConstants.YellowVictim):
    #             grid = surround_by_rubble(grid, r, c)
    #         for r, c in np.argwhere(grid == GridConstants.RedVictim):
    #             grid = surround_by_rubble(grid, r, c)

    #     return grid, states

    def get_terminateds_truncateds(self) -> tuple:
        """
        returns dones only when all targets have been located.
        """
        green_targets_in_grid = any(
            [
                isinstance(obj, search_rescue_grid_objects.GreenVictim)
                for obj in self.grid.grid
            ]
        )
        yellow_targets_in_grid = any(
            [
                isinstance(obj, search_rescue_grid_objects.YellowVictim)
                for obj in self.grid.grid
            ]
        )
        red_targets_in_grid = any(
            [
                isinstance(obj, search_rescue_grid_objects.RedVictim)
                for obj in self.grid.grid
            ]
        )

        all_targets_reached = (
            not green_targets_in_grid
            and not yellow_targets_in_grid
            and not red_targets_in_grid
        )

        if all_targets_reached:
            for agent in self.env_agents.values():
                agent.terminated = True

        return super().get_terminateds_truncateds()


registry.register("search_rescue", SearchRescueEnv)
