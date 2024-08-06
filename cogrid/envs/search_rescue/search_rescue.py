from cogrid import cogrid_env
from cogrid.envs.search_rescue import search_rescue_grid_objects
from cogrid.envs import registry


class SearchRescueEnv(cogrid_env.CoGridEnv):
    """
    The Search & Rescue task is a reproduction of the __Minimap__ game.
    """

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
