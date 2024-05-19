from cogrid.core.agent import Agent
from cogrid.core.grid_object import GridObj
from cogrid.envs.overcooked import overcooked_grid_objects


class OvercookedAgent(Agent):
    def can_pickup(self, grid_object: GridObj) -> bool:
        """Determine if the agent can pickup a specified grid object.
        The Overcooked agent can pick up objects until it reaches capacity,


        :param grid_object: The grid object to check if the agent can pick up.
        :type grid_object: GridObj
        :return: True if the agent can pick up the object, False otherwise.
        :rtype: bool
        """
        if isinstance(grid_object, overcooked_grid_objects.Pot) and any(
            [
                isinstance(inv_obj, overcooked_grid_objects.Plate)
                for inv_obj in self.inventory
            ]
        ):
            return True

        return len(self.inventory) < self.inventory_capacity
