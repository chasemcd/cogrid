from cogrid.core.agent import Agent
from cogrid.core.grid_object import GridObj


class OvercookedAgent(Agent):
    def __init__(self, agent_id, start_position, start_direction, **kwargs):
        super().__init__(agent_id, start_position, start_direction, **kwargs)

    @property
    def inventory_capacity(self) -> int:
        return 1

    def can_pickup(self, grid_object: GridObj) -> bool:
        if grid_object.__class__.object_id == "pot" and any(
            [
                inv_obj.__class__.object_id == "plate"
                for inv_obj in self.inventory
            ]
        ):
            return True

        return len(self.inventory) < self.inventory_capacity
