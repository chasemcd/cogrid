from cogrid.core.agent import Agent


class SRRoles:
    Medic = "medic"
    Engineer = "engineer"


class CookingAgent(Agent):
    def __init__(self, agent_id, start_position, start_direction, **kwargs):
        super().__init__(agent_id, start_position, start_direction, **kwargs)

    @property
    def inventory_capacity(self) -> int:
        return 1
