from cogrid.core.agent import Agent


class SRRoles:
    Medic = "medic"
    Engineer = "engineer"


class SRAgent(Agent):
    def __init__(self, agent_id, start_position, start_direction, role, **kwargs):
        super().__init__(agent_id, start_position, start_direction, **kwargs)
        self.role = role

        if role is not None:
            self.role_idx = 0 if role == SRRoles.Medic else 1
        else:
            self.role_idx = None

    @property
    def inventory_capacity(self) -> int:
        return 1
