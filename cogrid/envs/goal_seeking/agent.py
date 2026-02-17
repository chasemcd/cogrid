"""Goal-seeking environment agent."""

from cogrid.core.agent import Agent


class GoalSeekingAgent(Agent):
    """Agent subclass for goal-seeking with goal position tracking."""

    def __init__(
        self,
        agent_id,
        start_position,
        env,
        target_values,
        config,
    ):
        """Initialize goal-seeking agent with target values."""
        super().__init__(agent_id, start_position, env, config)

        self.target_values = target_values

        self.step_penalty = 0.01
        self.collision_penalty = 0.05

    def interact(self, char):
        """Handle interaction with a grid character."""
        raise NotImplementedError("Must add in reward modules!")
        # return " "

    def create_inventory_ob(self):
        """Return binary vector of collected target objects."""
        return [1 if obj in self.inventory else 0 for obj in self.target_values.keys()]

    @property
    def inventory_capacity(self):
        """Return the number of distinct target types."""
        return len([*self.target_values.keys()])
