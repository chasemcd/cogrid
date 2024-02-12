from cogrid.core.agent import Agent


class GoalSeekingAgent(Agent):
    def __init__(
        self,
        agent_id,
        start_position,
        env,
        target_values,
        config,
    ):
        super().__init__(agent_id, start_position, env, config)

        self.target_values = target_values

        self.step_penalty = 0.01
        self.collision_penalty = 0.05

    def interact(self, char):
        if char in self.target_values.keys():
            self.reward += self.target_values[char]
            self.inventory.append(char)
            self.done = True
        elif self.collision:
            self.reward -= self.collision_penalty
            self.collision = False
        else:
            self.reward -= self.step_penalty

        return " "

    def create_inventory_ob(self):
        return [1 if obj in self.inventory else 0 for obj in self.target_values.keys()]

    @property
    def inventory_capacity(self):
        return len([*self.target_values.keys()])
