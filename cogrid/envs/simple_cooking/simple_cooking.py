import copy

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core import grid_utils
from cogrid.core.actions import Actions
from cogrid.core import directions
from cogrid import gridworld_env
from cogrid.envs.simple_cooking import agent


class SimpleCooking(gridworld_env.GridWorld):
    """
    The Search & Rescue task is a reproduction of the __Minimap__ game.
    """

    def __init__(self, config, render_mode=None, **kwargs):
        super().__init__(
            config=config,
            render_mode=render_mode,
            env_actions=Actions,
            num_roles=2,
            **kwargs,
        )

        self._setup_agents()

        # When an agent interacts, they must play the interact action 5 times in a row (or until it is successful).
        # This is only triggered when action masking is used.
        self.toggle_seq_len = 5
        self.toggle_sequences = {a_id: 0 for a_id in self.agent_ids}

    def _setup_agents(self) -> None:
        if self.roles:
            assert self.config["num_agents"] % 2 == 0, (
                "Must have an even number of agents for Search and Rescue env with roles"
                "to ensure that there's an equal number of medics and engineers."
            )
        for i in range(self.config["num_agents"]):
            agent_id = f"agent-{i}"
            agent = agent.CookingAgent(
                agent_id=agent_id,
                start_position=self.select_spawn_point(),
                start_direction=self.np_random.choice(directions.Directions),
            )
            self.agents[agent_id] = agent

    def get_terminateds_truncateds(self) -> tuple:
        """ """
        return super().get_terminateds_truncateds()

    def get_action_mask(self, agent_id):
        if 1 <= self.toggle_sequences[agent_id] <= self.toggle_seq_len:
            action_mask = np.zeros((self.action_space.n,))
            action_mask[self.env_actions.Toggle] = 1
            return action_mask
        elif self.can_toggle(agent_id):
            return np.ones((self.action_space.n))
        else:
            mask = np.ones((self.action_space.n,))
            mask[self.env_actions.Toggle] = 0
            return mask

    def can_toggle(self, agent_id):
        # check if we can toggle by just making a copy of the forward cell and attempting to toggle it
        agent = self.agents[agent_id]
        fwd_cell = copy.deepcopy(self.grid.get(*agent.front_pos))
        return fwd_cell.toggle(env=self, toggling_agent=agent)
