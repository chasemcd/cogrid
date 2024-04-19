from cogrid.core import reward
from cogrid.core import actions
from cogrid.core.grid import Grid
from cogrid.envs.overcooked import overcooked_grid_objects


class SoupDeliveryReward(reward.Reward):
    """Provide a reward for delivery an OnionSoup to a DeliveryZone."""

    def __init__(self, agent_ids: list[str | int], **kwargs):
        super().__init__(
            name="delivery_reward", agent_ids=agent_ids, coefficient=1.0, **kwargs
        )

    def calculate_reward(
        self, state: Grid, agent_actions: dict[int | str, int | float], new_state: Grid
    ) -> dict[str | int, float]:
        """Calcaute the reward for delivering a soup dish.

        :param state: The previous state of the grid.
        :type state: Grid
        :param actions: Actions taken by each agent in the previous state of the grid.
        :type actions: dict[int  |  str, int  |  float]
        :param new_state: The new state of the grid.
        :type new_state: Grid
        """
        rewards = {agent_id: 0 for agent_id in self.agent_ids}

        for agent_id, action in agent_actions.items():
            # Check if agent is performing a PickupDrop action
            if action != actions.Actions.PickupDrop:
                continue

            # Check if an agent is holding an OnionSoup
            agent = state.grid_agents[agent_id]
            agent_holding_soup = any(
                [
                    isinstance(obj, overcooked_grid_objects.OnionSoup)
                    for obj in agent.inventory
                ]
            )

            # Check if the agent is facing a delivery zone
            fwd_pos = agent.front_pos
            fwd_cell = state.get(*fwd_pos)
            agent_facing_delivery = isinstance(
                fwd_cell, overcooked_grid_objects.DeliveryZone
            )

            if agent_holding_soup and agent_facing_delivery:
                rewards[agent_id] = self.coefficient

        return rewards


class OnionInPotReward(reward.Reward):
    """Provide a reward for putting an onion in the pot. This is for reward shaping."""

    def __init__(self, agent_ids: list[str | int], **kwargs):
        super().__init__(
            name="onion_in_pot_reward", agent_ids=agent_ids, coefficient=1.0, **kwargs
        )

    def calculate_reward(
        self, state: Grid, agent_actions: dict[int | str, int | float], new_state: Grid
    ) -> dict[str | int, float]:
        """Calcaute the reward for delivering a soup dish.

        :param state: The previous state of the grid.
        :type state: Grid
        :param actions: Actions taken by each agent in the previous state of the grid.
        :type actions: dict[int  |  str, int  |  float]
        :param new_state: The new state of the grid.
        :type new_state: Grid
        """
        rewards = {agent_id: 0 for agent_id in self.agent_ids}

        for agent_id, action in agent_actions.items():
            # Check if agent is performing a PickupDrop action
            if action != actions.Actions.PickupDrop:
                continue

            # Check if an agent is holding an Onion
            agent = state.grid_agents[agent_id]
            agent_holding_soup = any(
                [
                    isinstance(obj, overcooked_grid_objects.Onion)
                    for obj in agent.inventory
                ]
            )

            # Check if the agent is facing a Pot
            fwd_pos = agent.front_pos
            fwd_cell = state.get(*fwd_pos)
            agent_facing_pot = isinstance(fwd_cell, overcooked_grid_objects.Pot)

            if agent_holding_soup and agent_facing_pot:
                rewards[agent_id] = self.coefficient

        return rewards


class SoupInDishReward(reward.Reward):
    """Provide a reward for putting the soup into a dish."""

    def __init__(self, agent_ids: list[str | int], **kwargs):
        super().__init__(
            name="soup_in_dish_reward", agent_ids=agent_ids, coefficient=1.0, **kwargs
        )

    def calculate_reward(
        self, state: Grid, agent_actions: dict[int | str, int | float], new_state: Grid
    ) -> dict[str | int, float]:
        """Calcaute the reward for delivering a soup dish.

        :param state: The previous state of the grid.
        :type state: Grid
        :param actions: Actions taken by each agent in the previous state of the grid.
        :type actions: dict[int  |  str, int  |  float]
        :param new_state: The new state of the grid.
        :type new_state: Grid
        """
        rewards = {agent_id: 0 for agent_id in self.agent_ids}

        for agent_id, action in agent_actions.items():
            # Check if agent is performing a PickupDrop action
            if action != actions.Actions.PickupDrop:
                continue

            # Check if an agent is holding a Plate
            agent = state.grid_agents[agent_id]
            agent_holding_soup = any(
                [
                    isinstance(obj, overcooked_grid_objects.Plate)
                    for obj in agent.inventory
                ]
            )

            # Check if the agent is facing a Pot with a ready soup
            fwd_pos = agent.front_pos
            fwd_cell = state.get(*fwd_pos)
            agent_facing_pot = isinstance(fwd_cell, overcooked_grid_objects.Pot)
            facing_pot_and_pot_is_ready = agent_facing_pot and fwd_cell.dish_ready

            if agent_holding_soup and facing_pot_and_pot_is_ready:
                rewards[agent_id] = self.coefficient

        return rewards
