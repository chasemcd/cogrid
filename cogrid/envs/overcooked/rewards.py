import functools

from cogrid.core import reward
from cogrid.core import actions
from cogrid.core.grid import Grid
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.core import typing


class SoupDeliveryReward(reward.Reward):
    """Provide a reward for delivery an OnionSoup to a DeliveryZone."""

    def __init__(
        self,
        agent_ids: list[str | int],
        common_reward: bool = True,
        coefficient=1.0,
        **kwargs
    ):
        super().__init__(
            name="delivery_reward",
            agent_ids=agent_ids,
            coefficient=coefficient,
            **kwargs
        )
        self.common_reward = common_reward

    def calculate_reward(
        self,
        state: Grid,
        agent_actions: dict[typing.AgentID, typing.ActionType],
        new_state: Grid,
    ) -> dict[typing.AgentID, float]:
        """Calcaute the reward for delivering a soup dish.

        :param state: The previous state of the grid.
        :type state: Grid
        :param actions: Actions taken by each agent in the previous state of the grid.
        :type actions: dict[int  |  str, int  |  float]
        :param new_state: The new state of the grid.
        :type new_state: Grid
        """
        # Reward is shared among all agents, so calculate once
        # then distribute to all agents

        common_reward = 0
        individual_rewards = {agent_id: 0 for agent_id in self.agent_ids}

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
                common_reward += self.coefficient
                individual_rewards[agent_id] += self.coefficient

        if self.common_reward:
            return {agent_id: common_reward for agent_id in self.agent_ids}

        return individual_rewards


reward.register_reward("delivery_reward", SoupDeliveryReward)
reward.register_reward(
    "delivery_reward_individual",
    functools.partial(SoupDeliveryReward, common_reward=False),
)


class OnionInPotReward(reward.Reward):
    """Provide a reward for putting an onion in the pot. This is for reward shaping."""

    def __init__(
        self, agent_ids: list[str | int], coefficient: float = 0.1, **kwargs
    ):
        super().__init__(
            name="onion_in_pot_reward",
            agent_ids=agent_ids,
            coefficient=coefficient,
            **kwargs
        )

    def calculate_reward(
        self,
        state: Grid,
        agent_actions: dict[int | str, int | float],
        new_state: Grid,
    ) -> dict[str | int, float]:
        """Calcaute the reward putting an onion in the pot.

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
            agent_holding_onion = any(
                [
                    isinstance(obj, overcooked_grid_objects.Onion)
                    for obj in agent.inventory
                ]
            )
            if not agent_holding_onion:
                continue

            # Check if the agent is facing a Pot
            fwd_pos = agent.front_pos
            fwd_cell = state.get(*fwd_pos)
            agent_facing_pot = isinstance(fwd_cell, overcooked_grid_objects.Pot)
            pot_has_capacity = agent_facing_pot and fwd_cell.can_place_on(
                agent=agent, cell=agent.inventory[0]
            )

            if agent_holding_onion and agent_facing_pot and pot_has_capacity:
                rewards[agent_id] = self.coefficient

        return rewards


reward.register_reward("onion_in_pot_reward", OnionInPotReward)


class SoupInDishReward(reward.Reward):
    """Provide a reward for putting the soup into a dish."""

    def __init__(
        self, agent_ids: list[str | int], coefficient: float = 0.3, **kwargs
    ):
        super().__init__(
            name="soup_in_dish_reward",
            agent_ids=agent_ids,
            coefficient=coefficient,
            **kwargs
        )

    def calculate_reward(
        self,
        state: Grid,
        agent_actions: dict[int | str, int | float],
        new_state: Grid,
    ) -> dict[str | int, float]:
        """Calcaute the reward putting the soup .

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
            facing_pot_and_pot_is_ready = (
                agent_facing_pot and fwd_cell.dish_ready
            )

            if agent_holding_soup and facing_pot_and_pot_is_ready:
                rewards[agent_id] = self.coefficient

        return rewards


reward.register_reward("soup_in_dish_reward", SoupInDishReward)
