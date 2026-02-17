import functools
import sys
import unittest

from cogrid.cogrid_env import CoGridEnv
from cogrid.core import layouts
from cogrid.core.actions import Actions, ActionSets
from cogrid.envs import registry
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.envs.overcooked.agent import OvercookedAgent
from cogrid.envs.overcooked.config import overcooked_interaction_fn

# Map string action names to their integer indices for the cardinal action set.
_ACTION_IDX = {action: idx for idx, action in enumerate(ActionSets.CardinalActions)}

layouts.register_layout(
    "overcooked_cramped_room_v1",
    [
        "#######",
        "#CCUCC#",
        "#T   O#",
        "#C   C#",
        "#C=C@C#",
        "#######",
    ],
)

N_agent_overcooked_config = {
    "name": "NAgentOvercooked-V0",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": [
        "agent_dir",
        "overcooked_inventory",
        "next_to_counter",
        "next_to_pot",
        "closest_onion",
        "closest_plate",
        "closest_plate_stack",
        "closest_onion_stack",
        "closest_onion_soup",
        "closest_delivery_zone",
        "closest_counter",
        "ordered_pot_features",
        "dist_to_other_players",
        "agent_position",
        "can_move_direction",
        "layout_id",
        "environment_layout",
    ],
    "scope": "overcooked",
    "interaction_fn": overcooked_interaction_fn,
    "grid": {"layout": "overcooked_cramped_room_v1"},
    "max_steps": 1000,
}


def make_env(num_agents=4, layout="overcooked_cramped_room_v1", render_mode="human"):
    config = N_agent_overcooked_config.copy()  # get config obj
    config["num_agents"] = num_agents
    config["grid"]["layout"] = layout

    registry.register(
        "NAgentOvercooked-V0",
        functools.partial(
            CoGridEnv,
            config=config,
            agent_class=OvercookedAgent,
        ),
    )
    return registry.make(
        "NAgentOvercooked-V0",
        render_mode=render_mode,
    )


def pause_until_keypress():
    print("Press any key to continue...")
    while sys.stdin.read(1):
        break


class TestOvercookedEnv(unittest.TestCase):
    def setUp(self):
        from cogrid.backend._dispatch import _reset_backend_for_testing

        _reset_backend_for_testing()
        self.env = make_env(num_agents=2, layout="overcooked_cramped_room_v1", render_mode=None)
        self.env.reset()

    def _a(self, action_str):
        """Convert an Actions string to its integer index."""
        return _ACTION_IDX[action_str]

    def _step(self, actions):
        """Step the environment and sync Grid objects from array state."""
        result = self.env.step(actions)
        self.env._sync_objects_from_state()
        return result

    def pick_tomato_and_move_to_pot(self):
        """Move agent 0 to the pot.

        agent 0 assume to start next to the tomato facing up. in cramped room
        """
        NOOP = self._a(Actions.Noop)

        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveLeft), 1: NOOP})
        obs, reward, _, _, _ = self._step({0: self._a(Actions.PickupDrop), 1: NOOP})
        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveRight), 1: NOOP})
        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveUp), 1: NOOP})

        # now agent 0 is in front of the pot and facing the pot
        agent_0 = self.env.grid.grid_agents[0]
        agent_0_forward_pos = agent_0.front_pos
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        # make sure that object in front is a pot
        self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)

    def test_tomato_in_pot(self):
        """Test that we can get tomato from the stack and put it in the pot.

        Tests Pot.can_place_on() for Tomato objects
        """
        self.pick_tomato_and_move_to_pot()

        # now agent 0 is in front of the pot and facing the pot
        agent_0 = self.env.grid.grid_agents[0]
        agent_0_forward_pos = agent_0.front_pos
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)

        can_place_tomato = pot_tile.can_place_on(agent_0, overcooked_grid_objects.Tomato())

        # testing can_place_on to return the rigt boolean
        self.assertTrue(can_place_tomato)

        # agent 0 Drop the tomato in the pot
        obs, reward, _, _, _ = self._step(
            {0: self._a(Actions.PickupDrop), 1: self._a(Actions.Noop)}
        )
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        # assert that tomato is in the pot
        self.assertTrue(
            any(isinstance(obj, overcooked_grid_objects.Tomato) for obj in pot_tile.objects_in_pot)
        )

        return

    def test_cooking_tomato_soup(self):
        """Test tat puts 3 tomatoes in the pot and then simulates cooking.

        Lastly, check if the soup is ready and can be picked up.

        Tests Pot.can_pickup_from() for TomatoSoup objects
        Tests Pot.pick_up_from() for TomatoSoup objects
        """
        NOOP = self._a(Actions.Noop)
        PICKUP = self._a(Actions.PickupDrop)
        LEFT = self._a(Actions.MoveLeft)

        # put a tomato in the pot
        self.pick_tomato_and_move_to_pot()
        self._step({0: PICKUP, 1: NOOP})
        # go back initial position
        obs, reward, _, _, _ = self._step({0: LEFT, 1: NOOP})
        obs, reward, _, _, _ = self._step({0: LEFT, 1: NOOP})

        # put 2nd tomato in the pot
        self.pick_tomato_and_move_to_pot()
        self._step({0: PICKUP, 1: NOOP})
        # go back to initial position
        obs, reward, _, _, _ = self._step({0: LEFT, 1: NOOP})
        obs, reward, _, _, _ = self._step({0: LEFT, 1: NOOP})

        # put 3rd tomato in the pot
        self.pick_tomato_and_move_to_pot()
        self._step({0: PICKUP, 1: NOOP})

        # simulate cooking
        agent_0 = self.env.grid.grid_agents[0]
        agent_0.inventory.append(
            overcooked_grid_objects.Plate()
        )  # just to make sure the agent has a plate in inventory
        while True:
            # ticks cooking_timer down per step
            obs, reward, _, _, _ = self._step({0: NOOP, 1: NOOP})
            # check if the pot is cooking
            pot_tile = self.env.grid.get(*agent_0.front_pos)
            if pot_tile.can_pickup_from(
                agent_0
            ):  # true if pot timer is 0 and agent has plate in inventory
                # FOOD IS READY
                break
        soup = pot_tile.pick_up_from(agent_0)
        self.assertIsInstance(soup, overcooked_grid_objects.TomatoSoup)

        self.assertEqual(1, 1)
        return

    def test_pot_can_place_on(self):
        self.pick_tomato_and_move_to_pot()

        # now agent 0 is in front of the pot and facing the pot
        agent_0 = self.env.grid.grid_agents[0]
        agent_0_forward_pos = agent_0.front_pos
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        # make sure that object in front is a pot
        self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)

        # test that we can place a tomato on the pot
        can_place_tomato = pot_tile.can_place_on(agent_0, overcooked_grid_objects.Tomato())

        self.assertTrue(can_place_tomato)

        # place the tomato on the pot
        self._step({0: self._a(Actions.PickupDrop), 1: self._a(Actions.Noop)})

        # assert that we can place more tomatoes on the pot
        can_place_tomato = pot_tile.can_place_on(agent_0, overcooked_grid_objects.Tomato())
        self.assertTrue(can_place_tomato)

        # assert that we can't place onion since tomato is already on the pot
        can_place_onion = pot_tile.can_place_on(agent_0, overcooked_grid_objects.Onion())
        self.assertFalse(can_place_onion)

        return

    def test_delivery_zone_can_place_on(self):
        NOOP = self._a(Actions.Noop)
        # agent 0 move right 2 times
        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveRight), 1: NOOP})
        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveRight), 1: NOOP})
        # agent 0 move down 1 time
        obs, reward, _, _, _ = self._step({0: self._a(Actions.MoveDown), 1: NOOP})

        # now agent 0 is in front of the delivery zone
        agent_0 = self.env.grid.grid_agents[0]
        agent_0_forward_pos = agent_0.front_pos
        delivery_zone_tile = self.env.grid.get(*agent_0_forward_pos)

        # make sure that object in front is a delivery zone
        self.assertIsInstance(delivery_zone_tile, overcooked_grid_objects.DeliveryZone)

        # put Tomato soup agent inventory
        agent_0.inventory.append(overcooked_grid_objects.TomatoSoup())

        # test that we can place a tomato soup on the delivery zone
        can_place_tomato_soup = delivery_zone_tile.can_place_on(
            agent_0, overcooked_grid_objects.TomatoSoup()
        )
        self.assertTrue(can_place_tomato_soup)

        # put onion soup agent inventory
        agent_0.inventory[0] = overcooked_grid_objects.OnionSoup()
        # test that we can place a onion soup on the delivery zone
        can_place_onion_soup = delivery_zone_tile.can_place_on(
            agent_0, overcooked_grid_objects.OnionSoup()
        )
        self.assertTrue(can_place_onion_soup)
        return

    def test_random_actions(self):
        """Test that random actions are valid and do not crash the environment."""
        for _ in range(100):
            action = {0: self.env.action_spaces[0].sample(), 1: self.env.action_spaces[1].sample()}
            obs, reward, _, _, _ = self.env.step(action)


if __name__ == "__main__":
    unittest.main()
