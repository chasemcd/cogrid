import sys
import time
import unittest
import functools
import numpy as np
from cogrid.core.actions import Actions
from cogrid.core.directions import Directions
from cogrid.envs.overcooked import overcooked_grid_objects
from cogrid.core import grid_object

from cogrid.feature_space import feature
from cogrid.feature_space import features
from cogrid.feature_space import feature_space
from cogrid.envs.overcooked import overcooked_features
from cogrid.envs.overcooked import overcooked
from cogrid import cogrid_env
from cogrid.core import layouts
from cogrid.envs import registry

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

class NAgentOvercookedFeatureSpace(feature.Feature):
    """
    A wrapper class to generate all encoded Overcooked features as a single array.

    For each agent j, calculate:

        - Agent j Direction
        - Agent j Inventory
        - Agent j Adjacent to Counter
        - Agent j Dist to closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
        - Agent j Pot Features for the two closest pots
            - pot_k_reachable: {0, 1}  # NOTE(chase): This is hardcoded to 1 currently.
            - pot_k_status: onehot of {empty | full | is_cooking | is_ready}
            - pot_k_contents: integer of the number of onions in the pot
            - pot_k_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
            - pot_k_distance: (dy, dx) from the player's location
            - pot_k_location: (row, column) of the pot on the grid
        - Agent j Distance to other agents j != i
        - Agent j Position

    The observation is the concatenation of all these features for all players.
    """

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        self.agent_features = [
            # Represent the direction of the agent
            features.AgentDir(),
            # The current inventory of the agent (max=1 item)
            overcooked_features.OvercookedInventory(),
            # One-hot indicator if there is a counter or pot in each of the four cardinal directions
            overcooked_features.NextToCounter(),
            overcooked_features.NextToPot(),
            # The (dy, dx) distance to the closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Onion, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.Plate, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.PlateStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionStack, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionSoup, n=4
            ),
            overcooked_features.ClosestObj(
                focal_object_type=overcooked_grid_objects.DeliveryZone, n=2
            ),
            overcooked_features.ClosestObj(
                focal_object_type=grid_object.Counter, n=4
            ),
            # All pot features for the closest two pots
            overcooked_features.NClosestPotFeatures(num_pots=2),
            # The (dy, dx) distance to the closest other agent
            overcooked_features.DistToOtherPlayers(
                num_other_players=num_agents - 1
            ),
            # The (row, column) position of the agent
            features.AgentPosition(),
            # The direction the agent can move in
            features.CanMoveDirection(),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        )
        #feature_sum = 0
        #feature_dict = {

        #}
        #for feature in self.agent_features:
        #    print(
        #        f"Feature: {feature.name}, shape: {feature.shape}"
        #    )
        #    if feature.name not in feature_dict:
        #        feature_dict[feature.name] = 0
        #    feature_dict[feature.name] += 1
        #    feature_sum += feature.shape[0]
        #print(f"Total feature shape: {feature_sum}")
        #print(f"Feature dict: {feature_dict}")

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="n_agent_overcooked_features",
            **kwargs,
        )

    def generate(
        self, env: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndarray:
        player_encodings = [self.generate_player_encoding(env, player_id)]

        for pid in env.agent_ids:
            if pid == player_id:
                continue
            player_encodings.append(self.generate_player_encoding(env, pid))

        encoding = np.hstack(player_encodings).astype(np.float32)

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)

feature_space.register_feature(
    "n_agent_overcooked_features", NAgentOvercookedFeatureSpace
)


N_agent_overcooked_config = {
    "name": "NAgentOvercooked-V0",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": "n_agent_overcooked_features",
    "rewards": ["onion_in_pot_reward", "soup_in_dish_reward"],
    "scope": "overcooked",
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
            overcooked.Overcooked, config=config
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
        self.env = make_env(num_agents=2, layout="overcooked_cramped_room_v1", render_mode="human")
        self.env.reset()
    
    def test_tomato_in_pot(self):
        """
        Test that we can get tomato from the stack and put it in the pot 
        """
        obs, reward, _, _, _ = self.env.step({0: Actions.MoveLeft, 1: Actions.Noop})
        time.sleep(1)
        obs, reward, _, _, _ = self.env.step({0: Actions.PickupDrop, 1: Actions.Noop})
        # agent 0 move right
        time.sleep(1)
        obs, reward, _, _, _ = self.env.step({0: Actions.MoveRight, 1: Actions.Noop})
        time.sleep(1)
        # agent 0 move up
        obs, reward, _, _, _ = self.env.step({0: Actions.MoveUp, 1: Actions.Noop})
        time.sleep(1)

        # now agent 0 is in front of the pot and facing the pot
        agent_0 = self.env.grid.grid_agents[0]
        agent_0_forward_pos = agent_0.front_pos
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        self.assertIsInstance(pot_tile, overcooked_grid_objects.Pot)

        can_place_tomato = pot_tile.can_place_on(agent_0, overcooked_grid_objects.Tomato())  

        self.assertTrue(can_place_tomato)

        # agent 0 PickupDrop
        obs, reward, _, _, _ = self.env.step({0: Actions.PickupDrop, 1: Actions.Noop})
        pot_tile = self.env.grid.get(*agent_0_forward_pos)

        self.assertTrue(any(  # assert that tomato is in the pot
            isinstance(obj, overcooked_grid_objects.Tomato)
            for obj in pot_tile.objects_in_pot
        ))

        return


if __name__ == "__main__":
    unittest.main() 
