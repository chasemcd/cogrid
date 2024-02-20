import collections

from gymnasium import Space
from cogrid.feature_space import features
from cogrid.core import grid_utils
from cogrid import cogrid_env
from cogrid.core import grid_object
from cogrid.envs.overcooked import overcooked_grid_objects
import numpy as np


def euclidian_distance(pos_1: tuple[int, int], pos_2: tuple[int, int]) -> int:
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


"""
            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)

"""


class OvercookedCollectedFeatures(features.Feature):
    """A wrapper class to create all overcooked features as a single array."""

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):
        num_pots = np.sum(
            [
                int(isinstance(grid_obj, overcooked_grid_objects.Pot))
                for grid_obj in env.grid.grid
            ]
        )
        num_agents = len(env.agent_ids)

        full_shape = len(env.agent_ids) * (
            4 + 2 * 5 + num_pots * 9 + (num_agents - 1) * 2 + 4
        )
        super().__init__(
            low=-1,
            high=np.inf,
            shape=(full_shape,),
            name="overcooked_features",
            **kwargs,
        )
        self.features = [
            features.AgentDir(),
            OvercookedInventory(),
            NextToCounter(),
            ClosestObj(focal_object_type=overcooked_grid_objects.Onion),
            ClosestObj(focal_object_type=overcooked_grid_objects.Plate),
            ClosestObj(focal_object_type=overcooked_grid_objects.PlateStack),
            ClosestObj(focal_object_type=overcooked_grid_objects.OnionStack),
            ClosestObj(focal_object_type=overcooked_grid_objects.OnionSoup),
            ClosestObj(focal_object_type=overcooked_grid_objects.DeliveryZone),
            ClosestObj(focal_object_type=grid_object.Counter),
            OrderedPotFeatures(num_pots=num_pots),
            DistToOtherPlayers(num_other_players=len(env.agent_ids) - 1),
            features.AgentPosition(),
        ]

    def generate(
        self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs
    ) -> np.ndaray:
        cur_player_encoding = self.generate_player_encoding(gridworld, player_id)

        other_player_encodings = []
        for pid in gridworld.agent_ids:
            if pid == player_id:
                continue
            other_player_encodings.append(self.generate_player_encoding(gridworld, pid))

        return np.hstack([cur_player_encoding] + other_player_encodings)

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.features:
            encoded_features.append(feature.generate(env, player_id))
        return np.hstack(encoded_features)


class OvercookedInventory(features.Feature):
    shape = (3,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="overcooked_inventory", **kwargs)

    def generate(self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        agent = gridworld.grid.grid_agents[player_id]

        if not agent.inventory:
            return encoding

        objs = [
            overcooked_grid_objects.Onion,
            overcooked_grid_objects.OnionSoup,
            overcooked_grid_objects.Plate,
        ]
        encoding[objs.index(type(agent.inventory[0]))] = 1

        return encoding


class NextToCounter(features.Feature):
    """This feature represents a multi-hot encoding of whether or not there is a counter
    immediately in each of the four cardinal directions.

    For example, let '#' be the counter and '@' be the player. The following situation
        ####
        #  #
        # @#
        ####

    would result in the feature [0, 1, 1, 0], corresponding to having a counter
    immediately to the east and south, but not north and west.
    """

    shape = (4,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="next_to_counter", **kwargs)

    def generate(self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros((4,), dtype=np.int32)
        agent = gridworld.grid.grid_agents[player_id]

        for i, (row, col) in enumerate(grid_utils.adjacent_positions(*agent.pos)):
            adj_cell = gridworld.grid.get(row, col)
            if isinstance(adj_cell, grid_object.Counter):
                encoding[i] = 1

        return encoding


class ClosestObj(features.Feature):
    """
    This feature calculates (dy, dx) to the closest instance of a specified object.

    # TODO(chase): Use BFS here, right now we're just doing (agent_y - pos_y, agent_x - pos_x)!
    It uses BFS to calculate a path from the player's position to the target, if
    there is no possible path, returns (-1, -1).

    If the object is in the inventory of the player, it returns (0, 0).
    """

    shape = (2,)

    def __init__(self, focal_object_type: grid_object.GridObj, **kwargs):
        super().__init__(
            low=-1,
            high=np.inf,
            name=f"closest_{focal_object_type.object_id}_dist",
            **kwargs,
        )
        self.focal_object_type = focal_object_type

    def generate(self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs):
        agent = gridworld.grid.grid_agents[player_id]

        # If the agent is holding the specified item, return (0, 0)
        if agent.inventory and any(
            [
                isinstance(held_obj, self.focal_object_type)
                for held_obj in agent.inventory
            ]
        ):
            return np.zeros(self.shape, dtype=np.int32)

        # collect the distances
        distances: list[tuple[int, int]] = []
        euc_distances: list[float] = []
        for grid_obj in gridworld.grid.grid:
            if isinstance(grid_obj, self.focal_object_type):
                distances.append(np.array(agent.pos) - np.array(grid_obj.pos))
                euc_distances.append(euclidian_distance(agent.pos, grid_obj.pos))

        # if there were no instances of that object, return (-1, -1)
        if not distances:
            return np.array([-1, -1], dtype=np.int32)

        # find the closest instance and return that array
        min_dist_idx = np.argmin(euc_distances)
        return np.asarray(distances[min_dist_idx], dtype=np.int32)


class OrderedPotFeatures(features.Feature):
    """
    Encode features related to the pot. Note that this assumes the number of pots is fixed, otherwise
    the feature size will vary and will cause errors. For each pot, calculate:
        - pot_j_reachable: {0, 1}  # TODO(chase): use BFS to calculate this, currently fixed at 1.
        - pot_j_status: onehot of {empty | full | is_cooking | is_ready}
        - pot_j_contents: integer of the number of onions in the pot
        - pot_j_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
        - pot_j_distance: (dy, dx) from the player's location
        - pot_j_location: (row, column) of the pot on the grid

    We will sort based on the euclidian distance to the player's location and then concatenate all pot features.
    """

    def __init__(self, num_pots=1, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shap=(num_pots * 9,),
            name="pot_features",
            **kwargs,
        )

    def generate(self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs):
        pot_feature_dict = {}
        agent = gridworld.grid.grid_agents[player_id]

        for grid_obj in gridworld.grid.grid:
            if not isinstance(grid_obj, overcooked_grid_objects.Pot):
                continue

            # Encode if the pot is reachable (size 1)
            pot_reachable = [1]  # TODO(chase): use search to determine

            # Encode if the pot is empty, cooking, or ready (size 3)
            pot_status = np.zeros((3,), dtype=np.int32)  # empty, cooking, ready
            if grid_obj.dish_ready:
                pot_status[0] = 1
            elif len(grid_obj.objects_in_pot) == 0:
                pot_status[1] = 1
            elif len(grid_obj.objects_in_pot) == grid_obj.capacity:
                pot_status[2] = 1
            else:
                raise ValueError("Pot status encoding failed.")

            # Encode the number of each legal content in the pot (size legal_contents)
            pot_contents = np.zeros((len(grid_obj.legal_contents),))
            item_types_in_pot = [
                grid_obj.legal_contents.index(type(pot_content_obj))
                for pot_content_obj in grid_obj.objects_in_pot
            ]
            for obj_index, obj_count in collections.Counter(item_types_in_pot).items():
                pot_contents[obj_index] = obj_count

            # Encode cooking time (size 1)
            pot_cooking_time = np.array(
                (grid_obj.cooking_timer if grid_obj.is_cooking else -1,), dtype=np.int32
            )

            # encode the distance from agent to pot (size 2)
            pot_distance = np.asarray(agent.pos) - np.asarray(grid_obj.pos)

            # encode the pot location (size 2)
            pot_location = np.asarray(grid_obj.pos)

            # concatenate all features and store distance
            euc_distance = euc_distance(agent.pos, grid_obj.pos)

            pot_features = np.hstack(
                [
                    pot_reachable,
                    pot_contents,
                    pot_cooking_time,
                    pot_distance,
                    pot_location,
                ]
            )
            pot_feature_dict[grid_obj.uuid] = (euc_distance, pot_features)

        # sort based on euclidian distance
        # 1 indexes the value, 0 the euc_distance from above
        pot_feature_dict = dict(
            sorted(pot_feature_dict.items(), key=lambda item: item[1][0])
        )

        return np.hstack(pot_feature_dict.values()).astype(np.int32)


class DistToOtherPlayers(features.Feature):
    """Return an encoding of the distance to all other players, unsorted."""

    def __init__(self, num_other_players=1, **kwargs):
        super().__init__(
            low=1,
            high=np.inf,
            shape=(num_other_players * 2,),
            name="dist_to_other_players",
            **kwargs,
        )

    def generate(self, gridworld: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros((2 * (len(gridworld.agent_ids) - 1),), dtype=np.int32)
        agent = gridworld.grid.grid_agents[player_id]

        for i, (pid, other_agent) in enumerate(gridworld.grid.grid_agents.items()):
            if pid == player_id:
                continue

            encoding[i : i + 2] = np.asarray(agent.pos) - np.asarray(other_agent.pos)

        return encoding
