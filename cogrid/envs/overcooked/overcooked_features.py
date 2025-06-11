import collections

from cogrid.feature_space import feature
from cogrid.feature_space import features
from cogrid.feature_space import feature_space
from cogrid.core import grid_utils
from cogrid.core import agent
from cogrid import cogrid_env
from cogrid.core import grid_object
from cogrid.envs.overcooked import overcooked_grid_objects
import numpy as np


def euclidian_distance(pos_1: tuple[int, int], pos_2: tuple[int, int]) -> int:
    """Calculate the euclidian distance between two points.

    :param pos_1: The first point on the grid.
    :type pos_1: tuple[int, int]
    :param pos_2: The second point on the grid.
    :type pos_2: tuple[int, int]
    :return: The euclidian distance between the two points.
    :rtype: int
    """
    return np.sqrt((pos_1[0] - pos_2[0]) ** 2 + (pos_1[1] - pos_2[1]) ** 2)


class OvercookedCollectedFeatures(feature.Feature):
    """
    A wrapper class to create all overcooked features as a single array.

    For each agent j, calculate:

        - Agent j Direction
        - Agent j Inventory
        - Agent j Adjacent to Counter
        - Agent j Dist to closest {onion, plate, platestack, onionstack, onionsoup, deliveryzone}
        - Agent j Pot Features
            - pot_j_reachable: {0, 1}
            - pot_j_status: onehot of {empty | full | is_cooking | is_ready}
            - pot_j_contents: integer of the number of onions in the pot
            - pot_j_cooking_timer: integer for the number of ts remaining if cooking, 0 if finished, -1 if not cooking
            - pot_j_distance: (dy, dx) from the player's location
            - pot_j_location: (row, column) of the pot on the grid
        - Agent j Distance to other agents j != i
        - Agent j Position

    The observation is the concatenation of all these features for all players.
    """

    def __init__(self, env: cogrid_env.CoGridEnv, **kwargs):

        num_agents = env.config["num_agents"]

        # NOTE: This is hardcoded because we can have a changing number of pots in the environment.
        # We need it to be the max possible number of pots to keep the feature shape constant.
        max_num_pots = 2

        self.agent_features = [
            features.AgentDir(),
            OvercookedInventory(),
            NextToCounter(),
            NextToPot(),
            ClosestObj(focal_object_type=overcooked_grid_objects.Onion, n=4),
            ClosestObj(focal_object_type=overcooked_grid_objects.Plate, n=4),
            ClosestObj(
                focal_object_type=overcooked_grid_objects.PlateStack, n=2
            ),
            ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionStack, n=2
            ),
            ClosestObj(
                focal_object_type=overcooked_grid_objects.OnionSoup, n=4
            ),
            ClosestObj(
                focal_object_type=overcooked_grid_objects.DeliveryZone, n=2
            ),
            ClosestObj(focal_object_type=grid_object.Counter, n=4),
            OrderedPotFeatures(num_pots=max_num_pots),
            DistToOtherPlayers(num_other_players=num_agents - 1),
            features.AgentPosition(),
            features.CanMoveDirection(),
        ]

        self.global_features = [
            LayoutID(),
            EnvironmentLayout(max_shape=(11, 7)),
        ]

        full_shape = num_agents * np.sum(
            [feature.shape for feature in self.agent_features]
        ) + np.sum([feature.shape for feature in self.global_features])

        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(full_shape,),
            name="overcooked_features",
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

        global_encoding = self.generate_global_encoding(env, player_id)
        encoding = np.hstack([encoding, global_encoding])

        assert np.array_equal(self.shape, encoding.shape)

        return encoding

    def generate_player_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.agent_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)

    def generate_global_encoding(
        self, env: cogrid_env.CoGridEnv, player_id: str | int
    ) -> np.ndarray:
        encoded_features = []
        for feature in self.global_features:
            encoded_features.append(feature.generate(env, player_id))

        return np.hstack(encoded_features)


feature_space.register_feature(
    "overcooked_features", OvercookedCollectedFeatures
)


class LayoutID(feature.Feature):
    shape = (5,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="layout_id", **kwargs)
        self.overcooked_layouts = [
            "overcooked_cramped_room_v0",
            "overcooked_asymmetric_advantages_v0",
            "overcooked_coordination_ring_v0",
            "overcooked_forced_coordination_v0",
            "overcooked_counter_circuit_v0",
        ]

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        encoding[self.overcooked_layouts.index(env.current_layout_id)] = 1
        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class OvercookedInventory(feature.Feature):
    shape = (5,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="overcooked_inventory", **kwargs)

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        agent = env.grid.grid_agents[player_id]

        if not agent.inventory:
            return encoding

        objs = [
            overcooked_grid_objects.Onion,
            overcooked_grid_objects.OnionSoup,
            overcooked_grid_objects.Plate,
            overcooked_grid_objects.Tomato,
            overcooked_grid_objects.TomatoSoup
        ]
        encoding[objs.index(type(agent.inventory[0]))] = 1
        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class NextToCounter(feature.Feature):
    """A feature that represents a multi-hot encoding of whether or not there is a counter
    immediately in each of the four cardinal directions [E, W, S, N].

    For example, let '#' be the counter and '@' be the player. The following situation
        ####
        #  #
        # @#
        ####

    would result in the feature [1, 0, 1, 0], corresponding to having a counter
    immediately to the east and south, but not north and west.
    """

    shape = (4,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="next_to_counter", **kwargs)

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros((4,), dtype=np.int32)
        agent = env.grid.grid_agents[player_id]

        for i, (row, col) in enumerate(
            grid_utils.adjacent_positions(*agent.pos)
        ):
            adj_cell = env.grid.get(row, col)
            if isinstance(adj_cell, grid_object.Counter):
                encoding[i] = 1

        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class NextToPot(feature.Feature):
    """A feature that represents a multi-hot encoding of whether or not there is a pot
    in each of the four cardinal directions, for each of the four pot statuses.

    The feature is a concatenation of four vectors of shape (4,), resulting in a shape of (16,).
    The four vectors correspond to:
    1. Empty pot
    2. Pot with less than three onions
    3. Pot that is currently cooking
    4. Full pot (3 onions, ready to serve)

    For example, if there's an empty pot to the north and a cooking pot to the east,
    the feature would be [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
    """

    shape = (16,)

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, name="next_to_pot", **kwargs)

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        agent = env.grid.grid_agents[player_id]

        for i, (row, col) in enumerate(
            grid_utils.adjacent_positions(*agent.pos)
        ):
            adj_cell = env.grid.get(row, col)
            if isinstance(adj_cell, overcooked_grid_objects.Pot):
                if len(adj_cell.objects_in_pot) == 0:
                    encoding[i] = 1
                elif 0 < len(adj_cell.objects_in_pot) < adj_cell.capacity:
                    encoding[i + 4] = 1
                elif adj_cell.is_cooking:
                    encoding[i + 8] = 1
                elif adj_cell.dish_ready:
                    encoding[i + 12] = 1

        assert np.array_equal(self.shape, encoding.shape)
        return encoding


class ClosestObj(feature.Feature):
    """
    This feature calculates (dy, dx) to the closest instance of a specified object.

    # TODO(chase): Use BFS here, right now we're just doing (agent_y - pos_y, agent_x - pos_x)!
    It uses BFS to calculate a path from the player's position to the target, if
    there is no possible path, returns (0, 0).

    n is the number of closest objects to return.

    Note that this does not consider item's that the agent is holding in its inventory.

    """

    def __init__(
        self, focal_object_type: grid_object.GridObj, n: int = 1, **kwargs
    ):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            name=f"closest_{focal_object_type.object_id}_dist",
            shape=(2 * n,),
            **kwargs,
        )
        self.focal_object_type = focal_object_type
        self.n = n

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        agent = env.grid.grid_agents[player_id]
        encoding = np.zeros(self.shape, dtype=np.int32)

        # collect the distances
        distances: list[int] = []
        deltas: list[tuple[int, int]] = []

        for grid_obj in env.grid.grid:
            if grid_obj is None:
                continue
            # Check if the grid obj is what we're looking for
            is_focal_obj = isinstance(
                grid_obj, self.focal_object_type
            ) and not np.array_equal(agent.pos, grid_obj.pos)

            obj_is_placed_on = isinstance(
                grid_obj.obj_placed_on, self.focal_object_type
            )

            if is_focal_obj or obj_is_placed_on:
                delta = np.array(agent.pos) - np.array(grid_obj.pos)
                manhattan_distance = np.sum(np.abs(delta))
                deltas.append(delta)
                distances.append(manhattan_distance)

        # if there were no instances of that object, return (0, 0)
        if not distances:
            return np.zeros(self.shape, dtype=np.int32)

        # sort the distances and retrieve the corresponding deltas
        sorted_indices = np.argsort(distances)
        sorted_deltas = [deltas[i] for i in sorted_indices][: self.n]

        # flatten the sorted deltas into a flat array
        flat_deltas = [item for sublist in sorted_deltas for item in sublist]

        encoding[: len(flat_deltas)] = flat_deltas

        # # find the closest instance and return that array
        # min_dist_idx = np.argmin(euc_distances)
        # encoding = np.asarray(distances[min_dist_idx], dtype=np.int32)
        assert np.array_equal(self.shape, encoding.shape)

        return encoding


def _calc_pot_features(pot: overcooked_grid_objects.Pot, agent: agent.Agent):
    # Encode if the pot is reachable (size 1)
    pot_reachable = [1]  # TODO(chase): use search to determine

    # Encode if the pot is empty, cooking, or ready (size 4)
    pot_status = np.zeros((4,), dtype=np.int32)  # empty, cooking, ready, ptr
    if pot.dish_ready:
        pot_status[0] = 1
    elif len(pot.objects_in_pot) == 0:
        pot_status[1] = 1
    elif len(pot.objects_in_pot) == pot.capacity:
        pot_status[2] = 1
    else:
        pot_status[3] = 1

    # Encode the number of each legal content in the pot (size legal_contents)
    pot_contents = np.zeros((len(pot.legal_contents),))
    item_types_in_pot = [
        pot.legal_contents.index(type(pot_content_obj))
        for pot_content_obj in pot.objects_in_pot
    ]
    for obj_index, obj_count in collections.Counter(item_types_in_pot).items():
        pot_contents[obj_index] = obj_count

    # Encode cooking time (size 1)
    pot_cooking_time = np.array(
        (pot.cooking_timer if pot.is_cooking else -1,),
        dtype=np.int32,
    )

    # encode the distance from agent to pot (size 2)
    pot_distance = np.asarray(agent.pos) - np.asarray(pot.pos)

    # encode the pot location (size 2)
    pot_location = np.asarray(pot.pos)

    pot_features = np.hstack(
        [
            pot_reachable,
            pot_status,
            pot_contents,
            pot_cooking_time,
            pot_distance,
            pot_location,
        ]
    )

    return pot_features


class OrderedPotFeatures(feature.Feature):
    """Encode features related to the pot. Note that this assumes the number of pots is fixed,
    otherwise the feature size will vary and will cause errors. For each pot, calculate:
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
            shape=(num_pots * 11,),
            name="pot_features",
            **kwargs,
        )
        self.num_pots = num_pots

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        pot_feature_dict = {}
        agent = env.grid.grid_agents[player_id]
        count = 0
        for grid_obj in env.grid.grid:
            if not isinstance(grid_obj, overcooked_grid_objects.Pot):
                continue
            count += 1
            # concatenate all features and store distance
            pot_feature_dict[grid_obj.uuid] = _calc_pot_features(
                grid_obj, agent
            )
            if count >= self.num_pots:
                break

        pot_feature_values = [*pot_feature_dict.values()]
        encoding = np.hstack(pot_feature_values)

        padded_encoding = np.zeros(self.shape, dtype=np.float32)
        padded_encoding[: len(encoding)] = encoding

        return padded_encoding


class NClosestPotFeatures(feature.Feature):
    def __init__(self, num_pots=2, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(num_pots * 11,),
            name="n_closest_pot_features",
            **kwargs,
        )
        self.num_pots = num_pots

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        agent = env.grid.grid_agents[player_id]
        pots_and_dists = []
        for grid_obj in env.grid.grid:
            if not isinstance(grid_obj, overcooked_grid_objects.Pot):
                continue

            euc_dist = euclidian_distance(agent.pos, grid_obj.pos)
            pots_and_dists.append((euc_dist, grid_obj))

        # Retrieve the N closest pots
        closest_pots = [
            pot[1]
            for pot in sorted(pots_and_dists, key=lambda x: x[0])[
                : self.num_pots
            ]
        ]

        pot_features = []
        for pot in closest_pots:
            pot_features.append(_calc_pot_features(pot, agent))

        encoding = np.hstack(pot_features)

        # If we're in an environment with less than N pots, pad with zeros
        padded_encoding = np.zeros(self.shape, dtype=np.float32)
        padded_encoding[: len(encoding)] = encoding

        return padded_encoding


class DistToOtherPlayers(feature.Feature):
    """Return an encoding of the distance to all other players, unsorted."""

    def __init__(self, num_other_agents=1, **kwargs):
        super().__init__(
            low=1,
            high=np.inf,
            shape=(num_other_agents * 2,),
            name="dist_to_other_players",
            **kwargs,
        )
        self.num_other_agents = num_other_agents

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros((2 * (self.num_other_agents),), dtype=np.int32)
        agent = env.grid.grid_agents[player_id]

        other_agent_nums = 0
        for pid, other_agent in env.grid.grid_agents.items():
            if pid == player_id:
                continue

            encoding[other_agent_nums * 2 : other_agent_nums * 2 + 2] = (
                np.asarray(agent.pos) - np.asarray(other_agent.pos)
            )
        other_agent_nums += 1

        assert np.array_equal(self.shape, encoding.shape), (
            "DistToOtherPlayers shape is off. You likely are using an environment with > 2 "
            "agents, in which case you should change the feature encoding."
        )
        return encoding


class EnvironmentLayout(feature.Feature):
    """
    A 1D array representation of the entire environment for each kind of object.

    For each object type, there is a flattened 1D array where each element corresponds
    to a cell in the grid. The value is 1 if the corresponding cell contains that object,
    and 0 otherwise.
    """

    def __init__(self, max_shape, **kwargs):
        self.max_shape = max_shape
        self.object_types = [
            grid_object.Counter,
            overcooked_grid_objects.Pot,
            overcooked_grid_objects.Onion,
            overcooked_grid_objects.Plate,
            overcooked_grid_objects.OnionStack,
            overcooked_grid_objects.PlateStack,
        ]
        shape = (len(self.object_types) * (max_shape[0] * max_shape[1]),)
        super().__init__(
            low=0, high=1, shape=shape, name="environment_layout", **kwargs
        )

    def generate(self, env: cogrid_env.CoGridEnv, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        current_height, current_width = env.grid.height, env.grid.width

        for obj_index, obj_type in enumerate(self.object_types):
            start_index = obj_index * (self.max_shape[0] * self.max_shape[1])
            for row in range(current_height):
                for col in range(current_width):
                    cell = env.grid.get(row, col)
                    if isinstance(cell, obj_type):
                        flat_index = row * self.max_shape[1] + col
                        encoding[start_index + flat_index] = 1

        assert np.array_equal(self.shape, encoding.shape)
        return encoding
