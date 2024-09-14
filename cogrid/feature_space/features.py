"""Feature generators takes the CoGridEnv object and turns it into the desired observation"""

import numpy as np
from gymnasium import spaces
from collections import deque

from cogrid.core.grid_object import OBJECT_NAMES
from cogrid.feature_space import feature, feature_space
from cogrid.core import grid_utils

try:
    import cv2
except ImportError:
    cv2 = None


class FullMapImage(feature.Feature):
    def __init__(self, map_size, tile_size, **kwargs):
        rows, cols = map_size
        super().__init__(
            low=0,
            high=1,
            shape=(rows * tile_size, cols * tile_size, 3),
            name="full_map_image",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        img = env.get_full_render(highlight=False)
        img = (img / 255.0).astype(np.float32)
        return img


feature_space.register_feature("full_map_image", FullMapImage)


class StackedFullMapResizedGrayscale(feature.Feature):
    def __init__(self, **kwargs):
        assert (
            cv2 is not None
        ), "Must install cv2 to use image resizing. Run `pip install opencv-python` then try again."
        super().__init__(
            low=0,
            high=1,
            shape=(84, 84, 4),
            name="stacked_full_map_resized_grayscale_image",
            **kwargs
        )
        self.frames = deque(maxlen=4)
        for _ in range(4):
            self.frames.append(np.zeros((84, 84, 1)))

        self.player_id = None

    def generate(self, env, player_id, **kwargs):
        if self.player_id is not None:
            assert player_id == self.player_id
        else:
            self.player_id = player_id

        img_rgb = env.get_full_render(highlight=False)

        assert img_rgb.shape[-1] == 3

        img_resized = cv2.resize(
            img_rgb, (84, 84), interpolation=cv2.INTER_AREA
        )
        img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_grayscale = np.expand_dims(img_grayscale, -1)
        self.frames.append(img_grayscale / 255.0)

        stacked = np.stack(self.frames, axis=-1).reshape(self.shape)

        return stacked


feature_space.register_feature(
    "stacked_full_map_resized_grayscale_image",
    StackedFullMapResizedGrayscale,
)


class FullMapResizedGrayscale(feature.Feature):
    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(42, 42, 1),
            name="full_map_resized_grayscale_image",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        img_rgb = env.get_full_render(highlight=False)
        img_resized = cv2.resize(
            img_rgb, (42, 42), interpolation=cv2.INTER_AREA
        )
        img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_grayscale = np.expand_dims(img_grayscale, -1)

        return img_grayscale / 255.0


feature_space.register_feature(
    "full_map_resized_grayscale_image",
    FullMapResizedGrayscale,
)


class FoVImage(feature.Feature):
    def __init__(self, view_len, tile_size, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(view_len * tile_size, view_len * tile_size, 3),
            name="fov_image",
            **kwargs
        )
        self.view_len = view_len

    def generate(self, env, player_id, **kwargs):
        img = env.get_pov_render(agent_id=player_id)
        img = (img / 255.0).astype(np.float32)
        return img


feature_space.register_feature("fov_image", FoVImage)


class FullMapEncoding(feature.Feature):
    def __init__(self, map_size, **kwargs):
        # TODO(chase): We need to determine a high value for the encodings
        super().__init__(
            low=0,
            high=np.inf,
            shape=(*map_size, 3),
            name="full_map_encoding",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        encoded_map = env.grid.encode(encode_char=False)
        return encoded_map


feature_space.register_feature("full_map_encoding", FullMapEncoding)


class FoVEncoding(feature.Feature):
    def __init__(self, view_len, **kwargs):
        # TODO(chase): We need to determine a high value for the encodings
        super().__init__(
            low=0,
            high=100,
            shape=(view_len, view_len, 3),
            name="fov_encoding",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        agent_grid, _ = env.gen_obs_grid(agent_id=player_id)
        encoded_agent_grid = agent_grid.encode(encode_char=False)
        return encoded_agent_grid


feature_space.register_feature("fov_encoding", FoVEncoding)


class FullMapASCII(feature.Feature):
    def __init__(self, map_size, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(*map_size, 3),
            name="full_map_ascii",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        encoded_map = env.grid.encode(encode_char=True)
        return encoded_map


feature_space.register_feature("full_map_ascii", FullMapASCII)


class FoVASCII(feature.Feature):
    def __init__(self, view_len, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(view_len, view_len, 3),
            name="fov_ascii",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        agent_grid, _ = env.gen_obs_grid(agent_id=player_id)
        encoded_agent_grid = agent_grid.encode(encode_char=True)

        # TODO(chase): Confirm that this shouldn't already be correct
        # assert encoded_agent_grid[0, -1, agent_grid.width // 2] == "^"
        encoded_agent_grid[0, -1, agent_grid.width // 2] = "^"

        return encoded_agent_grid


feature_space.register_feature("fov_ascii", FoVASCII)


class AgentPosition(feature.Feature):

    def __init__(self, **kwargs):
        super().__init__(
            low=0, high=np.inf, shape=(2,), name="agent_position", **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        return np.asarray(env.env_agents[player_id].pos, dtype=np.int32)


feature_space.register_feature("agent_position", AgentPosition)


class AgentPositions(feature.Feature):
    def __init__(self, map_shape, **kwargs):
        self.rgb = True
        super().__init__(
            low=0,
            high=1,
            shape=(
                *map_shape,
                1 if not self.rgb else 3,
            ),
            name="agent_positions",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        channel_dim = 1 if not self.rgb else 3
        grid = np.full((*env.map_with_agents.shape, channel_dim), fill_value=0)
        for a_id, agent in env.env_agents.items():
            if (
                agent is not None
            ):  # will be None before being set by subclassed env
                assert not self.rgb, "RGB not implemented for new grid."
                # if self.rgb:
                #     grid[:, agent.pos[0], agent.pos[1]] = (
                #         np.array(constants.DEFAULT_COLORS[str(env.id_to_numeric(a_id))]) / 255.0
                #     )
                # else:
                grid[:, agent.pos[0], agent.pos[1]] = int(
                    env.id_to_numeric(a_id)
                )

        return grid


feature_space.register_feature("agent_positions", AgentPositions)


class AgentDir(feature.Feature):
    """One-hot encoding of the agent's direction."""

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, shape=(4,), name="agent_dir")

    def generate(self, env, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        encoding[env.env_agents[player_id].dir] = 1
        return encoding


feature_space.register_feature("agent_dir", AgentDir)


class OtherAgentActions(feature.Feature):
    def __init__(self, num_agents, num_actions, **kwargs):
        super().__init__(
            low=0,
            high=num_actions,
            shape=(num_actions * (num_agents - 1),),
            name="other_agent_actions",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        return (
            np.array(
                [
                    self.one_hot_encode_actions(
                        env.prev_actions[a_id], self.high
                    )
                    for a_id in env.agent_ids
                    if a_id is not player_id
                ]
            )
            .reshape(-1)
            .astype(np.uint8)
        )

    @staticmethod
    def one_hot_encode_actions(action, num_actions):
        oh_action = np.zeros(num_actions)
        oh_action[action] = 1
        return oh_action


feature_space.register_feature("other_agent_actions", OtherAgentActions)


class OtherAgentVisibility(feature.Feature):
    def __init__(self, num_agents, view_len, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(num_agents - 1,),
            name="other_agent_visibility",
            **kwargs
        )
        self.view_len = view_len
        self.num_other_agents = num_agents - 1

    def generate(self, env, player_id, **kwargs):
        raise NotImplementedError
        # agent = env.env_agents[player_id]
        # view = ascii_view(env.ascii_map, agent.pos, self.view_len)
        # visibility = np.zeros((len(env.agent_ids) - 1,))
        # other_agent_ids = [pid for pid in env.agent_ids if pid != player_id]
        # for i, other_agent_id in enumerate(other_agent_ids):
        #     numeric_id = env.id_to_numeric(other_agent_id)
        #     visibility[i] = int(numeric_id in view)
        # return visibility


feature_space.register_feature("other_agent_visibility", OtherAgentVisibility)


class Role(feature.Feature):
    def __init__(self, num_roles, **kwargs):
        super().__init__(
            low=0, high=num_roles - 1, shape=(num_roles,), name="role", **kwargs
        )
        self.num_roles = num_roles

    def generate(self, env, player_id, **kwargs):
        agent = env.env_agents[player_id]
        role_encoding = np.zeros((self.num_roles,), dtype=np.uint8)
        role_encoding[agent.role_idx] = 1
        return role_encoding


feature_space.register_feature("role", Role)


class Inventory(feature.Feature):
    def __init__(self, inventory_capacity, **kwargs):
        if inventory_capacity == 1:
            super().__init__(
                low=0,
                high=len(OBJECT_NAMES),
                shape=(inventory_capacity,),
                name="inventory",
                **kwargs
            )
        else:
            raise NotImplementedError(
                "RLLib has a deserializing bug with the multi-discrete shape."
            )
            # space = MultiDiscrete([len(OBJECT_NAMES) for _ in range(inventory_capacity)])
            # super().__init__(low=0, high=len(OBJECT_NAMES), shape=space.shape, space=space, name="inventory", **kwargs)

    def generate(self, env, player_id, **kwargs):
        agent = env.env_agents[player_id]
        idxs = []
        for obj in agent.inventory:
            idxs.append(OBJECT_NAMES.index(obj.object_id) + 1)
        sorted_idxs = sorted(idxs)

        encoding = np.zeros(self.shape, dtype=np.uint8)
        for i, idx in enumerate(sorted_idxs):
            encoding[i] = idx

        return encoding


feature_space.register_feature("inventory", Inventory)


class CanMoveDirection(feature.Feature):
    """
    Returns a multi-hot encoding of the agent's ability to move in each direction.
    """

    def __init__(self, **kwargs):
        super().__init__(
            low=0, high=1, shape=(4,), name="can_move_direction", **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        agent = env.env_agents[player_id]
        can_move = np.zeros(self.shape, dtype=np.int32)

        # check if the agent can move in each direction by checking if the next tile in each direction
        # is overlappable (grid_obj.can_overlap())
        for i, pos in enumerate(grid_utils.adjacent_positions(*agent.pos)):
            obj = env.grid.get(*pos)

            if obj is None or obj.can_overlap(agent):
                can_move[i] = 1

        return can_move


feature_space.register_feature("can_move_direction", CanMoveDirection)


class ActionMask(feature.Feature):
    def __init__(self, env, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=None,
            space=spaces.Box(
                env.action_spaces[0].n,
            ),
            name="action_mask",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        action_mask = env.get_action_mask(player_id)
        return action_mask


feature_space.register_feature("action_mask", ActionMask)


class AgentID(feature.Feature):
    def __init__(self, env, **kwargs):
        super().__init__(
            low=0,
            high=len(env.agent_ids) - 1,
            shape=(1,),
            name="agent_id",
            **kwargs
        )

    def generate(self, env, player_id, **kwargs):
        agent_number = (
            env.env_agents[player_id].agent_number - 1
        )  # subtract 1 so we start from 0
        return np.array([agent_number])


feature_space.register_feature("agent_id", AgentID)
