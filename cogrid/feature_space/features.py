"""Feature generators takes the CoGridEnv object and turns it into the desired observation"""

import numpy as np
from gymnasium import spaces
from collections import deque

from cogrid.core.grid_object import OBJECT_NAMES


try:
    import cv2
except ImportError:
    cv2 = None


class Feature:
    """Base class for"""

    shape = None

    def __init__(
        self,
        low: float,
        high: float,
        name: str,
        shape: tuple | np.ndarray = None,
        space: spaces.Space | None = None,
        **kwargs
    ):
        self.low = low
        self.high = high

        if shape is not None:
            self.shape = shape
        assert self.shape is not None, "Must specify shape via class or init!"

        self.space = spaces.Box(low, high, shape) if space is None else space
        self.name = name

    def generate(self, gridworld, player_id, **kwargs):
        raise NotImplementedError


class FullMapImage(Feature):
    def __init__(self, map_size, tile_size, **kwargs):
        rows, cols = map_size
        super().__init__(
            low=0,
            high=1,
            shape=(rows * tile_size, cols * tile_size, 3),
            name="full_map_image",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        img = gridworld.get_full_render(highlight=False)
        img = (img / 255.0).astype(np.float32)
        return img


class StackedFullMapResizedGrayscale(Feature):
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

    def generate(self, gridworld, player_id, **kwargs):
        if self.player_id is not None:
            assert player_id == self.player_id
        else:
            self.player_id = player_id

        img_rgb = gridworld.get_full_render(highlight=False)

        assert img_rgb.shape[-1] == 3

        img_resized = cv2.resize(
            img_rgb, (84, 84), interpolation=cv2.INTER_AREA
        )
        img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_grayscale = np.expand_dims(img_grayscale, -1)
        self.frames.append(img_grayscale / 255.0)

        stacked = np.stack(self.frames, axis=-1).reshape(self.shape)

        return stacked


class FullMapResizedGrayscale(Feature):
    def __init__(self, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(84, 84, 1),
            name="full_map_resized_grayscale_image",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        img_rgb = gridworld.get_full_render(highlight=False)
        img_resized = cv2.resize(
            img_rgb, (84, 84), interpolation=cv2.INTER_AREA
        )
        img_grayscale = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_grayscale = np.expand_dims(img_grayscale, -1)

        return img_grayscale / 255.0


class FoVImage(Feature):
    def __init__(self, view_len, tile_size, **kwargs):
        super().__init__(
            low=0,
            high=1,
            shape=(view_len * tile_size, view_len * tile_size, 3),
            name="fov_image",
            **kwargs
        )
        self.view_len = view_len

    def generate(self, gridworld, player_id, **kwargs):
        img = gridworld.get_pov_render(agent_id=player_id)
        img = (img / 255.0).astype(np.float32)
        return img


class FullMapEncoding(Feature):
    def __init__(self, map_size, **kwargs):
        # TODO(chase): We need to determine a high value for the encodings
        super().__init__(
            low=0,
            high=np.inf,
            shape=(*map_size, 3),
            name="full_map_encoding",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        encoded_map = gridworld.grid.encode(encode_char=False)
        return encoded_map


class FoVEncoding(Feature):
    def __init__(self, view_len, **kwargs):
        # TODO(chase): We need to determine a high value for the encodings
        super().__init__(
            low=0,
            high=100,
            shape=(view_len, view_len, 3),
            name="fov_encoding",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        agent_grid, _ = gridworld.gen_obs_grid(agent_id=player_id)
        encoded_agent_grid = agent_grid.encode(encode_char=False)
        return encoded_agent_grid


class FullMapASCII(Feature):
    def __init__(self, map_size, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(*map_size, 3),
            name="full_map_ascii",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        encoded_map = gridworld.grid.encode(encode_char=True)
        return encoded_map


class FoVASCII(Feature):
    def __init__(self, view_len, **kwargs):
        super().__init__(
            low=-np.inf,
            high=np.inf,
            shape=(view_len, view_len, 3),
            name="fov_ascii",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        agent_grid, _ = gridworld.gen_obs_grid(agent_id=player_id)
        encoded_agent_grid = agent_grid.encode(encode_char=True)

        # TODO(chase): Confirm that this shouldn't already be correct
        # assert encoded_agent_grid[0, -1, agent_grid.width // 2] == "^"
        encoded_agent_grid[0, -1, agent_grid.width // 2] = "^"

        return encoded_agent_grid


class AgentPosition(Feature):

    def __init__(self, **kwargs):
        super().__init__(
            low=0, high=np.inf, shape=(2,), name="agent_position", **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        return np.asarray(gridworld.agents[player_id].pos, dtype=np.int32)


class AgentPositions(Feature):
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

    def generate(self, gridworld, player_id, **kwargs):
        channel_dim = 1 if not self.rgb else 3
        grid = np.full(
            (*gridworld.map_with_agents.shape, channel_dim), fill_value=0
        )
        for a_id, agent in gridworld.agents.items():
            if (
                agent is not None
            ):  # will be None before being set by subclassed env
                assert not self.rgb, "RGB not implemented for new grid."
                # if self.rgb:
                #     grid[:, agent.pos[0], agent.pos[1]] = (
                #         np.array(constants.DEFAULT_COLORS[str(gridworld.id_to_numeric(a_id))]) / 255.0
                #     )
                # else:
                grid[:, agent.pos[0], agent.pos[1]] = int(
                    gridworld.id_to_numeric(a_id)
                )

        return grid


class AgentDir(Feature):
    """One-hot encoding of the agent's direction."""

    def __init__(self, **kwargs):
        super().__init__(low=0, high=1, shape=(4,), name="agent_dir")

    def generate(self, gridworld, player_id, **kwargs):
        encoding = np.zeros(self.shape, dtype=np.int32)
        encoding[gridworld.agents[player_id].dir] = 1
        return encoding


class OtherAgentActions(Feature):
    def __init__(self, num_agents, num_actions, **kwargs):
        super().__init__(
            low=0,
            high=num_actions,
            shape=(num_actions * (num_agents - 1),),
            name="other_agent_actions",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        return (
            np.array(
                [
                    self.one_hot_encode_actions(
                        gridworld.prev_actions[a_id], self.high
                    )
                    for a_id in gridworld.agent_ids
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


class OtherAgentVisibility(Feature):
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

    def generate(self, gridworld, player_id, **kwargs):
        raise NotImplementedError
        # agent = gridworld.agents[player_id]
        # view = ascii_view(gridworld.ascii_map, agent.pos, self.view_len)
        # visibility = np.zeros((len(gridworld.agent_ids) - 1,))
        # other_agent_ids = [pid for pid in gridworld.agent_ids if pid != player_id]
        # for i, other_agent_id in enumerate(other_agent_ids):
        #     numeric_id = gridworld.id_to_numeric(other_agent_id)
        #     visibility[i] = int(numeric_id in view)
        # return visibility


class Role(Feature):
    def __init__(self, num_roles, **kwargs):
        super().__init__(
            low=0, high=num_roles - 1, shape=(num_roles,), name="role", **kwargs
        )
        self.num_roles = num_roles

    def generate(self, gridworld, player_id, **kwargs):
        agent = gridworld.agents[player_id]
        role_encoding = np.zeros((self.num_roles,), dtype=np.uint8)
        role_encoding[agent.role_idx] = 1
        return role_encoding


class Inventory(Feature):
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

    def generate(self, gridworld, player_id, **kwargs):
        agent = gridworld.agents[player_id]
        idxs = []
        for obj in agent.inventory:
            idxs.append(OBJECT_NAMES.index(obj.object_id) + 1)
        sorted_idxs = sorted(idxs)

        encoding = np.zeros(self.shape, dtype=np.uint8)
        for i, idx in enumerate(sorted_idxs):
            encoding[i] = idx

        return encoding


class ActionMask(Feature):
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

    def generate(self, gridworld, player_id, **kwargs):
        action_mask = gridworld.get_action_mask(player_id)
        return action_mask


class AgentID(Feature):
    def __init__(self, env, **kwargs):
        super().__init__(
            low=0,
            high=len(env.agent_ids) - 1,
            shape=(1,),
            name="agent_id",
            **kwargs
        )

    def generate(self, gridworld, player_id, **kwargs):
        agent_number = (
            gridworld.agents[player_id].agent_number - 1
        )  # subtract 1 so we start from 0
        return np.array([agent_number])
