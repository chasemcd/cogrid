"""
Testing Copied from the SSD Repo:
https://github.com/eugenevinitsky/sequential_social_dilemma_games/blob/master/tests/test_envs.py
"""

import unittest

import numpy as np
from gymnasium.spaces import Dict

from cogrid.cogrid_env import CoGridEnv
from cogrid.core.agent import Agent
from cogrid.core.actions import Actions
from cogrid.core.directions import Directions
from cogrid.constants import FIXED_GRIDS
from cogrid.constants import GridConstants
from cogrid.core import grid_object
from cogrid.envs.search_rescue import search_rescue


BASE_MAP = (
    [
        "#######",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#     #",
        "#######",
    ],
    np.zeros((7, 7)),
)
BASE_MAP_2 = (
    ["######", "# S  #", "#    #", "#    #", "#   S#", "######"],
    np.zeros((6, 6)),
)
SR_TEST_MAP = (
    [
        "##########",
        "#SS      #",
        "#        #",
        "#        #",
        "#       G#",
        "#        #",
        "#P     K #",
        "#XX M ##D#",
        "#GX Y #GG#",
        "##########",
    ],
    np.zeros((10, 10)),
)


class DummyAgent(Agent):
    def interact(self, char):
        return GridConstants.FreeSpace


class DummyMapEnv(CoGridEnv):
    def __init__(
        self, config: dict, test_grid_data: tuple[np.ndarray, np.ndarray]
    ):
        self.test_grid_data: tuple[np.ndarray, np.ndarray] = test_grid_data
        self.start_positions: list[tuple] | None = config.get("start_positions")
        self.start_directions: list[int] | None = config.get("start_directions")
        super().__init__(config=config)

    def _setup_agents(self) -> None:
        for agent_id in range(self.config["num_agents"]):
            agent = Agent(
                agent_id=agent_id,
                start_position=(
                    self.start_positions.pop(0)
                    if self.start_positions
                    else self.select_spawn_point()
                ),
                start_direction=(
                    self.start_directions.pop(0)
                    if self.start_directions
                    else self.np_random.choice(Directions)
                ),
            )
            self.env_agents[agent_id] = agent

    def _generate_encoded_grid_states(self, **kwargs):
        return self.test_grid_data


class TestMapEnv(unittest.TestCase):
    def tearDown(self) -> None:
        self.env = None

    def _construct_map(
        self,
        map_encoding: tuple[np.ndarray | list[str], np.ndarray],
        num_agents=1,
        start_positions=None,
        start_directions=None,
        view_size=7,
        obs=(
            "full_map_image",
            "full_map_encoding",
            "full_map_ascii",
            "fov_image",
            "fov_encoding",
        ),
        env_cls=DummyMapEnv,
    ):
        dummy_config = {
            "name": "dummyenv",
            "max_steps": 1000,
            "view_size": view_size,
            "num_agents": num_agents,
            "start_positions": start_positions,
            "start_directions": start_directions,
            "features": obs,
            "grid": {
                "layout_fn": lambda *args, **kwargs: (
                    map_encoding if type(map_encoding) == str else None
                )
            },
        }
        self.env = env_cls(dummy_config, map_encoding)
        self.env.reset()
        self.env.update_grid_agents()

    def change_agent_position(self, agent_id, new_pos, new_dir):
        agent = self.env.env_agents[agent_id]
        agent.pos = new_pos
        agent.dir = new_dir
        self.env.update_grid_agents()

    def add_agent_to_env(self, agent_id, start_position, start_direction):
        self.env.env_agents[agent_id] = Agent(
            agent_id,
            start_position=start_position,
            start_direction=start_direction,
        )
        self.env.feature_generators[agent_id] = [
            self.env._fetch_feature_generator(feature_name)
            for feature_name in self.env.config["features"]
        ]
        self.env.observation_spaces[agent_id] = Dict(
            {
                feature.name: feature.space
                for feature in self.env.feature_generators[agent_id]
            }
        )
        self.env.update_grid_agents()

    def test_step(self):
        """Check that the step method works at all for all possible actions"""
        self._construct_map(BASE_MAP_2, num_agents=1)
        aid = self.env.agent_ids[0]
        action_dim = self.env.action_space.n
        for i in range(action_dim):
            self.env.step({aid: i})

    def test_walls(self):
        """Check that the spawned map and base map have walls in the right place"""
        self._construct_map(map_encoding=BASE_MAP, num_agents=0)

        # Make sure wall objects surround the grid
        h, w = self.env.shape
        for row in range(h):
            for col in range(w):
                if row in [0, h - 1] or col in [0, w - 1]:
                    cell = self.env.grid.get(row=row, col=col)
                    assert isinstance(cell, grid_object.Wall)

        # Make sure it encodes the correct representation
        ascii_map = self.env.map_with_agents
        np.testing.assert_array_equal(ascii_map[0, :], ["#"] * w)
        np.testing.assert_array_equal(ascii_map[-1, :], ["#"] * w)
        np.testing.assert_array_equal(ascii_map[:, 0], ["#"] * h)
        np.testing.assert_array_equal(ascii_map[:, -1], ["#"] * h)

    #     env_image = self.env.get_full_render(highlight=False)
    #     wall_img = np.full_like(
    #         env_image.shape,
    #         fill_value=DEFAULT_COLORS[GridConstants.Wall],
    #     )
    #     wall_img = np.moveaxis(wall_img, -1, 0)  # need channel dim to be last for the above fill to work
    #
    #     np.testing.assert_array_equal(
    #         self.env.world_map_color[:, pad, pad:-pad],
    #         wall_img[:, pad, pad:-pad],
    #     )
    #     np.testing.assert_array_equal(
    #         self.env.world_map_color[:, -pad - 1, pad:-pad],
    #         wall_img[:, -pad - 1, pad:-pad],
    #     )
    #     np.testing.assert_array_equal(
    #         self.env.world_map_color[:, pad:-pad, pad],
    #         wall_img[:, pad:-pad, pad],
    #     )
    #     np.testing.assert_array_equal(
    #         self.env.world_map_color[:, pad:-pad, -pad - 1],
    #         wall_img[:, pad:-pad, -pad - 1],
    #     )
    #
    # def assert_logical_and_image_view(self, agent_id, expected_logical_view):
    #     obs = self.env.get_obs()
    #     image_view = obs[agent_id]["image_view"]
    #     logical_view = obs[agent_id]["ascii_view"]
    #
    #     expected_image_view = map_to_image(expected_logical_view)
    #
    #     np.testing.assert_array_equal(logical_view, expected_logical_view)
    #     np.testing.assert_array_equal(image_view, expected_image_view)
    #
    # def test_view(self):
    #     self._construct_map(BASE_MAP, num_agents=1, start_positions=[(3, 3)], view_size=2)
    #     a_id = self.env.agent_ids[0]
    #
    #     # Test that the view is empty around the agent
    #     expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "     ", "     "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Move agent one row closer to the top, check that the top row is walls
    #     self.change_agent_position(a_id, (2, 3))
    #     expected_logical_view = ascii_to_numpy(["#####", "     ", "  1  ", "     ", "     "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check that the agent sees that padded view when the row above them is out of view
    #     self.change_agent_position(a_id, (1, 3))
    #     expected_logical_view = ascii_to_numpy(["     ", "#####", "  1  ", "     ", "     "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check that the view is correct if the left wall is in view
    #     self.change_agent_position(a_id, (3, 2))
    #     expected_logical_view = ascii_to_numpy(["#    ", "#    ", "# 1  ", "#    ", "#    "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if it exceeds the left view
    #     self.change_agent_position(a_id, (3, 1))
    #     expected_logical_view = ascii_to_numpy([" #   ", " #   ", " #1  ", " #   ", " #   "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if the bottom is in view
    #     self.change_agent_position(a_id, (4, 3))
    #     expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "     ", "#####"])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if exceeding bottom view
    #     self.change_agent_position(a_id, (5, 3))
    #     expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "#####", "     "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if the right wall is in view
    #     self.change_agent_position(a_id, (3, 4))
    #     expected_logical_view = ascii_to_numpy(["    #", "    #", "  1 #", "    #", "    #"])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if exceeding the right wall
    #     self.change_agent_position(a_id, (3, 5))
    #     expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "   # ", "   # "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    #     # Check if the view is correct if in the bottom right corner
    #     self.change_agent_position(a_id, (5, 5))
    #     expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "#### ", "     "])
    #     self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # def test_agent_visibility(self):
    #     self._construct_map(BASE_MAP.copy(), 0, None, view_size=2, obs=["image_view", "other_agent_visibility"])
    #     self.add_agent_to_env("agent-0", (1, 1))
    #     self.add_agent_to_env("agent-1", (1, 3))
    #     self.add_agent_to_env("agent-2", (1, 5))
    #     obs, *_ = self.env.step({})
    #     visibility = [a_obs["other_agent_visibility"] for a_obs in obs.values()]
    #     np.testing.assert_array_equal(visibility[0], [1, 0])
    #     np.testing.assert_array_equal(visibility[1], [1, 1])
    #     np.testing.assert_array_equal(visibility[2], [0, 1])

    def test_agent_actions(self):
        a_id = 0
        self._construct_map(BASE_MAP, 1, [(2, 2)], [Directions.Up])

        self.env.step({a_id: Actions.Noop})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 2])
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [2, 2]
        )
        np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")

        self.env.step({a_id: Actions.Left})
        assert self.env.grid.grid_agents[a_id].dir == Directions.Left
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 1])
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [2, 1]
        )
        np.testing.assert_array_equal(self.env.map_with_agents[2, 1], "1")

        self.env.step({a_id: Actions.Right})  # Rotate to face up
        assert self.env.grid.grid_agents[a_id].dir == Directions.Up
        self.env.step({a_id: Actions.Right})  # Rotate to face right
        assert self.env.grid.grid_agents[a_id].dir == Directions.Right
        self.env.step({a_id: Actions.Forward})  # Move back to (2, 2)
        assert self.env.grid.grid_agents[a_id].dir == Directions.Right
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 2])
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [2, 2]
        )
        np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")

        self.env.step({a_id: Actions.Left})  # Rotate to face up
        assert self.env.grid.grid_agents[a_id].dir == Directions.Up
        self.env.step({a_id: Actions.Forward})
        assert self.env.grid.grid_agents[a_id].dir == Directions.Up

        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [1, 2])
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [1, 2]
        )
        np.testing.assert_array_equal(self.env.map_with_agents[1, 2], "1")

        self.env.step({a_id: Actions.Left})  # Rotate to face left
        assert self.env.grid.grid_agents[a_id].dir == Directions.Left
        self.env.step({a_id: Actions.Left})  # Rotate to face down
        assert self.env.grid.grid_agents[a_id].dir == Directions.Down
        self.env.step({a_id: Actions.Forward})
        assert self.env.grid.grid_agents[a_id].dir == Directions.Down

        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, [2, 2])
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [2, 2]
        )
        np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")

        # if an agent tries to move through a wall they should stay in the same place
        # we check that this works correctly for both corner and non-corner edges
        self.change_agent_position(a_id, new_pos=(1, 1), new_dir=Directions.Up)
        assert self.env.env_agents[a_id].dir == Directions.Up
        assert self.env.grid.grid_agents[a_id].dir == Directions.Up
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (1, 1))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [1, 1]
        )

        self.env.step({a_id: Actions.Left})
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (1, 1))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [1, 1]
        )

        self.change_agent_position(
            a_id, new_pos=(4, 4), new_dir=Directions.Right
        )
        self.env.step({a_id: Actions.Forward})
        self.env.step({a_id: Actions.Right})
        self.env.step({a_id: Actions.Forward})
        self.env.step({a_id: Actions.Left})
        self.env.step({a_id: Actions.Forward})
        self.env.step({a_id: Actions.Right})

        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (5, 5))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [5, 5]
        )

        assert self.env.grid.grid_agents[a_id].dir == Directions.Down
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (5, 5))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [5, 5]
        )

        self.env.step({a_id: Actions.Right})
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (5, 4))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [5, 4]
        )

        self.change_agent_position(
            a_id, new_pos=(4, 5), new_dir=Directions.Right
        )
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (4, 5))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [4, 5]
        )

        self.change_agent_position(
            a_id, new_pos=(2, 1), new_dir=Directions.Left
        )
        self.env.step({a_id: Actions.Left})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (2, 1))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [2, 1]
        )

        self.change_agent_position(a_id, new_pos=(1, 2), new_dir=Directions.Up)
        self.env.step({a_id: Actions.Forward})
        np.testing.assert_array_equal(self.env.env_agents[a_id].pos, (1, 2))
        np.testing.assert_array_equal(
            self.env.grid.grid_agents[a_id].pos, [1, 2]
        )

    # def test_agent_conflict(self):
    #     self._construct_map(BASE_MAP_2, 2, [(1, 2), (4, 4)], [Directions.Up, Directions.Up])
    #
    #     # test that agents can't walk into other agents
    #     self.change_agent_position("agent-0", (3, 3))
    #     self.change_agent_position("agent-1", (3, 4))
    #     self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"]})
    #     self.env.step({"agent-1": BASE_ACTION_MAP["LEFT"]})
    #     np.testing.assert_array_equal(self.env.env_agents["agent-0"].pos, (3, 3))
    #     np.testing.assert_array_equal(self.env.env_agents["agent-1"].pos, (3, 4))
    #
    #     # test that agents can't walk through each other if they move simultaneously
    #     self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["LEFT"]})
    #     np.testing.assert_array_equal(self.env.env_agents["agent-0"].pos, (3, 3))
    #     np.testing.assert_array_equal(self.env.env_agents["agent-1"].pos, (3, 4))
    #     # also check that the map looks correct, no agent has disappeared
    #     expected_map = ascii_to_numpy(["######", "#    #", "#    #", "#  12#", "#    #", "######"])
    #
    #     np.testing.assert_array_equal(expected_map, self.env.map_with_agents)
    #
    #     # test that agents can walk into other agents if moves are de-conflicting
    #     # conflict only occurs stochastically so try it 50 times
    #     np.random.seed(1)
    #     self.change_agent_position("agent-0", (2, 4))
    #     self.change_agent_position("agent-1", (3, 4))
    #     expected_map = ascii_to_numpy(["######", "#    #", "#   1#", "#   2#", "#    #", "######"])
    #     self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["UP"]})
    #     np.testing.assert_array_equal(expected_map, self.env.map_with_agents)
    #     self.env.step({"agent-0": BASE_ACTION_MAP["LEFT"], "agent-1": BASE_ACTION_MAP["DOWN"]})
    #     self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["UP"]})
    #     np.testing.assert_array_equal(expected_map, self.env.map_with_agents)

    # def test_map_encoding(self):
    #     self._construct_map(
    #         BASE_MAP, num_agents=1, start_positions=[(3, 3)], start_directions=[Directions.Up], view_size=3
    #     )
    #     a_id = self.env.agent_ids[0]

    # # Test that the view is empty around the agent
    # expected_logical_view = ascii_to_numpy(["   ", "   ", " ^ "])
    # self.assert_encoding_and_image_view(a_id, expected_logical_view)
    #
    # # Move agent one row closer to the top, check that the top row is walls
    # self.change_agent_position(a_id, (2, 3))
    # expected_logical_view = ascii_to_numpy(["#####", "     ", "  1  ", "     ", "     "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check that the agent sees that padded view when the row above them is out of view
    # self.change_agent_position(a_id, (1, 3))
    # expected_logical_view = ascii_to_numpy(["     ", "#####", "  1  ", "     ", "     "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check that the view is correct if the left wall is in view
    # self.change_agent_position(a_id, (3, 2))
    # expected_logical_view = ascii_to_numpy(["#    ", "#    ", "# 1  ", "#    ", "#    "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if it exceeds the left view
    # self.change_agent_position(a_id, (3, 1))
    # expected_logical_view = ascii_to_numpy([" #   ", " #   ", " #1  ", " #   ", " #   "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if the bottom is in view
    # self.change_agent_position(a_id, (4, 3))
    # expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "     ", "#####"])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if exceeding bottom view
    # self.change_agent_position(a_id, (5, 3))
    # expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "#####", "     "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if the right wall is in view
    # self.change_agent_position(a_id, (3, 4))
    # expected_logical_view = ascii_to_numpy(["    #", "    #", "  1 #", "    #", "    #"])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if exceeding the right wall
    # self.change_agent_position(a_id, (3, 5))
    # expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "   # ", "   # "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)
    #
    # # Check if the view is correct if in the bottom right corner
    # self.change_agent_position(a_id, (5, 5))
    # expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "#### ", "     "])
    # self.assert_logical_and_image_view(a_id, expected_logical_view)

    def test_interact_sequence(self):
        self._construct_map(
            FIXED_GRIDS["small_test_map"],
            num_agents=1,
            start_positions=[(1, 1)],
            start_directions=[Directions.Down],
            view_size=7,
        )

        env_config = {
            "name": "search_rescue",
            "num_agents": 1,
            "features": [
                "full_map_encoding",
                "fov_encoding",
            ],
            # TODO(chase): register this layout
            "grid": {
                "layout": "small_test_map",
            },
            "roles": False,
            "agent_view_size": 7,
            "max_steps": 1000,
        }

        self.env = search_rescue.SearchRescueEnv(env_config)
        agent_id = self.env.agent_ids[0]
        self.change_agent_position(agent_id, (1, 1), Directions.Down)

        # Starts in top left facing down, move 4 units down to the Pickaxe
        self.env.step({agent_id: Actions.Forward})
        self.env.step({agent_id: Actions.Forward})
        self.env.step({agent_id: Actions.Forward})
        obs, *_ = self.env.step({agent_id: Actions.Forward})

        # Pick up the pickaxe and move towards the rubble
        obs, *_ = self.env.step({agent_id: Actions.Pickup})
        self.env.step({agent_id: Actions.Forward})


if __name__ == "__main__":
    unittest.main()
