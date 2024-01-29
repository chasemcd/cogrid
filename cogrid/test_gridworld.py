# """
# Testing Copied from the SSD Repo:
# https://github.com/eugenevinitsky/sequential_social_dilemma_games/blob/master/tests/test_envs.py
# """
# import unittest
#
# import numpy as np
# from gymnasium.spaces import Dict
#
# from cogrid.gridworld_env import GridWorld
# from cogrid.core.agent import Agent
# from cogrid.core.grid_utils import ascii_to_numpy
# from cogrid.constants import DEFAULT_COLORS, GridConstants, BASE_ACTION_MAP
# from cogrid.features import map_to_image
#
#
# BASE_MAP = {"grid": ["#######", "#     #", "#     #", "#     #", "#     #", "#     #", "#######"], "values": {}}
# BASE_MAP_2 = {"grid": ["######", "# S  #", "#    #", "#    #", "#   S#", "######"], "values": {}}
#
#
# class DummyAgent(Agent):
#     def interact(self, char):
#         return GridConstants.FreeSpace
#
#
# class DummyMapEnv(GridWorld):
#     def __init__(self, config, test_grid_data):
#         self.test_grid_data = test_grid_data
#         self.start_positions = config.get("start_positions")
#         super().__init__(config=config, grid_creation_fn=self.grid_fn)
#
#     def _setup_agents(self):
#         map_with_agents = self.map_with_agents
#         for i in range(self.config["num_agents"]):
#             agent_id = f"agent-{i}"
#             spawn_point = self.start_positions.pop(0) if self.start_positions else self.select_spawn_point()
#             agent = DummyAgent(
#                 agent_id=agent_id,
#                 start_position=spawn_point,
#                 env=map_with_agents,
#                 config=self.config,
#             )
#             self.agents[agent_id] = agent
#
#     def grid_fn(self, **kwargs):
#         return self.test_grid_data
#
#
# class TestMapEnv(unittest.TestCase):
#     def tearDown(self) -> None:
#         self.env = None
#
#     def _construct_map(self, map, num_agents, start_positions, view_size=2, obs=("image_view", "ascii_view")):
#         dummy_config = {
#             "name": "dummyenv",
#             "horizon": 1000,
#             "view_size": view_size,
#             "num_agents": num_agents,
#             "start_positions": start_positions,
#             "obs": obs,
#         }
#         self.env = DummyMapEnv(dummy_config, map)
#         self.env.reset()
#
#     def change_agent_position(self, agent_id, new_pos):
#         agent = self.env.agents[agent_id]
#         self.env.remove_agents_from_color_map()
#         agent.pos = new_pos
#         self.env.add_agents_to_color_map()
#
#     def add_agent_to_env(self, agent_id, start_position):
#         self.env.agents[agent_id] = DummyAgent(agent_id, start_position, self.env.map_with_agents, self.env.config)
#         self.env.feature_generators[agent_id] = [
#             self.env._fetch_feature_generator(ob_name) for ob_name in self.env.config["obs"]
#         ]
#         self.env.observation_space[agent_id] = Dict(
#             {feature.name: feature.space for feature in self.env.feature_generators[agent_id]}
#         )
#         self.env.add_agents_to_color_map()
#
#     def test_step(self):
#         """Check that the step method works at all for all possible actions"""
#         self._construct_map(BASE_MAP_2, 1, None, 2)
#         aid = self.env.agent_ids[0]
#         action_dim = self.env.action_space.n
#         for i in range(action_dim):
#             self.env.step({aid: i})
#
#     def test_walls(self):
#         """Check that the spawned map and base map have walls in the right place"""
#         pad = 2
#         self._construct_map(BASE_MAP, 0, None, pad)
#         base_grid_np = ascii_to_numpy(BASE_MAP["grid"])
#         np.testing.assert_array_equal(self.env.base_grid[0, :], base_grid_np[0, :])
#         np.testing.assert_array_equal(self.env.base_grid[-1, :], base_grid_np[-1, :])
#         np.testing.assert_array_equal(self.env.base_grid[:, 0], base_grid_np[0, :])
#         np.testing.assert_array_equal(self.env.base_grid[:, -1], base_grid_np[:, -1])
#
#         wall_img = np.full(
#             (self.env.world_map_color.shape[1], self.env.world_map_color.shape[2], 3),
#             fill_value=DEFAULT_COLORS[GridConstants.Wall],
#         )
#         wall_img = np.moveaxis(wall_img, -1, 0)  # need channel dim to be last for the above fill to work
#         np.testing.assert_array_equal(
#             self.env.world_map_color[:, pad, pad:-pad],
#             wall_img[:, pad, pad:-pad],
#         )
#         np.testing.assert_array_equal(
#             self.env.world_map_color[:, -pad - 1, pad:-pad],
#             wall_img[:, -pad - 1, pad:-pad],
#         )
#         np.testing.assert_array_equal(
#             self.env.world_map_color[:, pad:-pad, pad],
#             wall_img[:, pad:-pad, pad],
#         )
#         np.testing.assert_array_equal(
#             self.env.world_map_color[:, pad:-pad, -pad - 1],
#             wall_img[:, pad:-pad, -pad - 1],
#         )
#
#     def assert_logical_and_image_view(self, agent_id, expected_logical_view):
#         obs = self.env.get_obs()
#         image_view = obs[agent_id]["image_view"]
#         logical_view = obs[agent_id]["ascii_view"]
#
#         expected_image_view = map_to_image(expected_logical_view)
#
#         np.testing.assert_array_equal(logical_view, expected_logical_view)
#         np.testing.assert_array_equal(image_view, expected_image_view)
#
#     def test_view(self):
#         self._construct_map(BASE_MAP, num_agents=1, start_positions=[(3, 3)], view_size=2)
#         a_id = self.env.agent_ids[0]
#
#         # Test that the view is empty around the agent
#         expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "     ", "     "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Move agent one row closer to the top, check that the top row is walls
#         self.change_agent_position(a_id, (2, 3))
#         expected_logical_view = ascii_to_numpy(["#####", "     ", "  1  ", "     ", "     "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check that the agent sees that padded view when the row above them is out of view
#         self.change_agent_position(a_id, (1, 3))
#         expected_logical_view = ascii_to_numpy(["     ", "#####", "  1  ", "     ", "     "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check that the view is correct if the left wall is in view
#         self.change_agent_position(a_id, (3, 2))
#         expected_logical_view = ascii_to_numpy(["#    ", "#    ", "# 1  ", "#    ", "#    "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if it exceeds the left view
#         self.change_agent_position(a_id, (3, 1))
#         expected_logical_view = ascii_to_numpy([" #   ", " #   ", " #1  ", " #   ", " #   "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if the bottom is in view
#         self.change_agent_position(a_id, (4, 3))
#         expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "     ", "#####"])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if exceeding bottom view
#         self.change_agent_position(a_id, (5, 3))
#         expected_logical_view = ascii_to_numpy(["     ", "     ", "  1  ", "#####", "     "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if the right wall is in view
#         self.change_agent_position(a_id, (3, 4))
#         expected_logical_view = ascii_to_numpy(["    #", "    #", "  1 #", "    #", "    #"])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if exceeding the right wall
#         self.change_agent_position(a_id, (3, 5))
#         expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "   # ", "   # "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#         # Check if the view is correct if in the bottom right corner
#         self.change_agent_position(a_id, (5, 5))
#         expected_logical_view = ascii_to_numpy(["   # ", "   # ", "  1# ", "#### ", "     "])
#         self.assert_logical_and_image_view(a_id, expected_logical_view)
#
#     def test_agent_visibility(self):
#         self._construct_map(BASE_MAP.copy(), 0, None, view_size=2, obs=["image_view", "other_agent_visibility"])
#         self.add_agent_to_env("agent-0", (1, 1))
#         self.add_agent_to_env("agent-1", (1, 3))
#         self.add_agent_to_env("agent-2", (1, 5))
#         obs, *_ = self.env.step({})
#         visibility = [a_obs["other_agent_visibility"] for a_obs in obs.values()]
#         np.testing.assert_array_equal(visibility[0], [1, 0])
#         np.testing.assert_array_equal(visibility[1], [1, 1])
#         np.testing.assert_array_equal(visibility[2], [0, 1])
#
#     def test_agent_actions(self):
#         a_id = "agent-0"
#         pad = 2
#         self._construct_map(
#             BASE_MAP.copy(),
#             1,
#             [(2, 2)],
#             pad,
#         )
#
#         self.env.step({a_id: BASE_ACTION_MAP["NOOP"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, [2, 2])
#         np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")
#         np.testing.assert_array_equal(self.env.ascii_map[pad + 2, pad + 2], "1")
#         np.testing.assert_array_equal(self.env.world_map_color[:, pad + 2, pad + 2], DEFAULT_COLORS["1"])
#
#         self.env.step({a_id: BASE_ACTION_MAP["LEFT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, [2, 1])
#         np.testing.assert_array_equal(self.env.map_with_agents[2, 1], "1")
#         np.testing.assert_array_equal(self.env.ascii_map[pad + 2, pad + 1], "1")
#         np.testing.assert_array_equal(self.env.world_map_color[:, pad + 2, pad + 1], DEFAULT_COLORS["1"])
#
#         self.env.step({a_id: BASE_ACTION_MAP["RIGHT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, [2, 2])
#         np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")
#         np.testing.assert_array_equal(self.env.ascii_map[pad + 2, pad + 2], "1")
#         np.testing.assert_array_equal(self.env.world_map_color[:, pad + 2, pad + 2], DEFAULT_COLORS["1"])
#
#         self.env.step({a_id: BASE_ACTION_MAP["UP"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, [1, 2])
#         np.testing.assert_array_equal(self.env.map_with_agents[1, 2], "1")
#         np.testing.assert_array_equal(self.env.ascii_map[pad + 1, pad + 2], "1")
#         np.testing.assert_array_equal(self.env.world_map_color[:, pad + 1, pad + 2], DEFAULT_COLORS["1"])
#
#         self.env.step({a_id: BASE_ACTION_MAP["DOWN"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, [2, 2])
#         np.testing.assert_array_equal(self.env.map_with_agents[2, 2], "1")
#         np.testing.assert_array_equal(self.env.ascii_map[pad + 2, pad + 2], "1")
#         np.testing.assert_array_equal(self.env.world_map_color[:, pad + 2, pad + 2], DEFAULT_COLORS["1"])
#
#         # if an agent tries to move through a wall they should stay in the same place
#         # we check that this works correctly for both corner and non-corner edges
#         self.change_agent_position(a_id, (1, 1))
#         self.env.step({a_id: BASE_ACTION_MAP["UP"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (1, 1))
#
#         self.env.step({a_id: BASE_ACTION_MAP["LEFT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (1, 1))
#
#         self.change_agent_position(a_id, (4, 4))
#         self.env.step({a_id: BASE_ACTION_MAP["RIGHT"]})
#         self.env.step({a_id: BASE_ACTION_MAP["DOWN"]})
#         self.env.step({a_id: BASE_ACTION_MAP["RIGHT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (5, 5))
#
#         self.env.step({a_id: BASE_ACTION_MAP["DOWN"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (5, 5))
#
#         self.env.step({a_id: BASE_ACTION_MAP["LEFT"]})
#         self.env.step({a_id: BASE_ACTION_MAP["DOWN"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (5, 4))
#
#         self.change_agent_position(a_id, (4, 5))
#         self.env.step({a_id: BASE_ACTION_MAP["RIGHT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (4, 5))
#
#         self.change_agent_position(a_id, (2, 1))
#         self.env.step({a_id: BASE_ACTION_MAP["LEFT"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (2, 1))
#
#         self.change_agent_position(a_id, (1, 2))
#         self.env.step({a_id: BASE_ACTION_MAP["UP"]})
#         np.testing.assert_array_equal(self.env.agents[a_id].pos, (1, 2))
#
#     def test_agent_conflict(self):
#         self._construct_map(BASE_MAP_2.copy(), 2, [(1, 2), (4, 4)])
#
#         # test that agents can't walk into other agents
#         self.change_agent_position("agent-0", (3, 3))
#         self.change_agent_position("agent-1", (3, 4))
#         self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"]})
#         self.env.step({"agent-1": BASE_ACTION_MAP["LEFT"]})
#         np.testing.assert_array_equal(self.env.agents["agent-0"].pos, (3, 3))
#         np.testing.assert_array_equal(self.env.agents["agent-1"].pos, (3, 4))
#
#         # test that agents can't walk through each other if they move simultaneously
#         self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["LEFT"]})
#         np.testing.assert_array_equal(self.env.agents["agent-0"].pos, (3, 3))
#         np.testing.assert_array_equal(self.env.agents["agent-1"].pos, (3, 4))
#         # also check that the map looks correct, no agent has disappeared
#         expected_map = ascii_to_numpy(["######", "#    #", "#    #", "#  12#", "#    #", "######"])
#
#         np.testing.assert_array_equal(expected_map, self.env.map_with_agents)
#
#         # test that agents can walk into other agents if moves are de-conflicting
#         # conflict only occurs stochastically so try it 50 times
#         np.random.seed(1)
#         self.change_agent_position("agent-0", (2, 4))
#         self.change_agent_position("agent-1", (3, 4))
#         expected_map = ascii_to_numpy(["######", "#    #", "#   1#", "#   2#", "#    #", "######"])
#         self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["UP"]})
#         np.testing.assert_array_equal(expected_map, self.env.map_with_agents)
#         self.env.step({"agent-0": BASE_ACTION_MAP["LEFT"], "agent-1": BASE_ACTION_MAP["DOWN"]})
#         self.env.step({"agent-0": BASE_ACTION_MAP["RIGHT"], "agent-1": BASE_ACTION_MAP["UP"]})
#         np.testing.assert_array_equal(expected_map, self.env.map_with_agents)
#
#
# if __name__ == "__main__":
#     unittest.main()
