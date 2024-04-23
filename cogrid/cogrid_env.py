import collections
from itertools import combinations
from typing import Any
import copy

import numpy as np
import pygame
import pygame.freetype
from gymnasium.spaces import Discrete, Dict
import pettingzoo
from cogrid.constants import GridConstants, FIXED_GRIDS
from cogrid.core import actions as grid_actions
from cogrid.core.constants import CoreConstants
from cogrid.core.directions import Directions
from cogrid.core.grid import Grid
from cogrid.core.grid_object import GridObj, GridAgent
from cogrid.core.grid_utils import ascii_to_numpy
from cogrid.feature_space.feature_space import FeatureSpace


RNG = RandomNumberGenerator = np.random.Generator


# pettingzoo.ParallelEnv
class CoGridEnv(pettingzoo.ParallelEnv):
    """
    The CoGridEnv class is a base environment for any other CoGridEnv environment that you may want to create.
    Any subclass should be sure to define rewards
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 35,
        "screen_size": 480,
        "render_message": "",
    }

    def __init__(
        self,
        config: dict,
        render_mode: str | None = None,
        num_roles: int | None = None,
        highlight: bool = False,
        agent_pov: str | None = None,
        **kwargs,
    ):
        super(CoGridEnv, self).__init__()
        self._np_random: np.random.Generator | None = None  # set in reset()

        self.clock = None
        self.render_size = None
        self.config = config
        self.render_mode = render_mode
        self.render_message = (
            kwargs.get("render_message") or self.metadata["render_message"]
        )
        self.highlight = highlight
        self.agent_pov = agent_pov
        self.tile_size = CoreConstants.TilePixels
        self.screen_size = kwargs.get("screen_size") or self.metadata["screen_size"]
        self.window = None
        self.name = config["name"]
        self.cumulative_score = 0

        self.max_steps = config["max_steps"]
        self.visualizer = None
        self.common_reward = self.config.get("common_reward", False)
        self.roles = self.config.get("roles", True)
        self.num_roles = num_roles  # some envs have agent roles that we need to specify at init for obs space

        self.t = 0

        # grid data is set by _gen_grid()
        self.grid: Grid | None = None
        self.spawn_points: list = []
        self.load = self.config["grid_gen_kwargs"].get(
            "load", None
        )  # load a specific env configuration
        self._gen_grid()
        self.shape = (self.grid.height, self.grid.width)

        self.agent_view_size = self.config.get("agent_view_size", 7)

        self.agents = {
            f"agent-{i}": None for i in range(config["num_agents"])
        }  # will contain: {'agent_id': agent}
        self._agent_ids = set(self.agents.keys())

        # Action space describes the set of actions available to agents.
        action_str = config.get("action_set")
        if action_str == "rotation_actions":
            self.action_set = grid_actions.ActionSets.RotationActions
        elif action_str == "cardinal_actions":
            self.action_set = grid_actions.ActionSets.CardinalActions
        else:
            raise ValueError(f"Invalid or None action set string: {action_str}.")

        # Set the action space for the gym environment
        self.action_spaces = {
            a_id: Discrete(len(self.action_set)) for a_id in self.agent_ids
        }

        # If False, the observations (ascii, images, etc) will be obscured so that agents cannot see through walls
        self.see_through_walls = self.config.get("see_through_walls", True)

        # Establish the observation space. This provides the general form of what an agent's observation
        # looks like so that they can be properly initialized.
        self.feature_spaces = {
            a_id: FeatureSpace(feature_names=config["obs"], env=self, agent_id=a_id)
            for a_id in self.agent_ids
        }
        self.observation_spaces = {
            a_id: self.feature_spaces[a_id].observation_space for a_id in self.agent_ids
        }

        self.prev_actions = None
        self.trajectories = collections.defaultdict(list)

    def _gen_grid(self):
        self.spawn_points = []
        grid, states = self._generate_encoded_grid_states()

        # If the grid is a list of strings this will turn it into the correct np format
        grid = ascii_to_numpy(grid)
        spawn_points = np.where(grid == GridConstants.Spawn)
        grid[spawn_points] = GridConstants.FreeSpace
        self.spawn_points = list(zip(*spawn_points))
        grid_encoding = np.stack([grid, states], axis=0)
        self.grid, _ = Grid.decode(grid_encoding)

    def _generate_encoded_grid_states(self) -> tuple[np.ndarray, np.ndarray]:
        if self.load is not None:  # load a specific grid instead of generating one
            return FIXED_GRIDS[self.load]

        shape = self.config["grid_gen_kwargs"]["shape"]
        grid = np.full(shape, fill_value=GridConstants.FreeSpace)
        states = np.zeros(shape=grid.shape)

        # Fill outside border with walls
        grid[0, :] = GridConstants.Wall
        grid[-1, :] = GridConstants.Wall
        grid[:, 0] = GridConstants.Wall
        grid[:, -1] = GridConstants.Wall

        return grid, states

    @staticmethod
    def _set_np_random(seed: int | None = None):
        if seed is not None and not (isinstance(seed, int) and 0 <= seed):
            if isinstance(seed, int) is False:
                raise ValueError(
                    f"Seed must be a python integer, actual type: {type(seed)}"
                )
            else:
                raise ValueError(
                    f"Seed must be greater or equal to zero, actual value: {seed}"
                )

        seed_seq = np.random.SeedSequence(seed)
        np_seed = seed_seq.entropy
        rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
        return rng, np_seed

    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, _ = self._set_np_random()

        return self._np_random

    def reset(
        self, *, seed: int | None = 42, options: dict[str, Any] | None = None
    ) -> tuple:
        """
        Reset the map and return the initial observations. Must be implemented for each environment.
        """
        if seed is not None:
            self._np_random, _ = self._set_np_random(seed=seed)

        self._gen_grid()

        # Clear out past agents and re-initialize
        self.agents = {}
        self.setup_agents()

        # Initialize previous actions as no-op
        self.prev_actions = {a_id: grid_actions.Actions.Noop for a_id in self.agent_ids}

        self.trajectories = collections.defaultdict(list)

        self.t = 0

        self.on_reset()

        self.update_grid_agents()

        self.cumulative_score = 0

        self.render()

        obs = self.get_obs()

        return obs, {}

    def _action_idx_to_str(self, actions: dict[str, int | str]) -> dict[str, str]:
        """If not already, convert the action from index to string representation"""
        str_actions = {
            a_id: self.action_set[action] if not isinstance(action, str) else action
            for a_id, action in actions.items()
        }

        return str_actions

    def step(self, actions: dict) -> tuple:
        """
        :param actions: (dict) Dictionary of actions keyed by agent IDs. Actions are integers in self.action_space
        :return observations: (dict) New observations for each agent, keyed by agent IDs.
        :return rewards: (dict) The reward returned to each agent, keyed by agent IDs.
        :return dones: (dict) Indicator of whether or not an agent is done, keyed by agent IDs.
        :return info: (dict) supplementary information for each agent, defined on a per-environment basis.
        """
        self.t += 1

        # Track the previous state so that we have the delta for computing rewards
        self.prev_grid = copy.deepcopy(self.grid)

        # Update attributes of the grid objects that are timestep dependent
        self.grid.tick()

        # Convert the integer actions to strings (helpful for debugging!)
        actions = self._action_idx_to_str(actions)

        # Agents who are moving have their positions updated (and conflicts resolved)
        self.move_agents(actions)

        # Given any new position(s), agents interact with the environment
        self.interact(actions)

        # Updates the GridAgent objects to reflect new positions/interactions
        self.update_grid_agents()

        # Store the actions taken by the agents
        self.prev_actions = actions.copy()

        # setup return values
        observations, rewards, info, (terminateds, truncateds) = (
            self.get_obs(),
            self.compute_rewards(),
            self.get_info(),
            self.get_terminateds_truncateds(),
        )

        self.render()

        self.cumulative_score += sum([*rewards.values()])

        # Custom hook if a subclass wants to make any updates
        self.on_step()

        return observations, rewards, terminateds, truncateds, info

    def update_grid_agents(self):
        # TODO(chase): this is inefficient; just update the existing objects?
        self.grid.grid_agents = {
            a_id: GridAgent(agent) for a_id, agent in self.agents.items()
        }

    def setup_agents(self):
        self._setup_agents()

    def _setup_agents(self):
        raise NotImplementedError

    def move_agents(self, actions: dict[str, str]) -> None:
        """This function executes all agent movements"""

        # Take any terminated agents or those who we do not have an action for and keep them in the same position
        new_positions = {
            a_id: agent.pos
            for a_id, agent in self.agents.items()
            if agent.terminated or a_id not in actions
        }
        agents_to_move = [
            a_id for a_id in actions.keys() if not self.agents[a_id].terminated
        ]

        # If we're using cardinal actions, change the agent direction if they aren't already facing the desired dir
        # if they are, then they move in that direction.
        if self.action_set == grid_actions.ActionSets.CardinalActions:
            move_action_to_dir = {
                grid_actions.Actions.MoveRight: Directions.Right,
                grid_actions.Actions.MoveLeft: Directions.Left,
                grid_actions.Actions.MoveUp: Directions.Up,
                grid_actions.Actions.MoveDown: Directions.Down,
            }
            for a_id, action in actions.items():
                if action in move_action_to_dir.keys():
                    agent = self.agents[a_id]
                    desired_direction = move_action_to_dir[action]

                    # rotate to the desired direction and move that way
                    agent.dir = desired_direction
                    actions[a_id] = grid_actions.Actions.Forward

        # Determine the position each agent is attempting to move to
        attempted_positions = {}
        for a_id, action in actions.items():
            attempted_positions[a_id] = self.determine_attempted_pos(a_id, action)

        # First, give priority to agents staying in the same position
        for a_id, attemped_pos in attempted_positions.items():
            agent = self.agents[a_id]
            if np.array_equal(attemped_pos, agent.pos):
                new_positions[a_id] = agent.pos
                if a_id in agents_to_move:
                    agents_to_move.remove(a_id)

        # Randomize agent priority and move agents to new positions
        self.np_random.shuffle(agents_to_move)
        for a_id in agents_to_move:
            agent = self.agents[a_id]
            attempted_pos = attempted_positions[a_id]
            # If an agent is already moving to the desired position, keep them at the current position
            if tuple(attempted_pos) in [tuple(npos) for npos in new_positions.values()]:
                new_positions[a_id] = agent.pos
            else:
                new_positions[a_id] = attempted_pos

        # Make sure no two agents moved through each other
        for a_id1, a_id2 in combinations(self.agent_ids, r=2):
            agent1, agent2 = self.agents[a_id1], self.agents[a_id2]
            if np.array_equal(new_positions[a_id1], agent2.pos) and np.array_equal(
                new_positions[a_id2], agent1.pos
            ):
                new_positions[a_id1] = agent1.pos
                new_positions[a_id2] = agent2.pos

        # assign the new positions and store the agent position
        for a_id, agent in self.agents.items():
            agent.pos = new_positions[a_id]
            self.trajectories[a_id].append(agent.pos)

        assert len(self.agent_pos) == len(
            set(self.agent_pos)
        ), "Agents do not have unique positions!"

    def determine_attempted_pos(self, agent_id, action) -> tuple[int, int]:
        agent = self.agents[agent_id]
        fwd_pos = agent.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if action == grid_actions.Actions.Forward:
            if fwd_cell is None or fwd_cell.can_overlap(agent=agent):
                return fwd_pos

        return agent.pos

    def interact(self, actions) -> None:
        for a_id, action in actions.items():
            agent: GridAgent = self.agents[a_id]
            agent.cell_toggled = None
            agent.cell_placed_on = None
            agent.cell_picked_up_from = None
            agent.cell_overlapped = self.grid.get(*agent.pos)

            if action == grid_actions.Actions.RotateRight:
                agent.rotate_right()
            elif action == grid_actions.Actions.RotateLeft:
                agent.rotate_left()

            # Attempt to pick up the object in front of the agent
            elif action == grid_actions.Actions.PickupDrop:
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                agent_ahead = tuple(fwd_pos) in self.agent_pos

                # If there's an agent in front of you, you can't
                # pick up or drop.
                if agent_ahead:
                    continue

                # TODO(chase): we need to fix this logic so that we check if an agent
                # can pick up the type of object that is returned from pick_up_from
                if (
                    fwd_cell
                    and fwd_cell.can_pickup(agent=agent)
                    and agent.can_pickup(grid_object=fwd_cell)
                ):
                    pos = fwd_cell.pos
                    agent.inventory.append(fwd_cell)
                    fwd_cell.pos = None
                    self.grid.set(*pos, None)
                elif (
                    fwd_cell
                    and fwd_cell.can_pickup_from(agent=agent)
                    and agent.can_pickup(grid_object=fwd_cell)
                ):
                    pickup_cell = fwd_cell.pick_up_from(agent=agent)
                    pickup_cell.pos = None
                    agent.inventory.append(pickup_cell)
                    agent.cell_picked_up_from = fwd_cell
                elif not agent_ahead and not fwd_cell and agent.inventory:
                    drop_cell = agent.inventory.pop(0)
                    drop_cell.pos = fwd_pos
                    self.grid.set(fwd_pos[0], fwd_pos[1], drop_cell)
                elif (
                    fwd_cell
                    and agent.inventory
                    and fwd_cell.can_place_on(cell=agent.inventory[0], agent=agent)
                ):
                    drop_cell = agent.inventory.pop(0)
                    drop_cell.pos = fwd_pos
                    fwd_cell.place_on(cell=drop_cell, agent=agent)
                    agent.cell_placed_on = fwd_cell

            # Attempt to toggle the object in front of the agent
            elif action == grid_actions.Actions.Toggle:
                fwd_cell = self.grid.get(*agent.front_pos)
                if fwd_cell:
                    toggle_success = fwd_cell.toggle(env=self, agent=agent)
                    if toggle_success:
                        agent.cell_toggled = fwd_cell

        self.on_interact(actions)

    def on_interact(self, actions) -> None:
        pass

    def on_reset(self) -> None:
        pass

    def on_step(self) -> None:
        pass

    def get_obs(self) -> dict:
        """
        Fetch new observations for the agents
        """
        obs = {}
        for a_id in self.agent_ids:
            # obs[a_id] = {feature.name: feature.generate(self, a_id) for feature in self.feature_generators[a_id]}
            obs[a_id] = self.feature_spaces[a_id].generate_features()
        return obs

    def compute_rewards(self) -> dict:
        """
        Calculate each agent's reward at the current step
        """
        per_agent_reward = {
            agent_id: agent.compute_and_reset_step_reward()
            for agent_id, agent in self.agents.items()
        }
        if self.common_reward:
            collective_reward = sum([*per_agent_reward.values()])
            per_agent_reward = {
                a_id: collective_reward for a_id in per_agent_reward.keys()
            }

            # TODO(chase): if we have a step penalty, it'll be step_penalty * num_agents for common_reward. Decide
            #   how to deal with it.

        return per_agent_reward

    def get_terminateds_truncateds(self) -> tuple:
        """
        Determine the done status for each agent.
        """
        terminateds = {
            agent_id: agent.terminated for agent_id, agent in self.agents.items()
        }
        terminateds["__all__"] = all([*terminateds.values()])

        if self.t >= self.max_steps:
            truncateds = {agent_id: True for agent_id in self.agent_ids}
        else:
            truncateds = {agent_id: False for agent_id in self.agent_ids}

        truncateds["__all__"] = all([*truncateds.values()])

        return terminateds, truncateds

    def get_info(self) -> dict:
        """fetch env info"""
        return {}

    @property
    def map_with_agents(self):
        """
        retrieve a version of the environment where 'P' chars have agent IDs.
        :return: 1d array of strings representing the map
        """

        grid_encoding = self.grid.encode(encode_char=True)
        grid = grid_encoding[:, :, 0]

        for a_id, agent in self.agents.items():
            if agent is not None:  # will be None before being set by subclassed env
                grid[agent.pos[0], agent.pos[1]] = self.id_to_numeric(a_id)

        return grid

    def select_spawn_point(self) -> tuple:
        if self.spawn_points:
            return self.spawn_points.pop(0)

        available_spawns = self.available_positions
        return self.np_random.choice(available_spawns)

    @property
    def available_positions(self):
        spawns = []
        for r in range(self.grid.height):
            for c in range(self.grid.width):
                cell = self.grid.get(row=r, col=c)
                if (r, c) not in self.agent_pos and (
                    not cell or cell.object_id == "floor"
                ):
                    spawns.append((r, c))
        return spawns

    def put_obj(self, obj: GridObj, row: int, col: int):
        """
        Place an object at a specific point in the grid.
        """
        self.grid.set(row=row, col=col, v=obj)
        obj.pos = (row, col)
        obj.init_pos = (row, col)

    def get_view_exts(self, agent_id, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """
        agent = self.agents[agent_id]
        agent_view_size = agent_view_size or self.agent_view_size

        if agent.dir == Directions.Right:
            topY = agent.pos[0] - agent_view_size // 2
            topX = agent.pos[1]
        elif agent.dir == Directions.Down:
            topY = agent.pos[0]
            topX = agent.pos[1] - agent_view_size // 2
        elif agent.dir == Directions.Left:
            topY = agent.pos[0] - agent_view_size // 2
            topX = agent.pos[1] - agent_view_size + 1
        elif agent.dir == Directions.Up:
            topY = agent.pos[0] - agent_view_size + 1
            topX = agent.pos[1] - agent_view_size // 2
        else:
            raise ValueError("Invalid agent direction.")

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def gen_obs_grid(self, agent_id, agent_view_size=None):
        """Generate the sub-grid observed by a specific agent"""
        topX, topY, botX, botY = self.get_view_exts(agent_id, agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        assert agent_id in grid.grid_agents

        agent = self.agents[agent_id]
        for i in range(agent.dir + 1):
            grid = grid.rotate_left()

        if not self.see_through_walls:
            # Mask view from the agents position at the bottom-center of the grid view
            vis_mask = grid.process_vis(agent_pos=(-1, grid.width // 2))
        else:
            vis_mask = np.ones(shape=(grid.height, grid.width), dtype=bool)

        assert len(grid.grid_agents) >= 1
        assert grid.grid_agents[agent_id].dir == Directions.Up

        # NOTE: In Minigrid, they replace the agent's position with the item they're carrying. We don't do that
        #   here. Rather, we'll provide an additional observation space that represents the item(s) in inventory.

        return grid, vis_mask

    def get_pov_render(self, agent_id, tile_size=CoreConstants.TilePixels):
        """Render a specific agent's POV"""
        grid, vis_mask = self.gen_obs_grid(agent_id)
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size - 1, self.agent_view_size // 2),
            agent_dir=Directions.Up,
            highlight_mask=vis_mask,
        )
        return img

    def get_full_render(self, highlight: bool, tile_size=CoreConstants.TilePixels):
        """Return a render of the full environment"""

        if highlight:
            highlight_mask = np.zeros(
                shape=(self.grid.height, self.grid.width), dtype=bool
            )
            for a_id, agent in self.agents.items():
                # Determine cell visibility for the agent
                _, vis_mask = self.gen_obs_grid(a_id)

                # compute the world coordinates of the bottom-left corner of the agent's view area
                f_vec = agent.dir_vec
                r_vec = agent.right_vec
                top_left = (
                    agent.pos
                    + f_vec * (self.agent_view_size - 1)
                    - r_vec * (self.agent_view_size // 2)
                )

                # identify the cells to highlight as visible in the render
                for vis_row in range(self.agent_view_size):
                    for vis_col in range(self.agent_view_size):
                        if not vis_mask[vis_row, vis_col]:
                            continue

                        # compute world coordinates of agent view
                        abs_row, abs_col = (
                            top_left - (f_vec * vis_row) + (r_vec * vis_col)
                        )

                        if abs_row < 0 or abs_row >= self.grid.height:
                            continue
                        if abs_col < 0 or abs_col >= self.grid.width:
                            continue

                        highlight_mask[abs_row, abs_col] = True
        else:
            highlight_mask = None

        img = self.grid.render(tile_size=tile_size, highlight_mask=highlight_mask)
        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = CoreConstants.TilePixels,
        agent_pov: str | None = None,
    ):
        """Return RGB image corresponding to the whole environment or an agent's POV"""
        if agent_pov:
            frame = self.get_pov_render(agent_id=self.agent_pov, tile_size=tile_size)
        else:
            frame = self.get_full_render(highlight=highlight, tile_size=tile_size)

        return frame

    def render(self) -> None | np.ndarray:
        if self.visualizer is not None:
            orientations = {
                self.id_to_numeric(a_id): agent.orientation
                for a_id, agent in self.agents.items()
            }
            inventories = {
                self.id_to_numeric(a_id): agent.inventory
                for a_id, agent in self.agents.items()
                if len(agent.inventory) > 0
            }
            self.visualizer.render(
                self.map_with_agents,
                self.t,
                orientations=orientations,
                subitems=inventories,
            )

        if self.render_mode is None:
            return

        img = self.get_frame(self.highlight, self.tile_size, self.agent_pov)
        if self.render_mode == "human":
            # TODO(chase): move all pygame logic to run_interactive.py so it's not needed here.
            # if img.shape[0] == 3:  # move the channels last
            #     img = np.moveaxis(img, 0, -1)
            if self.render_size is None:
                self.render_size = img.shape[:2]
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.screen_size, self.screen_size)
                )
                pygame.display.set_caption(self.name)
            if self.clock is None:
                self.clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(img)

            # For some reason, pygame is rotating/flipping the image...
            surf = pygame.transform.flip(surf, False, True)
            surf = pygame.transform.rotate(surf, 270)

            # Create background with mission description
            offset = surf.get_size()[0] * 0.1
            bg = pygame.Surface(
                (int(surf.get_size()[0] + offset), int(surf.get_size()[1] + offset))
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(bg, (self.screen_size, self.screen_size))

            font_size = 22
            text = f"Score: {np.round(self.cumulative_score, 2)}" + self.render_message

            font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)
            text_rect = font.get_rect(text, size=font_size)
            text_rect.center = bg.get_rect().center
            text_rect.y = bg.get_height() - font_size * 1.5
            font.render_to(bg, text_rect, text, size=font_size)

            self.window.blit(bg, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()

        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        if self.window:
            pygame.quit()

    def get_action_mask(self, agent_id):
        raise NotImplementedError

    @property
    def agent_ids(self) -> list:
        return list(self.agents.keys())

    @property
    def agent_pos(self) -> list:
        return [tuple(agent.pos) for agent in self.agents.values() if agent is not None]

    def id_to_numeric(self, agent_id) -> str:
        """Converts agent id to integer, beginning with 1,
        e.g., agent-0 -> 1, agent-1 -> 2, etc.
        """
        agent = self.agents[agent_id]
        return str(agent.agent_number)

    # @staticmethod
    # def create_gif_from_frames(frames):
    #     frames = [PIL.Image.fromarray(np.moveaxis(f, 0, -1).astype(np.uint8)) for f in frames]
    #
    #     save_path = f'render.gif'
    #     frames[0].save(save_path, save_all=True, optimize=True, quality=90, append_images=frames[1:], loop=0, duration=80)
    #
    #     print(f"Video saved to {save_path}.")
