import collections
from itertools import combinations
import copy

import numpy as np


try:
    import pygame
except ImportError:
    pygame = None


from gymnasium import spaces
import pettingzoo
from cogrid import constants
from cogrid.core import actions as grid_actions
from cogrid.core.constants import CoreConstants
from cogrid.core import directions
from cogrid.core import grid
from cogrid.core import grid_object
from cogrid.core import grid_utils
from cogrid.feature_space import feature_space
from cogrid.core import reward
from cogrid.core import typing
from cogrid.core import agent
from cogrid.core import directions
from cogrid.core import reward

from cogrid.core import layouts

# Vectorized components (Plans 01-06)
from cogrid.backend import set_backend
from cogrid.core.grid_object import build_lookup_tables
from cogrid.core.grid_utils import layout_to_array_state
from cogrid.core.agent import create_agent_arrays, sync_arrays_to_agents, get_dir_vec_table
from cogrid.core.movement import move_agents


RNG = RandomNumberGenerator = np.random.Generator


class CoGridEnv(pettingzoo.ParallelEnv):
    """CoGridEnv is the base environment class for a multi-agent grid-world environment.

    This class inherits from the ``pettingzoo.ParallelEnv`` class and implements the necessary methods
    for a parallel environment.

    :param config: Configuration dictionary for the environment.
    :type config: dict
    :param render_mode: Rendering method for local visualization, defaults to None
    :type render_mode: str | None, optional
    :param agent_class: Agent class for the environment if using a custom Agent, defaults to None
    :type agent_class: agent.Agent | None, optional
    :raises ValueError: ValueError if an invalid or None action set string is provided.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 35,
        "screen_size": 480,
        "render_message": "",
        "agent_pov": None,
        "highlight": False,
        "see_through_walls": True,
    }

    def __init__(
        self,
        config: dict,
        render_mode: str | None = None,
        agent_class: agent.Agent | None = None,
        backend: str = "numpy",
        **kwargs,
    ):
        """_summary_

        :param config: _description_
        :type config: dict
        :param render_mode: _description_, defaults to None
        :type render_mode: str | None, optional
        :param agent_class: _description_, defaults to None
        :type agent_class: agent.Agent | None, optional
        :param backend: Array backend name ('numpy' or 'jax'). The first
            environment created sets the global backend; subsequent envs
            must use the same backend or a RuntimeError is raised.
        :type backend: str
        :raises ValueError: _description_
        """
        super(CoGridEnv, self).__init__()
        self._np_random: np.random.Generator | None = None  # set in reset()

        # Backend dispatch: first env sets global backend, subsequent verify match
        set_backend(backend)
        self._backend = backend

        # TODO(chase): Move PyGame/rendering logic outside of this class.
        self.clock = None
        self.render_size = None
        self.config = config
        self.render_mode = render_mode
        self.render_message = (
            kwargs.get("render_message") or self.metadata["render_message"]
        )
        self.tile_size = CoreConstants.TilePixels
        self.screen_size = (
            kwargs.get("screen_size") or self.metadata["screen_size"]
        )
        self.window = None
        self.name = config["name"]
        self.cumulative_score = 0

        self.max_steps = config["max_steps"]
        self.visualizer = None
        self.roles = self.config.get("roles", True)
        self.agent_class = agent_class or agent.Agent
        self.t = 0

        # grid data is set by _gen_grid()
        self.scope: str = config.get("scope", "global")
        self.grid: grid.Grid | None = None
        self.spawn_points: list = []
        self.current_layout_id: str | None = None
        self._gen_grid()
        self.shape = (self.grid.height, self.grid.width)

        self.agent_view_size = self.config.get("agent_view_size", 7)

        self.possible_agents = [i for i in range(config["num_agents"])]
        self.agents = copy.copy(self.possible_agents)
        self._agent_ids: set[typing.AgentID] = set(self.agents)
        self.env_agents: dict[typing.AgentID, agent.Agent] = {
            i: None for i in self.agents
        }  # will contain: {'agent_id': agent}

        # Establish reward function through reward modules
        reward_names = config.get("rewards", [])
        self.rewards = []
        for reward_name in reward_names:
            self.rewards.append(
                reward.make_reward(reward_name, agent_ids=self._agent_ids)
            )

        # The reward at each timestep is the sum of the rewards from each reward module
        # but can also be added to via environment hooks.
        self.per_agent_reward: dict[typing.AgentID, float] = (
            self.get_empty_reward_dict()
        )
        self.per_component_reward: dict[str, dict[typing.AgentID, float]] = {}
        self.reward_this_step = self.get_empty_reward_dict()

        # Action space describes the set of actions available to agents.
        action_str = config.get("action_set")
        if action_str == "rotation_actions":
            self.action_set = grid_actions.ActionSets.RotationActions
        elif action_str == "cardinal_actions":
            self.action_set = grid_actions.ActionSets.CardinalActions
        else:
            raise ValueError(
                f"Invalid or None action set string: {action_str}."
            )

        # Set the action space for the gym environment
        self.action_spaces = {
            a_id: spaces.Discrete(len(self.action_set))
            for a_id in self.agent_ids
        }

        # Establish the observation space. This provides the general form of what an agent's observation
        # looks like so that they can be properly initialized.
        features = config.get("features", [])
        if isinstance(features, list):
            features = {agent_id: features for agent_id in self.agent_ids}
        elif isinstance(features, str):
            features = {agent_id: [features] for agent_id in self.agent_ids}
        else:
            assert isinstance(features, dict) and set(features.keys()) == set(
                self.agent_ids
            ), (
                "Must pass a feature dictionary keyed by agent IDs, "
                "a list of features (universal for all agents), or the name "
                "of a single feature."
            )

        self.feature_spaces = {
            a_id: feature_space.FeatureSpace(
                feature_names=features[a_id], env=self, agent_id=a_id
            )
            for a_id in self.agent_ids
        }

        self.observation_spaces = {
            a_id: self.feature_spaces[a_id].observation_space
            for a_id in self.agent_ids
        }

        self.prev_actions = None

        # -------------------------------------------------------------------
        # Vectorized infrastructure (Plans 01-06 integration)
        # -------------------------------------------------------------------
        # Build lookup tables for the current scope
        self._lookup_tables = build_lookup_tables(scope=self.scope)

        # Build scope config for environment-specific array logic
        from cogrid.core.scope_config import get_scope_config
        self._scope_config = get_scope_config(self.scope)
        self._type_ids = self._scope_config["type_ids"]
        self._interaction_tables = self._scope_config["interaction_tables"]

        # Array state is built in reset() after agents are placed
        self._array_state = None

        # Enable shadow parity validation (set to False for performance)
        self._validate_array_parity = False

        # -------------------------------------------------------------------
        # JAX backend infrastructure (Plan 03-02)
        # -------------------------------------------------------------------
        if self._backend == 'jax':
            import jax.numpy as jnp
            from cogrid.feature_space.array_features import build_feature_fn_jax
            from cogrid.core.jax_step import make_jitted_step, make_jitted_reset
            from cogrid.backend.env_state import register_envstate_pytree

            # Register EnvState as JAX pytree (idempotent)
            register_envstate_pytree()

            # Convert lookup tables to JAX arrays for JIT compatibility
            for key in self._lookup_tables:
                self._lookup_tables[key] = jnp.array(
                    self._lookup_tables[key], dtype=jnp.int32
                )

            # Convert static_tables in scope_config to JAX arrays
            if "static_tables" in self._scope_config:
                import numpy as _np
                st = self._scope_config["static_tables"]
                for key in st:
                    if isinstance(st[key], _np.ndarray):
                        st[key] = jnp.array(st[key], dtype=jnp.int32)

            # Convert interaction_tables arrays to JAX
            if self._scope_config.get("interaction_tables") is not None:
                import numpy as _np
                it = self._scope_config["interaction_tables"]
                for key in it:
                    if isinstance(it[key], _np.ndarray):
                        it[key] = jnp.array(it[key], dtype=jnp.int32)

            # Build JAX feature function using array-based feature names
            # The JAX path uses these low-level array features regardless of
            # what higher-level feature space the config specifies.
            jax_feature_names = [
                "agent_position", "agent_dir", "full_map_encoding",
                "can_move_direction", "inventory",
            ]
            self._jax_feature_fn = build_feature_fn_jax(
                jax_feature_names, scope=self.scope,
            )

            # Compute action indices for PickupDrop and Toggle
            self._action_pickup_drop_idx = self.action_set.index(
                grid_actions.Actions.PickupDrop
            )
            self._action_toggle_idx = self.action_set.index(
                grid_actions.Actions.Toggle
            )

            # Build reward config for JAX path
            # Map config reward names (e.g. "delivery_reward") to JAX fn names
            # (e.g. "delivery") by stripping the "_reward" suffix if present.
            jax_reward_specs = []
            for name in self.config.get("rewards", []):
                fn_name = name.replace("_reward", "") if name.endswith("_reward") else name
                jax_reward_specs.append({
                    "fn": fn_name,
                    "coefficient": 1.0,
                    "common_reward": True,
                })
            from cogrid.envs.overcooked.array_rewards import compute_rewards

            self._jax_reward_config = {
                "type_ids": self._type_ids,
                "n_agents": self.config["num_agents"],
                "rewards": jax_reward_specs,
                "action_pickup_drop_idx": self._action_pickup_drop_idx,
                "compute_fn": compute_rewards,
            }

            # These are populated lazily in reset() once layout arrays are available
            self._jax_layout_arrays = None
            self._jax_spawn_positions = None
            self._env_state = None
            self._jitted_step = None
            self._jitted_reset = None

            # Sorted agent ID order for dict<->array conversion
            self._agent_id_order = sorted(self.possible_agents)

    def _gen_grid(self) -> None:
        """Generates the grid for the environment.

        This method generates the grid for the environment by calling the ``_generate_encoded_grid_states`` method,
        converting the grid to the correct numpy format, finding the spawn points, and encoding the grid and states.
        The resulting grid is then decoded and stored in the ``self.grid`` attribute.
        """
        self.spawn_points = []
        encoded_grid, states = self._generate_encoded_grid_states()

        # If the grid is a list of strings this will turn it into the correct np format
        np_grid = grid_utils.ascii_to_numpy(encoded_grid)
        spawn_points = np.where(np_grid == constants.GridConstants.Spawn)
        np_grid[spawn_points] = constants.GridConstants.FreeSpace
        self.spawn_points = list(zip(*spawn_points))
        grid_encoding = np.stack([np_grid, states], axis=0)
        self.grid, _ = grid.Grid.decode(grid_encoding, scope=self.scope)

    def _generate_encoded_grid_states(self) -> tuple[np.ndarray, np.ndarray]:
        """Generates a grid encoding from the configuration.

        :return: A tuple containing the encoded grid and state arrays.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        grid_config: dict = self.config.get("grid", {})
        layout_name = grid_config.get("layout", None)
        layout_fn = grid_config.get("layout_fn", None)

        if layout_name is None and layout_fn is None:
            raise ValueError(
                "Must provide either a `layout` name or layout-generating function in config['grid']"
            )
        elif layout_name is not None:
            layout, state_encoding = layouts.get_layout(layout_name)
            self.current_layout_id = layout_name
        elif layout_fn is not None:
            layout_name, layout, state_encoding = layout_fn(**grid_config)
            self.current_layout_id = layout_name

        return layout, state_encoding

    @staticmethod
    def _set_np_random(
        seed: int | None = None,
    ) -> tuple[RandomNumberGenerator, int]:
        """Set the numpy random number generator. This is copied from
        the

        :param seed: _description_, defaults to None
        :type seed: int | None, optional
        :raises ValueError: Invalid seed value.
        :return: Random number generator and seed.
        :rtype: tuple[RandomNumberGenerator, int]
        """
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
        """Get the numpy random number generator.

        :return: The numpy random number generator.
        :rtype: np.random.Generator
        """
        if self._np_random is None:
            self._np_random, _ = self._set_np_random()

        return self._np_random

    def observation_space(self, agent: typing.AgentID) -> spaces.Space:
        """Takes in agent and returns the observation space for that agent.

        :param agent: The agent ID.
        :type agent: typing.AgentID
        :return: The observation space for the agent.
        :rtype: spaces.Space
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: typing.AgentID) -> spaces.Space:
        """Takes in agent and returns the action space for that agent.

        :param agent: The agent ID.
        :type agent: typing.AgentID
        :return: The action space for the agent.
        :rtype: spaces.Space
        """
        return self.action_spaces[agent]

    def reset(
        self,
        *,
        seed: int | None = 42,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[dict[typing.AgentID, typing.ObsType], dict[str, typing.Any]]:
        """Reset the environement and return the initial observations. Must be implemented for each environment.

        :param seed: NumPy random seed, defaults to 42
        :type seed: int | None, optional
        :param options: Environment reset options, defaults to None
        :type options: dict[str, typing.Any] | None, optional
        :return: Tuple of observations and info.
        :rtype: tuple[dict[typing.AgentID, typing.ObsType], dict[str, typing.Any]]
        """
        if seed is not None:
            self._np_random, _ = self._set_np_random(seed=seed)

        self.agents = copy.copy(self.possible_agents)

        self._gen_grid()

        # Clear out past agents and re-initialize
        self.env_agents = {}
        self.setup_agents()

        # Initialize previous actions as no-op
        self.prev_actions = {
            a_id: grid_actions.Actions.Noop for a_id in self.agent_ids
        }

        self.t = 0

        self.on_reset()

        self.update_grid_agents()

        # Build array state from grid and agents
        self._array_state = layout_to_array_state(self.grid, scope=self.scope, scope_config=self._scope_config)
        agent_arrays = create_agent_arrays(self.env_agents, scope=self.scope)
        self._array_state.update(agent_arrays)

        self.cumulative_score = 0

        # -------------------------------------------------------------------
        # JAX backend: build EnvState from array_state, run JIT-compiled reset
        # -------------------------------------------------------------------
        if self._backend == 'jax':
            import jax
            import jax.numpy as jnp
            from cogrid.core.jax_step import make_jitted_step, make_jitted_reset

            # Convert layout arrays to JAX arrays
            pot_positions = self._array_state.get("pot_positions", [])
            if isinstance(pot_positions, list):
                if len(pot_positions) > 0:
                    pot_positions = jnp.array(pot_positions, dtype=jnp.int32)
                else:
                    pot_positions = jnp.zeros((0, 2), dtype=jnp.int32)
            else:
                pot_positions = jnp.array(pot_positions, dtype=jnp.int32)

            self._jax_layout_arrays = {
                "wall_map": jnp.array(self._array_state["wall_map"], dtype=jnp.int32),
                "object_type_map": jnp.array(self._array_state["object_type_map"], dtype=jnp.int32),
                "object_state_map": jnp.array(self._array_state["object_state_map"], dtype=jnp.int32),
                "pot_contents": jnp.array(self._array_state["pot_contents"], dtype=jnp.int32),
                "pot_timer": jnp.array(self._array_state["pot_timer"], dtype=jnp.int32),
                "pot_positions": pot_positions,
            }

            # Convert spawn positions from the parsed layout
            self._jax_spawn_positions = jnp.array(
                self._array_state["agent_pos"], dtype=jnp.int32
            )

            n_agents = self.config["num_agents"]

            # Determine action set name
            if self.action_set == grid_actions.ActionSets.CardinalActions:
                action_set_name = "cardinal"
            else:
                action_set_name = "rotation"

            # Build JIT-compiled reset and step functions
            self._jitted_reset = make_jitted_reset(
                self._jax_layout_arrays,
                self._jax_spawn_positions,
                n_agents,
                self._jax_feature_fn,
                self._scope_config,
                action_set_name,
            )

            self._jitted_step = make_jitted_step(
                self._scope_config,
                self._lookup_tables,
                self._jax_feature_fn,
                self._jax_reward_config,
                self._action_pickup_drop_idx,
                self._action_toggle_idx,
                self.max_steps,
            )

            # Create JAX PRNG key and run reset
            rng_key = jax.random.key(seed if seed is not None else 42)
            self._env_state, jax_obs = self._jitted_reset(rng_key)

            # Convert JAX obs to PettingZoo dict format
            obs = {
                aid: np.array(jax_obs[i])
                for i, aid in enumerate(self._agent_id_order)
            }

            self.per_agent_reward = self.get_empty_reward_dict()
            self.per_component_reward = {}

            return obs, {agent_id: {} for agent_id in self.agent_ids}

        # -------------------------------------------------------------------
        # Numpy backend: existing path
        # -------------------------------------------------------------------
        self.render()

        obs = self.get_obs()

        self.per_agent_reward = self.get_empty_reward_dict()
        self.per_component_reward = {}

        return obs, {agent_id: {} for agent_id in self.agent_ids}

    def _action_idx_to_str(
        self, actions: dict[typing.AgentID, int | str]
    ) -> dict[typing.AgentID, str]:
        """Convert the action from index to string representation


        :param actions: Dictionary of agent IDs and actions.
        :type actions: dict[str, int  |  str]
        :return: _description_
        :rtype: dict[str, str]
        """
        str_actions = {
            a_id: (
                self.action_set[action]
                if not isinstance(action, str)
                else action
            )
            for a_id, action in actions.items()
        }

        return str_actions

    def step(self, actions: dict[typing.AgentID, typing.ActionType]) -> tuple[
        dict[typing.AgentID, typing.ObsType],
        dict[typing.AgentID, float],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, dict[typing.Any, typing.Any]],
    ]:
        """Transitition the environment forward by one step, given the actions of the agents.

        :param actions: Dictionary of agent IDs and actions.
        :type actions: dict
        :return: Tuple of observations, rewards, terminateds, truncateds, and infos.
        :rtype: tuple[ dict[typing.AgentID, typing.ObsType], dict[typing.AgentID, float], dict[typing.AgentID, bool], dict[typing.AgentID, bool], dict[typing.AgentID, dict[typing.Any, typing.Any]], ]
        """
        # JAX backend: bypass all numpy-path logic, go straight to JIT step
        if self._backend == 'jax':
            return self._jax_step_wrapper(actions)

        self.t += 1

        # Reset the rewards for this step
        self.per_agent_reward = self.get_empty_reward_dict()
        self.per_component_reward = {}

        # Track the previous state so that we have the delta for computing rewards
        self.prev_grid = copy.deepcopy(self.grid)

        # Save previous array state for reward computation
        if self._array_state is not None:
            self._prev_array_state = {
                k: v.copy() if hasattr(v, 'copy') else v
                for k, v in self._array_state.items()
            }
        else:
            self._prev_array_state = None

        # Update attributes of the grid objects that are timestep dependent
        self.grid.tick()

        # Convert the integer actions to strings (helpful for debugging!)
        actions = self._action_idx_to_str(actions)

        # -------------------------------------------------------------------
        # Vectorized movement (PRIMARY path) -- replaces self.move_agents()
        # -------------------------------------------------------------------
        if self._array_state is not None:
            self._vectorized_move(actions)
        else:
            # Fallback to original if array state not initialized
            self.move_agents(actions)

        # Given any new position(s), agents interact with the environment
        self.interact(actions)

        # Updates the GridAgent objects to reflect new positions/interactions
        self.update_grid_agents()

        # Sync array state from objects after interactions
        if self._array_state is not None:
            self._sync_array_state_from_objects()

        # Store the actions taken by the agents
        self.prev_actions = actions.copy()

        # Setup the return values
        observations = self.get_obs()
        self.compute_rewards()
        infos = self.get_infos(**self.per_component_reward)
        terminateds, truncateds = self.get_terminateds_truncateds()

        self.render()

        self.cumulative_score += sum([*self.per_agent_reward.values()])

        # Custom hook if a subclass wants to make any updates
        self.on_step()

        return (
            observations,
            self.per_agent_reward,
            terminateds,
            truncateds,
            infos,
        )

    def update_grid_agents(self) -> None:
        """Update the grid agents to reflect the current state of each Agent."""
        self.grid.grid_agents = {
            a_id: grid_object.GridAgent(
                agent, num_agents=self.config["num_agents"], scope=self.scope
            )
            for a_id, agent in self.env_agents.items()
        }

    def _vectorized_move(self, actions: dict) -> None:
        """Use vectorized movement as the primary path, then sync back to Agent objects.

        Replaces :meth:`move_agents` with ``move_agents_array()`` for the position
        computation, then writes the results back to Agent objects so that the rest
        of the step loop (interact, get_obs, compute_rewards) works unchanged.
        """
        # Build action array in the same order as agent_arrays
        agent_ids = self._array_state["agent_ids"]
        n_agents = self._array_state["n_agents"]

        # Map string actions to integer indices for vectorized path
        action_to_idx = {name: i for i, name in enumerate(self.action_set)}
        actions_arr = np.array(
            [action_to_idx.get(actions.get(a_id, grid_actions.Actions.Noop),
                               len(self.action_set) - 1)
             for a_id in agent_ids],
            dtype=np.int32,
        )

        # Determine action set type for move_agents_array
        if self.action_set == grid_actions.ActionSets.CardinalActions:
            action_set_str = "cardinal"
        else:
            action_set_str = "rotation"

        # Run vectorized movement
        priority = self.np_random.permutation(n_agents).astype(np.int32)
        new_pos, new_dir = move_agents(
            self._array_state["agent_pos"],
            self._array_state["agent_dir"],
            actions_arr,
            self._array_state["wall_map"],
            self._array_state["object_type_map"],
            self._lookup_tables["CAN_OVERLAP"],
            priority,
            action_set_str,
        )

        # Update array state
        self._array_state["agent_pos"] = new_pos
        self._array_state["agent_dir"] = new_dir

        # Sync results back to Agent objects
        sync_arrays_to_agents(self._array_state, self.env_agents)

        # Call on_move hooks for each agent
        for a_id in self.agent_ids:
            self.on_move(a_id)

        # Verify unique positions (matching original assertion)
        assert len(self.agent_pos) == len(
            set(self.agent_pos)
        ), "Agents do not have unique positions!"

    def _sync_array_state_from_objects(self) -> None:
        """Rebuild array state from Grid and Agent objects after interactions.

        Called after interact() and update_grid_agents() to ensure the array
        state reflects any changes made by the object-based interaction code.
        This is the Phase 1 sync approach -- Phase 2 will remove the object
        path entirely.
        """
        self._array_state = layout_to_array_state(self.grid, scope=self.scope, scope_config=self._scope_config)
        agent_arrays = create_agent_arrays(self.env_agents, scope=self.scope)
        self._array_state.update(agent_arrays)

    def _jax_step_wrapper(
        self, action_dict: dict[typing.AgentID, typing.ActionType]
    ) -> tuple:
        """Execute one step using the JIT-compiled JAX step function.

        Converts between PettingZoo dict-based API and array-based JAX core.
        Bypasses all numpy-path logic (no grid.tick, no interact, no render).

        Args:
            action_dict: Dictionary mapping agent IDs to integer actions.

        Returns:
            PettingZoo-format tuple (obs, rewards, terminateds, truncateds, infos).
        """
        import jax.numpy as jnp

        # Convert action dict to ordered array
        actions_arr = jnp.array(
            [action_dict[aid] for aid in self._agent_id_order],
            dtype=jnp.int32,
        )

        # Call JIT-compiled step
        self._env_state, obs_arr, rewards_arr, done_scalar, infos = (
            self._jitted_step(self._env_state, actions_arr)
        )

        # Convert to PettingZoo dict format
        obs = {
            aid: np.array(obs_arr[i])
            for i, aid in enumerate(self._agent_id_order)
        }
        rewards = {
            aid: float(rewards_arr[i])
            for i, aid in enumerate(self._agent_id_order)
        }

        truncated = bool(done_scalar)
        terminateds = {aid: False for aid in self._agent_id_order}
        truncateds = {aid: truncated for aid in self._agent_id_order}

        if truncated:
            self.agents = []

        infos = {aid: {} for aid in self._agent_id_order}

        # Increment PettingZoo timestep counter
        self.t += 1

        return obs, rewards, terminateds, truncateds, infos

    @property
    def jax_step(self):
        """Raw JIT-compiled step function for direct JIT/vmap usage.

        Signature: (EnvState, actions) -> (EnvState, obs, rewards, done, infos)

        Returns:
            The JIT-compiled step function.

        Raises:
            RuntimeError: If backend is not 'jax' or reset() has not been called.
        """
        if self._backend != 'jax':
            raise RuntimeError("jax_step is only available with backend='jax'")
        if self._jitted_step is None:
            raise RuntimeError("Must call reset() before accessing jax_step")
        return self._jitted_step

    @property
    def jax_reset(self):
        """Raw JIT-compiled reset function for direct JIT/vmap usage.

        Signature: (rng_key) -> (EnvState, obs)

        Returns:
            The JIT-compiled reset function.

        Raises:
            RuntimeError: If backend is not 'jax' or reset() has not been called.
        """
        if self._backend != 'jax':
            raise RuntimeError("jax_reset is only available with backend='jax'")
        if self._jitted_reset is None:
            raise RuntimeError("Must call reset() before accessing jax_reset")
        return self._jitted_reset

    def setup_agents(self):
        self._setup_agents()

    def _setup_agents(self):
        for agent_id in range(self.config["num_agents"]):
            agent = self.agent_class(
                agent_id=agent_id,
                start_position=self.select_spawn_point(),
                start_direction=self.np_random.choice(directions.Directions),
            )
            self.env_agents[agent_id] = agent

    def move_agents(
        self, actions: dict[typing.AgentID, typing.ActionType]
    ) -> None:
        """Move agents to new positions based on the actions they take.

        :param actions: A dictionary of agent IDs and the actions they are taking.
        :type actions: dict[typing.AgentID, typing.ActionType]
        """
        # All terminated agents or those we don't have an action for will stay in the same position
        new_positions = {
            a_id: agent.pos
            for a_id, agent in self.env_agents.items()
            if agent.terminated or a_id not in actions
        }

        # All agents that are taking an action will be moved to a new position
        agents_to_move = [
            a_id
            for a_id in actions.keys()
            if not self.env_agents[a_id].terminated
        ]

        # If we're using cardinal actions, change the agent direction if they aren't
        # already facing the desired dir if they are, then they move in that direction.
        if self.action_set == grid_actions.ActionSets.CardinalActions:
            move_action_to_dir = {
                grid_actions.Actions.MoveRight: directions.Directions.Right,
                grid_actions.Actions.MoveLeft: directions.Directions.Left,
                grid_actions.Actions.MoveUp: directions.Directions.Up,
                grid_actions.Actions.MoveDown: directions.Directions.Down,
            }
            for a_id, action in actions.items():
                if action in move_action_to_dir.keys():
                    agent = self.env_agents[a_id]
                    desired_direction = move_action_to_dir[action]

                    # rotate to the desired direction and move that way
                    agent.dir = desired_direction
                    actions[a_id] = grid_actions.Actions.Forward

        # Determine the position each agent is attempting to move to
        attempted_positions = {}
        for a_id, action in actions.items():
            attempted_positions[a_id] = self.determine_attempted_pos(
                a_id, action
            )

        # First, give priority to agents staying in the same position
        for a_id, attemped_pos in attempted_positions.items():
            agent = self.env_agents[a_id]
            if np.array_equal(attemped_pos, agent.pos):
                new_positions[a_id] = agent.pos
                if a_id in agents_to_move:
                    agents_to_move.remove(a_id)

        # Randomize agent priority and move agents to new positions
        self.np_random.shuffle(agents_to_move)
        for a_id in agents_to_move:
            agent = self.env_agents[a_id]
            attempted_pos = attempted_positions[a_id]

            # Enhanced collision check - check both new_positions and current positions of other agents
            position_blocked = tuple(attempted_pos) in [
                tuple(npos) for npos in new_positions.values()
            ] or tuple(attempted_pos) in [
                tuple(other.pos)
                for other_id, other in self.env_agents.items()
                if other_id != a_id and other_id not in new_positions
            ]

            if position_blocked:
                new_positions[a_id] = agent.pos
            else:
                new_positions[a_id] = attempted_pos

        # Make sure no two agents moved through each other
        for a_id1, a_id2 in combinations(self.agent_ids, r=2):
            agent1, agent2 = self.env_agents[a_id1], self.env_agents[a_id2]
            if np.array_equal(
                new_positions[a_id1], agent2.pos
            ) and np.array_equal(new_positions[a_id2], agent1.pos):
                new_positions[a_id1] = agent1.pos
                new_positions[a_id2] = agent2.pos

        # assign the new positions and store the agent position
        for a_id, agent in self.env_agents.items():
            agent.pos = new_positions[a_id]
            self.on_move(a_id)

        assert len(self.agent_pos) == len(
            set(self.agent_pos)
        ), "Agents do not have unique positions!"

    def determine_attempted_pos(
        self, agent_id: typing.AgentID, action: typing.ActionType
    ) -> tuple[int, int]:
        """Determine the position an agent is attempting to move to.

        :param agent_id: The ID of the agent attempting to move.
        :type agent_id: typing.AgentID
        :param action: The action the agent is attempting to take.
        :type action: typing.ActionType
        :return: The position the agent is attempting to move to.
        :rtype: tuple[int, int]
        """
        agent = self.env_agents[agent_id]
        fwd_pos = agent.front_pos
        fwd_cell = self.grid.get(*fwd_pos)

        if action == grid_actions.Actions.Forward:
            if fwd_cell is None or fwd_cell.can_overlap(agent=agent):
                return fwd_pos

        return agent.pos

    def can_toggle(self, agent_id: typing.AgentID) -> bool:
        """Check if an agent can toggle the object in front of them.

        :param agent_id: The ID of the agent attempting to toggle the object.
        :type agent_id: typing.AgentID
        :return: True if the agent can toggle the object in front of them, False otherwise.
        :rtype: bool
        """
        agent = self.env_agents[agent_id]
        fwd_cell = copy.deepcopy(self.grid.get(*agent.front_pos))
        return fwd_cell.toggle(env=self, toggling_agent=agent)

    def interact(
        self, actions: dict[typing.AgentID, typing.ActionType]
    ) -> None:
        """After agents have moved, let them interact with the environment
        based on their actions (e.g., picking up, dropping, toggling, etc.).

        :param actions: Dictionary of agent IDs and actions.
        :type actions: dict[typing.AgentID, typing.ActionType]
        """
        for a_id, action in actions.items():
            agent: grid_object.GridAgent = self.env_agents[a_id]
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
                    and fwd_cell.can_place_on(
                        cell=agent.inventory[0], agent=agent
                    )
                ):
                    drop_cell = agent.inventory.pop(0)
                    drop_cell.pos = fwd_pos

                    fwd_cell.place_on(cell=drop_cell, agent=agent)
                    agent.cell_placed_on = fwd_cell

                self.on_pickup_drop(a_id)

            # Attempt to toggle the object in front of the agent
            elif action == grid_actions.Actions.Toggle:
                fwd_cell = self.grid.get(*agent.front_pos)
                if fwd_cell:
                    toggle_success = fwd_cell.toggle(env=self, agent=agent)
                    if toggle_success:
                        agent.cell_toggled = fwd_cell

                self.on_toggle(a_id)

        self.on_interact(actions)

    def on_interact(
        self, actions: dict[typing.AgentID, typing.ActionType]
    ) -> None:
        """Hook for subclasses to implement custom logic after agents interact with the environment.

        :param actions: Dictionary of agent IDs and actions.
        :type actions: dict[typing.AgentID, typing.ActionType]
        """
        pass

    def on_toggle(self, agent_id: typing.AgentID) -> None:
        """Hook for subclasses to implement custom logic after an agent toggles an object.

        :param agent_id: The ID of the agent toggling the object.
        :type agent_id: typing.AgentID
        """
        pass

    def on_pickup_drop(self, agent_id: typing.AgentID) -> None:
        """Hook for subclasses to implement custom logic after an agent picks up or drops an object.

        :param agent_id: The ID of the agent picking up or dropping an object.
        :type agent_id: typing.AgentID
        """
        pass

    def on_reset(self) -> None:
        """Hook for subclasses to implement custom logic after the environment is reset."""
        pass

    def on_step(self) -> None:
        """Hook for subclasses to implement custom logic after each step."""
        pass

    def on_move(self, agent_id: typing.AgentID) -> None:
        """Hook for subclasses to implement custom logic after an agent moves.

        :param agent_id: The ID of the agent moving.
        :type agent_id: typing.AgentID
        """
        pass

    def get_obs(self) -> dict[typing.AgentID, typing.ObsType]:
        """Fetch new observations for the agents

        :return: Dictionary of agent IDs and their observations.
        :rtype: dict[typing.AgentID, typing.ObsType]
        """
        obs = {}
        for a_id in self.agent_ids:
            obs[a_id] = self.feature_spaces[a_id].generate_features()
        return obs

    def compute_rewards(
        self,
    ) -> None:
        """Compute the per agent and per component rewards for the current state transition
        using the reward modules provided in the environment configuration.

        The rewards are added to self.per_agent_rewards and self.per_component_rewards.
        """

        for reward in self.rewards:
            calculated_rewards = reward.calculate_reward(
                state=self.prev_grid,
                agent_actions=self.prev_actions,
                new_state=self.grid,
            )

            # Add component rewards to per agent reward
            for agent_id, reward_value in calculated_rewards.items():
                self.per_agent_reward[agent_id] += reward_value

            # Save reward by component
            self.per_component_reward[reward.name] = calculated_rewards

        for agent_id, val in self.reward_this_step.items():
            self.per_agent_reward[agent_id] += val

        self.reward_this_step = self.get_empty_reward_dict()

    def get_terminateds_truncateds(
        self,
    ) -> tuple[dict[typing.AgentID, bool], dict[typing.AgentID, bool]]:
        """Determine the done status for each agent."""
        terminateds = {
            agent_id: agent.terminated
            for agent_id, agent in self.env_agents.items()
        }
        # terminateds["__all__"] = all([*terminateds.values()])

        if self.t >= self.max_steps:
            truncateds = {agent_id: True for agent_id in self.agent_ids}
        else:
            truncateds = {agent_id: False for agent_id in self.agent_ids}

        # truncateds["__all__"] = all([*truncateds.values()])

        # Update active agents
        self.agents = [
            agent_id
            for agent_id in self.possible_agents
            if not terminateds[agent_id] and not truncateds[agent_id]
        ]

        return terminateds, truncateds

    def get_infos(
        self, **kwargs
    ) -> dict[typing.AgentID, dict[typing.Any, typing.Any]]:
        """Get info dictionaries for each agent.

        :return: Dictionary keyed by agent IDs containing info dictionaries.
        :rtype: dict[typing.Any, typing.Any]
        """
        infos = {agent_id: {} for agent_id in self._agent_ids}
        for info_key, info_dict in kwargs.items():
            for agent_id, val in info_dict.items():
                assert (
                    agent_id in self._agent_ids
                ), f"Must pass dicts keyed by AgentIDs to get_infos(), got invalid key: {agent_id}"
                infos[agent_id][info_key] = val
        return infos

    def get_empty_reward_dict(self) -> dict[typing.AgentID, float]:
        """Get a dictionary of rewards for each agent, initialized to 0.

        :return: Dictionary of rewards for each agent, initialized to 0.
        :rtype: dict[typing.AgentID, float]
        """
        return {a_id: 0 for a_id in self.agent_ids}

    @property
    def map_with_agents(self) -> np.ndarray:
        """retrieve a version of the environment where 'P' chars have agent IDs.

        :return: Map of the environment with agent IDs.
        :rtype: np.ndarray
        """

        grid_encoding = self.grid.encode(encode_char=True, scope=self.scope)
        grid = grid_encoding[:, :, 0]

        for a_id, agent in self.env_agents.items():
            if (
                agent is not None
            ):  # will be None before being set by subclassed env
                grid[agent.pos[0], agent.pos[1]] = self.id_to_numeric(a_id)

        return grid

    def select_spawn_point(self) -> tuple[int, int]:
        """Select a spawn point for an agent.

        :return: A spawn point for an agent.
        :rtype: tuple[int, int]
        """
        if self.spawn_points:
            return self.spawn_points.pop(0)

        available_spawns = self.available_positions
        return self.np_random.choice(available_spawns)

    @property
    def available_positions(self) -> list[tuple[int, int]]:
        """Get a list of available positions for agents to spawn.

        :return: List of available positions for agents to spawn.
        :rtype: list[tuple[int, int]]
        """
        spawns = []
        for r in range(self.grid.height):
            for c in range(self.grid.width):
                cell = self.grid.get(row=r, col=c)
                if (r, c) not in self.agent_pos and (
                    not cell or cell.object_id == "floor"
                ):
                    spawns.append((r, c))
        return spawns

    def put_obj(self, obj: grid_object.GridObj, row: int, col: int):
        """Place an object at a specific point in the grid.

        :param obj: The object to place.
        :type obj: grid_object.GridObj
        :param row: The row to place the object.
        :type row: int
        :param col: The column to place the object.
        :type col: int
        """
        self.grid.set(row=row, col=col, v=obj)
        obj.pos = (row, col)
        obj.init_pos = (row, col)

    def get_view_exts(
        self, agent_id: typing.AgentID, agent_view_size: int = None
    ) -> tuple[int, int, int, int]:
        """Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size


        :param agent_id: Agent ID of the agent to get the view extents for.
        :type agent_id: typing.AgentID
        :param agent_view_size: View distance, defaults to None
        :type agent_view_size: int, optional
        :raises ValueError: Invalid agent direction.
        :return: Tuple of the top-left and bottom-right extents of the agent's view.
        :rtype: tuple[int, int, int, int]
        """
        agent = self.env_agents[agent_id]
        agent_view_size = agent_view_size or self.agent_view_size

        if agent.dir == directions.Directions.Right:
            topY = agent.pos[0] - agent_view_size // 2
            topX = agent.pos[1]
        elif agent.dir == directions.Directions.Down:
            topY = agent.pos[0]
            topX = agent.pos[1] - agent_view_size // 2
        elif agent.dir == directions.Directions.Left:
            topY = agent.pos[0] - agent_view_size // 2
            topX = agent.pos[1] - agent_view_size + 1
        elif agent.dir == directions.Directions.Up:
            topY = agent.pos[0] - agent_view_size + 1
            topX = agent.pos[1] - agent_view_size // 2
        else:
            raise ValueError("Invalid agent direction.")

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def gen_obs_grid(
        self, agent_id: typing.AgentID, agent_view_size: int = None
    ) -> grid.Grid:
        """Generate the sub-grid observed by a specific agent

        :param agent_id: Agent ID of the agent to generate the observation for.
        :type agent_id: typing.AgentID
        :param agent_view_size: Size of the agents view area, defaults to None
        :type agent_view_size: int, optional
        :return: A sub-grid observed by the agent.
        :rtype: grid.Grid
        """
        topX, topY, *_ = self.get_view_exts(agent_id, agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        assert agent_id in grid.grid_agents

        agent = self.env_agents[agent_id]
        for i in range(agent.dir + 1):
            grid = grid.rotate_left()

        if not self.metadata.get("see_through_walls", True):
            # Mask view from the agents position at the bottom-center of the grid view
            vis_mask = grid.process_vis(agent_pos=(-1, grid.width // 2))
        else:
            vis_mask = np.ones(shape=(grid.height, grid.width), dtype=bool)

        assert len(grid.grid_agents) >= 1
        assert grid.grid_agents[agent_id].dir == directions.Directions.Up

        # NOTE: In Minigrid, they replace the agent's position with the item they're carrying. We don't do that
        #   here. Rather, we'll provide an additional observation space that represents the item(s) in inventory.

        return grid, vis_mask

    def get_pov_render(
        self,
        agent_id: typing.AgentID,
        tile_size: int = CoreConstants.TilePixels,
    ) -> np.ndarray:
        """Render a specific agent's POV

        :param agent_id: Agent ID of the POV to render.
        :type agent_id: typing.AgentID
        :param tile_size: The pixel height/width of each rendered grid cell, defaults to CoreConstants.TilePixels
        :type tile_size: int, optional
        :return: Render of an agent's POV.
        :rtype: np.ndarray
        """
        grid, vis_mask = self.gen_obs_grid(agent_id)
        img = grid.render(
            tile_size,
            highlight_mask=vis_mask,
        )
        return img

    def get_full_render(
        self, highlight: bool = False, tile_size: int = CoreConstants.TilePixels
    ) -> np.ndarray:
        """Generate a render of the full environment.

        :param highlight: Highlight the visible region for each agent, defaults to False.
        :type highlight: bool
        :param tile_size: The pixel height/width of each rendered grid cell, defaults to CoreConstants.TilePixels
        :type tile_size: int, optional
        :return: Render of the full environment.
        :rtype: np.ndarray
        """
        if highlight:
            highlight_mask = np.zeros(
                shape=(self.grid.height, self.grid.width), dtype=bool
            )
            for a_id, agent in self.env_agents.items():
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

        img = self.grid.render(
            tile_size=tile_size, highlight_mask=highlight_mask
        )
        return img

    def get_frame(
        self,
        highlight: bool = True,
        tile_size: int = CoreConstants.TilePixels,
        agent_pov: typing.AgentID | None = None,
    ) -> np.ndarray:
        """Return RGB image corresponding to the whole environment or an agent's POV

        :param highlight: Highlight the visible region for each agent, defaults to True
        :type highlight: bool, optional
        :param tile_size: The pixel width height of each grid cell, defaults to CoreConstants.TilePixels
        :type tile_size: int, optional
        :param agent_pov: If specified, gets the frame view for the specified agent, defaults to None
        :type agent_pov: str | None, optional
        :return: The rendered image of the environment or agent perspective.
        :rtype: np.ndarray
        """
        if agent_pov:
            frame = self.get_pov_render(
                agent_id=self.agent_pov, tile_size=tile_size
            )
        else:
            frame = self.get_full_render(
                highlight=highlight, tile_size=tile_size
            )

        return frame

    def render(self) -> None | np.ndarray:
        """Render the environment.

        :return: None if rendering is not enabled, otherwise the rendered image.
        :rtype: None | np.ndarray
        """
        if self.visualizer is not None:
            orientations = {
                self.id_to_numeric(a_id): agent.orientation
                for a_id, agent in self.env_agents.items()
            }
            inventories = {
                self.id_to_numeric(a_id): agent.inventory
                for a_id, agent in self.env_agents.items()
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

        img = self.get_frame(
            self.metadata.get("highlight", False),
            self.tile_size,
            self.metadata.get("agent_pov", None),
        )

        if self.render_mode == "human":
            if pygame is None:
                raise ImportError(
                    "Must install pygame to use interactive mode."
                )
            # TODO(chase): move all pygame logic to run_interactive.py so it's not needed here.
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
                (
                    int(surf.get_size()[0] + offset),
                    int(surf.get_size()[1] + offset),
                )
            )
            bg.convert()
            bg.fill((255, 255, 255))
            bg.blit(surf, (offset / 2, 0))

            bg = pygame.transform.smoothscale(
                bg, (self.screen_size, self.screen_size)
            )

            font_size = 22
            text = (
                f"Score: {np.round(self.cumulative_score, 2)}"
                + self.render_message
            )

            font = pygame.freetype.SysFont(
                pygame.font.get_default_font(), font_size
            )
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
        return list(self.env_agents.keys())

    @property
    def agent_pos(self) -> list:
        return [
            tuple(agent.pos)
            for agent in self.env_agents.values()
            if agent is not None
        ]

    def id_to_numeric(self, agent_id) -> str:
        """Converts agent id to integer, beginning with 1,
        e.g., agent-0 -> 1, agent-1 -> 2, etc.
        """
        agent = self.env_agents[agent_id]
        return str(agent.agent_number)
