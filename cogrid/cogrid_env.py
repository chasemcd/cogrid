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

# Vectorized components
from cogrid.backend import set_backend
from cogrid.core.grid_object import build_lookup_tables
from cogrid.core.grid_utils import layout_to_array_state
from cogrid.core.agent import create_agent_arrays, sync_arrays_to_agents, get_dir_vec_table
from cogrid.core.movement import move_agents


RNG = RandomNumberGenerator = np.random.Generator


class CoGridEnv(pettingzoo.ParallelEnv):
    """CoGridEnv is the base environment class for a multi-agent grid-world environment.

    This class inherits from the ``pettingzoo.ParallelEnv`` class and implements the necessary methods
    for a parallel environment. Both the numpy and JAX backends delegate step/reset to the
    unified step pipeline (``build_step_fn``/``build_reset_fn``), making this class a thin
    stateful wrapper around the functional core.

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
        """Initialize the CoGrid environment.

        :param config: Environment configuration dictionary.
        :type config: dict
        :param render_mode: Rendering mode, defaults to None
        :type render_mode: str | None, optional
        :param agent_class: Custom Agent class, defaults to None
        :type agent_class: agent.Agent | None, optional
        :param backend: Array backend name ('numpy' or 'jax'). The first
            environment created sets the global backend; subsequent envs
            must use the same backend or a RuntimeError is raised.
        :type backend: str
        :raises ValueError: If an invalid action set is provided.
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
        # Vectorized infrastructure (backend-agnostic)
        # -------------------------------------------------------------------
        # Build lookup tables for the current scope
        self._lookup_tables = build_lookup_tables(scope=self.scope)

        # Build scope config via auto-wiring from registered components
        from cogrid.core.autowire import (
            build_scope_config_from_components,
            build_reward_config_from_components,
        )
        self._scope_config = build_scope_config_from_components(self.scope)
        self._type_ids = self._scope_config["type_ids"]
        self._interaction_tables = self._scope_config.get("interaction_tables")

        # Array state is built in reset() after agents are placed
        self._array_state = None

        # Enable shadow parity validation (set to False for performance)
        self._validate_array_parity = False

        # Build feature function (works for both backends)
        from cogrid.feature_space.array_features import build_feature_fn
        jax_feature_names = [
            "agent_position", "agent_dir", "full_map_encoding",
            "can_move_direction", "inventory",
        ]
        self._feature_fn = build_feature_fn(
            jax_feature_names, scope=self.scope,
        )

        # Compute action indices for PickupDrop and Toggle
        self._action_pickup_drop_idx = self.action_set.index(
            grid_actions.Actions.PickupDrop
        )
        self._action_toggle_idx = self.action_set.index(
            grid_actions.Actions.Toggle
        )

        # Auto-wire reward config from registered ArrayReward subclasses
        self._reward_config = build_reward_config_from_components(
            self.scope,
            n_agents=self.config["num_agents"],
            type_ids=self._type_ids,
            action_pickup_drop_idx=self._action_pickup_drop_idx,
        )

        # Sorted agent ID order for dict<->array conversion (both backends)
        self._agent_id_order = sorted(self.possible_agents)

        # Optional per-agent termination function.  Can be provided via
        # config["terminated_fn"] or set_terminated_fn() before reset().
        self._terminated_fn = config.get("terminated_fn")

        # Step/reset pipeline functions -- initialized lazily in reset()
        self._step_fn = None
        self._reset_fn = None
        self._env_state = None

        # -------------------------------------------------------------------
        # JAX-specific setup: pytree registration, array conversion
        # -------------------------------------------------------------------
        if self._backend == 'jax':
            import jax.numpy as jnp
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
        """Set the numpy random number generator.

        :param seed: Random seed, defaults to None
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

    def set_terminated_fn(self, fn):
        """Set a per-agent termination function.

        Must be called before the first ``reset()`` call so the function
        is closed over by the JIT-compiled step pipeline.

        Args:
            fn: Callable ``(prev_state_dict, state_dict, reward_config)``
                returning a bool array of shape ``(n_agents,)``.
        """
        self._terminated_fn = fn

    def reset(
        self,
        *,
        seed: int | None = 42,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[dict[typing.AgentID, typing.ObsType], dict[str, typing.Any]]:
        """Reset the environment and return the initial observations.

        Both numpy and JAX backends build step/reset functions from the
        unified step pipeline and delegate to them.

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

        # Build extra state from autowired builder (e.g. pot arrays for Overcooked)
        extra_state_builder = self._scope_config.get("extra_state_builder")
        if extra_state_builder is not None:
            extra = extra_state_builder(self._array_state, self.scope)
            # Strip scope prefix from keys: the step_pipeline.reset() will
            # re-add the prefix when building EnvState.extra_state.
            scope_prefix = f"{self.scope}."
            stripped = {
                (k[len(scope_prefix):] if k.startswith(scope_prefix) else k): v
                for k, v in extra.items()
            }
            self._array_state.update(stripped)

        agent_arrays = create_agent_arrays(self.env_agents, scope=self.scope)
        self._array_state.update(agent_arrays)

        self.cumulative_score = 0

        # -------------------------------------------------------------------
        # Build layout arrays and step/reset pipeline (both backends)
        # -------------------------------------------------------------------
        # Determine the array module for this backend
        if self._backend == 'jax':
            import jax
            import jax.numpy as jnp
            xp = jnp
        else:
            xp = np

        # Build layout arrays from array_state (scope-generic)
        skip_keys = {"agent_pos", "agent_dir", "agent_inv", "spawn_points"}
        layout_arrays = {}
        for key, val in self._array_state.items():
            if key in skip_keys:
                continue
            if isinstance(val, list):
                val = np.array(val, dtype=np.int32) if len(val) > 0 else np.zeros((0, 2), dtype=np.int32)
            layout_arrays[key] = xp.array(val, dtype=xp.int32)

        spawn_positions = xp.array(
            self._array_state["agent_pos"], dtype=xp.int32
        )

        n_agents = self.config["num_agents"]

        # Determine action set name
        if self.action_set == grid_actions.ActionSets.CardinalActions:
            action_set_name = "cardinal"
        else:
            action_set_name = "rotation"

        # Build step and reset functions from the unified pipeline
        from cogrid.core.step_pipeline import build_step_fn, build_reset_fn

        self._reset_fn = build_reset_fn(
            layout_arrays,
            spawn_positions,
            n_agents,
            self._feature_fn,
            self._scope_config,
            action_set_name,
        )

        self._step_fn = build_step_fn(
            self._scope_config,
            self._lookup_tables,
            self._feature_fn,
            self._reward_config,
            self._action_pickup_drop_idx,
            self._action_toggle_idx,
            self.max_steps,
            terminated_fn=self._terminated_fn,
        )

        # Create RNG and run reset
        if self._backend == 'jax':
            rng = jax.random.key(seed if seed is not None else 42)
        else:
            rng = seed if seed is not None else 42

        self._env_state, obs_arr = self._reset_fn(rng)

        # Convert obs array to PettingZoo dict format
        obs = {
            aid: np.array(obs_arr[i])
            for i, aid in enumerate(self._agent_id_order)
        }

        self.per_agent_reward = self.get_empty_reward_dict()
        self.per_component_reward = {}

        # Sync rendering objects from state if needed
        if self.render_mode is not None:
            self._sync_objects_from_state()
            self.render()

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
        """Transition the environment forward by one step.

        Both numpy and JAX backends delegate to the unified step pipeline
        via self._step_fn, then convert outputs to PettingZoo dict format.

        :param actions: Dictionary of agent IDs and actions.
        :type actions: dict
        :return: Tuple of observations, rewards, terminateds, truncateds, and infos.
        :rtype: tuple
        """
        # Select array module
        if self._backend == 'jax':
            import jax.numpy as jnp
            xp = jnp
        else:
            xp = np

        # Convert action dict to ordered array
        actions_arr = xp.array(
            [actions[aid] for aid in self._agent_id_order],
            dtype=xp.int32,
        )

        # Delegate to the unified step pipeline
        self._env_state, obs_arr, rewards_arr, terminateds_arr, truncateds_arr, infos = self._step_fn(
            self._env_state, actions_arr
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

        terminateds = {aid: bool(terminateds_arr[i]) for i, aid in enumerate(self._agent_id_order)}
        truncateds = {aid: bool(truncateds_arr[i]) for i, aid in enumerate(self._agent_id_order)}

        if any(terminateds.values()) or any(truncateds.values()):
            self.agents = []

        infos = {aid: {} for aid in self._agent_id_order}

        # Increment PettingZoo timestep counter
        self.t += 1

        self.cumulative_score += sum(rewards.values())

        # Sync rendering objects from state if needed
        if self.render_mode is not None:
            self._sync_objects_from_state()
            self.render()

        # Custom hook if a subclass wants to make any updates
        self.on_step()

        return obs, rewards, terminateds, truncateds, infos

    def get_state(self) -> dict:
        """Export a JSON-serializable snapshot of the environment state.

        Returns a plain dict of Python lists and scalars that fully
        describes the current simulation state.  The dict can be passed to
        :meth:`set_state` to restore the environment to this exact point,
        or serialized with ``json.dumps()`` for saving/transmission.

        Must be called after :meth:`reset`.

        Returns:
            Dict with keys matching :class:`EnvState` dynamic/static fields
            plus ``t`` (PettingZoo timestep) and ``cumulative_score``.
        """
        if self._env_state is None:
            raise RuntimeError("Must call reset() before get_state()")

        state = self._env_state
        snapshot = {
            "agent_pos": np.array(state.agent_pos).tolist(),
            "agent_dir": np.array(state.agent_dir).tolist(),
            "agent_inv": np.array(state.agent_inv).tolist(),
            "wall_map": np.array(state.wall_map).tolist(),
            "object_type_map": np.array(state.object_type_map).tolist(),
            "object_state_map": np.array(state.object_state_map).tolist(),
            "extra_state": {
                k: np.array(v).tolist() for k, v in state.extra_state.items()
            },
            "rng_key": np.array(state.rng_key).tolist() if state.rng_key is not None else None,
            "time": int(state.time),
            "done": np.array(state.done).tolist(),
            "n_agents": state.n_agents,
            "height": state.height,
            "width": state.width,
            "action_set": state.action_set,
            "t": self.t,
            "cumulative_score": float(self.cumulative_score),
        }
        return snapshot

    def set_state(self, snapshot: dict) -> None:
        """Restore the environment to a previously exported state.

        Accepts a dict produced by :meth:`get_state` (plain Python lists
        and scalars, as returned by JSON deserialization) and rebuilds the
        internal :class:`EnvState`, converting to the active backend's
        array type.

        Must be called after :meth:`reset` (the step/reset pipeline must
        already be initialized).

        Args:
            snapshot: Dict previously returned by :meth:`get_state`.
        """
        if self._step_fn is None:
            raise RuntimeError("Must call reset() before set_state()")

        from cogrid.backend.env_state import create_env_state

        if self._backend == "jax":
            import jax.numpy as jnp
            xp = jnp
        else:
            xp = np

        extra_state = {
            k: xp.array(v, dtype=xp.int32) for k, v in snapshot["extra_state"].items()
        }

        rng_key = snapshot["rng_key"]
        if rng_key is not None:
            if self._backend == "jax":
                import jax
                rng_key = jax.numpy.array(rng_key, dtype=jax.numpy.uint32)
            else:
                rng_key = np.array(rng_key)

        self._env_state = create_env_state(
            agent_pos=xp.array(snapshot["agent_pos"], dtype=xp.int32),
            agent_dir=xp.array(snapshot["agent_dir"], dtype=xp.int32),
            agent_inv=xp.array(snapshot["agent_inv"], dtype=xp.int32),
            wall_map=xp.array(snapshot["wall_map"], dtype=xp.int32),
            object_type_map=xp.array(snapshot["object_type_map"], dtype=xp.int32),
            object_state_map=xp.array(snapshot["object_state_map"], dtype=xp.int32),
            extra_state=extra_state,
            rng_key=rng_key,
            time=xp.int32(snapshot["time"]),
            done=xp.array(snapshot["done"], dtype=xp.bool_),
            n_agents=snapshot["n_agents"],
            height=snapshot["height"],
            width=snapshot["width"],
            action_set=snapshot["action_set"],
        )

        self.t = snapshot["t"]
        self.cumulative_score = snapshot["cumulative_score"]

        if self.render_mode is not None:
            self._sync_objects_from_state()

    def _sync_objects_from_state(self) -> None:
        """Sync Agent/Grid objects from EnvState arrays for rendering.

        Reads agent_pos, agent_dir, agent_inv, object_type_map, and
        object_state_map back to Agent/Grid objects so the tile renderer
        reflects current simulation state.

        Scope-specific rendering (e.g. counter obj_placed_on, pot contents)
        is delegated to the ``render_sync`` hook in scope_config, composed
        from ``build_render_sync_fn`` classmethods on registered GridObj
        subclasses.

        Only called when render_mode is not None. Not part of the simulation loop.
        """
        if self._env_state is None:
            return

        state = self._env_state
        from cogrid.core.grid_object import idx_to_object, make_object

        # --- Sync agent positions, directions, and inventory ---
        for i, aid in enumerate(self._agent_id_order):
            if aid not in self.env_agents or self.env_agents[aid] is None:
                continue
            agent_obj = self.env_agents[aid]
            pos = np.array(state.agent_pos[i])
            agent_obj.pos = (int(pos[0]), int(pos[1]))
            agent_obj.dir = int(np.array(state.agent_dir[i]))

            inv_type_id = int(np.array(state.agent_inv[i, 0]))
            if inv_type_id <= 0:
                agent_obj.inventory = []
            else:
                obj_id = idx_to_object(inv_type_id, scope=self.scope)
                agent_obj.inventory = (
                    [make_object(obj_id, scope=self.scope)] if obj_id else []
                )

        # --- Sync grid cells from object_type_map / object_state_map ---
        otm = np.array(state.object_type_map)
        osm = np.array(state.object_state_map)

        for r in range(self.grid.height):
            for c in range(self.grid.width):
                type_id = int(otm[r, c])
                state_val = int(osm[r, c])

                if type_id == 0:
                    self.grid.set(r, c, None)
                    continue

                obj_id = idx_to_object(type_id, scope=self.scope)
                if obj_id is None or obj_id == "free_space":
                    self.grid.set(r, c, None)
                    continue

                existing = self.grid.get(r, c)
                if existing is not None and existing.object_id == obj_id:
                    existing.state = state_val
                else:
                    new_obj = make_object(obj_id, scope=self.scope, state=state_val)
                    if new_obj is not None:
                        new_obj.pos = (r, c)
                    self.grid.set(r, c, new_obj)

        # --- Delegate scope-specific rendering sync ---
        render_sync = self._scope_config.get("render_sync")
        if render_sync is not None:
            render_sync(self.grid, state, self.scope)

        # Rebuild GridAgent wrappers for rendering
        self.update_grid_agents()

    def update_grid_agents(self) -> None:
        """Update the grid agents to reflect the current state of each Agent."""
        self.grid.grid_agents = {
            a_id: grid_object.GridAgent(
                agent, num_agents=self.config["num_agents"], scope=self.scope
            )
            for a_id, agent in self.env_agents.items()
        }

    @property
    def jax_step(self):
        """Raw step function for direct JIT/vmap usage.

        Signature: (EnvState, actions) -> (EnvState, obs, rewards, terminateds, truncateds, infos)

        Returns:
            The step function (JIT-compiled on JAX backend).

        Raises:
            RuntimeError: If backend is not 'jax' or reset() has not been called.
        """
        if self._backend != 'jax':
            raise RuntimeError("jax_step is only available with backend='jax'")
        if self._step_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_step")
        return self._step_fn

    @property
    def jax_reset(self):
        """Raw reset function for direct JIT/vmap usage.

        Signature: (rng_key) -> (EnvState, obs)

        Returns:
            The reset function (JIT-compiled on JAX backend).

        Raises:
            RuntimeError: If backend is not 'jax' or reset() has not been called.
        """
        if self._backend != 'jax':
            raise RuntimeError("jax_reset is only available with backend='jax'")
        if self._reset_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_reset")
        return self._reset_fn

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

    # ------------------------------------------------------------------
    # Hooks for subclass customization
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Observation and reward helpers
    # ------------------------------------------------------------------

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

        if self.t >= self.max_steps:
            truncateds = {agent_id: True for agent_id in self.agent_ids}
        else:
            truncateds = {agent_id: False for agent_id in self.agent_ids}

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
