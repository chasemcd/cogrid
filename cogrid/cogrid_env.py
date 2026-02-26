"""CoGrid environment: stateful wrapper around the functional step/reset pipeline."""

import copy

import numpy as np
import pettingzoo
from gymnasium import spaces

from cogrid import constants

# Vectorized components
from cogrid.backend import get_backend, set_backend
from cogrid.core import actions as grid_actions
from cogrid.core import agent, directions, grid, grid_object, grid_utils, layouts, typing
from cogrid.core.agent import create_agent_arrays
from cogrid.core.constants import CoreConstants
from cogrid.core.grid_object import build_lookup_tables
from cogrid.core.grid_utils import layout_to_state
from cogrid.rendering import EnvRenderer

RNG = RandomNumberGenerator = np.random.Generator


class CoGridEnv(pettingzoo.ParallelEnv):
    """Thin stateful wrapper around the functional step/reset pipeline.

    Both numpy and JAX backends delegate to ``build_step_fn``/``build_reset_fn``.
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
        backend: str | None = None,
        **kwargs,
    ):
        """Initialize the CoGrid environment.

        The first environment created sets the global backend; subsequent
        envs must use the same backend or a RuntimeError is raised.

        If ``backend`` is None, the already-set backend is used (defaulting
        to ``"numpy"`` if ``set_backend()`` was never called). If ``backend``
        is provided explicitly, it is forwarded to ``set_backend()``.
        """
        super().__init__()
        self._np_random: np.random.Generator | None = None  # set in reset()

        if backend is not None:
            set_backend(backend)
        self._backend = backend if backend is not None else get_backend()
        self.config = config
        self.name = config["name"]
        self.cumulative_score = 0
        self.max_steps = config["max_steps"]
        self.roles = config.get("roles", True)
        self.agent_class = agent_class or agent.Agent
        self.t = 0

        if "features" not in config or not isinstance(config["features"], list):
            raise ValueError("config['features'] must be a list of feature names.")

        self._init_rendering(render_mode, kwargs)
        self._init_grid(config)
        self._init_agents(config)
        self._init_action_space(config)
        self._init_vectorized_infrastructure()
        self._init_jax_arrays()

    def _init_rendering(self, render_mode, kwargs):
        """Set up rendering attributes and optional EnvRenderer."""
        self.render_mode = render_mode
        self.render_message = kwargs.get("render_message") or self.metadata["render_message"]
        self.tile_size = CoreConstants.TilePixels
        self.screen_size = kwargs.get("screen_size") or self.metadata["screen_size"]
        self._renderer = (
            EnvRenderer(
                name=self.name,
                screen_size=self.screen_size,
                render_fps=self.metadata["render_fps"],
            )
            if render_mode
            else None
        )
        self.visualizer = None

    def _init_grid(self, config):
        """Initialize grid, spawn points, and shape from config."""
        self.scope: str = config.get("scope", "global")
        self.grid: grid.Grid | None = None
        self.spawn_points: list = []
        self.current_layout_id: str | None = None
        self._gen_grid()
        self.shape = (self.grid.height, self.grid.width)

    def _init_agents(self, config):
        """Initialize agent bookkeeping: IDs, env_agents dict, rewards."""
        self.possible_agents = [i for i in range(config["num_agents"])]
        self.agents = copy.copy(self.possible_agents)
        self._agent_ids: set[typing.AgentID] = set(self.agents)
        self.env_agents: dict[typing.AgentID, agent.Agent] = {i: None for i in self.agents}
        self.per_agent_reward: dict[typing.AgentID, float] = self.get_empty_reward_dict()
        self.per_component_reward: dict[str, dict[typing.AgentID, float]] = {}
        self.reward_this_step = self.get_empty_reward_dict()
        self.agent_view_size = self.config.get("agent_view_size", 7)

    def _init_action_space(self, config):
        """Parse action set from config and build per-agent action spaces."""
        action_str = config.get("action_set")
        if action_str == "rotation_actions":
            self.action_set = grid_actions.ActionSets.RotationActions
        elif action_str == "cardinal_actions":
            self.action_set = grid_actions.ActionSets.CardinalActions
        else:
            raise ValueError(f"Invalid or None action set string: {action_str}.")

        self.action_spaces = {
            a_id: spaces.Discrete(len(self.action_set)) for a_id in self.agent_ids
        }

        self._action_pickup_drop_idx = self.action_set.index(grid_actions.Actions.PickupDrop)
        self._action_toggle_idx = self.action_set.index(grid_actions.Actions.Toggle)
        self.prev_actions = None

    def _init_vectorized_infrastructure(self):
        """Build lookup tables, scope config, reward config, and pipeline placeholders."""
        self._lookup_tables = build_lookup_tables(scope=self.scope)

        from cogrid.core.autowire import (
            build_reward_config,
            build_scope_config_from_components,
        )
        from cogrid.core.component_registry import get_layout_index, get_pre_compose_hook

        # Run pre-compose hook before scope config (e.g. to set layout index).
        pre_hook = get_pre_compose_hook(self.scope)
        if pre_hook is not None:
            _layout_idx = get_layout_index(self.scope, self.current_layout_id)
            pre_hook(layout_idx=_layout_idx, scope=self.scope, env_config=self.config)

        self._scope_config = build_scope_config_from_components(self.scope)
        if "interaction_fn" in self.config:
            self._scope_config["interaction_fn"] = self.config["interaction_fn"]
        self._type_ids = self._scope_config["type_ids"]
        self._interaction_tables = self._scope_config.get("interaction_tables")

        self._state = None
        self._feature_fn = None

        reward_instances = self.config.get("rewards", [])
        self._reward_config = build_reward_config(
            reward_instances,
            n_agents=self.config["num_agents"],
            type_ids=self._type_ids,
            action_pickup_drop_idx=self._action_pickup_drop_idx,
            action_toggle_idx=self._action_toggle_idx,
            static_tables=self._scope_config.get("static_tables"),
        )

        self._agent_id_order = sorted(self.possible_agents)
        self._terminated_fn = self.config.get("terminated_fn")

        self._step_fn = None
        self._reset_fn = None
        self._env_state = None

    def _init_jax_arrays(self):
        """Convert lookup and scope config tables to JAX arrays (JAX backend only)."""
        if self._backend != "jax":
            return

        import jax.numpy as jnp

        from cogrid.backend.env_state import register_envstate_pytree

        register_envstate_pytree()

        for key in self._lookup_tables:
            self._lookup_tables[key] = jnp.array(self._lookup_tables[key], dtype=jnp.int32)

        if "static_tables" in self._scope_config:
            import numpy as _np

            st = self._scope_config["static_tables"]
            for key in st:
                if isinstance(st[key], _np.ndarray):
                    st[key] = jnp.array(st[key], dtype=jnp.int32)

        if self._scope_config.get("interaction_tables") is not None:
            import numpy as _np

            it = self._scope_config["interaction_tables"]
            for key in it:
                if isinstance(it[key], _np.ndarray):
                    it[key] = jnp.array(it[key], dtype=jnp.int32)

    def _gen_grid(self) -> None:
        """Parse the ASCII layout into a Grid, extracting spawn points."""
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
        """Generate grid and state arrays from config layout or layout_fn."""
        grid_config: dict = self.config.get("grid", {})
        layout_name = grid_config.get("layout", None)
        layout_fn = grid_config.get("layout_fn", None)

        if layout_name is None and layout_fn is None:
            raise ValueError(
                "Must provide either a `layout` name or layout-generating"
                " function in config['grid']"
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
        """Create a PCG64-backed Generator from an optional integer seed."""
        if seed is not None and not (isinstance(seed, int) and 0 <= seed):
            if isinstance(seed, int) is False:
                raise ValueError(f"Seed must be a python integer, actual type: {type(seed)}")
            else:
                raise ValueError(f"Seed must be greater or equal to zero, actual value: {seed}")

        seed_seq = np.random.SeedSequence(seed)
        np_seed = seed_seq.entropy
        rng = RandomNumberGenerator(np.random.PCG64(seed_seq))
        return rng, np_seed

    @property
    def np_random(self) -> np.random.Generator:
        """Lazily initialize and return the numpy RNG."""
        if self._np_random is None:
            self._np_random, _ = self._set_np_random()

        return self._np_random

    def action_space(self, agent: typing.AgentID) -> spaces.Space:
        """Return the action space for the given agent."""
        return self.action_spaces[agent]

    def set_terminated_fn(self, fn):
        """Set a per-agent termination function.

        Must be called before the first ``reset()`` so the function
        is closed over by the JIT-compiled step pipeline.

        ``fn(prev_state, state, reward_config) -> (n_agents,) bool``
        """
        self._terminated_fn = fn

    def reset(
        self,
        *,
        seed: int | None = 42,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[dict[typing.AgentID, typing.ObsType], dict[str, typing.Any]]:
        """Reset the environment and return initial observations.

        Builds step/reset pipeline on first call, then delegates to
        the functional reset.
        """
        self._reset_agents(seed)
        self._build_state()
        layout_arrays, spawn_positions, action_set_name = self._build_layout_arrays()
        obs = self._build_pipeline(layout_arrays, spawn_positions, action_set_name, seed)

        if self.render_mode is not None:
            self._sync_objects_from_state()
            self.render()

        return obs, {agent_id: {} for agent_id in self.agent_ids}

    def _reset_agents(self, seed):
        """Reset RNG, regenerate grid, and re-initialize agents."""
        if seed is not None:
            self._np_random, _ = self._set_np_random(seed=seed)

        self.agents = copy.copy(self.possible_agents)
        self._gen_grid()

        self.env_agents = {}
        self.setup_agents()

        self.prev_actions = {a_id: grid_actions.Actions.Noop for a_id in self.agent_ids}

        self.t = 0
        self.on_reset()
        self.update_grid_agents()

    def _build_state(self):
        """Build array state from grid layout, extra state, and agent arrays."""
        self._state = layout_to_state(self.grid, scope=self.scope, scope_config=self._scope_config)

        extra_state_builder = self._scope_config.get("extra_state_builder")
        if extra_state_builder is not None:
            extra = extra_state_builder(self._state, self.scope)
            scope_prefix = f"{self.scope}."
            stripped = {
                (k[len(scope_prefix) :] if k.startswith(scope_prefix) else k): v
                for k, v in extra.items()
            }
            self._state.update(stripped)

        agent_arrays = create_agent_arrays(self.env_agents, scope=self.scope)
        self._state.update(agent_arrays)
        self.cumulative_score = 0

    def _build_layout_arrays(self):
        """Convert array state to typed layout arrays for the active backend."""
        if self._backend == "jax":
            import jax.numpy as jnp

            xp = jnp
        else:
            xp = np

        skip_keys = {"agent_pos", "agent_dir", "agent_inv", "spawn_points"}
        layout_arrays = {}
        for key, val in self._state.items():
            if key in skip_keys:
                continue
            if isinstance(val, list):
                val = (
                    np.array(val, dtype=np.int32)
                    if len(val) > 0
                    else np.zeros((0, 2), dtype=np.int32)
                )
            layout_arrays[key] = xp.array(val, dtype=xp.int32)

        spawn_positions = xp.array(self._state["agent_pos"], dtype=xp.int32)

        if self.action_set == grid_actions.ActionSets.CardinalActions:
            action_set_name = "cardinal"
        else:
            action_set_name = "rotation"

        return layout_arrays, spawn_positions, action_set_name

    def _build_pipeline(self, layout_arrays, spawn_positions, action_set_name, seed):
        """Build feature/step/reset pipeline and run initial reset."""
        from cogrid.core.autowire import build_feature_config_from_components
        from cogrid.core.component_registry import get_layout_index
        from cogrid.core.step_pipeline import build_reset_fn, build_step_fn

        n_agents = self.config["num_agents"]
        _layout_idx = get_layout_index(self.scope, self.current_layout_id)

        feature_config = build_feature_config_from_components(
            self.scope,
            self.config["features"],
            n_agents=n_agents,
            layout_idx=_layout_idx,
            env_config=self.config,
        )
        self._feature_fn = feature_config["feature_fn"]

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

        if self._backend == "jax":
            import jax

            rng = jax.random.key(seed if seed is not None else 42)
        else:
            rng = seed if seed is not None else 42

        obs_arr, self._env_state, _ = self._reset_fn(rng)

        obs = {aid: np.array(obs_arr[i]) for i, aid in enumerate(self._agent_id_order)}

        self.per_agent_reward = self.get_empty_reward_dict()
        self.per_component_reward = {}
        return obs

    def _action_idx_to_str(
        self, actions: dict[typing.AgentID, int | str]
    ) -> dict[typing.AgentID, str]:
        """Convert action indices to their string names."""
        str_actions = {
            a_id: (self.action_set[action] if not isinstance(action, str) else action)
            for a_id, action in actions.items()
        }

        return str_actions

    def step(
        self, actions: dict[typing.AgentID, typing.ActionType]
    ) -> tuple[
        dict[typing.AgentID, typing.ObsType],
        dict[typing.AgentID, float],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, bool],
        dict[typing.AgentID, dict[typing.Any, typing.Any]],
    ]:
        """Advance one timestep via the unified step pipeline.

        Converts dict actions to an ordered array, delegates to
        ``self._step_fn``, and converts outputs back to PettingZoo dicts.
        """
        # Select array module
        if self._backend == "jax":
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
        obs_arr, self._env_state, rewards_arr, terminateds_arr, truncateds_arr, infos = (
            self._step_fn(self._env_state, actions_arr)
        )

        # Convert to PettingZoo dict format
        obs = {aid: np.array(obs_arr[i]) for i, aid in enumerate(self._agent_id_order)}
        rewards = {aid: float(rewards_arr[i]) for i, aid in enumerate(self._agent_id_order)}

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
        """Export a JSON-serializable snapshot of the full environment state.

        Returns plain Python lists/scalars suitable for ``json.dumps()``.
        Restorable via :meth:`set_state`. Must be called after :meth:`reset`.
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
            "extra_state": {k: np.array(v).tolist() for k, v in state.extra_state.items()},
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
        """Restore from a dict produced by :meth:`get_state`.

        Must be called after :meth:`reset` (pipeline must be initialized).
        Arrays are converted to the active backend's type.
        """
        if self._step_fn is None:
            raise RuntimeError("Must call reset() before set_state()")

        from cogrid.backend.env_state import create_env_state

        if self._backend == "jax":
            import jax.numpy as jnp

            xp = jnp
        else:
            xp = np

        extra_state = {k: xp.array(v, dtype=xp.int32) for k, v in snapshot["extra_state"].items()}

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

        Writes array state back to Agent/Grid objects so the tile renderer
        reflects current simulation state. Scope-specific rendering is
        delegated to the ``render_sync`` hook in scope_config.

        Only called when render_mode is not None.
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
                agent_obj.inventory = [make_object(obj_id, scope=self.scope)] if obj_id else []

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
            a_id: grid_object.GridAgent(agent, n_agents=self.config["num_agents"], scope=self.scope)
            for a_id, agent in self.env_agents.items()
        }

    @property
    def jax_step(self):
        """Raw JIT-compiled step function for direct JIT/vmap usage.

        ``(EnvState, actions) -> (obs, EnvState, rewards, terminateds, truncateds, infos)``
        """
        if self._backend != "jax":
            raise RuntimeError("jax_step is only available with backend='jax'")
        if self._step_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_step")
        return self._step_fn

    @property
    def jax_reset(self):
        """Raw JIT-compiled reset function for direct JIT/vmap usage.

        ``(rng_key) -> (obs, EnvState, infos)``
        """
        if self._backend != "jax":
            raise RuntimeError("jax_reset is only available with backend='jax'")
        if self._reset_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_reset")
        return self._reset_fn

    def setup_agents(self):
        """Set up agents using the default agent factory."""
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

    def on_interact(self, actions: dict[typing.AgentID, typing.ActionType]) -> None:
        """Hook for subclass logic after agents interact with the environment."""
        pass

    def on_toggle(self, agent_id: typing.AgentID) -> None:
        """Hook for subclass logic after an agent toggles an object."""
        pass

    def on_pickup_drop(self, agent_id: typing.AgentID) -> None:
        """Hook for subclass logic after an agent picks up or drops an object."""
        pass

    def on_reset(self) -> None:
        """Hook for subclasses to implement custom logic after the environment is reset."""
        pass

    def on_step(self) -> None:
        """Hook for subclasses to implement custom logic after each step."""
        pass

    def on_move(self, agent_id: typing.AgentID) -> None:
        """Hook for subclass logic after an agent moves."""
        pass

    # ------------------------------------------------------------------
    # Observation and reward helpers
    # ------------------------------------------------------------------

    def get_infos(self, **kwargs) -> dict[typing.AgentID, dict[typing.Any, typing.Any]]:
        """Build per-agent info dicts from keyword argument dicts."""
        infos = {agent_id: {} for agent_id in self._agent_ids}
        for info_key, info_dict in kwargs.items():
            for agent_id, val in info_dict.items():
                assert agent_id in self._agent_ids, (
                    f"Must pass dicts keyed by AgentIDs to get_infos(), got invalid key: {agent_id}"
                )
                infos[agent_id][info_key] = val
        return infos

    def get_empty_reward_dict(self) -> dict[typing.AgentID, float]:
        """Return a reward dict with all agents set to 0."""
        return {a_id: 0 for a_id in self.agent_ids}

    @property
    def map_with_agents(self) -> np.ndarray:
        """Return the encoded grid with agents overlaid as numeric IDs."""
        grid_encoding = self.grid.encode(encode_char=True, scope=self.scope)
        grid = grid_encoding[:, :, 0]

        for a_id, ag in self.env_agents.items():
            if ag is not None:  # will be None before being set by subclassed env
                grid[ag.pos[0], ag.pos[1]] = self.id_to_numeric(a_id)

        return grid

    def select_spawn_point(self) -> tuple[int, int]:
        """Pop a spawn point from the queue, or pick a random free cell."""
        if self.spawn_points:
            return self.spawn_points.pop(0)

        available_spawns = self.available_positions
        return self.np_random.choice(available_spawns)

    @property
    def available_positions(self) -> list[tuple[int, int]]:
        """Return grid cells that are empty and not occupied by an agent."""
        spawns = []
        for r in range(self.grid.height):
            for c in range(self.grid.width):
                cell = self.grid.get(row=r, col=c)
                if (r, c) not in self.agent_pos and (not cell or cell.object_id == "floor"):
                    spawns.append((r, c))
        return spawns

    def put_obj(self, obj: grid_object.GridObj, row: int, col: int):
        """Place an object at (row, col) and set its init_pos."""
        self.grid.set(row=row, col=col, obj=obj)
        obj.pos = (row, col)
        obj.init_pos = (row, col)

    def get_view_exts(
        self, agent_id: typing.AgentID, agent_view_size: int = None
    ) -> tuple[int, int, int, int]:
        """Return (topX, topY, botX, botY) of the agent's visible tile square.

        Bottom extent indices are exclusive.
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

    def gen_obs_grid(self, agent_id: typing.AgentID, agent_view_size: int = None) -> grid.Grid:
        """Return the rotated sub-grid and visibility mask for an agent's POV."""
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

        # NOTE: In Minigrid, they replace the agent's position with the item
        #   they're carrying. We don't do that here. Rather, we'll provide an
        #   additional observation space that represents the item(s) in inventory.

        return grid, vis_mask

    def get_pov_render(
        self,
        agent_id: typing.AgentID,
        tile_size: int = CoreConstants.TilePixels,
    ) -> np.ndarray:
        """Render a specific agent's POV as an RGB array."""
        grid, vis_mask = self.gen_obs_grid(agent_id)
        img = grid.render(
            tile_size,
            highlight_mask=vis_mask,
        )
        return img

    def get_full_render(
        self, highlight: bool = False, tile_size: int = CoreConstants.TilePixels
    ) -> np.ndarray:
        """Render the full grid, optionally highlighting each agent's visible region."""
        if highlight:
            highlight_mask = np.zeros(shape=(self.grid.height, self.grid.width), dtype=bool)
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
                        abs_row, abs_col = top_left - (f_vec * vis_row) + (r_vec * vis_col)

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
        agent_pov: typing.AgentID | None = None,
    ) -> np.ndarray:
        """Return RGB image of the full environment or a single agent's POV."""
        if agent_pov:
            frame = self.get_pov_render(agent_id=self.agent_pov, tile_size=tile_size)
        else:
            frame = self.get_full_render(highlight=highlight, tile_size=tile_size)

        return frame

    def render(self) -> None | np.ndarray:
        """Render the environment (human window or rgb_array return)."""
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
            self._renderer.render_human(img, self.cumulative_score, self.render_message)
        elif self.render_mode == "rgb_array":
            return img

    def close(self):
        """Close the renderer if active."""
        if self._renderer is not None:
            self._renderer.close()

    def get_action_mask(self, agent_id):
        """Return action mask for the given agent (not implemented)."""
        raise NotImplementedError

    @property
    def agent_ids(self) -> list:
        """Return list of active agent IDs."""
        return list(self.env_agents.keys())

    @property
    def agent_pos(self) -> list:
        """Return list of agent positions."""
        return [tuple(agent.pos) for agent in self.env_agents.values() if agent is not None]

    def id_to_numeric(self, agent_id) -> str:
        """Convert agent id to its numeric string representation.

        For example, agent-0 -> 1, agent-1 -> 2, etc.
        """
        agent = self.env_agents[agent_id]
        return str(agent.agent_number)
