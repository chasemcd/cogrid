"""End-to-end tests for the unified step pipeline.

Tests both numpy and JAX backends, verifying that step() and reset()
produce correct outputs and that build_step_fn/build_reset_fn
factories work with optional JIT compilation.
"""
import pytest
import numpy as np

_OVERCOOKED_FEATURES = [
    "agent_dir",
    "overcooked_inventory",
    "next_to_counter",
    "next_to_pot",
    "closest_onion",
    "closest_plate",
    "closest_plate_stack",
    "closest_onion_stack",
    "closest_onion_soup",
    "closest_delivery_zone",
    "closest_counter",
    "ordered_pot_features",
    "dist_to_other_players",
    "agent_position",
    "can_move_direction",
    "layout_id",
    "environment_layout",
]


def _setup_overcooked_config():
    """Create Overcooked env config on numpy backend, return all pipeline inputs.

    Creates a CrampedRoom env via registry (numpy), extracts array state,
    and builds scope config, lookup tables, feature function, and reward
    config.  Returns everything needed to call step()/reset()/build_*().
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.backend import set_backend

    _reset_backend_for_testing()
    set_backend("numpy")

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry
    from cogrid.core.grid_object import build_lookup_tables
    from cogrid.core.autowire import (
        build_scope_config_from_components,
        build_reward_config_from_components,
        build_feature_config_from_components,
    )

    env = registry.make("Overcooked-CrampedRoom-V0")
    env.reset(seed=42)
    array_state = env._array_state

    scope_config = build_scope_config_from_components("overcooked")
    lookup_tables = build_lookup_tables(scope="overcooked")

    n_agents = array_state["n_agents"]
    feature_config = build_feature_config_from_components("overcooked", _OVERCOOKED_FEATURES, n_agents=n_agents)
    feature_fn = feature_config["feature_fn"]

    type_ids = scope_config["type_ids"]

    reward_config = build_reward_config_from_components(
        "overcooked",
        n_agents=n_agents,
        type_ids=type_ids,
        action_pickup_drop_idx=4,
    )

    action_pickup_drop_idx = 4
    action_toggle_idx = 5
    max_steps = 1000

    # Extract pot_positions -- may be list or array
    pot_positions = array_state.get("pot_positions", [])
    if isinstance(pot_positions, list):
        if len(pot_positions) > 0:
            pot_positions = np.array(pot_positions, dtype=np.int32)
        else:
            pot_positions = np.zeros((0, 2), dtype=np.int32)
    else:
        pot_positions = np.array(pot_positions, dtype=np.int32)

    layout_arrays = {
        "wall_map": np.array(array_state["wall_map"], dtype=np.int32),
        "object_type_map": np.array(array_state["object_type_map"], dtype=np.int32),
        "object_state_map": np.array(array_state["object_state_map"], dtype=np.int32),
        "pot_contents": np.array(array_state["pot_contents"], dtype=np.int32),
        "pot_timer": np.array(array_state["pot_timer"], dtype=np.int32),
        "pot_positions": pot_positions,
    }

    spawn_positions = np.array(array_state["agent_pos"], dtype=np.int32)

    return {
        "scope_config": scope_config,
        "lookup_tables": lookup_tables,
        "feature_fn": feature_fn,
        "reward_config": reward_config,
        "action_pickup_drop_idx": action_pickup_drop_idx,
        "action_toggle_idx": action_toggle_idx,
        "max_steps": max_steps,
        "layout_arrays": layout_arrays,
        "spawn_positions": spawn_positions,
        "n_agents": n_agents,
    }


def test_step_numpy_backend():
    """step() and reset() produce correct outputs on the numpy backend."""
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.backend import set_backend
    from cogrid.core.step_pipeline import step, reset
    from cogrid.backend.env_state import EnvState

    cfg = _setup_overcooked_config()
    n_agents = cfg["n_agents"]

    try:
        _reset_backend_for_testing()
        set_backend("numpy")

        # Reset
        state, obs = reset(
            42,
            layout_arrays=cfg["layout_arrays"],
            spawn_positions=cfg["spawn_positions"],
            n_agents=n_agents,
            feature_fn=cfg["feature_fn"],
            scope_config=cfg["scope_config"],
            action_set="cardinal",
        )

        assert isinstance(state, EnvState)
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == n_agents
        assert len(obs.shape) == 2

        # Step with noop actions
        actions = np.zeros(n_agents, dtype=np.int32)
        result = step(
            state, actions,
            scope_config=cfg["scope_config"],
            lookup_tables=cfg["lookup_tables"],
            feature_fn=cfg["feature_fn"],
            reward_config=cfg["reward_config"],
            action_pickup_drop_idx=cfg["action_pickup_drop_idx"],
            action_toggle_idx=cfg["action_toggle_idx"],
            max_steps=cfg["max_steps"],
        )

        assert len(result) == 6
        state, obs, rewards, terminateds, truncateds, infos = result
        assert isinstance(state, EnvState)
        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
        assert terminateds.shape == (n_agents,)
        assert truncateds.shape == (n_agents,)

        # Run 5 more steps
        for _ in range(5):
            state, obs, rewards, terminateds, truncateds, infos = step(
                state, actions,
                scope_config=cfg["scope_config"],
                lookup_tables=cfg["lookup_tables"],
                feature_fn=cfg["feature_fn"],
                reward_config=cfg["reward_config"],
                action_pickup_drop_idx=cfg["action_pickup_drop_idx"],
                action_toggle_idx=cfg["action_toggle_idx"],
                max_steps=cfg["max_steps"],
            )

        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
    finally:
        _reset_backend_for_testing()


def test_step_jax_backend_eager():
    """step() and reset() produce correct outputs on the JAX backend (eager)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.backend import set_backend
    from cogrid.core.step_pipeline import step, reset
    from cogrid.backend.env_state import EnvState, register_envstate_pytree

    cfg = _setup_overcooked_config()
    n_agents = cfg["n_agents"]

    try:
        _reset_backend_for_testing()
        set_backend("jax")
        register_envstate_pytree()

        # Convert all arrays to JAX
        lookup_tables = {
            k: jnp.array(v, dtype=jnp.int32) for k, v in cfg["lookup_tables"].items()
        }
        layout_arrays = {
            k: jnp.array(v, dtype=jnp.int32) for k, v in cfg["layout_arrays"].items()
        }
        spawn_positions = jnp.array(cfg["spawn_positions"], dtype=jnp.int32)

        # Convert scope_config tables to JAX arrays
        scope_config = cfg["scope_config"]
        if "static_tables" in scope_config:
            st = scope_config["static_tables"]
            for key in st:
                if isinstance(st[key], np.ndarray):
                    st[key] = jnp.array(st[key], dtype=jnp.int32)
        if scope_config.get("interaction_tables") is not None:
            it = scope_config["interaction_tables"]
            for key in it:
                if isinstance(it[key], np.ndarray):
                    it[key] = jnp.array(it[key], dtype=jnp.int32)

        # Rebuild feature function on JAX backend
        from cogrid.core.autowire import build_feature_config_from_components
        feature_config = build_feature_config_from_components("overcooked", _OVERCOOKED_FEATURES, n_agents=n_agents)
        feature_fn = feature_config["feature_fn"]

        # Reset
        rng = jax.random.key(42)
        state, obs = reset(
            rng,
            layout_arrays=layout_arrays,
            spawn_positions=spawn_positions,
            n_agents=n_agents,
            feature_fn=feature_fn,
            scope_config=scope_config,
            action_set="cardinal",
        )

        assert isinstance(state, EnvState)
        assert obs.shape[0] == n_agents
        assert len(obs.shape) == 2

        # Step
        actions = jnp.zeros(n_agents, dtype=jnp.int32)
        state, obs, rewards, terminateds, truncateds, infos = step(
            state, actions,
            scope_config=scope_config,
            lookup_tables=lookup_tables,
            feature_fn=feature_fn,
            reward_config=cfg["reward_config"],
            action_pickup_drop_idx=cfg["action_pickup_drop_idx"],
            action_toggle_idx=cfg["action_toggle_idx"],
            max_steps=cfg["max_steps"],
        )

        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
        assert terminateds.shape == (n_agents,)
        assert truncateds.shape == (n_agents,)

        # Run 5 more steps
        for _ in range(5):
            state, obs, rewards, terminateds, truncateds, infos = step(
                state, actions,
                scope_config=scope_config,
                lookup_tables=lookup_tables,
                feature_fn=feature_fn,
                reward_config=cfg["reward_config"],
                action_pickup_drop_idx=cfg["action_pickup_drop_idx"],
                action_toggle_idx=cfg["action_toggle_idx"],
                max_steps=cfg["max_steps"],
            )

        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
    finally:
        _reset_backend_for_testing()


def test_build_step_fn_jit_compiles():
    """build_step_fn/build_reset_fn produce JIT-compiled functions that execute correctly."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.backend import set_backend
    from cogrid.core.step_pipeline import build_step_fn, build_reset_fn
    from cogrid.backend.env_state import EnvState, register_envstate_pytree, get_extra

    cfg = _setup_overcooked_config()
    n_agents = cfg["n_agents"]

    try:
        _reset_backend_for_testing()
        set_backend("jax")
        register_envstate_pytree()

        # Convert all arrays to JAX
        lookup_tables = {
            k: jnp.array(v, dtype=jnp.int32) for k, v in cfg["lookup_tables"].items()
        }
        layout_arrays = {
            k: jnp.array(v, dtype=jnp.int32) for k, v in cfg["layout_arrays"].items()
        }
        spawn_positions = jnp.array(cfg["spawn_positions"], dtype=jnp.int32)

        # Convert scope_config tables to JAX arrays
        scope_config = cfg["scope_config"]
        if "static_tables" in scope_config:
            st = scope_config["static_tables"]
            for key in st:
                if isinstance(st[key], np.ndarray):
                    st[key] = jnp.array(st[key], dtype=jnp.int32)
        if scope_config.get("interaction_tables") is not None:
            it = scope_config["interaction_tables"]
            for key in it:
                if isinstance(it[key], np.ndarray):
                    it[key] = jnp.array(it[key], dtype=jnp.int32)

        # Rebuild feature function on JAX backend
        from cogrid.core.autowire import build_feature_config_from_components
        feature_config = build_feature_config_from_components("overcooked", _OVERCOOKED_FEATURES, n_agents=n_agents)
        feature_fn = feature_config["feature_fn"]

        # Build factories (should auto-JIT on JAX backend)
        reset_fn = build_reset_fn(
            layout_arrays, spawn_positions, n_agents, feature_fn,
            scope_config, "cardinal",
        )
        step_fn = build_step_fn(
            scope_config, lookup_tables, feature_fn, cfg["reward_config"],
            cfg["action_pickup_drop_idx"], cfg["action_toggle_idx"],
            cfg["max_steps"],
        )

        # Reset -- first call triggers JIT compilation
        state, obs = reset_fn(jax.random.key(42))
        assert isinstance(state, EnvState)
        assert obs.shape[0] == n_agents

        # Step -- first call triggers JIT compilation
        actions = jnp.zeros(n_agents, dtype=jnp.int32)
        state, obs, rewards, terminateds, truncateds, infos = step_fn(state, actions)
        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
        assert terminateds.shape == (n_agents,)
        assert truncateds.shape == (n_agents,)

        # Run 10 more steps to verify repeated execution
        for _ in range(10):
            state, obs, rewards, terminateds, truncateds, infos = step_fn(state, actions)

        assert obs.shape[0] == n_agents
        assert rewards.shape == (n_agents,)
        assert terminateds.shape == (n_agents,)
        assert truncateds.shape == (n_agents,)

        # Verify extra_state persistence through JIT-compiled steps
        pc = get_extra(state, "pot_contents", scope="overcooked")
        pt = get_extra(state, "pot_timer", scope="overcooked")
        pp = get_extra(state, "pot_positions", scope="overcooked")
        assert pc.shape[1] == 3, f"Expected pot_contents cols=3, got {pc.shape[1]}"
        assert pt.shape[0] == pp.shape[0], "pot_timer and pot_positions row mismatch"
    finally:
        _reset_backend_for_testing()


