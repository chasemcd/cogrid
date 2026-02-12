"""End-to-end JAX step and reset functions for CoGrid environments.

Composes all Phase 2 JAX sub-functions (move_agents_jax,
process_interactions_jax, get_all_agent_obs_jax, compute_rewards_jax,
overcooked_tick_jax) into pure functions that operate on
:class:`~cogrid.backend.env_state.EnvState` pytrees.

The main API consists of five functions:

- :func:`envstate_to_dict` -- zero-cost conversion from EnvState to
  the dict format expected by observation and reward sub-functions.
- :func:`jax_step` -- end-to-end step composing tick, move, interact,
  obs, rewards, dones. Non-array args designed to be closed over.
- :func:`jax_reset` -- build initial EnvState from pre-computed layout
  arrays and compute initial observations.
- :func:`make_jitted_step` -- factory returning a JIT-compiled step
  closure with all static config closed over.
- :func:`make_jitted_reset` -- factory returning a JIT-compiled reset
  closure with all layout config closed over.

Usage::

    from cogrid.core.jax_step import make_jitted_step, make_jitted_reset

    jitted_reset = make_jitted_reset(layout_arrays, spawn_positions, ...)
    jitted_step = make_jitted_step(scope_config, lookup_tables, ...)

    state, obs = jitted_reset(jax.random.key(42))
    state, obs, rewards, done, infos = jitted_step(state, actions)
"""

from __future__ import annotations

import dataclasses


def envstate_to_dict(state) -> dict:
    """Convert an EnvState to the dict format used by sub-functions.

    Creates Python-level aliases for EnvState fields. This is zero-cost
    under JIT -- no array copies occur. The returned dict has the keys
    expected by :func:`get_all_agent_obs_jax` and
    :func:`compute_rewards_jax`.

    Extra_state entries are flattened into the dict with their scope
    prefix stripped (e.g. ``"overcooked.pot_contents"`` becomes
    ``"pot_contents"``). This maintains backward compatibility with
    sub-functions that expect pot arrays as top-level keys.

    Args:
        state: An :class:`EnvState` instance.

    Returns:
        Dict with keys: ``agent_pos``, ``agent_dir``, ``agent_inv``,
        ``wall_map``, ``object_type_map``, ``object_state_map``,
        plus any extra_state entries with scope prefix stripped.
    """
    result = {
        "agent_pos": state.agent_pos,
        "agent_dir": state.agent_dir,
        "agent_inv": state.agent_inv,
        "wall_map": state.wall_map,
        "object_type_map": state.object_type_map,
        "object_state_map": state.object_state_map,
    }
    # Merge extra_state into the dict for backward compatibility
    # with sub-functions that expect pot_contents, pot_timer, pot_positions
    # as top-level keys. Strip the scope prefix for now.
    for key, val in state.extra_state.items():
        # "overcooked.pot_contents" -> "pot_contents"
        short_key = key.split(".", 1)[-1] if "." in key else key
        result[short_key] = val
    return result


def jax_step(
    state,
    actions,
    *,
    scope_config,
    lookup_tables,
    feature_fn,
    reward_config,
    action_pickup_drop_idx,
    action_toggle_idx,
    max_steps,
):
    """End-to-end JAX step function operating on EnvState.

    Composes tick, movement, interactions, observations, rewards, and
    done computation into a single pure function. Non-array keyword
    arguments (``scope_config``, ``lookup_tables``, etc.) are designed
    to be closed over via :func:`make_jitted_step` or
    ``functools.partial``.

    The step ordering matches the existing ``CoGridEnv.step()`` pipeline
    (``cogrid_env.py`` lines 406-471):

    1. Capture ``prev_state`` before any mutations
    2. Tick (pot cooking timers via scope config handler)
    3. Movement (``move_agents_jax``)
    4. Interactions (``process_interactions_jax``)
    5. Observations (``get_all_agent_obs_jax``)
    6. Rewards (``compute_rewards_jax`` using prev_state)
    7. Done check (``time >= max_steps``)
    8. ``lax.stop_gradient`` on obs, rewards, done (JaxMARL pattern)

    Args:
        state: Current :class:`EnvState`.
        actions: int32 array of shape ``(n_agents,)`` with action indices.
        scope_config: Scope config dict (static, closed over).
        lookup_tables: Dict of property arrays (static, closed over).
        feature_fn: Composed JAX feature function (static, closed over).
        reward_config: Reward config dict for ``compute_rewards_jax``
            (static, closed over).
        action_pickup_drop_idx: int, PickupDrop action index.
        action_toggle_idx: int, Toggle action index.
        max_steps: int, maximum timesteps per episode.

    Returns:
        Tuple ``(state, obs, rewards, done, infos)`` where:
        - ``state``: Updated :class:`EnvState`.
        - ``obs``: float/int array of shape ``(n_agents, obs_dim)``.
        - ``rewards``: float32 array of shape ``(n_agents,)``.
        - ``done``: scalar bool.
        - ``infos``: empty dict ``{}``.
    """
    import jax.numpy as jnp
    import jax.lax as lax
    from cogrid.core.movement import move_agents_jax
    from cogrid.core.interactions import process_interactions_jax
    from cogrid.feature_space.array_features import get_all_agent_obs_jax
    from cogrid.envs.overcooked.array_rewards import compute_rewards_jax

    # a. Capture prev_state before ANY mutations (zero-cost, immutable)
    prev_state = state

    # b. Tick: update pot cooking timers
    tick_handler_jax = scope_config.get("tick_handler_jax") if scope_config else None
    if tick_handler_jax is not None:
        pot_contents = state.extra_state["overcooked.pot_contents"]
        pot_timer = state.extra_state["overcooked.pot_timer"]
        pot_positions = state.extra_state["overcooked.pot_positions"]
        n_pots = pot_positions.shape[0]

        pot_contents, pot_timer, pot_state = tick_handler_jax(
            pot_contents, pot_timer
        )
        # Write pot_state into object_state_map at pot positions
        osm = state.object_state_map

        def set_pot_state_fn(i, osm):
            r = pot_positions[i, 0]
            c = pot_positions[i, 1]
            return osm.at[r, c].set(pot_state[i])

        osm = lax.fori_loop(0, n_pots, set_pot_state_fn, osm)
        new_extra = {
            **state.extra_state,
            "overcooked.pot_contents": pot_contents,
            "overcooked.pot_timer": pot_timer,
        }
        state = dataclasses.replace(
            state, object_state_map=osm, extra_state=new_extra
        )

    # c. Movement
    new_pos, new_dir, new_key = move_agents_jax(
        state.agent_pos,
        state.agent_dir,
        actions,
        state.wall_map,
        state.object_type_map,
        lookup_tables["CAN_OVERLAP"],
        state.rng_key,
        state.action_set,
    )
    state = dataclasses.replace(
        state, agent_pos=new_pos, agent_dir=new_dir, rng_key=new_key
    )

    # d. Interactions
    dir_vec_table = jnp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32
    )
    agent_inv, otm, osm, pc, pt = process_interactions_jax(
        state.agent_pos,
        state.agent_dir,
        state.agent_inv,
        actions,
        state.object_type_map,
        state.object_state_map,
        lookup_tables,
        scope_config,
        dir_vec_table,
        action_pickup_drop_idx,
        action_toggle_idx,
        pot_contents=state.extra_state["overcooked.pot_contents"],
        pot_timer=state.extra_state["overcooked.pot_timer"],
        pot_positions=state.extra_state["overcooked.pot_positions"],
    )
    new_extra = {
        **state.extra_state,
        "overcooked.pot_contents": pc,
        "overcooked.pot_timer": pt,
    }
    state = dataclasses.replace(
        state,
        agent_inv=agent_inv,
        object_type_map=otm,
        object_state_map=osm,
        extra_state=new_extra,
        time=state.time + 1,
    )

    # e. Observations
    state_dict = envstate_to_dict(state)
    obs = get_all_agent_obs_jax(feature_fn, state_dict, state.n_agents)

    # f. Rewards
    prev_dict = envstate_to_dict(prev_state)
    rewards = compute_rewards_jax(prev_dict, state_dict, actions, reward_config)

    # g. Dones
    done = state.time >= max_steps

    # h. Stop gradient (JaxMARL pattern for RL)
    obs = lax.stop_gradient(obs)
    rewards = lax.stop_gradient(rewards)
    done = lax.stop_gradient(done)

    # i. Return
    return state, obs, rewards, done, {}


def jax_reset(
    rng_key,
    *,
    layout_arrays,
    spawn_positions,
    n_agents,
    feature_fn,
    scope_config,
    action_set,
    max_inv_size=1,
    pot_capacity=3,
    cooking_time=30,
):
    """Build initial EnvState from pre-computed layout arrays.

    All layout-dependent data (``layout_arrays``, ``spawn_positions``)
    is pre-computed at ``__init__`` time from the ASCII layout. This
    function only randomizes agent initial directions.

    Args:
        rng_key: JAX PRNG key.
        layout_arrays: Dict with keys ``wall_map``, ``object_type_map``,
            ``object_state_map``, ``pot_contents``, ``pot_timer``,
            ``pot_positions`` -- all pre-computed jnp arrays.
        spawn_positions: int32 array of shape ``(n_agents, 2)`` with
            fixed spawn points ``[row, col]``.
        n_agents: Number of agents.
        feature_fn: Composed JAX feature function for initial obs.
        scope_config: Scope config dict (used for consistency, not
            directly in reset).
        action_set: ``"cardinal"`` or ``"rotation"``.
        max_inv_size: Maximum inventory slots per agent (default 1).
        pot_capacity: Maximum items per pot (default 3).
        cooking_time: Initial pot cooking timer (default 30).

    Returns:
        Tuple ``(state, obs)`` where:
        - ``state``: Initial :class:`EnvState`.
        - ``obs``: Initial observations, shape ``(n_agents, obs_dim)``.
    """
    import jax
    import jax.numpy as jnp
    import jax.lax as lax
    from cogrid.backend.env_state import create_env_state
    from cogrid.feature_space.array_features import get_all_agent_obs_jax

    # Split key for agent directions and future use
    key, subkey = jax.random.split(rng_key)

    # Agent positions from fixed spawn points
    agent_pos = spawn_positions.astype(jnp.int32)

    # Random initial directions
    agent_dir = jax.random.randint(subkey, (n_agents,), 0, 4).astype(jnp.int32)

    # Empty inventory
    agent_inv = jnp.full((n_agents, max_inv_size), -1, dtype=jnp.int32)

    # Time starts at 0
    time = jnp.int32(0)

    # Extract layout arrays
    wall_map = layout_arrays["wall_map"]
    object_type_map = layout_arrays["object_type_map"]
    object_state_map = layout_arrays["object_state_map"]

    H, W = wall_map.shape

    # Build extra_state from pot arrays in layout_arrays
    extra_state = {
        "overcooked.pot_contents": layout_arrays["pot_contents"],
        "overcooked.pot_timer": layout_arrays["pot_timer"],
        "overcooked.pot_positions": layout_arrays["pot_positions"],
    }

    # Build EnvState
    state = create_env_state(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        agent_inv=agent_inv,
        wall_map=wall_map,
        object_type_map=object_type_map,
        object_state_map=object_state_map,
        extra_state=extra_state,
        rng_key=key,
        time=time,
        n_agents=n_agents,
        height=H,
        width=W,
        action_set=action_set,
    )

    # Compute initial observations
    state_dict = envstate_to_dict(state)
    obs = get_all_agent_obs_jax(feature_fn, state_dict, n_agents)
    obs = lax.stop_gradient(obs)

    return state, obs


def make_jitted_step(
    scope_config,
    lookup_tables,
    feature_fn,
    reward_config,
    action_pickup_drop_idx,
    action_toggle_idx,
    max_steps,
):
    """Create a JIT-compiled step closure with all static config closed over.

    Returns a function with signature
    ``(state: EnvState, actions: jnp.ndarray) -> (EnvState, obs, rewards, done, infos)``.

    Args:
        scope_config: Scope config dict.
        lookup_tables: Dict of property arrays.
        feature_fn: Composed JAX feature function.
        reward_config: Reward config dict.
        action_pickup_drop_idx: int, PickupDrop action index.
        action_toggle_idx: int, Toggle action index.
        max_steps: int, maximum timesteps per episode.

    Returns:
        JIT-compiled step function.
    """
    import jax

    def step_fn(state, actions):
        return jax_step(
            state,
            actions,
            scope_config=scope_config,
            lookup_tables=lookup_tables,
            feature_fn=feature_fn,
            reward_config=reward_config,
            action_pickup_drop_idx=action_pickup_drop_idx,
            action_toggle_idx=action_toggle_idx,
            max_steps=max_steps,
        )

    return jax.jit(step_fn)


def make_jitted_reset(
    layout_arrays,
    spawn_positions,
    n_agents,
    feature_fn,
    scope_config,
    action_set,
    **kwargs,
):
    """Create a JIT-compiled reset closure with all layout config closed over.

    Returns a function with signature ``(rng_key) -> (EnvState, obs)``.

    Args:
        layout_arrays: Dict of pre-computed layout arrays.
        spawn_positions: int32 array of shape ``(n_agents, 2)``.
        n_agents: Number of agents.
        feature_fn: Composed JAX feature function.
        scope_config: Scope config dict.
        action_set: ``"cardinal"`` or ``"rotation"``.
        **kwargs: Additional keyword args forwarded to ``jax_reset``
            (e.g., ``max_inv_size``, ``pot_capacity``, ``cooking_time``).

    Returns:
        JIT-compiled reset function.
    """
    import jax

    def reset_fn(rng_key):
        return jax_reset(
            rng_key,
            layout_arrays=layout_arrays,
            spawn_positions=spawn_positions,
            n_agents=n_agents,
            feature_fn=feature_fn,
            scope_config=scope_config,
            action_set=action_set,
            **kwargs,
        )

    return jax.jit(reset_fn)


if __name__ == "__main__":
    # Smoke test: end-to-end JIT compilation on Overcooked cramped_room
    #
    # Strategy: Create the env using numpy backend (default), extract all
    # needed arrays and configs, then convert to JAX arrays for JIT testing.
    # The scope config's static tables (numpy arrays) are auto-converted
    # by JAX during tracing.
    import cogrid.envs  # trigger registration
    from cogrid.envs import registry
    from cogrid.core.grid_object import build_lookup_tables
    from cogrid.envs.overcooked.array_config import build_overcooked_scope_config

    # --- Step 1: Create numpy env, extract layout arrays ---
    env = registry.make("Overcooked-CrampedRoom-V0")
    env.reset(seed=42)
    array_state = env._array_state

    # Build scope config and lookup tables (uses numpy backend, which is fine)
    scope_config = build_overcooked_scope_config()
    lookup_tables = build_lookup_tables(scope="overcooked")

    # --- Step 2: Convert everything to JAX arrays ---
    import jax
    import jax.numpy as jnp

    # Register EnvState as pytree (required for JAX operations)
    from cogrid.backend.env_state import register_envstate_pytree
    register_envstate_pytree()

    # Convert lookup tables to JAX arrays
    for key in lookup_tables:
        lookup_tables[key] = jnp.array(lookup_tables[key], dtype=jnp.int32)

    # Convert static_tables in scope_config to JAX arrays
    # These are used inside lax.fori_loop bodies and must be JAX arrays,
    # not numpy arrays, to avoid TracerArrayConversionError.
    if "static_tables" in scope_config:
        import numpy as np
        st = scope_config["static_tables"]
        for key in st:
            if isinstance(st[key], np.ndarray):
                st[key] = jnp.array(st[key], dtype=jnp.int32)

    # Also convert interaction_tables arrays
    if "interaction_tables" in scope_config:
        import numpy as np
        it = scope_config["interaction_tables"]
        for key in it:
            if isinstance(it[key], np.ndarray):
                it[key] = jnp.array(it[key], dtype=jnp.int32)

    # Convert pot_positions from list to array
    pot_positions = array_state.get("pot_positions", [])
    if isinstance(pot_positions, list):
        if len(pot_positions) > 0:
            pot_positions = jnp.array(pot_positions, dtype=jnp.int32)
        else:
            pot_positions = jnp.zeros((0, 2), dtype=jnp.int32)
    else:
        pot_positions = jnp.array(pot_positions, dtype=jnp.int32)

    # Build layout arrays dict (all converted to jnp arrays)
    layout_arrays = {
        "wall_map": jnp.array(array_state["wall_map"], dtype=jnp.int32),
        "object_type_map": jnp.array(array_state["object_type_map"], dtype=jnp.int32),
        "object_state_map": jnp.array(array_state["object_state_map"], dtype=jnp.int32),
        "pot_contents": jnp.array(array_state["pot_contents"], dtype=jnp.int32),
        "pot_timer": jnp.array(array_state["pot_timer"], dtype=jnp.int32),
        "pot_positions": pot_positions,
    }

    # Extract spawn positions
    spawn_positions = jnp.array(array_state["agent_pos"], dtype=jnp.int32)
    n_agents = array_state["n_agents"]

    # --- Step 3: Build feature function ---
    # build_feature_fn_jax imports jax internally, works regardless of backend
    from cogrid.feature_space.array_features import build_feature_fn_jax

    feature_names = ["agent_position", "agent_dir", "full_map_encoding",
                     "can_move_direction", "inventory"]
    feature_fn = build_feature_fn_jax(feature_names, scope="overcooked")

    # --- Step 4: Build reward config ---
    type_ids = scope_config["type_ids"]
    reward_config = {
        "type_ids": type_ids,
        "n_agents": n_agents,
        "rewards": [
            {"fn": "delivery", "coefficient": 1.0, "common_reward": True},
        ],
        "action_pickup_drop_idx": 4,
    }

    # Action indices for cardinal actions
    action_pickup_drop_idx = 4
    action_toggle_idx = 5
    max_steps = 1000

    # --- Step 5: Create JIT-compiled functions ---
    print("Building JIT-compiled reset...")
    jitted_reset = make_jitted_reset(
        layout_arrays, spawn_positions, n_agents, feature_fn,
        scope_config, "cardinal",
    )

    print("Building JIT-compiled step...")
    jitted_step = make_jitted_step(
        scope_config, lookup_tables, feature_fn, reward_config,
        action_pickup_drop_idx, action_toggle_idx, max_steps,
    )

    # --- Step 6: Run reset ---
    print("Running jitted_reset...")
    state, obs = jitted_reset(jax.random.key(42))
    print(f"  state.agent_pos shape: {state.agent_pos.shape}")
    print(f"  state.agent_dir shape: {state.agent_dir.shape}")
    print(f"  state.time: {state.time}")
    print(f"  obs shape: {obs.shape}")

    # --- Step 7: Run one step ---
    print("Running jitted_step (first call, includes JIT compile)...")
    actions = jnp.zeros(n_agents, dtype=jnp.int32)
    state, obs, rewards, done, infos = jitted_step(state, actions)
    print(f"  obs shape: {obs.shape}")
    print(f"  rewards shape: {rewards.shape}")
    print(f"  rewards: {rewards}")
    print(f"  done: {done}")
    print(f"  state.time: {state.time}")

    # --- Step 8: Run 10 steps in sequence ---
    print("Running 10 steps in sequence...")
    for i in range(10):
        actions = jnp.zeros(n_agents, dtype=jnp.int32)  # Noop for all
        state, obs, rewards, done, infos = jitted_step(state, actions)
    print(f"  After 10 more steps: time={state.time}, done={done}")
    print(f"  obs shape: {obs.shape}")
    print(f"  rewards shape: {rewards.shape}")

    # --- Step 9: Verify shapes ---
    assert obs.shape[0] == n_agents, f"Expected obs[0]={n_agents}, got {obs.shape[0]}"
    assert rewards.shape == (n_agents,), f"Expected rewards shape ({n_agents},), got {rewards.shape}"
    assert done.shape == (), f"Expected scalar done, got shape {done.shape}"

    print("\nSMOKE TEST PASSED: End-to-end JIT compilation successful.")
    print(f"  Reset + 11 steps completed without ConcretizationTypeError.")
    print(f"  obs: ({obs.shape[0]}, {obs.shape[1]}), rewards: {rewards.shape}, done: scalar bool")
