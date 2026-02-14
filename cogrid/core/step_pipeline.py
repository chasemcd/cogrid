"""Unified step and reset pipeline for CoGrid environments.

Replaces the JAX-only ``jax_step.py`` with backend-agnostic functions
that work on both numpy and JAX via ``xp``.  Backend-specific code
(RNG splitting, ``stop_gradient``) is isolated to ``get_backend()``
conditionals -- everything else uses the ``xp`` array namespace.

The three public functions compose all Phase 5-7 sub-functions
(``move_agents``, ``process_interactions``, ``get_all_agent_obs``,
``compute_rewards``, scope-config tick handler) into a pure pipeline
operating on :class:`~cogrid.backend.env_state.EnvState`.

Public API:

- :func:`envstate_to_dict` -- zero-cost EnvState-to-dict conversion.
- :func:`step` -- end-to-end step: tick, move, interact, obs, rewards,
  dones.
- :func:`reset` -- build initial EnvState and compute initial
  observations.
- :func:`build_step_fn` -- init-time factory: closes over static config,
  returns ``(state, actions) -> ...`` closure (auto-JIT on JAX).
- :func:`build_reset_fn` -- init-time factory: closes over layout config,
  returns ``(rng) -> (state, obs)`` closure (auto-JIT on JAX).

Usage::

    from cogrid.core.step_pipeline import step, reset, envstate_to_dict
    from cogrid.core.step_pipeline import build_step_fn, build_reset_fn
"""

from __future__ import annotations

import dataclasses


def envstate_to_dict(state) -> dict:
    """Convert an EnvState to the dict format used by sub-functions.

    Creates Python-level aliases for EnvState fields. This is zero-cost
    under JIT -- no array copies occur. The returned dict has the keys
    expected by :func:`get_all_agent_obs` and
    :func:`compute_rewards`.

    Extra_state entries are flattened into the dict with their scope
    prefix stripped (e.g. ``"scope.key"`` becomes ``"key"``).
    This maintains backward compatibility with sub-functions that
    expect scope-specific arrays as top-level keys.

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
    # with sub-functions that expect scope-specific arrays as top-level
    # keys. Strip the scope prefix ("scope.key" -> "key").
    for key, val in state.extra_state.items():
        short_key = key.split(".", 1)[-1] if "." in key else key
        result[short_key] = val
    return result


def step(
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
    terminated_fn=None,
):
    """End-to-end step function operating on EnvState.

    Composes tick, movement, interactions, observations, rewards, and
    done computation into a single pure function. Non-array keyword
    arguments are designed to be closed over via ``functools.partial``
    or a ``build_step_fn`` factory.

    Uses ``xp`` for all array operations.  Backend-specific code is
    limited to two conditional blocks: RNG for movement priority and
    ``stop_gradient`` for RL training.

    Step ordering:

    1. Capture ``prev_state`` before any mutations
    2. Tick (object timers via scope config handler)
    3. Movement (``move_agents``)
    4. Interactions (``process_interactions``)
    5. Observations (``get_all_agent_obs``)
    6. Rewards (``compute_rewards`` using prev_state)
    7. Terminateds/truncateds (``terminated_fn`` + ``time >= max_steps``)
    8. Stop gradient (JAX only)

    Args:
        state: Current :class:`EnvState`.
        actions: int32 array of shape ``(n_agents,)`` with action indices.
        scope_config: Scope config dict (static, closed over).
        lookup_tables: Dict of property arrays (static, closed over).
        feature_fn: Composed feature function (static, closed over).
        reward_config: Reward config dict (static, closed over). Must
            contain a ``"compute_fn"`` key pointing to the reward function.
        action_pickup_drop_idx: int, PickupDrop action index.
        action_toggle_idx: int, Toggle action index.
        max_steps: int, maximum timesteps per episode.
        terminated_fn: Optional callable
            ``(prev_state_dict, state_dict, reward_config) -> bool array``.
            Returns per-agent termination flags. If None, all-False.

    Returns:
        Tuple ``(state, obs, rewards, terminateds, truncateds, infos)`` where:
        - ``state``: Updated :class:`EnvState`.
        - ``obs``: array of shape ``(n_agents, obs_dim)``.
        - ``rewards``: float32 array of shape ``(n_agents,)``.
        - ``terminateds``: bool array of shape ``(n_agents,)`` from
          ``terminated_fn`` (or all-False).
        - ``truncateds``: bool array of shape ``(n_agents,)`` broadcast
          from ``time >= max_steps``.
        - ``infos``: empty dict ``{}``.
    """
    from cogrid.backend import xp
    from cogrid.backend._dispatch import get_backend
    from cogrid.core.movement import move_agents
    from cogrid.core.interactions import process_interactions
    from cogrid.feature_space.array_features import get_all_agent_obs

    # a. Capture prev_state before ANY mutations (zero-cost, immutable)
    prev_state = state

    # b. Tick: delegate to scope config handler (if any)
    tick_handler = scope_config.get("tick_handler") if scope_config else None
    if tick_handler is not None:
        state = tick_handler(state, scope_config)

    # c. Movement -- backend-specific RNG for priority
    if get_backend() == "jax":
        import jax

        key, subkey = jax.random.split(state.rng_key)
        priority = jax.random.permutation(subkey, state.n_agents)
    else:
        import numpy as _np

        priority = _np.random.default_rng().permutation(state.n_agents).astype(_np.int32)
        key = state.rng_key

    new_pos, new_dir = move_agents(
        state.agent_pos,
        state.agent_dir,
        actions,
        state.wall_map,
        state.object_type_map,
        lookup_tables["CAN_OVERLAP"],
        priority,
        state.action_set,
    )
    state = dataclasses.replace(
        state, agent_pos=new_pos, agent_dir=new_dir, rng_key=key
    )

    # d. Interactions -- pass extra_state generically (strip scope prefix)
    dir_vec_table = xp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32
    )
    extra_state = {}
    for es_key, val in state.extra_state.items():
        short_key = es_key.split(".", 1)[-1] if "." in es_key else es_key
        extra_state[short_key] = val

    agent_inv, otm, osm, extra_out = process_interactions(
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
        extra_state=extra_state,
    )
    # Re-prefix returned extra_out keys back into extra_state
    scope_prefix = next(
        (k.split(".")[0] for k in state.extra_state if "." in k), None
    )
    new_extra = dict(state.extra_state)
    for k, v in extra_out.items():
        prefixed = f"{scope_prefix}.{k}" if scope_prefix else k
        if prefixed in new_extra:
            new_extra[prefixed] = v

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
    obs = get_all_agent_obs(feature_fn, state_dict, state.n_agents)

    # f. Rewards -- compute_fn comes from reward_config (no env-specific import)
    prev_dict = envstate_to_dict(prev_state)
    compute_fn = reward_config["compute_fn"]
    rewards = compute_fn(prev_dict, state_dict, actions, reward_config)

    # g. Terminateds and truncateds
    if terminated_fn is not None:
        terminateds = terminated_fn(prev_dict, state_dict, reward_config)
    else:
        terminateds = xp.zeros(state.n_agents, dtype=xp.bool_)

    truncated = state.time >= max_steps
    truncateds = xp.full(state.n_agents, truncated, dtype=xp.bool_)

    # h. Zero out rewards for agents already done before this step
    already_done = prev_state.done
    rewards = xp.where(already_done, xp.zeros_like(rewards), rewards)

    # i. Update done mask: once done, stays done
    new_done = already_done | terminateds | truncateds
    state = dataclasses.replace(state, done=new_done)

    # j. Stop gradient (JAX only, no-op on numpy)
    if get_backend() == "jax":
        import jax.lax as lax

        obs = lax.stop_gradient(obs)
        rewards = lax.stop_gradient(rewards)
        terminateds = lax.stop_gradient(terminateds)
        truncateds = lax.stop_gradient(truncateds)

    # k. Return
    return state, obs, rewards, terminateds, truncateds, {}


def reset(
    rng,
    *,
    layout_arrays,
    spawn_positions,
    n_agents,
    feature_fn,
    scope_config,
    action_set,
    max_inv_size=1,
):
    """Build initial EnvState from pre-computed layout arrays.

    All layout-dependent data (``layout_arrays``, ``spawn_positions``)
    is pre-computed at init time from the ASCII layout.  This function
    only randomizes agent initial directions.

    Uses ``xp`` for all array operations.  Backend-specific code is
    limited to RNG (random directions) and ``stop_gradient``.

    Args:
        rng: JAX PRNG key, integer seed, or None.
        layout_arrays: Dict with keys ``wall_map``, ``object_type_map``,
            ``object_state_map``, plus any scope-specific arrays
            (e.g. scope-specific extra_state arrays).
        spawn_positions: int32 array of shape ``(n_agents, 2)`` with
            fixed spawn points ``[row, col]``.
        n_agents: Number of agents.
        feature_fn: Composed feature function for initial obs.
        scope_config: Scope config dict with ``"scope"`` key for prefix.
        action_set: ``"cardinal"`` or ``"rotation"``.
        max_inv_size: Maximum inventory slots per agent (default 1).

    Returns:
        Tuple ``(state, obs)`` where:
        - ``state``: Initial :class:`EnvState`.
        - ``obs``: Initial observations, shape ``(n_agents, obs_dim)``.
    """
    from cogrid.backend import xp
    from cogrid.backend._dispatch import get_backend
    from cogrid.backend.env_state import create_env_state
    from cogrid.feature_space.array_features import get_all_agent_obs

    # Backend-specific RNG for random initial directions
    if get_backend() == "jax":
        import jax

        key, subkey = jax.random.split(rng)
        agent_dir = jax.random.randint(subkey, (n_agents,), 0, 4).astype(xp.int32)
    else:
        import numpy as _np

        agent_dir = _np.random.default_rng(
            rng if isinstance(rng, int) else None
        ).integers(0, 4, size=(n_agents,)).astype(_np.int32)
        key = None

    # Agent positions from fixed spawn points
    agent_pos = spawn_positions.astype(xp.int32)

    # Empty inventory
    agent_inv = xp.full((n_agents, max_inv_size), -1, dtype=xp.int32)

    # Time starts at 0
    time = xp.int32(0)

    # Extract layout arrays
    wall_map = layout_arrays["wall_map"]
    object_type_map = layout_arrays["object_type_map"]
    object_state_map = layout_arrays["object_state_map"]

    H, W = wall_map.shape

    # Build extra_state generically: any layout_arrays key not in
    # the base set is a scope-specific extra, prefixed with {scope}.
    base_keys = {"wall_map", "object_type_map", "object_state_map"}
    scope = scope_config.get("scope", "global") if scope_config else "global"
    extra_state = {}
    for la_key, val in layout_arrays.items():
        if la_key not in base_keys:
            extra_state[f"{scope}.{la_key}"] = val

    # Build EnvState
    done = xp.zeros(n_agents, dtype=xp.bool_)
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
        done=done,
        n_agents=n_agents,
        height=H,
        width=W,
        action_set=action_set,
    )

    # Compute initial observations
    state_dict = envstate_to_dict(state)
    obs = get_all_agent_obs(feature_fn, state_dict, n_agents)

    # Stop gradient (JAX only, no-op on numpy)
    if get_backend() == "jax":
        import jax.lax as lax

        obs = lax.stop_gradient(obs)

    return state, obs


def build_step_fn(
    scope_config,
    lookup_tables,
    feature_fn,
    reward_config,
    action_pickup_drop_idx,
    action_toggle_idx,
    max_steps,
    terminated_fn=None,
    jit_compile=None,
):
    """Build a step function with all static config closed over.

    Returns a function with signature
    ``(state, actions) -> (state, obs, rewards, terminateds, truncateds, infos)``.

    Args:
        scope_config: Scope config dict.
        lookup_tables: Dict of property arrays.
        feature_fn: Composed feature function.
        reward_config: Reward config dict.
        action_pickup_drop_idx: int, PickupDrop action index.
        action_toggle_idx: int, Toggle action index.
        max_steps: int, maximum timesteps per episode.
        terminated_fn: Optional callable for per-agent termination.
        jit_compile: If None, auto-detect from backend. If True/False, force.

    Returns:
        Step function (optionally JIT-compiled on JAX backend).
    """
    from cogrid.backend._dispatch import get_backend

    def step_fn(state, actions):
        return step(
            state,
            actions,
            scope_config=scope_config,
            lookup_tables=lookup_tables,
            feature_fn=feature_fn,
            reward_config=reward_config,
            action_pickup_drop_idx=action_pickup_drop_idx,
            action_toggle_idx=action_toggle_idx,
            max_steps=max_steps,
            terminated_fn=terminated_fn,
        )

    should_jit = jit_compile if jit_compile is not None else (get_backend() == "jax")
    if should_jit:
        import jax

        return jax.jit(step_fn)
    return step_fn


def build_reset_fn(
    layout_arrays,
    spawn_positions,
    n_agents,
    feature_fn,
    scope_config,
    action_set,
    jit_compile=None,
    **kwargs,
):
    """Build a reset function with all layout config closed over.

    Returns a function with signature ``(rng) -> (state, obs)``.

    Args:
        layout_arrays: Dict of pre-computed layout arrays.
        spawn_positions: int32 array of shape (n_agents, 2).
        n_agents: Number of agents.
        feature_fn: Composed feature function.
        scope_config: Scope config dict.
        action_set: "cardinal" or "rotation".
        jit_compile: If None, auto-detect from backend. If True/False, force.
        **kwargs: Additional keyword args forwarded to reset()
            (e.g., max_inv_size).

    Returns:
        Reset function (optionally JIT-compiled on JAX backend).
    """
    from cogrid.backend._dispatch import get_backend

    def reset_fn(rng):
        return reset(
            rng,
            layout_arrays=layout_arrays,
            spawn_positions=spawn_positions,
            n_agents=n_agents,
            feature_fn=feature_fn,
            scope_config=scope_config,
            action_set=action_set,
            **kwargs,
        )

    should_jit = jit_compile if jit_compile is not None else (get_backend() == "jax")
    if should_jit:
        import jax

        return jax.jit(reset_fn)
    return reset_fn
