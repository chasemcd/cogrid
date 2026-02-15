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

import dataclasses

from cogrid.backend import xp
from cogrid.backend._dispatch import get_backend
from cogrid.backend.env_state import create_env_state
from cogrid.backend.state_view import StateView
from cogrid.core.interactions import process_interactions
from cogrid.core.movement import move_agents
from cogrid.feature_space.array_features import get_all_agent_obs


# ---------------------------------------------------------------------------
# Backend-conditional helpers (each contains exactly one get_backend check)
# ---------------------------------------------------------------------------


def _backend_rng(rng_key, mode, n_agents):
    """Generate random arrays (permutation or directions) via the active backend.

    Returns ``(new_key, result_array)`` -- splits JAX key or uses numpy Generator.
    """
    if get_backend() == "jax":
        import jax

        key, subkey = jax.random.split(rng_key)
        if mode == "permutation":
            return key, jax.random.permutation(subkey, n_agents)
        else:
            return key, jax.random.randint(subkey, (n_agents,), 0, 4)
    else:
        import numpy as _np

        if mode == "permutation":
            return rng_key, _np.random.default_rng().permutation(n_agents).astype(_np.int32)
        else:
            seed = rng_key if isinstance(rng_key, int) else None
            return None, _np.random.default_rng(seed).integers(0, 4, size=(n_agents,)).astype(
                _np.int32
            )


def _maybe_stop_gradient(*arrays):
    """Apply jax.lax.stop_gradient on JAX backend; identity on numpy."""
    if get_backend() == "jax":
        import jax.lax as lax

        return tuple(lax.stop_gradient(a) for a in arrays)
    return arrays


def _maybe_jit(fn, jit_compile=None):
    """Optionally JIT-compile fn. Auto-JIT when backend is JAX unless disabled."""
    if get_backend() == "jax" and jit_compile is not False:
        import jax

        return jax.jit(fn)
    return fn


def envstate_to_dict(state):
    """Convert EnvState to a StateView with dot access and scope-stripped extras.

    Zero-cost under JIT (no array copies). Extra_state keys like
    ``"scope.key"`` become ``state_view.key``.
    """
    extra = {}
    for key, val in state.extra_state.items():
        short_key = key.split(".", 1)[-1] if "." in key else key
        extra[short_key] = val

    return StateView(
        agent_pos=state.agent_pos,
        agent_dir=state.agent_dir,
        agent_inv=state.agent_inv,
        wall_map=state.wall_map,
        object_type_map=state.object_type_map,
        object_state_map=state.object_state_map,
        extra=extra,
    )


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
    """End-to-end step: tick, move, interact, observe, reward, done.

    Keyword arguments are designed to be closed over via ``build_step_fn``.

    Pipeline order: (1) capture prev_state, (2) tick, (3) movement,
    (4) interactions, (5) observations, (6) rewards, (7) terminateds/
    truncateds, (8) stop_gradient (JAX only).

    Returns ``(state, obs, rewards, terminateds, truncateds, infos)``.
    """
    # a. Capture prev_state before ANY mutations (zero-cost, immutable)
    prev_state = state

    # b. Tick: delegate to scope config handler (if any)
    tick_handler = scope_config.get("tick_handler") if scope_config else None
    if tick_handler is not None:
        state = tick_handler(state, scope_config)

    # c. Movement -- backend-specific RNG for priority
    key, priority = _backend_rng(state.rng_key, "permutation", state.n_agents)

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
    sv = envstate_to_dict(state)
    obs = get_all_agent_obs(feature_fn, sv, state.n_agents)

    # f. Rewards -- compute_fn comes from reward_config (no env-specific import)
    prev_sv = envstate_to_dict(prev_state)
    compute_fn = reward_config["compute_fn"]
    rewards = compute_fn(prev_sv, sv, actions, reward_config)

    # g. Terminateds and truncateds
    if terminated_fn is not None:
        terminateds = terminated_fn(prev_sv, sv, reward_config)
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
    obs, rewards, terminateds, truncateds = _maybe_stop_gradient(
        obs, rewards, terminateds, truncateds
    )

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

    Layout data is pre-computed at init time; this function only
    randomizes agent initial directions. Returns ``(state, obs)``.
    """
    # Backend-specific RNG for random initial directions
    key, agent_dir = _backend_rng(rng, "directions", n_agents)
    agent_dir = agent_dir.astype(xp.int32)

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
    sv = envstate_to_dict(state)
    obs = get_all_agent_obs(feature_fn, sv, n_agents)

    # Stop gradient (JAX only, no-op on numpy)
    (obs,) = _maybe_stop_gradient(obs)

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
    """Close over static config and return a step function.

    ``(state, actions) -> (state, obs, rewards, terminateds, truncateds, infos)``

    Auto-JIT on JAX backend unless ``jit_compile=False``.
    """
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

    return _maybe_jit(step_fn, jit_compile)


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
    """Close over layout config and return a reset function.

    ``(rng) -> (state, obs)``

    Auto-JIT on JAX backend unless ``jit_compile=False``.
    """
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

    return _maybe_jit(reset_fn, jit_compile)
