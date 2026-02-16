"""Cross-backend parity tests for numpy vs JAX backends.

Verifies that:
1. Scripted action sequences produce identical array state and rewards
   across numpy and JAX backends (TEST-01, TEST-02)
2. Random action sequences produce structurally valid outputs on the JAX
   backend (TEST-01 supplement)
3. Eager and JIT execution produce identical results (TEST-03)
4. Core JAX functions are deterministic (no hidden statefulness)

The test strategy accounts for two key differences between backends:

- **Observation encoding:** The numpy path uses high-level OvercookedCollectedFeatures
  while the JAX path uses low-level array features (agent_position, agent_dir,
  full_map_encoding, can_move_direction, inventory). These are fundamentally
  different encodings, so parity is verified at the array state level (agent_pos,
  agent_dir, agent_inv, object_type_map, etc.) and reward level, not at the raw
  observation level.

- **RNG divergence:** numpy PCG64 and JAX ThreeFry produce different random
  sequences. Scripted non-colliding actions are used for exact parity; random
  actions verify structural correctness only.
"""

import numpy as np
import pytest

# Registry IDs for the 3 test layouts
LAYOUTS = [
    "Overcooked-CrampedRoom-V0",
    "Overcooked-AsymmetricAdvantages-V0",
    "Overcooked-CoordinationRing-V0",
]

# Number of scripted steps for parity comparison
N_SCRIPTED_STEPS = 50

# Number of random steps for structural validation
N_RANDOM_STEPS = 100


# State fields to compare between backends (all dynamic arrays).
# Core EnvState fields are direct attributes; extra_state fields
# (pot_contents, pot_timer) are in the extra_state dict with scope prefix.
CORE_STATE_FIELDS = [
    "agent_pos",
    "agent_dir",
    "agent_inv",
    "object_type_map",
    "object_state_map",
]
EXTRA_STATE_FIELDS = ["pot_contents", "pot_timer"]
STATE_FIELDS = CORE_STATE_FIELDS + EXTRA_STATE_FIELDS


def _create_env(registry_id, backend, seed=42, max_steps=200):
    """Create and reset an Overcooked environment on a given backend.

    Handles backend switching via _reset_backend_for_testing() so that
    numpy and JAX envs can coexist within a single test process.

    Args:
        registry_id: Registered environment ID string.
        backend: 'numpy' or 'jax'.
        seed: RNG seed for reset.
        max_steps: Episode length limit.

    Returns:
        Tuple of (env, obs_dict) after reset.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    env = registry.make(registry_id, backend=backend)
    env.max_steps = max_steps
    obs, _ = env.reset(seed=seed)
    return env, obs


def _get_numpy_state(env):
    """Extract the array state dict from a numpy-backend env.

    Returns a dict with numpy arrays for the STATE_FIELDS keys.
    Both backends now use EnvState internally.
    """
    es = env._env_state
    return {k: np.array(_get_state_field(es, k)) for k in STATE_FIELDS}


def _get_state_field(es, field):
    """Get a state field from EnvState, checking extra_state for scope-prefixed keys."""
    if hasattr(es, field) and field not in EXTRA_STATE_FIELDS:
        return getattr(es, field)
    # Look in extra_state with any scope prefix
    for key, val in es.extra_state.items():
        short_key = key.split(".", 1)[-1] if "." in key else key
        if short_key == field:
            return val
    raise AttributeError(f"Field {field} not found in EnvState or extra_state")


def _get_jax_env_state_as_numpy(env):
    """Extract the EnvState from a JAX-backend env as numpy arrays.

    Returns a dict with numpy arrays for the STATE_FIELDS keys.
    """
    es = env._env_state
    return {k: np.array(_get_state_field(es, k)) for k in STATE_FIELDS}


def _scripted_actions(agent_ids, n_steps):
    """Generate a deterministic action sequence that avoids collisions.

    Uses a cycling pattern where agents move in different directions,
    then perform noop. This avoids same-cell competition so both backends
    produce identical outcomes.

    Cardinal actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=PickupDrop, 5=Toggle, 6=Noop

    The key property: agents never attempt to move to the same cell
    simultaneously, so collision resolution is never triggered and
    both backends produce identical results.

    Args:
        agent_ids: List of agent IDs (sorted).
        n_steps: Number of steps to generate.

    Returns:
        List of action dicts, one per step.
    """
    if len(agent_ids) >= 2:
        cycle = [
            {agent_ids[0]: 0, agent_ids[1]: 1},  # Up, Down
            {agent_ids[0]: 2, agent_ids[1]: 3},  # Left, Right
            {agent_ids[0]: 6, agent_ids[1]: 6},  # Noop, Noop
            {agent_ids[0]: 1, agent_ids[1]: 0},  # Down, Up
            {agent_ids[0]: 3, agent_ids[1]: 2},  # Right, Left
            {agent_ids[0]: 6, agent_ids[1]: 6},  # Noop, Noop
        ]
    else:
        cycle = [
            {agent_ids[0]: 0},
            {agent_ids[0]: 2},
            {agent_ids[0]: 6},
            {agent_ids[0]: 1},
            {agent_ids[0]: 3},
            {agent_ids[0]: 6},
        ]

    return [cycle[step % len(cycle)] for step in range(n_steps)]


# ======================================================================
# TEST-01 / TEST-02: Scripted parity across backends
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_scripted_parity(layout):
    """Scripted actions produce identical array state and rewards on numpy vs JAX.

    Compares the underlying array state (agent_pos, agent_dir, agent_inv,
    object_type_map, object_state_map, pot_contents, pot_timer) and reward
    values at every step. Observations are NOT compared directly because
    the numpy and JAX backends use different feature encodings (high-level
    OvercookedCollectedFeatures vs low-level array features).

    Uses assert_array_equal for int-valued state arrays and
    assert_allclose(atol=1e-7) for float32 rewards.
    """
    pytest.importorskip("jax")

    # --- Phase 1: Run numpy env, save state + rewards at each step ---
    np_env, _ = _create_env(layout, backend="numpy", seed=42, max_steps=N_SCRIPTED_STEPS + 10)
    agent_ids = sorted(np_env.possible_agents)
    scripted = _scripted_actions(agent_ids, N_SCRIPTED_STEPS)

    np_states = [_get_numpy_state(np_env)]
    np_rewards = []

    for action_dict in scripted:
        _, rewards, _, _, _ = np_env.step(action_dict)
        np_states.append(_get_numpy_state(np_env))
        np_rewards.append(rewards)

    # --- Phase 2: Run JAX env, compare at each step ---
    jax_env, _ = _create_env(layout, backend="jax", seed=42, max_steps=N_SCRIPTED_STEPS + 10)
    jax_agent_ids = sorted(jax_env.possible_agents)
    assert jax_agent_ids == agent_ids, f"Agent IDs differ: numpy={agent_ids}, jax={jax_agent_ids}"

    # Compare initial state (skip agent_dir: it is randomly initialized
    # and numpy PCG64 vs JAX ThreeFry produce different values from the
    # same seed. After the first cardinal move action, directions become
    # deterministic and match between backends.)
    jax_init_state = _get_jax_env_state_as_numpy(jax_env)
    init_fields = [f for f in STATE_FIELDS if f != "agent_dir"]
    for field in init_fields:
        np.testing.assert_array_equal(
            np_states[0][field],
            jax_init_state[field],
            err_msg=f"Layout {layout}, initial state, field {field}: mismatch",
        )

    # Step through and compare (all fields from step 1 onward, since
    # step 0 actions are cardinal moves that set agent_dir deterministically)
    for step_idx, action_dict in enumerate(scripted):
        _, rewards, _, _, _ = jax_env.step(action_dict)

        # Compare state -- include agent_dir from step 1 onward
        jax_state = _get_jax_env_state_as_numpy(jax_env)
        compare_fields = STATE_FIELDS if step_idx >= 1 else init_fields
        for field in compare_fields:
            np.testing.assert_array_equal(
                np_states[step_idx + 1][field],
                jax_state[field],
                err_msg=(f"Layout {layout}, step {step_idx}, field {field}: state mismatch"),
            )

        # Compare rewards
        for aid in agent_ids:
            np.testing.assert_allclose(
                np_rewards[step_idx][aid],
                rewards[aid],
                atol=1e-7,
                err_msg=(f"Layout {layout}, step {step_idx}, agent {aid}: reward mismatch"),
            )


# ======================================================================
# TEST-01 supplement: Random structural validation
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_random_structural(layout):
    """Random actions produce structurally valid outputs on the JAX backend.

    Verifies shapes, dtypes, finite values, and correct dict keys over
    100+ steps. Does NOT compare against numpy (RNG divergence means
    collision resolution may differ).
    """
    pytest.importorskip("jax")

    env, obs = _create_env(layout, backend="jax", seed=42, max_steps=N_RANDOM_STEPS + 10)
    agent_ids = sorted(env.possible_agents)
    rng = np.random.default_rng(123)

    # Track initial obs shape for consistency across steps
    obs_shapes = {aid: obs[aid].shape for aid in agent_ids}

    for step_idx in range(N_RANDOM_STEPS):
        # Random actions from all available cardinal actions (0-6)
        action_dict = {aid: int(rng.integers(0, 7)) for aid in agent_ids}
        obs, rewards, terminateds, truncateds, infos = env.step(action_dict)

        for aid in agent_ids:
            # Shape check (consistent with first step)
            assert obs[aid].shape == obs_shapes[aid], (
                f"Step {step_idx}, agent {aid}: obs shape changed. "
                f"Got {obs[aid].shape}, expected {obs_shapes[aid]}"
            )

            # Dtype check: numeric (int or float)
            assert np.issubdtype(obs[aid].dtype, np.number), (
                f"Step {step_idx}, agent {aid}: obs dtype {obs[aid].dtype} is not numeric"
            )

            # Finite values (no NaN, no inf)
            assert np.all(np.isfinite(obs[aid])), (
                f"Step {step_idx}, agent {aid}: obs contains non-finite values"
            )

            # Reward is finite
            assert np.isfinite(rewards[aid]), (
                f"Step {step_idx}, agent {aid}: reward is not finite: {rewards[aid]}"
            )

        # Correct dict keys
        assert set(terminateds.keys()) == set(agent_ids), (
            f"Step {step_idx}: terminateds keys mismatch"
        )
        assert set(truncateds.keys()) == set(agent_ids), (
            f"Step {step_idx}: truncateds keys mismatch"
        )

        # Bool types for terminateds and truncateds
        for aid in agent_ids:
            assert isinstance(terminateds[aid], bool), (
                f"Step {step_idx}, agent {aid}: terminated is not bool"
            )
            assert isinstance(truncateds[aid], bool), (
                f"Step {step_idx}, agent {aid}: truncated is not bool"
            )


# ======================================================================
# TEST-03: Eager vs JIT for end-to-end step and reset
# ======================================================================


def _compare_states(s1, s2, label):
    """Compare two EnvState instances field by field."""
    for field_name in STATE_FIELDS + ["wall_map", "time"]:
        v1 = _get_state_field(s1, field_name)
        v2 = _get_state_field(s2, field_name)
        np.testing.assert_array_equal(
            np.array(v1), np.array(v2), err_msg=f"{label}: state.{field_name} mismatch"
        )


def test_eager_vs_jit():
    """Eager and JIT execution produce identical results for step and reset.

    Creates a JAX env, obtains the raw step and reset functions,
    and compares outputs of eager vs jax.jit execution for both reset
    and 10 sequential steps.
    """
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.core.step_pipeline import reset as reset_fn_eager
    from cogrid.core.step_pipeline import step as step_fn_eager

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    # Create env to extract config
    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=42)

    # Get the JIT-compiled functions from the env
    reset_fn = env.jax_reset
    step_fn = env.jax_step

    # Extract configs for eager calls
    scope_config = env._scope_config
    lookup_tables = env._lookup_tables
    feature_fn = env._feature_fn
    reward_config = env._reward_config
    action_pickup_drop_idx = env._action_pickup_drop_idx
    action_toggle_idx = env._action_toggle_idx
    max_steps = env.max_steps
    n_agents = env.config["num_agents"]

    # Reconstruct layout_arrays and spawn_positions from the env's array state
    layout_arrays = {
        "wall_map": jnp.array(env._state["wall_map"], dtype=jnp.int32),
        "object_type_map": jnp.array(env._state["object_type_map"], dtype=jnp.int32),
        "object_state_map": jnp.array(env._state["object_state_map"], dtype=jnp.int32),
        "pot_contents": jnp.array(env._state["pot_contents"], dtype=jnp.int32),
        "pot_timer": jnp.array(env._state["pot_timer"], dtype=jnp.int32),
        "pot_positions": jnp.array(
            env._state.get("pot_positions", np.zeros((0, 2), dtype=np.int32)), dtype=jnp.int32
        ),
    }
    spawn_positions = jnp.array(env._state["agent_pos"], dtype=jnp.int32)

    rng_key = jax.random.key(42)

    # --- Reset comparison ---
    state_eager, obs_eager = reset_fn_eager(
        rng_key,
        layout_arrays=layout_arrays,
        spawn_positions=spawn_positions,
        n_agents=n_agents,
        feature_fn=feature_fn,
        scope_config=scope_config,
        action_set="cardinal",
    )

    state_jit, obs_jit = reset_fn(rng_key)

    np.testing.assert_array_equal(
        np.array(obs_eager), np.array(obs_jit), err_msg="Reset: obs mismatch between eager and JIT"
    )
    _compare_states(state_eager, state_jit, "Reset")

    # --- Step comparison (10 steps) ---
    state_e = state_eager
    state_j = state_jit

    for step_i in range(10):
        actions = jnp.array([0, 1], dtype=jnp.int32)  # Up, Down

        state_e, obs_e, rew_e, term_e, trunc_e, _ = step_fn_eager(
            state_e,
            actions,
            scope_config=scope_config,
            lookup_tables=lookup_tables,
            feature_fn=feature_fn,
            reward_config=reward_config,
            action_pickup_drop_idx=action_pickup_drop_idx,
            action_toggle_idx=action_toggle_idx,
            max_steps=max_steps,
        )

        state_j, obs_j, rew_j, term_j, trunc_j, _ = step_fn(state_j, actions)

        np.testing.assert_array_equal(
            np.array(obs_e),
            np.array(obs_j),
            err_msg=f"Step {step_i}: obs mismatch between eager and JIT",
        )
        np.testing.assert_allclose(
            np.array(rew_e),
            np.array(rew_j),
            atol=1e-7,
            err_msg=f"Step {step_i}: reward mismatch between eager and JIT",
        )
        np.testing.assert_array_equal(
            np.array(term_e),
            np.array(term_j),
            err_msg=f"Step {step_i}: terminateds mismatch between eager and JIT",
        )
        np.testing.assert_array_equal(
            np.array(trunc_e),
            np.array(trunc_j),
            err_msg=f"Step {step_i}: truncateds mismatch between eager and JIT",
        )
        _compare_states(state_e, state_j, f"Step {step_i}")


# ======================================================================
# Determinism test: step with same inputs produces same outputs
# ======================================================================


def test_step_determinism():
    """Calling step twice with identical state+actions gives identical outputs.

    Catches hidden statefulness (global counters, in-place mutation, etc.)
    that would cause non-deterministic behavior under the same inputs.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=42)

    step_fn = env.jax_step
    state = env._env_state

    actions = jnp.array([0, 3], dtype=jnp.int32)  # Up, Right

    # Call step twice with same state and actions
    state1, obs1, rew1, term1, trunc1, _ = step_fn(state, actions)
    state2, obs2, rew2, term2, trunc2, _ = step_fn(state, actions)

    np.testing.assert_array_equal(
        np.array(obs1), np.array(obs2), err_msg="Determinism: obs differ between identical calls"
    )
    np.testing.assert_array_equal(
        np.array(rew1),
        np.array(rew2),
        err_msg="Determinism: rewards differ between identical calls",
    )
    np.testing.assert_array_equal(
        np.array(term1),
        np.array(term2),
        err_msg="Determinism: terminateds differ between identical calls",
    )
    np.testing.assert_array_equal(
        np.array(trunc1),
        np.array(trunc2),
        err_msg="Determinism: truncateds differ between identical calls",
    )

    for field_name in STATE_FIELDS:
        np.testing.assert_array_equal(
            np.array(_get_state_field(state1, field_name)),
            np.array(_get_state_field(state2, field_name)),
            err_msg=f"Determinism: state.{field_name} differs",
        )


# ======================================================================
# TEST-03 sub-function variants: Eager vs JIT for core functions
# ======================================================================


def _setup_jax_env():
    """Create a JAX env on cramped_room and return (env, state, configs).

    Shared setup for all sub-function eager-vs-JIT tests. Returns
    everything needed to call individual JAX sub-functions.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=42)
    return env


def test_obs_eager_vs_jit():
    """get_all_agent_obs produces identical outputs eagerly and under jax.jit."""
    pytest.importorskip("jax")
    import jax

    from cogrid.core.step_pipeline import envstate_to_dict
    from cogrid.feature_space.features import get_all_agent_obs

    env = _setup_jax_env()
    state = env._env_state
    state_view = envstate_to_dict(state)
    feature_fn = env._feature_fn
    n_agents = env.config["num_agents"]

    # Eager call
    obs_e = get_all_agent_obs(feature_fn, state_view, n_agents)

    # JIT call -- StateView is a registered pytree, pass directly
    @jax.jit
    def jitted_obs(sv):
        return get_all_agent_obs(feature_fn, sv, n_agents)

    obs_j = jitted_obs(state_view)

    np.testing.assert_array_equal(
        np.array(obs_e),
        np.array(obs_j),
        err_msg="get_all_agent_obs: obs mismatch between eager and JIT",
    )


def test_rewards_eager_vs_jit():
    """Auto-wired compute_fn produces identical outputs eagerly and under jax.jit."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from cogrid.core.step_pipeline import envstate_to_dict

    env = _setup_jax_env()
    state = env._env_state

    # Run one step to get a prev_state and current state
    step_fn = env.jax_step
    actions = jnp.array([0, 1], dtype=jnp.int32)
    new_state, _, _, _, _, _ = step_fn(state, actions)

    prev_sv = envstate_to_dict(state)
    curr_sv = envstate_to_dict(new_state)
    reward_config = env._reward_config
    compute_fn = reward_config["compute_fn"]

    # Eager call
    rew_e = compute_fn(prev_sv, curr_sv, actions, reward_config)

    # JIT call -- StateView is a registered pytree, pass directly
    @jax.jit
    def jitted_rewards(prev, curr, acts):
        return compute_fn(prev, curr, acts, reward_config)

    rew_j = jitted_rewards(prev_sv, curr_sv, actions)

    np.testing.assert_allclose(
        np.array(rew_e),
        np.array(rew_j),
        atol=1e-7,
        err_msg="compute_fn: reward mismatch between eager and JIT",
    )
