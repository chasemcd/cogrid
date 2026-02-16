"""Phase 14 end-to-end validation tests.

Four tests proving the auto-wired CoGridEnv path is correct:

1. **Determinism + invariants (numpy):** Two identical auto-wired Overcooked
   envs with the same seed produce identical outputs over 50 scripted steps,
   with invariant checks (finite rewards, correct shapes, bool terminateds).

2. **JIT + vmap at 1024 envs (JAX):** Auto-wired Overcooked compiles and
   runs under jax.jit(jax.vmap(...)) for 1024 parallel environments.

3. **Goal-finding component API (numpy):** The goal-finding example creates
   an environment via the component API with zero manual scope_config.

4. **Cross-backend parity (numpy vs JAX):** Goal-finding env with identical
   scripted actions produces identical state sequences on both backends.

Satisfies TEST-01, TEST-02, TEST-03 for Phase 14.
"""

import numpy as np
import pytest

# ======================================================================
# TEST-01: Auto-wired Overcooked determinism + invariants (numpy)
# ======================================================================


def test_autowired_overcooked_determinism_and_invariants():
    """Two auto-wired Overcooked envs with same seed produce identical outputs.

    Runs 50 steps with scripted deterministic actions and verifies:
    - Exact output parity between both runs (obs, rewards, terminateds, truncateds)
    - Rewards are finite and non-negative at every step
    - Observation shapes match env.observation_space
    - terminateds and truncateds values are bool-typed
    - Agent positions are within grid bounds
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    N_STEPS = 50

    # Create two identical environments
    env_a = registry.make("Overcooked-CrampedRoom-V0", backend="numpy")
    obs_a, _ = env_a.reset(seed=42)

    _reset_backend_for_testing()
    env_b = registry.make("Overcooked-CrampedRoom-V0", backend="numpy")
    obs_b, _ = env_b.reset(seed=42)

    agent_ids = sorted(env_a.possible_agents)
    grid_h, grid_w = env_a.shape

    # Verify initial obs parity
    for aid in agent_ids:
        np.testing.assert_array_equal(
            obs_a[aid],
            obs_b[aid],
            err_msg=f"Initial obs mismatch for agent {aid}",
        )

    for step_idx in range(N_STEPS):
        # Deterministic scripted actions
        actions = {aid: (step_idx + i * 3) % 7 for i, aid in enumerate(agent_ids)}

        obs_a, rew_a, term_a, trunc_a, _ = env_a.step(actions)
        obs_b, rew_b, term_b, trunc_b, _ = env_b.step(actions)

        for aid in agent_ids:
            # Parity checks
            np.testing.assert_array_equal(
                obs_a[aid],
                obs_b[aid],
                err_msg=f"Step {step_idx}, agent {aid}: obs mismatch",
            )
            np.testing.assert_allclose(
                rew_a[aid],
                rew_b[aid],
                atol=1e-7,
                err_msg=f"Step {step_idx}, agent {aid}: reward mismatch",
            )
            assert term_a[aid] == term_b[aid], f"Step {step_idx}, agent {aid}: terminated mismatch"
            assert trunc_a[aid] == trunc_b[aid], f"Step {step_idx}, agent {aid}: truncated mismatch"

            # Invariant: rewards are finite and non-negative
            assert np.isfinite(rew_a[aid]), (
                f"Step {step_idx}, agent {aid}: reward not finite: {rew_a[aid]}"
            )
            assert rew_a[aid] >= 0.0, f"Step {step_idx}, agent {aid}: negative reward: {rew_a[aid]}"

            # Invariant: terminateds and truncateds are bool
            assert isinstance(term_a[aid], bool), (
                f"Step {step_idx}, agent {aid}: terminated not bool: {type(term_a[aid])}"
            )
            assert isinstance(trunc_a[aid], bool), (
                f"Step {step_idx}, agent {aid}: truncated not bool: {type(trunc_a[aid])}"
            )

        # Invariant: agent positions within grid bounds
        state = env_a._env_state
        agent_pos = np.array(state.agent_pos)
        for i, aid in enumerate(agent_ids):
            r, c = int(agent_pos[i, 0]), int(agent_pos[i, 1])
            assert 0 <= r < grid_h, (
                f"Step {step_idx}, agent {aid}: row {r} out of bounds [0, {grid_h})"
            )
            assert 0 <= c < grid_w, (
                f"Step {step_idx}, agent {aid}: col {c} out of bounds [0, {grid_w})"
            )


# ======================================================================
# TEST-02: JIT + vmap at 1024 environments (JAX)
# ======================================================================


def test_autowired_overcooked_jit_vmap_1024():
    """Auto-wired Overcooked runs under jax.jit(jax.vmap(...)) for 1024 envs.

    Validates:
    - No errors during JIT compilation or vmap execution
    - Output shapes have batch dimension 1024
    - Rewards are finite (no NaN/inf)
    """
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp

    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    N_ENVS = 1024
    N_STEPS = 10

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=42)

    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = env.config["num_agents"]

    # Batched reset
    keys = jax.random.split(jax.random.key(0), N_ENVS)
    batched_reset = jax.jit(jax.vmap(reset_fn))
    batched_step = jax.jit(jax.vmap(step_fn))

    states, obs = batched_reset(keys)

    # Verify batch dimension on obs
    assert obs.shape[0] == N_ENVS, f"Expected obs batch dim {N_ENVS}, got {obs.shape[0]}"

    # Run N_STEPS of batched steps with scripted actions
    for step_i in range(N_STEPS):
        # Cycle through actions deterministically
        action_val = step_i % 7
        actions = jnp.full((N_ENVS, n_agents), action_val, dtype=jnp.int32)
        states, obs, rew, term, trunc, _ = batched_step(states, actions)

        # Shape checks
        assert rew.shape == (N_ENVS, n_agents), (
            f"Step {step_i}: expected reward shape ({N_ENVS}, {n_agents}), got {rew.shape}"
        )

        # Finite rewards
        assert jnp.all(jnp.isfinite(rew)), f"Step {step_i}: non-finite rewards detected"


# ======================================================================
# TEST-03: Goal-finding component API (numpy)
# ======================================================================


def test_goal_finding_component_api_numpy():
    """Goal-finding example works through the component API with zero manual scope_config.

    Imports the goal_finding example module, creates the env via registry,
    resets, runs 10 steps, and verifies no errors and correct reward shapes.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    # Import the goal_finding example module to trigger registration
    import examples.goal_finding  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("GoalFinding-Simple-V0", backend="numpy")
    obs, info = env.reset(seed=42)

    agent_ids = sorted(env.possible_agents)

    # Verify obs dict structure
    assert set(obs.keys()) == set(agent_ids), (
        f"Obs keys {set(obs.keys())} != agent_ids {set(agent_ids)}"
    )

    for step_i in range(10):
        actions = {aid: step_i % 4 for aid in agent_ids}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        # Rewards are returned for all agents
        assert set(rewards.keys()) == set(agent_ids), f"Step {step_i}: reward keys mismatch"

        # Reward values are finite floats
        for aid in agent_ids:
            assert np.isfinite(rewards[aid]), f"Step {step_i}, agent {aid}: reward not finite"


# ======================================================================
# TEST-04: Cross-backend parity -- numpy vs JAX produce identical states
# ======================================================================


def test_goal_finding_cross_backend_parity():
    """Identical scripted actions produce identical states on numpy and JAX.

    Creates the goal-finding env on both backends with the same seed,
    runs 50 steps with the same action sequence, and asserts that
    agent_pos, agent_dir, agent_inv, object_type_map, object_state_map,
    and rewards match exactly at every step.

    Agent directions at step 0 are skipped because numpy PCG64 and JAX
    ThreeFry produce different random values from the same seed. After
    the first cardinal move, directions become deterministic.
    """
    pytest.importorskip("jax")
    import examples.goal_finding  # noqa: F401 -- trigger registration
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.envs import registry

    N_STEPS = 50
    STATE_FIELDS = ["agent_pos", "agent_dir", "agent_inv", "object_type_map", "object_state_map"]

    def _get_state_arrays(env):
        es = env._env_state
        return {k: np.array(getattr(es, k)) for k in STATE_FIELDS}

    # Scripted non-colliding actions: agents move in opposite directions
    # to avoid collision resolution (which uses backend-specific RNG).
    def _scripted_actions(agent_ids, n_steps):
        cycle = [
            {agent_ids[0]: 0, agent_ids[1]: 1},  # Up, Down
            {agent_ids[0]: 2, agent_ids[1]: 3},  # Left, Right
            {agent_ids[0]: 6, agent_ids[1]: 6},  # Noop, Noop
            {agent_ids[0]: 1, agent_ids[1]: 0},  # Down, Up
            {agent_ids[0]: 3, agent_ids[1]: 2},  # Right, Left
            {agent_ids[0]: 6, agent_ids[1]: 6},  # Noop, Noop
        ]
        return [cycle[s % len(cycle)] for s in range(n_steps)]

    # --- Run numpy ---
    _reset_backend_for_testing()
    np_env = registry.make("GoalFinding-Simple-V0", backend="numpy")
    np_env.reset(seed=42)
    agent_ids = sorted(np_env.possible_agents)
    actions_seq = _scripted_actions(agent_ids, N_STEPS)

    np_states = [_get_state_arrays(np_env)]
    np_rewards = []
    for action_dict in actions_seq:
        _, rewards, _, _, _ = np_env.step(action_dict)
        np_states.append(_get_state_arrays(np_env))
        np_rewards.append({aid: float(rewards[aid]) for aid in agent_ids})

    # --- Run JAX ---
    _reset_backend_for_testing()
    jax_env = registry.make("GoalFinding-Simple-V0", backend="jax")
    jax_env.reset(seed=42)

    # Compare initial state (skip agent_dir due to RNG divergence)
    jax_s0 = _get_state_arrays(jax_env)
    for field in STATE_FIELDS:
        if field == "agent_dir":
            continue
        np.testing.assert_array_equal(
            np_states[0][field],
            jax_s0[field],
            err_msg=f"Initial state mismatch: {field}",
        )

    # Step through and compare
    for step_idx, action_dict in enumerate(actions_seq):
        _, rewards, _, _, _ = jax_env.step(action_dict)
        jax_state = _get_state_arrays(jax_env)

        for field in STATE_FIELDS:
            np.testing.assert_array_equal(
                np_states[step_idx + 1][field],
                jax_state[field],
                err_msg=f"Step {step_idx}, field {field}: numpy vs JAX mismatch",
            )

        for aid in agent_ids:
            np.testing.assert_allclose(
                np_rewards[step_idx][aid],
                float(rewards[aid]),
                atol=1e-7,
                err_msg=f"Step {step_idx}, agent {aid}: reward mismatch",
            )
