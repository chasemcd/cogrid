"""vmap correctness tests for batched environment rollouts.

Verifies that:
1. jax.vmap(env.jax_reset)(keys) produces correctly shaped batched output
   at 1024 parallel environments (TEST-01 shapes).
2. jax.vmap(env.jax_step)(batched_state, batched_actions) executes without
   error and produces correct shapes at 1024 environments (TEST-02 shapes).
3. Single-env reset output matches the corresponding slice of the batched
   reset output for sampled indices (TEST-03 reset parity).
4. Single-env step output matches the corresponding slice of the batched
   step output after multiple steps with varied actions (TEST-04 step parity).
5. jax.jit(jax.vmap(fn)) composition works without error (TEST-05 jit+vmap).

These tests satisfy Phase 4 Success Criteria 1 (vmap executes correctly at
1024 envs with correct shapes) and 2 (each environment in the batch produces
identical results to running it individually).
"""

import pytest
import numpy as np


# Registry IDs for the 3 test layouts
LAYOUTS = [
    "Overcooked-CrampedRoom-V0",
    "Overcooked-AsymmetricAdvantages-V0",
    "Overcooked-CoordinationRing-V0",
]

# Batch size for vmap testing
BATCH_SIZE = 1024

# Number of steps for parity testing with varied actions
N_PARITY_STEPS = 5

# Number of sample indices to spot-check for parity
N_SAMPLE_INDICES = 8

# Dynamic state fields to compare between single-env and batched.
# Core fields are direct EnvState attributes; extra fields live in extra_state
# dict with scope prefix (e.g. "overcooked.pot_contents").
CORE_STATE_FIELDS = [
    "agent_pos", "agent_dir", "agent_inv",
    "wall_map", "object_type_map", "object_state_map",
    "rng_key", "time",
]
EXTRA_STATE_FIELDS = ["pot_contents", "pot_timer", "pot_positions"]
DYNAMIC_STATE_FIELDS = CORE_STATE_FIELDS + EXTRA_STATE_FIELDS


def _get_state_field(es, field):
    """Get a state field from EnvState, checking extra_state for scope-prefixed keys."""
    if field not in EXTRA_STATE_FIELDS:
        return getattr(es, field)
    for key, val in es.extra_state.items():
        short_key = key.split(".", 1)[-1] if "." in key else key
        if short_key == field:
            return val
    raise AttributeError(f"Field {field} not found in EnvState or extra_state")


def _create_jax_env(registry_id, seed=42):
    """Create and reset a JAX-backend Overcooked environment.

    Handles backend switching via _reset_backend_for_testing() so that
    multiple environments can coexist within a single test process.

    Args:
        registry_id: Registered environment ID string.
        seed: RNG seed for reset.

    Returns:
        The environment after reset.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    env = registry.make(registry_id, backend="jax")
    env.reset(seed=seed)
    return env


def _compare_state_fields(single_state, batched_state, idx, label):
    """Compare all dynamic state fields between single-env and batched slice.

    Args:
        single_state: EnvState from a single-env call.
        batched_state: EnvState from a vmapped call (with batch dim).
        idx: Index into the batch dimension to compare.
        label: Human-readable label for error messages.
    """
    jax = pytest.importorskip("jax")

    for field_name in DYNAMIC_STATE_FIELDS:
        single_val = _get_state_field(single_state, field_name)
        batched_val = _get_state_field(batched_state, field_name)

        if field_name == "rng_key":
            # JAX key types are opaque; extract underlying integer data
            single_data = np.array(jax.random.key_data(single_val))
            batched_data = np.array(jax.random.key_data(batched_val[idx]))
            np.testing.assert_array_equal(
                single_data, batched_data,
                err_msg=(
                    f"{label}, field {field_name}: "
                    f"rng_key mismatch at batch index {idx}"
                ),
            )
        else:
            np.testing.assert_array_equal(
                np.array(single_val),
                np.array(batched_val[idx]),
                err_msg=(
                    f"{label}, field {field_name}: "
                    f"mismatch at batch index {idx}"
                ),
            )


def _get_sample_indices():
    """Return the N_SAMPLE_INDICES indices to spot-check for parity."""
    return [
        0, 1, 2, 3,
        BATCH_SIZE // 2,
        BATCH_SIZE - 3,
        BATCH_SIZE - 2,
        BATCH_SIZE - 1,
    ]


# ======================================================================
# TEST 1: vmap reset produces correct shapes at 1024 envs
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_vmap_reset_shapes(layout):
    """vmap(jax_reset) at 1024 envs produces correctly shaped batched output.

    Verifies batch dimensions on observations, agent_pos, wall_map, and
    that static meta fields (n_agents) are NOT batched.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    env = _create_jax_env(layout)
    reset_fn = env.jax_reset
    n_agents = env.config["num_agents"]

    # Get obs_dim from a single-env reset
    single_state, single_obs = reset_fn(jax.random.key(99))
    obs_dim = single_obs.shape[-1]

    # Create batch of PRNG keys
    keys = jax.random.split(jax.random.key(0), BATCH_SIZE)

    # Batched reset
    batched_state, batched_obs = jax.vmap(reset_fn)(keys)

    # Observation shape: (BATCH_SIZE, n_agents, obs_dim)
    assert batched_obs.shape == (BATCH_SIZE, n_agents, obs_dim), (
        f"Layout {layout}: expected obs shape "
        f"({BATCH_SIZE}, {n_agents}, {obs_dim}), got {batched_obs.shape}"
    )

    # agent_pos shape: (BATCH_SIZE, n_agents, 2)
    assert batched_state.agent_pos.shape == (BATCH_SIZE, n_agents, 2), (
        f"Layout {layout}: expected agent_pos shape "
        f"({BATCH_SIZE}, {n_agents}, 2), "
        f"got {batched_state.agent_pos.shape}"
    )

    # wall_map has leading batch dim
    assert batched_state.wall_map.shape[0] == BATCH_SIZE, (
        f"Layout {layout}: expected wall_map leading dim {BATCH_SIZE}, "
        f"got {batched_state.wall_map.shape[0]}"
    )

    # All observations are finite (no NaN/inf)
    assert np.all(np.isfinite(np.array(batched_obs))), (
        f"Layout {layout}: batched obs contains non-finite values"
    )

    # Static meta field n_agents is NOT batched (scalar)
    assert batched_state.n_agents == n_agents, (
        f"Layout {layout}: expected n_agents={n_agents} (scalar), "
        f"got {batched_state.n_agents}"
    )


# ======================================================================
# TEST 2: vmap step produces correct shapes at 1024 envs
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_vmap_step_shapes(layout):
    """vmap(jax_step) at 1024 envs produces correctly shaped batched output.

    Verifies batch dimensions on observations, rewards, done, and new state.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    env = _create_jax_env(layout)
    reset_fn = env.jax_reset
    step_fn = env.jax_step
    n_agents = env.config["num_agents"]

    # Get obs_dim from single-env
    single_state, single_obs = reset_fn(jax.random.key(99))
    obs_dim = single_obs.shape[-1]

    # Batched reset
    keys = jax.random.split(jax.random.key(0), BATCH_SIZE)
    batched_state, _ = jax.vmap(reset_fn)(keys)

    # Batched actions: all noop (zeros)
    batched_actions = jnp.zeros((BATCH_SIZE, n_agents), dtype=jnp.int32)

    # Batched step
    new_state, obs, rew, terminateds, truncateds, infos = jax.vmap(step_fn)(
        batched_state, batched_actions
    )

    # obs shape: (BATCH_SIZE, n_agents, obs_dim)
    assert obs.shape == (BATCH_SIZE, n_agents, obs_dim), (
        f"Layout {layout}: expected obs shape "
        f"({BATCH_SIZE}, {n_agents}, {obs_dim}), got {obs.shape}"
    )

    # rew shape: (BATCH_SIZE, n_agents)
    assert rew.shape == (BATCH_SIZE, n_agents), (
        f"Layout {layout}: expected rew shape "
        f"({BATCH_SIZE}, {n_agents}), got {rew.shape}"
    )

    # terminateds shape: (BATCH_SIZE, n_agents)
    assert terminateds.shape == (BATCH_SIZE, n_agents), (
        f"Layout {layout}: expected terminateds shape "
        f"({BATCH_SIZE}, {n_agents}), got {terminateds.shape}"
    )

    # truncateds shape: (BATCH_SIZE, n_agents)
    assert truncateds.shape == (BATCH_SIZE, n_agents), (
        f"Layout {layout}: expected truncateds shape "
        f"({BATCH_SIZE}, {n_agents}), got {truncateds.shape}"
    )

    # infos is empty dict
    assert infos == {}, (
        f"Layout {layout}: expected infos={{}}, got {infos}"
    )

    # new_state.agent_pos shape: (BATCH_SIZE, n_agents, 2)
    assert new_state.agent_pos.shape == (BATCH_SIZE, n_agents, 2), (
        f"Layout {layout}: expected new agent_pos shape "
        f"({BATCH_SIZE}, {n_agents}, 2), "
        f"got {new_state.agent_pos.shape}"
    )


# ======================================================================
# TEST 3: vmap reset parity -- single-env matches batched slice
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_vmap_reset_parity(layout):
    """Single-env reset output matches the corresponding batched reset slice.

    For N_SAMPLE_INDICES spot-check indices, verifies that running reset_fn
    on a single key produces the same state and obs as the corresponding
    slice of the vmapped reset output.
    """
    jax = pytest.importorskip("jax")

    env = _create_jax_env(layout)
    reset_fn = env.jax_reset

    # Create batch of PRNG keys
    keys = jax.random.split(jax.random.key(0), BATCH_SIZE)

    # Batched reset
    batched_state, batched_obs = jax.vmap(reset_fn)(keys)

    # Sample indices to spot-check
    sample_indices = _get_sample_indices()

    for i in sample_indices:
        # Single-env reset with the same key
        single_state, single_obs = reset_fn(keys[i])

        # Compare observations
        np.testing.assert_array_equal(
            np.array(single_obs),
            np.array(batched_obs[i]),
            err_msg=(
                f"Layout {layout}, reset parity, index {i}: "
                f"obs mismatch"
            ),
        )

        # Compare all dynamic state fields
        _compare_state_fields(
            single_state, batched_state, i,
            label=f"Layout {layout}, reset parity",
        )


# ======================================================================
# TEST 4: vmap step parity -- single-env matches batched slice after
#          multiple steps with varied actions
# ======================================================================


@pytest.mark.parametrize("layout", LAYOUTS)
def test_vmap_step_parity(layout):
    """Single-env step output matches the corresponding batched step slice.

    Runs N_PARITY_STEPS steps with random varied actions and verifies
    that for N_SAMPLE_INDICES spot-check indices, the single-env trajectory
    matches the batched trajectory exactly.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    env = _create_jax_env(layout)
    reset_fn = env.jax_reset
    step_fn = env.jax_step
    n_agents = env.config["num_agents"]

    # Create batch of PRNG keys
    keys = jax.random.split(jax.random.key(0), BATCH_SIZE)

    # Batched reset
    batched_state, _ = jax.vmap(reset_fn)(keys)

    # Generate N_PARITY_STEPS batches of random actions
    rng = np.random.default_rng(123)
    actions_list = []
    for _ in range(N_PARITY_STEPS):
        actions_np = rng.integers(0, 7, size=(BATCH_SIZE, n_agents))
        actions_list.append(jnp.array(actions_np, dtype=jnp.int32))

    # Step through all steps with vmapped step
    vmapped_step = jax.vmap(step_fn)
    for step_i in range(N_PARITY_STEPS):
        batched_state, batched_obs, batched_rew, batched_term, batched_trunc, _ = (
            vmapped_step(batched_state, actions_list[step_i])
        )

    # Sample indices to spot-check
    sample_indices = _get_sample_indices()

    for i in sample_indices:
        # Run single-env trajectory with the same key and actions
        single_state, _ = reset_fn(keys[i])

        for step_i in range(N_PARITY_STEPS):
            single_actions = actions_list[step_i][i]
            single_state, single_obs, single_rew, single_term, single_trunc, _ = (
                step_fn(single_state, single_actions)
            )

        # Compare final observations
        np.testing.assert_array_equal(
            np.array(single_obs),
            np.array(batched_obs[i]),
            err_msg=(
                f"Layout {layout}, step parity, index {i}: "
                f"final obs mismatch after {N_PARITY_STEPS} steps"
            ),
        )

        # Compare final rewards
        np.testing.assert_allclose(
            np.array(single_rew),
            np.array(batched_rew[i]),
            atol=1e-7,
            err_msg=(
                f"Layout {layout}, step parity, index {i}: "
                f"final rewards mismatch after {N_PARITY_STEPS} steps"
            ),
        )

        # Compare final terminateds
        np.testing.assert_array_equal(
            np.array(single_term),
            np.array(batched_term[i]),
            err_msg=(
                f"Layout {layout}, step parity, index {i}: "
                f"final terminateds mismatch after {N_PARITY_STEPS} steps"
            ),
        )

        # Compare final truncateds
        np.testing.assert_array_equal(
            np.array(single_trunc),
            np.array(batched_trunc[i]),
            err_msg=(
                f"Layout {layout}, step parity, index {i}: "
                f"final truncateds mismatch after {N_PARITY_STEPS} steps"
            ),
        )

        # Compare all dynamic state fields
        _compare_state_fields(
            single_state, batched_state, i,
            label=(
                f"Layout {layout}, step parity after "
                f"{N_PARITY_STEPS} steps"
            ),
        )


# ======================================================================
# TEST 5: jit(vmap(fn)) composition -- the optimal pattern for perf
# ======================================================================


def test_vmap_jit_composition():
    """jax.jit(jax.vmap(fn)) composition works without error.

    Verifies that the recommended jit(vmap(...)) pattern for maximum
    performance executes correctly for both reset and 3 steps.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    env = _create_jax_env("Overcooked-CrampedRoom-V0")
    n_agents = env.config["num_agents"]

    # Wrap with jit(vmap(...)) -- the optimal pattern
    vmapped_reset = jax.jit(jax.vmap(env.jax_reset))
    vmapped_step = jax.jit(jax.vmap(env.jax_step))

    # Batched reset
    keys = jax.random.split(jax.random.key(0), BATCH_SIZE)
    batched_state, batched_obs = vmapped_reset(keys)

    # Verify shapes after reset
    assert batched_obs.shape[0] == BATCH_SIZE, (
        f"jit(vmap(reset)): expected batch dim {BATCH_SIZE}, "
        f"got {batched_obs.shape[0]}"
    )

    # Run 3 steps
    for step_i in range(3):
        actions = jnp.zeros((BATCH_SIZE, n_agents), dtype=jnp.int32)
        batched_state, batched_obs, batched_rew, batched_term, batched_trunc, _ = (
            vmapped_step(batched_state, actions)
        )

    # Verify shapes after steps
    assert batched_obs.shape[0] == BATCH_SIZE, (
        f"jit(vmap(step)): expected obs batch dim {BATCH_SIZE}, "
        f"got {batched_obs.shape[0]}"
    )
    assert batched_rew.shape == (BATCH_SIZE, n_agents), (
        f"jit(vmap(step)): expected rew shape "
        f"({BATCH_SIZE}, {n_agents}), got {batched_rew.shape}"
    )
    assert batched_term.shape == (BATCH_SIZE, n_agents), (
        f"jit(vmap(step)): expected terminateds shape "
        f"({BATCH_SIZE}, {n_agents}), got {batched_term.shape}"
    )
    assert batched_trunc.shape == (BATCH_SIZE, n_agents), (
        f"jit(vmap(step)): expected truncateds shape "
        f"({BATCH_SIZE}, {n_agents}), got {batched_trunc.shape}"
    )
    assert batched_state.agent_pos.shape == (BATCH_SIZE, n_agents, 2), (
        f"jit(vmap(step)): expected agent_pos shape "
        f"({BATCH_SIZE}, {n_agents}, 2), "
        f"got {batched_state.agent_pos.shape}"
    )
