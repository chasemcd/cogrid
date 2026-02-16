"""Cross-backend parity tests for unified reward functions.

Verifies that each unified reward function (delivery_reward, onion_in_pot_reward,
soup_in_dish_reward) and the auto-wired compute_fn produce identical float32
reward values on numpy and JAX backends for the same scripted state inputs.

Test pattern for each reward function:
1. Build scope config (pure Python dicts, backend-agnostic) to get type_ids.
2. Reset backend, set to "numpy", call reward function, capture numpy result.
3. Reset backend, set to "jax", convert same state to JAX arrays, call same
   function, capture JAX result.
4. Compare with np.testing.assert_allclose(atol=1e-7).

Satisfies TEST-01: cross-backend parity for unified reward functions.
"""

import numpy as np
import pytest

# Trigger Overcooked object and reward registration before building scope config
import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
import cogrid.envs.overcooked.rewards  # noqa: F401
from cogrid.backend.state_view import StateView

# Build type_ids once before any backend switching (pure Python dict, no array ops)
from cogrid.core.autowire import build_scope_config_from_components

_SC = build_scope_config_from_components("overcooked")
_TYPE_IDS = _SC["type_ids"]

# Number of agents in all test scenarios
N_AGENTS = 2

# Grid dimensions for test scenarios
H, W = 5, 5


def _build_base_otm(delivery_zone_pos=None, pot_pos=None):
    """Build a minimal 5x5 object_type_map with optional objects placed.

    All cells default to 0 (empty). Walls are not needed for reward tests
    since reward functions only check type IDs, not wall_map.

    Args:
        delivery_zone_pos: Optional (row, col) to place a delivery_zone.
        pot_pos: Optional (row, col) to place a pot.

    Returns:
        (H, W) int32 numpy array.
    """
    otm = np.zeros((H, W), dtype=np.int32)
    if delivery_zone_pos is not None:
        otm[delivery_zone_pos] = _TYPE_IDS["delivery_zone"]
    if pot_pos is not None:
        otm[pot_pos] = _TYPE_IDS["pot"]
    return otm


_CORE_FIELDS = {
    "agent_pos",
    "agent_dir",
    "agent_inv",
    "wall_map",
    "object_type_map",
    "object_state_map",
}


def _dict_to_sv(fields):
    """Convert a plain dict of arrays to a StateView, filling missing core fields."""
    import numpy as _np

    defaults = dict(
        agent_pos=_np.zeros((N_AGENTS, 2), dtype=_np.int32),
        agent_dir=_np.zeros(N_AGENTS, dtype=_np.int32),
        agent_inv=_np.full((N_AGENTS, 1), -1, dtype=_np.int32),
        wall_map=_np.zeros((H, W), dtype=_np.int32),
        object_type_map=_np.zeros((H, W), dtype=_np.int32),
        object_state_map=_np.zeros((H, W), dtype=_np.int32),
    )
    extra = {}
    for k, v in fields.items():
        if k in _CORE_FIELDS:
            defaults[k] = v
        else:
            extra[k] = v
    return StateView(**defaults, extra=extra)


def _run_on_both_backends(reward_fn_name, prev_state_np, actions_np, fn_kwargs=None):
    """Run a reward function on numpy and JAX backends, return both results.

    The reward function is accessed via module attribute to pick up the
    correct xp binding after each backend switch.

    Args:
        reward_fn_name: Name of the function in rewards module
            (e.g. "delivery_reward").
        prev_state_np: Dict of numpy arrays for prev_state.
        actions_np: (n_agents,) int32 numpy array.
        fn_kwargs: Additional keyword arguments for the reward function.

    Returns:
        Tuple of (result_numpy, result_jax) as numpy arrays.
    """
    import importlib

    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    if fn_kwargs is None:
        fn_kwargs = {}

    _AR_MOD = "cogrid.envs.overcooked.rewards"

    # --- numpy path ---
    _reset_backend_for_testing()
    set_backend("numpy")

    ar_mod = importlib.import_module(_AR_MOD)
    importlib.reload(ar_mod)

    fn_np = getattr(ar_mod, reward_fn_name)
    sv_np = _dict_to_sv(prev_state_np)
    result_np = fn_np(sv_np, sv_np, actions_np, **fn_kwargs)
    result_np = np.array(result_np)

    # --- JAX path ---
    _reset_backend_for_testing()
    set_backend("jax")
    import jax.numpy as jnp

    from cogrid.backend.state_view import register_stateview_pytree

    register_stateview_pytree()

    importlib.reload(ar_mod)
    fn_jax = getattr(ar_mod, reward_fn_name)

    prev_state_jax = {k: jnp.array(v) for k, v in prev_state_np.items()}
    sv_jax = _dict_to_sv(prev_state_jax)
    actions_jax = jnp.array(actions_np)

    result_jax = fn_jax(sv_jax, sv_jax, actions_jax, **fn_kwargs)
    result_jax = np.array(result_jax)

    # Reset backend to numpy for clean state
    _reset_backend_for_testing()

    return result_np, result_jax


# ======================================================================
# Test 1: delivery_reward parity
# ======================================================================


def test_reward_parity_delivery():
    """Unified delivery_reward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds onion_soup, faces a delivery_zone at (1,3),
    performs PickupDrop (action 4). Agent 1 is idle (Noop, action 6).
    With common_reward=True (default), both agents receive the reward.
    """
    pytest.importorskip("jax")

    # Agent 0 at (1,2) facing Right (dir=0), forward position is (1,3)
    # Agent 1 at (3,3) facing Down (dir=1), forward position is (4,3)
    prev_state = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),
        "agent_inv": np.array([[_TYPE_IDS["onion_soup"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(delivery_zone_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": np.full((1, 3), -1, dtype=np.int32),
        "pot_timer": np.array([30], dtype=np.int32),
        "pot_positions": np.array([[0, 0]], dtype=np.int32),
    }
    actions = np.array([4, 6], dtype=np.int32)  # PickupDrop, Noop

    result_np, result_jax = _run_on_both_backends(
        "delivery_reward",
        prev_state,
        actions,
        fn_kwargs={
            "type_ids": _TYPE_IDS,
            "n_agents": N_AGENTS,
            "coefficient": 1.0,
            "common_reward": True,
        },
    )

    # Agent 0 earns delivery reward; common_reward=True means both get 1.0
    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="delivery_reward: numpy vs JAX mismatch"
    )

    # Sanity check: agent 0 should earn the reward (both agents get 1.0)
    assert result_np[0] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 0 reward ~1.0, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 1 reward ~1.0 (common), got {result_np[1]}"
    )


# ======================================================================
# Test 2: onion_in_pot_reward parity
# ======================================================================


def test_reward_parity_onion_in_pot():
    """Unified onion_in_pot_reward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds onion, faces a pot at (1,3) with 1 onion
    already in slot 0 (slots 1-2 empty). Performs PickupDrop.
    Agent 1 is idle. common_reward=False (default), so only agent 0
    gets the reward.
    """
    pytest.importorskip("jax")

    pot_contents = np.full((1, 3), -1, dtype=np.int32)
    pot_contents[0, 0] = _TYPE_IDS["onion"]  # 1 onion in slot 0

    prev_state = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),  # Right, Down
        "agent_inv": np.array([[_TYPE_IDS["onion"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(pot_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": pot_contents,
        "pot_timer": np.array([30], dtype=np.int32),
        "pot_positions": np.array([[1, 3]], dtype=np.int32),
    }
    actions = np.array([4, 6], dtype=np.int32)

    result_np, result_jax = _run_on_both_backends(
        "onion_in_pot_reward",
        prev_state,
        actions,
        fn_kwargs={
            "type_ids": _TYPE_IDS,
            "n_agents": N_AGENTS,
            "coefficient": 0.1,
            "common_reward": False,
        },
    )

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="onion_in_pot_reward: numpy vs JAX mismatch"
    )

    # Only agent 0 earns (common_reward=False)
    assert result_np[0] == pytest.approx(0.1, abs=1e-7), (
        f"Expected agent 0 reward ~0.1, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(0.0, abs=1e-7), (
        f"Expected agent 1 reward ~0.0, got {result_np[1]}"
    )


# ======================================================================
# Test 3: soup_in_dish_reward parity
# ======================================================================


def test_reward_parity_soup_in_dish():
    """Unified soup_in_dish_reward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds plate, faces a pot at (1,3) with timer==0
    (soup is ready). Performs PickupDrop. Agent 1 is idle.
    """
    pytest.importorskip("jax")

    # Pot is full (3 onions) and ready (timer=0)
    pot_contents = np.full((1, 3), _TYPE_IDS["onion"], dtype=np.int32)

    prev_state = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),
        "agent_inv": np.array([[_TYPE_IDS["plate"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(pot_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": pot_contents,
        "pot_timer": np.array([0], dtype=np.int32),  # ready
        "pot_positions": np.array([[1, 3]], dtype=np.int32),
    }
    actions = np.array([4, 6], dtype=np.int32)

    result_np, result_jax = _run_on_both_backends(
        "soup_in_dish_reward",
        prev_state,
        actions,
        fn_kwargs={
            "type_ids": _TYPE_IDS,
            "n_agents": N_AGENTS,
            "coefficient": 0.3,
            "common_reward": False,
        },
    )

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="soup_in_dish_reward: numpy vs JAX mismatch"
    )

    assert result_np[0] == pytest.approx(0.3, abs=1e-7), (
        f"Expected agent 0 reward ~0.3, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(0.0, abs=1e-7), (
        f"Expected agent 1 reward ~0.0, got {result_np[1]}"
    )


# ======================================================================
# Test 4: compute_rewards parity (combined)
# ======================================================================


def test_reward_parity_compute_rewards():
    """Auto-wired compute_fn produces identical combined results on numpy and JAX.

    Uses the same delivery scenario from test 1 with the auto-wired
    compute_fn from build_reward_config_from_components. Only delivery
    should trigger (agent holds soup, not onion).
    """
    pytest.importorskip("jax")

    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.core.autowire import build_reward_config_from_components

    prev_state_np = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),
        "agent_inv": np.array([[_TYPE_IDS["onion_soup"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(delivery_zone_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": np.full((1, 3), -1, dtype=np.int32),
        "pot_timer": np.array([30], dtype=np.int32),
        "pot_positions": np.array([[0, 0]], dtype=np.int32),
    }
    actions_np = np.array([4, 6], dtype=np.int32)

    # --- numpy path ---
    _reset_backend_for_testing()
    set_backend("numpy")

    reward_config_np = build_reward_config_from_components(
        "overcooked",
        n_agents=N_AGENTS,
        type_ids=_TYPE_IDS,
        action_pickup_drop_idx=4,
    )
    compute_fn_np = reward_config_np["compute_fn"]
    sv_np = _dict_to_sv(prev_state_np)
    result_np = compute_fn_np(sv_np, sv_np, actions_np, reward_config_np)
    result_np = np.array(result_np)

    # --- JAX path ---
    _reset_backend_for_testing()
    set_backend("jax")
    import jax.numpy as jnp

    from cogrid.backend.state_view import register_stateview_pytree

    register_stateview_pytree()

    reward_config_jax = build_reward_config_from_components(
        "overcooked",
        n_agents=N_AGENTS,
        type_ids=_TYPE_IDS,
        action_pickup_drop_idx=4,
    )
    compute_fn_jax = reward_config_jax["compute_fn"]

    prev_state_jax = {k: jnp.array(v) for k, v in prev_state_np.items()}
    sv_jax = _dict_to_sv(prev_state_jax)
    actions_jax = jnp.array(actions_np)

    result_jax = compute_fn_jax(sv_jax, sv_jax, actions_jax, reward_config_jax)
    result_jax = np.array(result_jax)

    # Reset
    _reset_backend_for_testing()

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="compute_fn: numpy vs JAX mismatch"
    )

    # Only delivery triggers: both agents get 1.0 (common_reward=True)
    # onion_in_pot does NOT trigger (agent holds soup, not onion)
    assert result_np[0] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 0 combined reward ~1.0, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 1 combined reward ~1.0, got {result_np[1]}"
    )
