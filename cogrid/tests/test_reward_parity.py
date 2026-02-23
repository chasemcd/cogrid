"""Cross-backend parity tests for unified reward functions.

Verifies that each unified reward class (DeliveryReward, OnionInPotReward,
SoupInDishReward) and the auto-wired compute_fn produce identical float32
reward values on numpy and JAX backends for the same scripted state inputs.

Test pattern for each reward class:
1. Build scope config (pure Python dicts, backend-agnostic) to get type_ids.
2. Reset backend, set to "numpy", instantiate reward and call .compute(),
   capture numpy result.
3. Reset backend, set to "jax", convert same state to JAX arrays, instantiate
   reward and call .compute(), capture JAX result.
4. Compare with np.testing.assert_allclose(atol=1e-7).

Satisfies TEST-01: cross-backend parity for unified reward functions.
"""

import numpy as np
import pytest

# Trigger Overcooked object registration before building scope config
import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
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


def _run_on_both_backends(reward_cls, reward_kwargs, prev_state_np, actions_np, reward_config_base):
    """Run a reward class on numpy and JAX backends, return both results.

    Args:
        reward_cls: Reward class (e.g. DeliveryReward).
        reward_kwargs: kwargs for reward_cls.__init__.
        prev_state_np: Dict of numpy arrays for prev_state.
        actions_np: (n_agents,) int32 numpy array.
        reward_config_base: Base reward_config dict (type_ids, n_agents, etc.).

    Returns:
        Tuple of (result_numpy, result_jax) as numpy arrays.
    """
    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    # --- numpy path ---
    _reset_backend_for_testing()
    set_backend("numpy")

    inst_np = reward_cls(**reward_kwargs)
    sv_np = _dict_to_sv(prev_state_np)
    result_np = inst_np.compute(sv_np, sv_np, actions_np, reward_config_base)
    result_np = np.array(result_np)

    # --- JAX path ---
    _reset_backend_for_testing()
    set_backend("jax")
    import jax.numpy as jnp

    from cogrid.backend.state_view import register_stateview_pytree

    register_stateview_pytree()

    inst_jax = reward_cls(**reward_kwargs)
    prev_state_jax = {k: jnp.array(v) for k, v in prev_state_np.items()}
    sv_jax = _dict_to_sv(prev_state_jax)
    actions_jax = jnp.array(actions_np)

    result_jax = inst_jax.compute(sv_jax, sv_jax, actions_jax, reward_config_base)
    result_jax = np.array(result_jax)

    # Reset backend to numpy for clean state
    _reset_backend_for_testing()

    return result_np, result_jax


# ======================================================================
# Test 1: DeliveryReward parity
# ======================================================================


def test_reward_parity_delivery():
    """DeliveryReward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds onion_soup, faces a delivery_zone at (1,3),
    performs PickupDrop (action 4). Agent 1 is idle (Noop, action 6).
    With common_reward=True, both agents receive the reward.
    """
    pytest.importorskip("jax")

    from cogrid.envs.overcooked.rewards import DeliveryReward

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
    actions = np.array([4, 6], dtype=np.int32)

    reward_config = {
        "type_ids": _TYPE_IDS,
        "n_agents": N_AGENTS,
        "action_pickup_drop_idx": 4,
    }

    result_np, result_jax = _run_on_both_backends(
        DeliveryReward,
        {"coefficient": 1.0, "common_reward": True},
        prev_state,
        actions,
        reward_config,
    )

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="DeliveryReward: numpy vs JAX mismatch"
    )

    assert result_np[0] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 0 reward ~1.0, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(1.0, abs=1e-7), (
        f"Expected agent 1 reward ~1.0 (common), got {result_np[1]}"
    )


# ======================================================================
# Test 2: OnionInPotReward parity
# ======================================================================


def test_reward_parity_onion_in_pot():
    """OnionInPotReward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds onion, faces a pot at (1,3) with 1 onion
    already in slot 0 (slots 1-2 empty). Performs PickupDrop.
    Agent 1 is idle. common_reward=False, so only agent 0 gets the reward.
    """
    pytest.importorskip("jax")

    from cogrid.envs.overcooked.rewards import OnionInPotReward

    pot_contents = np.full((1, 3), -1, dtype=np.int32)
    pot_contents[0, 0] = _TYPE_IDS["onion"]

    prev_state = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),
        "agent_inv": np.array([[_TYPE_IDS["onion"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(pot_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": pot_contents,
        "pot_timer": np.array([30], dtype=np.int32),
        "pot_positions": np.array([[1, 3]], dtype=np.int32),
    }
    actions = np.array([4, 6], dtype=np.int32)

    reward_config = {
        "type_ids": _TYPE_IDS,
        "n_agents": N_AGENTS,
        "action_pickup_drop_idx": 4,
    }

    result_np, result_jax = _run_on_both_backends(
        OnionInPotReward,
        {"coefficient": 0.1, "common_reward": False},
        prev_state,
        actions,
        reward_config,
    )

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="OnionInPotReward: numpy vs JAX mismatch"
    )

    assert result_np[0] == pytest.approx(0.1, abs=1e-7), (
        f"Expected agent 0 reward ~0.1, got {result_np[0]}"
    )
    assert result_np[1] == pytest.approx(0.0, abs=1e-7), (
        f"Expected agent 1 reward ~0.0, got {result_np[1]}"
    )


# ======================================================================
# Test 3: SoupInDishReward parity
# ======================================================================


def test_reward_parity_soup_in_dish():
    """SoupInDishReward produces identical results on numpy and JAX.

    Scenario: Agent 0 holds plate, faces a pot at (1,3) with timer==0
    (soup is ready). Performs PickupDrop. Agent 1 is idle.
    """
    pytest.importorskip("jax")

    from cogrid.envs.overcooked.rewards import SoupInDishReward

    pot_contents = np.full((1, 3), _TYPE_IDS["onion"], dtype=np.int32)

    prev_state = {
        "agent_pos": np.array([[1, 2], [3, 3]], dtype=np.int32),
        "agent_dir": np.array([0, 1], dtype=np.int32),
        "agent_inv": np.array([[_TYPE_IDS["plate"]], [-1]], dtype=np.int32),
        "object_type_map": _build_base_otm(pot_pos=(1, 3)),
        "object_state_map": np.zeros((H, W), dtype=np.int32),
        "pot_contents": pot_contents,
        "pot_timer": np.array([0], dtype=np.int32),
        "pot_positions": np.array([[1, 3]], dtype=np.int32),
    }
    actions = np.array([4, 6], dtype=np.int32)

    reward_config = {
        "type_ids": _TYPE_IDS,
        "n_agents": N_AGENTS,
        "action_pickup_drop_idx": 4,
    }

    result_np, result_jax = _run_on_both_backends(
        SoupInDishReward,
        {"coefficient": 0.3, "common_reward": False},
        prev_state,
        actions,
        reward_config,
    )

    np.testing.assert_allclose(
        result_np, result_jax, atol=1e-7, err_msg="SoupInDishReward: numpy vs JAX mismatch"
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
    """build_reward_config compute_fn produces identical combined results on numpy and JAX.

    Uses the same delivery scenario from test 1 with the auto-wired
    compute_fn from build_reward_config. Only delivery should trigger
    (agent holds soup, not onion).
    """
    pytest.importorskip("jax")

    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.core.autowire import build_reward_config
    from cogrid.envs.overcooked.rewards import (
        DeliveryReward,
        OnionInPotReward,
        SoupInDishReward,
    )

    reward_instances = [
        DeliveryReward(coefficient=1.0, common_reward=True),
        OnionInPotReward(coefficient=0.1, common_reward=False),
        SoupInDishReward(coefficient=0.3, common_reward=False),
    ]

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

    reward_config_np = build_reward_config(
        reward_instances,
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

    reward_config_jax = build_reward_config(
        reward_instances,
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
