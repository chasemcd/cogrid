"""Tests for Overcooked array-based features.

Verifies shape correctness, individual feature functions, and parity
with the Python-based OvercookedCollectedFeatures across all 5 layouts.
"""

import copy

import numpy as np
import pytest

from cogrid.envs import cramped_room_config

OVERCOOKED_LAYOUTS = [
    "overcooked_cramped_room_v0",
    "overcooked_asymmetric_advantages_v0",
    "overcooked_coordination_ring_v0",
    "overcooked_forced_coordination_v0",
    "overcooked_counter_circuit_v0",
]


def _make_env(layout_name):
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    config = copy.deepcopy(cramped_room_config)
    config["grid"]["layout"] = layout_name
    from cogrid.cogrid_env import CoGridEnv
    from cogrid.envs.overcooked.agent import OvercookedAgent

    return CoGridEnv(config=config, agent_class=OvercookedAgent)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_name", OVERCOOKED_LAYOUTS)
def test_obs_shape_210(layout_name):
    """Each agent should get a (661,) observation."""
    env = _make_env(layout_name)
    obs, _ = env.reset(seed=42)
    for aid in env.possible_agents:
        assert obs[aid].shape == (661,), f"Agent {aid} obs shape {obs[aid].shape} != (661,)"


# ---------------------------------------------------------------------------
# Step test
# ---------------------------------------------------------------------------


def test_step_preserves_shape():
    """Observation shape should remain (661,) after stepping."""
    env = _make_env("overcooked_cramped_room_v0")
    obs, _ = env.reset(seed=42)

    for _ in range(10):
        actions = {aid: np.random.randint(0, 5) for aid in env.possible_agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break
        for aid in env.possible_agents:
            assert obs[aid].shape == (661,)


# ---------------------------------------------------------------------------
# Individual feature function tests
# ---------------------------------------------------------------------------


def test_overcooked_inventory_feature():
    """Inventory one-hot: empty -> all zeros, held item -> correct index."""
    from cogrid.envs.overcooked.features import overcooked_inventory_feature

    inv_type_ids = np.array([5, 6, 7, 8, 9], dtype=np.int32)  # mock type IDs
    agent_inv = np.array([[-1], [6]], dtype=np.int32)  # agent 0 empty, agent 1 holds type 6

    result_0 = overcooked_inventory_feature(agent_inv, 0, inv_type_ids)
    assert result_0.shape == (5,)
    np.testing.assert_array_equal(result_0, [0, 0, 0, 0, 0])

    result_1 = overcooked_inventory_feature(agent_inv, 1, inv_type_ids)
    np.testing.assert_array_equal(result_1, [0, 1, 0, 0, 0])


def test_next_to_counter_feature():
    """Counter adjacency multi-hot encoding."""
    from cogrid.envs.overcooked.features import next_to_counter_feature

    # 3x3 grid: counter at (0,1) and (1,2), agent at (1,1)
    counter_id = 2
    otm = np.array([[0, counter_id, 0], [0, 0, counter_id], [0, 0, 0]], dtype=np.int32)
    agent_pos = np.array([[1, 1]], dtype=np.int32)

    result = next_to_counter_feature(agent_pos, 0, otm, counter_id)
    assert result.shape == (4,)
    # Directions: R(0,1), L(0,-1), D(1,0), U(-1,0)
    # R: (1,2) = counter -> 1
    # L: (1,0) = 0 -> 0
    # D: (2,1) = 0 -> 0
    # U: (0,1) = counter -> 1
    np.testing.assert_array_equal(result, [1, 0, 0, 1])


def test_layout_id_feature():
    """One-hot layout encoding."""
    from cogrid.envs.overcooked.features import layout_id_feature

    result = layout_id_feature(2, num_layouts=5)
    assert result.shape == (5,)
    np.testing.assert_array_equal(result, [0, 0, 1, 0, 0])


def test_dist_to_other_players():
    """Distance to other players."""
    from cogrid.envs.overcooked.features import dist_to_other_players_feature

    agent_pos = np.array([[3, 4], [1, 2]], dtype=np.int32)
    result = dist_to_other_players_feature(agent_pos, 0, n_agents=2)
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [2, 2])  # (3-1, 4-2)


def test_closest_obj_feature():
    """Closest object deltas."""
    from cogrid.envs.overcooked.features import closest_obj_feature

    target_id = 5
    otm = np.array([[0, 0, 0], [0, 0, target_id], [target_id, 0, 0]], dtype=np.int32)
    osm = np.zeros_like(otm)
    agent_pos = np.array([[1, 1]], dtype=np.int32)

    # n=2 closest
    result = closest_obj_feature(agent_pos, 0, otm, osm, target_id, n=2)
    assert result.shape == (4,)

    # Two targets: (1,2) dist=1, (2,0) dist=2
    # Sorted by distance: (1,2) first, then (2,0)
    # Deltas: (1-1, 1-2) = (0, -1), (1-2, 1-0) = (-1, 1)
    np.testing.assert_array_equal(result, [0, -1, -1, 1])


# ---------------------------------------------------------------------------
# OrderObservation tests
# ---------------------------------------------------------------------------

from cogrid.backend.state_view import StateView  # noqa: E402


def _sv(**kwargs):
    """Build a minimal StateView, filling missing core fields with zeros."""
    defaults = dict(
        agent_pos=np.zeros((2, 2), dtype=np.int32),
        agent_dir=np.zeros(2, dtype=np.int32),
        agent_inv=np.full((2, 1), -1, dtype=np.int32),
        wall_map=np.zeros((5, 5), dtype=np.int32),
        object_type_map=np.zeros((5, 5), dtype=np.int32),
        object_state_map=np.zeros((5, 5), dtype=np.int32),
    )
    extra = {}
    for k, v in kwargs.items():
        if k in defaults:
            defaults[k] = v
        else:
            extra[k] = v
    return StateView(**defaults, extra=extra)


def test_order_observation_shape():
    """OrderObservation output shape is (9,) = 3 orders * (2 recipe_onehot + 1 norm_time)."""
    from cogrid.envs.overcooked.features import OrderObservation

    fn = OrderObservation.build_feature_fn("overcooked")

    state = _sv(
        order_recipe=np.array([0, -1, -1], dtype=np.int32),
        order_timer=np.array([100, 0, 0], dtype=np.int32),
    )
    result = fn(state)
    assert result.shape == (9,)
    assert result.dtype == np.float32


def test_order_observation_encoding():
    """Verify encoding: active order gets one-hot + normalized time, empty slots zeros."""
    from cogrid.envs.overcooked.features import OrderObservation

    fn = OrderObservation.build_feature_fn("overcooked")

    # One active order: recipe_idx=0, timer=100 (time_limit=200 -> 0.5)
    # Two empty slots: recipe=-1, timer=0
    state = _sv(
        order_recipe=np.array([0, -1, -1], dtype=np.int32),
        order_timer=np.array([100, 0, 0], dtype=np.int32),
    )
    result = fn(state)

    # Slot 0: recipe 0 one-hot = [1.0, 0.0], norm_time = 100/200 = 0.5
    np.testing.assert_allclose(result[0:3], [1.0, 0.0, 0.5])
    # Slot 1: empty -> [0.0, 0.0, 0.0]
    np.testing.assert_allclose(result[3:6], [0.0, 0.0, 0.0])
    # Slot 2: empty -> [0.0, 0.0, 0.0]
    np.testing.assert_allclose(result[6:9], [0.0, 0.0, 0.0])


def test_order_observation_no_orders():
    """When order arrays are absent, output is all zeros (9,)."""
    from cogrid.envs.overcooked.features import OrderObservation

    fn = OrderObservation.build_feature_fn("overcooked")

    # No order_recipe or order_timer in state
    state = _sv()
    result = fn(state)
    assert result.shape == (9,)
    np.testing.assert_array_equal(result, np.zeros(9, dtype=np.float32))


# ---------------------------------------------------------------------------
# Dynamic OvercookedInventory tests
# ---------------------------------------------------------------------------


def test_inventory_config_driven_matches_defaults():
    """Config-driven OvercookedInventory uses default types and encodes onion correctly."""
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.features import (
        _DEFAULT_PICKUPABLE_TYPES,
        OvercookedInventory,
    )

    standard = sorted(_DEFAULT_PICKUPABLE_TYPES)
    fn = OvercookedInventory.build_feature_fn("overcooked")

    onion_idx = standard.index("onion")
    onion_id = object_to_idx("onion", scope="overcooked")
    state = _sv(agent_inv=np.array([[onion_id], [-1]], dtype=np.int32))

    result = fn(state, 0)
    assert result.shape == (len(standard),)
    assert result[onion_idx] == 1
    assert np.sum(result) == 1  # exactly one-hot


def test_inventory_config_driven_all_types():
    """Config-driven OvercookedInventory: correct one-hot for each standard pickupable type."""
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.features import (
        _DEFAULT_PICKUPABLE_TYPES,
        OvercookedInventory,
    )

    standard = sorted(_DEFAULT_PICKUPABLE_TYPES)
    fn = OvercookedInventory.build_feature_fn("overcooked")

    for name in standard:
        type_id = object_to_idx(name, scope="overcooked")
        state = _sv(agent_inv=np.array([[type_id], [-1]], dtype=np.int32))
        result = fn(state, 0)
        assert result.shape == (len(standard),)

        expected_idx = standard.index(name)
        expected = np.zeros(len(standard), dtype=np.int32)
        expected[expected_idx] = 1
        np.testing.assert_array_equal(
            result, expected, err_msg=f"Failed for type '{name}' at index {expected_idx}"
        )


def test_inventory_obs_dim_matches_defaults():
    """Verify OvercookedInventory.obs_dim equals default pickupable count."""
    from cogrid.envs.overcooked.features import (
        _DEFAULT_PICKUPABLE_TYPES,
        OvercookedInventory,
    )

    assert OvercookedInventory.obs_dim == len(_DEFAULT_PICKUPABLE_TYPES), (
        f"OvercookedInventory.obs_dim={OvercookedInventory.obs_dim} "
        f"!= default count={len(_DEFAULT_PICKUPABLE_TYPES)}"
    )


def test_compute_obs_dim_without_config():
    """compute_obs_dim without env_config returns default obs_dim."""
    from cogrid.envs.overcooked.features import (
        OrderObservation,
        OvercookedInventory,
    )

    assert OvercookedInventory.compute_obs_dim("overcooked") == 5
    assert OrderObservation.compute_obs_dim("overcooked") == 9


def test_compute_obs_dim_with_config():
    """compute_obs_dim with env_config reads dimensions from config."""
    from cogrid.envs.overcooked.features import (
        OrderObservation,
        OvercookedInventory,
    )

    # Custom pickupable_types with 3 types -> obs_dim=3
    cfg = {"pickupable_types": ["onion", "plate", "tomato"]}
    assert OvercookedInventory.compute_obs_dim("overcooked", cfg) == 3

    # Custom recipes (3 recipes) + custom orders -> obs_dim changes
    cfg = {
        "recipes": [{"r": 1}, {"r": 2}, {"r": 3}],
        "orders": {"max_active": 2},
    }
    # 2 * (3 + 1) = 8
    assert OrderObservation.compute_obs_dim("overcooked", cfg) == 8


def test_inventory_isolation_from_registry():
    """Creating a standard env produces the same obs_dim even after make_ingredient_and_stack."""
    from cogrid.envs.overcooked.features import OvercookedInventory
    from cogrid.envs.overcooked.overcooked_grid_objects import make_ingredient_and_stack

    # Standard config
    cfg = {"pickupable_types": ["onion", "onion_soup", "plate", "tomato", "tomato_soup"]}

    dim_before = OvercookedInventory.compute_obs_dim("overcooked", cfg)

    # Pollute the global registry
    make_ingredient_and_stack(
        "mushroom_test_iso", "m", (128, 128, 0), "mushroom_test_iso_stack", "M"
    )

    dim_after = OvercookedInventory.compute_obs_dim("overcooked", cfg)
    assert dim_before == dim_after == 5

    # Without config, the default also stays at 5 (reads _DEFAULT_PICKUPABLE_TYPES)
    dim_no_cfg = OvercookedInventory.compute_obs_dim("overcooked")
    assert dim_no_cfg == 5
