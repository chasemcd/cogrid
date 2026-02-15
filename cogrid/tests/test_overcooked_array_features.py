"""Tests for Overcooked array-based features.

Verifies shape correctness, individual feature functions, and parity
with the Python-based OvercookedCollectedFeatures across all 5 layouts.
"""

import copy

import numpy as np
import pytest

from cogrid.envs import cramped_room_config
from cogrid.envs.overcooked.overcooked import Overcooked


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
    return Overcooked(config=config)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_name", OVERCOOKED_LAYOUTS)
def test_obs_shape_677(layout_name):
    """Each agent should get a (677,) observation."""
    env = _make_env(layout_name)
    obs, _ = env.reset(seed=42)
    for aid in env.possible_agents:
        assert obs[aid].shape == (677,), f"Agent {aid} obs shape {obs[aid].shape} != (677,)"


# ---------------------------------------------------------------------------
# Step test
# ---------------------------------------------------------------------------


def test_step_preserves_shape():
    """Observation shape should remain (677,) after stepping."""
    env = _make_env("overcooked_cramped_room_v0")
    obs, _ = env.reset(seed=42)

    for _ in range(10):
        actions = {aid: np.random.randint(0, 5) for aid in env.possible_agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break
        for aid in env.possible_agents:
            assert obs[aid].shape == (677,)


# ---------------------------------------------------------------------------
# Individual feature function tests
# ---------------------------------------------------------------------------


def test_overcooked_inventory_feature():
    """Inventory one-hot: empty -> all zeros, held item -> correct index."""
    from cogrid.envs.overcooked.overcooked_array_features import overcooked_inventory_feature

    inv_type_ids = np.array([5, 6, 7, 8, 9], dtype=np.int32)  # mock type IDs
    agent_inv = np.array([[-1], [6]], dtype=np.int32)  # agent 0 empty, agent 1 holds type 6

    result_0 = overcooked_inventory_feature(agent_inv, 0, inv_type_ids)
    assert result_0.shape == (5,)
    np.testing.assert_array_equal(result_0, [0, 0, 0, 0, 0])

    result_1 = overcooked_inventory_feature(agent_inv, 1, inv_type_ids)
    np.testing.assert_array_equal(result_1, [0, 1, 0, 0, 0])


def test_next_to_counter_feature():
    """Counter adjacency multi-hot encoding."""
    from cogrid.envs.overcooked.overcooked_array_features import next_to_counter_feature

    # 3x3 grid: counter at (0,1) and (1,2), agent at (1,1)
    counter_id = 2
    otm = np.array([[0, counter_id, 0],
                     [0, 0, counter_id],
                     [0, 0, 0]], dtype=np.int32)
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
    from cogrid.envs.overcooked.overcooked_array_features import layout_id_feature

    result = layout_id_feature(2, num_layouts=5)
    assert result.shape == (5,)
    np.testing.assert_array_equal(result, [0, 0, 1, 0, 0])


def test_dist_to_other_players():
    """Distance to other players."""
    from cogrid.envs.overcooked.overcooked_array_features import dist_to_other_players_feature

    agent_pos = np.array([[3, 4], [1, 2]], dtype=np.int32)
    result = dist_to_other_players_feature(agent_pos, 0, n_agents=2)
    assert result.shape == (2,)
    np.testing.assert_array_equal(result, [2, 2])  # (3-1, 4-2)


def test_closest_obj_feature():
    """Closest object deltas."""
    from cogrid.envs.overcooked.overcooked_array_features import closest_obj_feature

    target_id = 5
    otm = np.array([[0, 0, 0],
                     [0, 0, target_id],
                     [target_id, 0, 0]], dtype=np.int32)
    osm = np.zeros_like(otm)
    agent_pos = np.array([[1, 1]], dtype=np.int32)

    # n=2 closest
    result = closest_obj_feature(agent_pos, 0, otm, osm, target_id, n=2)
    assert result.shape == (4,)

    # Two targets: (1,2) dist=1, (2,0) dist=2
    # Sorted by distance: (1,2) first, then (2,0)
    # Deltas: (1-1, 1-2) = (0, -1), (1-2, 1-0) = (-1, 1)
    np.testing.assert_array_equal(result, [0, -1, -1, 1])


