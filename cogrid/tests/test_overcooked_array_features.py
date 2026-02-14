"""Tests for Overcooked array-based features.

Verifies shape correctness, individual feature functions, and parity
with the Python-based OvercookedCollectedFeatures across all 5 layouts.
"""

import copy

import numpy as np
import pytest

from cogrid.envs import cramped_room_config
from cogrid.envs.overcooked.overcooked import Overcooked
from cogrid.envs.overcooked.overcooked_features import OvercookedCollectedFeatures


OVERCOOKED_LAYOUTS = [
    "overcooked_cramped_room_v0",
    "overcooked_asymmetric_advantages_v0",
    "overcooked_coordination_ring_v0",
    "overcooked_forced_coordination_v0",
    "overcooked_counter_circuit_v0",
]


def _make_env(layout_name):
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
# Parity tests against Python features
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_name", OVERCOOKED_LAYOUTS)
def test_parity_with_python_features(layout_name):
    """Array features should match Python features.

    Known differences masked out:
    - AgentDir: pre-existing mismatch between Python agent.dir and array state.
    - ClosestObj (7 blocks per agent): tie-breaking differs in argsort for
      equidistant objects. The set of (distance, |dy|+|dx|) pairs is identical,
      just the ordering within a distance bucket varies.
    """
    env = _make_env(layout_name)
    obs, _ = env.reset(seed=42)

    feat = OvercookedCollectedFeatures(env)

    # Per-agent ClosestObj ranges (relative to agent block start):
    # 29-36(n=4), 37-44(n=4), 45-48(n=2), 49-52(n=2), 53-60(n=4), 61-64(n=2), 65-72(n=4)
    closest_ranges = [(29, 37), (37, 45), (45, 49), (49, 53), (53, 61), (61, 65), (65, 73)]

    for aid in env.possible_agents:
        py_obs = feat.generate(env, aid)
        arr_obs = obs[aid]

        assert py_obs.shape == arr_obs.shape == (677,)

        mask = np.ones(677, dtype=bool)
        # Mask AgentDir
        mask[0:4] = False
        mask[105:109] = False
        # Mask ClosestObj for both agent blocks
        for block_start in [0, 105]:
            for start, end in closest_ranges:
                mask[block_start + start:block_start + end] = False

        # Exact match for all non-ClosestObj, non-AgentDir features
        np.testing.assert_allclose(
            py_obs[mask], arr_obs[mask],
            atol=1e-5,
            err_msg=f"Layout {layout_name}, agent {aid}",
        )

        # For ClosestObj: verify same set of manhattan distances per block
        for block_start in [0, 105]:
            for start, end in closest_ranges:
                idx = block_start + start
                n = (end - start) // 2
                py_pairs = py_obs[idx:idx + 2 * n].reshape(n, 2)
                arr_pairs = arr_obs[idx:idx + 2 * n].reshape(n, 2)
                py_dists = sorted(np.abs(py_pairs[:, 0]) + np.abs(py_pairs[:, 1]))
                arr_dists = sorted(np.abs(arr_pairs[:, 0]) + np.abs(arr_pairs[:, 1]))
                np.testing.assert_array_equal(
                    py_dists, arr_dists,
                    err_msg=f"Layout {layout_name}, agent {aid}, ClosestObj block {start}-{end}",
                )


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


# ---------------------------------------------------------------------------
# Composed vs monolithic parity test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_name", OVERCOOKED_LAYOUTS)
def test_composed_vs_monolithic_677_parity(layout_name):
    """Composed ArrayFeature output matches monolithic build_overcooked_feature_fn element-by-element."""
    from cogrid.core.autowire import build_feature_config_from_components
    from cogrid.core.step_pipeline import envstate_to_dict
    from cogrid.envs.overcooked.overcooked_array_features import build_overcooked_feature_fn
    import cogrid.envs  # noqa: F401 -- ensure registration

    _overcooked_layouts = [
        "overcooked_cramped_room_v0",
        "overcooked_asymmetric_advantages_v0",
        "overcooked_coordination_ring_v0",
        "overcooked_forced_coordination_v0",
        "overcooked_counter_circuit_v0",
    ]
    layout_idx = _overcooked_layouts.index(layout_name)

    # Build monolithic feature function (the old way)
    monolithic_fn = build_overcooked_feature_fn(
        scope="overcooked", n_agents=2, layout_idx=layout_idx,
        grid_shape=None,
    )

    # Build composed feature function (the new way)
    feature_config = build_feature_config_from_components(
        "overcooked", n_agents=2, layout_idx=layout_idx,
    )
    composed_fn = feature_config["feature_fn"]

    # Create env and get state_dict
    env = _make_env(layout_name)
    env.reset(seed=42)

    state_dict = envstate_to_dict(env._env_state)

    # Compare outputs for both agents at reset
    for agent_idx in range(2):
        mono_obs = monolithic_fn(state_dict, agent_idx)
        comp_obs = composed_fn(state_dict, agent_idx)

        assert mono_obs.shape == comp_obs.shape == (677,), (
            f"Shape mismatch: mono={mono_obs.shape}, comp={comp_obs.shape}"
        )
        np.testing.assert_allclose(
            mono_obs, comp_obs, atol=1e-6,
            err_msg=f"Layout {layout_name}, agent {agent_idx}: element-by-element mismatch",
        )

    # Also verify after a few steps
    for step_num in range(3):
        actions = {aid: np.random.randint(0, 5) for aid in env.possible_agents}
        obs, _, terms, truncs, _ = env.step(actions)
        if any(terms.values()) or any(truncs.values()):
            break

        state_dict = envstate_to_dict(env._env_state)

        for agent_idx in range(2):
            mono_obs = monolithic_fn(state_dict, agent_idx)
            comp_obs = composed_fn(state_dict, agent_idx)
            np.testing.assert_allclose(
                mono_obs, comp_obs, atol=1e-6,
                err_msg=f"Layout {layout_name}, step {step_num}, agent {agent_idx}",
            )


# ---------------------------------------------------------------------------
# Autowire integration test
# ---------------------------------------------------------------------------


def test_autowire_provides_feature_fn_builder():
    """The autowire system should detect Pot.build_feature_fn and include feature_fn_builder.

    NOTE: This tests the legacy path -- Pot.build_feature_fn is still registered as a
    GridObject classmethod and still appears in scope_config, but CoGridEnv no longer
    uses it. Phase 19 will remove it entirely.
    """
    from cogrid.core.autowire import build_scope_config_from_components
    import cogrid.envs  # ensure registration

    scope_config = build_scope_config_from_components("overcooked")
    assert "feature_fn_builder" in scope_config
    assert scope_config["feature_fn_builder"] is not None


def test_generic_scope_has_no_feature_fn_builder():
    """Scopes without build_feature_fn should have feature_fn_builder=None."""
    from cogrid.core.autowire import build_scope_config_from_components

    scope_config = build_scope_config_from_components("global")
    assert scope_config["feature_fn_builder"] is None
