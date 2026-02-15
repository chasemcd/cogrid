"""Tests for ArrayFeature registration and composition layer.

Covers:
- Feature type registration (per_agent, global, duplicates, validation)
- compose_feature_fns: ego-centric ordering, alphabetical within sections
- obs_dim_for_features: dimension computation without calling feature functions
- Output dtype coercion to float32
"""

import numpy as np
import pytest

from cogrid.core.array_features import ArrayFeature
from cogrid.core.component_registry import (
    get_feature_types,
    register_feature_type,
)
from cogrid.backend.state_view import StateView


def _sv(**kwargs):
    """Build a minimal StateView for testing, filling missing core fields with zeros."""
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


# ===================================================================
# Registration tests (should PASS -- validates Plan 01 infrastructure)
# ===================================================================


def test_register_feature_type_per_agent():
    """Register a per_agent=True feature, verify get_feature_types returns it."""

    @register_feature_type("test_per_agent_reg", scope="test_af_reg_pa")
    class _PerAgentFeat(ArrayFeature):
        per_agent = True
        obs_dim = 3

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.zeros(3)
            return fn

    metas = get_feature_types(scope="test_af_reg_pa")
    assert len(metas) == 1
    assert metas[0].feature_id == "test_per_agent_reg"
    assert metas[0].per_agent is True
    assert metas[0].obs_dim == 3
    assert metas[0].cls is _PerAgentFeat


def test_register_feature_type_global():
    """Register a per_agent=False feature, verify metadata."""

    @register_feature_type("test_global_reg", scope="test_af_reg_gl")
    class _GlobalFeat(ArrayFeature):
        per_agent = False
        obs_dim = 5

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict):
                return np.zeros(5)
            return fn

    metas = get_feature_types(scope="test_af_reg_gl")
    assert len(metas) == 1
    assert metas[0].feature_id == "test_global_reg"
    assert metas[0].per_agent is False
    assert metas[0].obs_dim == 5


def test_duplicate_feature_id_raises():
    """Registering two different classes with the same feature_id raises ValueError."""

    @register_feature_type("dup_feat", scope="test_af_dup")
    class _Dup1(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd, ai: np.zeros(1)

    with pytest.raises(ValueError, match="Duplicate feature type"):

        @register_feature_type("dup_feat", scope="test_af_dup")
        class _Dup2(ArrayFeature):
            per_agent = True
            obs_dim = 1

            @classmethod
            def build_feature_fn(cls, scope):
                return lambda sd, ai: np.zeros(1)


def test_missing_per_agent_raises():
    """Class without per_agent attribute raises TypeError on decoration."""
    with pytest.raises(TypeError, match="per_agent"):

        @register_feature_type("no_pa", scope="test_af_no_pa")
        class _NoPa(ArrayFeature):
            obs_dim = 1

            @classmethod
            def build_feature_fn(cls, scope):
                return lambda sd, ai: np.zeros(1)


def test_missing_obs_dim_raises():
    """Class without obs_dim attribute raises TypeError on decoration."""
    with pytest.raises(TypeError, match="obs_dim"):

        @register_feature_type("no_od", scope="test_af_no_od")
        class _NoOd(ArrayFeature):
            per_agent = True

            @classmethod
            def build_feature_fn(cls, scope):
                return lambda sd, ai: np.zeros(1)


def test_features_sorted_alphabetically():
    """get_feature_types returns features sorted by feature_id."""

    @register_feature_type("charlie", scope="test_af_sort")
    class _Charlie(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd, ai: np.zeros(1)

    @register_feature_type("alpha", scope="test_af_sort")
    class _Alpha(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd, ai: np.zeros(1)

    @register_feature_type("bravo", scope="test_af_sort")
    class _Bravo(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd, ai: np.zeros(1)

    metas = get_feature_types(scope="test_af_sort")
    ids = [m.feature_id for m in metas]
    assert ids == ["alpha", "bravo", "charlie"]


# ===================================================================
# Composition tests (should FAIL -- compose_feature_fns not yet implemented)
# ===================================================================


def test_compose_single_per_agent_feature():
    """Single per_agent feature with 2 agents produces (6,) output."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("feat_pa", scope="test_af_compose_single_pa")
    class _Feat(ArrayFeature):
        per_agent = True
        obs_dim = 3

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array(
                    [agent_idx * 10, agent_idx * 10 + 1, agent_idx * 10 + 2],
                    dtype=np.int32,
                )
            return fn

    composed = compose_feature_fns(["feat_pa"], "test_af_compose_single_pa", n_agents=2)
    result = composed({}, agent_idx=0)
    # Agent 0 features: [0, 1, 2], Agent 1 features: [10, 11, 12]
    expected = np.array([0, 1, 2, 10, 11, 12], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (6,)


def test_compose_single_global_feature():
    """Single global feature with 2 agents produces (2,) output."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("feat_gl", scope="test_af_compose_single_gl")
    class _Feat(ArrayFeature):
        per_agent = False
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict):
                return np.array([42, 43], dtype=np.float32)
            return fn

    composed = compose_feature_fns(["feat_gl"], "test_af_compose_single_gl", n_agents=2)
    result = composed({}, agent_idx=0)
    expected = np.array([42, 43], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (2,)


def test_compose_per_agent_and_global():
    """Per-agent (3,) + global (2,) with 2 agents produces (8,) output."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("pa_feat", scope="test_af_compose_pa_gl")
    class _PA(ArrayFeature):
        per_agent = True
        obs_dim = 3

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array(
                    [agent_idx * 10, agent_idx * 10 + 1, agent_idx * 10 + 2],
                    dtype=np.float32,
                )
            return fn

    @register_feature_type("gl_feat", scope="test_af_compose_pa_gl")
    class _GL(ArrayFeature):
        per_agent = False
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict):
                return np.array([99, 100], dtype=np.float32)
            return fn

    composed = compose_feature_fns(
        ["pa_feat", "gl_feat"], "test_af_compose_pa_gl", n_agents=2
    )
    result = composed({}, agent_idx=0)
    # Per-agent focal(0): [0,1,2], other(1): [10,11,12], global: [99,100]
    expected = np.array([0, 1, 2, 10, 11, 12, 99, 100], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (8,)


def test_ego_centric_ordering():
    """Per-agent feature with 3 agents: focal first, others ascending, skip focal."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("ego_feat", scope="test_af_ego_order")
    class _EgoFeat(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([agent_idx], dtype=np.float32)
            return fn

    composed = compose_feature_fns(["ego_feat"], "test_af_ego_order", n_agents=3)
    # Focal is agent 1: expect [1, 0, 2]
    result = composed({}, agent_idx=1)
    expected = np.array([1, 0, 2], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_alphabetical_feature_ordering():
    """Per-agent features ordered alphabetically: alpha before beta."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("beta", scope="test_af_alpha_order")
    class _Beta(ArrayFeature):
        per_agent = True
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([20, 21], dtype=np.float32)
            return fn

    @register_feature_type("alpha", scope="test_af_alpha_order")
    class _Alpha(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([10], dtype=np.float32)
            return fn

    composed = compose_feature_fns(
        ["beta", "alpha"], "test_af_alpha_order", n_agents=1
    )
    result = composed({}, agent_idx=0)
    # alpha (1,) comes before beta (2,) alphabetically
    expected = np.array([10, 20, 21], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)


def test_obs_dim_for_features():
    """obs_dim_for_features computes total dim without calling feature functions."""
    from cogrid.core.array_features import obs_dim_for_features

    @register_feature_type("dim_pa", scope="test_af_obs_dim")
    class _PA(ArrayFeature):
        per_agent = True
        obs_dim = 4

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd, ai: np.zeros(4)

    @register_feature_type("dim_gl", scope="test_af_obs_dim")
    class _GL(ArrayFeature):
        per_agent = False
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            return lambda sd: np.zeros(2)

    # per_agent: 4 * 2 agents = 8, global: 2 => total = 10
    total = obs_dim_for_features(["dim_pa", "dim_gl"], "test_af_obs_dim", n_agents=2)
    assert total == 10


def test_output_is_float32():
    """Composed output is float32 even when individual features return int32."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("int_feat", scope="test_af_dtype")
    class _IntFeat(ArrayFeature):
        per_agent = True
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([1, 2], dtype=np.int32)
            return fn

    composed = compose_feature_fns(["int_feat"], "test_af_dtype", n_agents=1)
    result = composed({}, agent_idx=0)
    assert result.dtype == np.float32


# ===================================================================
# Core ArrayFeature subclass parity tests
# ===================================================================


def test_agent_dir_parity():
    """AgentDir ArrayFeature produces output identical to agent_dir_feature."""
    from cogrid.feature_space.array_features import AgentDir, agent_dir_feature

    state_dict = _sv(agent_dir=np.array([2, 0], dtype=np.int32))

    fn = AgentDir.build_feature_fn("global")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = agent_dir_feature(state_dict.agent_dir, idx)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (4,)
        assert result.dtype == np.int32


def test_agent_position_parity():
    """AgentPosition ArrayFeature produces output identical to agent_pos_feature."""
    from cogrid.feature_space.array_features import AgentPosition, agent_pos_feature

    state_dict = _sv(agent_pos=np.array([[3, 5], [1, 2]], dtype=np.int32))

    fn = AgentPosition.build_feature_fn("global")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = agent_pos_feature(state_dict.agent_pos, idx)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2,)
        assert result.dtype == np.int32


def test_can_move_direction_parity():
    """CanMoveDirection ArrayFeature produces output identical to can_move_direction_feature."""
    from cogrid.feature_space.array_features import (
        CanMoveDirection,
        can_move_direction_feature,
    )

    import cogrid.envs  # noqa: F401 -- triggers global scope registration

    wall_map = np.zeros((5, 5), dtype=np.int32)
    object_type_map = np.zeros((5, 5), dtype=np.int32)
    agent_pos = np.array([[0, 0], [2, 2]], dtype=np.int32)
    can_overlap_table = np.ones(10, dtype=np.int32)

    state_dict = _sv(
        agent_pos=agent_pos,
        wall_map=wall_map,
        object_type_map=object_type_map,
    )

    fn = CanMoveDirection.build_feature_fn("global")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = can_move_direction_feature(
            agent_pos, idx, wall_map, object_type_map, can_overlap_table
        )
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (4,)
        assert result.dtype == np.int32


def test_inventory_parity():
    """Inventory ArrayFeature produces output identical to inventory_feature."""
    from cogrid.feature_space.array_features import Inventory, inventory_feature

    state_dict = _sv(agent_inv=np.array([[-1], [3]], dtype=np.int32))

    fn = Inventory.build_feature_fn("global")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = inventory_feature(state_dict.agent_inv, idx)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (1,)
        assert result.dtype == np.int32


def test_all_four_registered_global():
    """All four core features are registered to global scope with correct metadata."""
    import cogrid.feature_space.array_features  # noqa: F401 -- triggers registration

    metas = get_feature_types(scope="global")
    meta_by_id = {m.feature_id: m for m in metas}

    expected = {
        "agent_dir": {"per_agent": True, "obs_dim": 4},
        "agent_position": {"per_agent": True, "obs_dim": 2},
        "can_move_direction": {"per_agent": True, "obs_dim": 4},
        "inventory": {"per_agent": True, "obs_dim": 1},
    }

    for feat_id, attrs in expected.items():
        assert feat_id in meta_by_id, f"{feat_id} not registered in global scope"
        assert meta_by_id[feat_id].per_agent == attrs["per_agent"]
        assert meta_by_id[feat_id].obs_dim == attrs["obs_dim"]


# ===================================================================
# Overcooked ArrayFeature subclass parity tests
# ===================================================================


def test_overcooked_inventory_parity():
    """OvercookedInventory ArrayFeature produces output identical to overcooked_inventory_feature."""
    import cogrid.envs  # noqa: F401 -- triggers overcooked scope registration
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import (
        OvercookedInventory,
        overcooked_inventory_feature,
    )

    inv_type_order = ["onion", "onion_soup", "plate", "tomato", "tomato_soup"]
    inv_type_ids = np.array(
        [object_to_idx(name, scope="overcooked") for name in inv_type_order],
        dtype=np.int32,
    )

    # Agent 0 holds onion, agent 1 holds nothing
    agent_inv = np.array(
        [[object_to_idx("onion", scope="overcooked")], [-1]], dtype=np.int32
    )
    state_dict = _sv(agent_inv=agent_inv)

    fn = OvercookedInventory.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = overcooked_inventory_feature(agent_inv, idx, inv_type_ids)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (5,)
        assert result.dtype == np.int32


def test_next_to_counter_parity():
    """NextToCounter ArrayFeature produces output identical to next_to_counter_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import (
        NextToCounter,
        next_to_counter_feature,
    )

    counter_type_id = object_to_idx("counter", scope="overcooked")

    object_type_map = np.zeros((5, 5), dtype=np.int32)
    # Place counters adjacent to agent at (2,2)
    object_type_map[2, 3] = counter_type_id  # right
    object_type_map[1, 2] = counter_type_id  # up

    agent_pos = np.array([[2, 2], [0, 0]], dtype=np.int32)
    state_dict = _sv(agent_pos=agent_pos, object_type_map=object_type_map)

    fn = NextToCounter.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = next_to_counter_feature(
            agent_pos, idx, object_type_map, counter_type_id
        )
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (4,)
        assert result.dtype == np.int32


def test_next_to_pot_parity():
    """NextToPot ArrayFeature produces output identical to next_to_pot_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import (
        NextToPot,
        next_to_pot_feature,
    )

    pot_type_id = object_to_idx("pot", scope="overcooked")

    object_type_map = np.zeros((5, 5), dtype=np.int32)
    # Place pot to the right of agent at (2,2)
    object_type_map[2, 3] = pot_type_id

    agent_pos = np.array([[2, 2], [0, 0]], dtype=np.int32)
    pot_positions = np.array([[2, 3]], dtype=np.int32)
    pot_contents = np.zeros((1, 3), dtype=np.int32)  # empty pot, capacity 3
    pot_timer = np.array([0], dtype=np.int32)

    state_dict = _sv(
        agent_pos=agent_pos,
        object_type_map=object_type_map,
        pot_positions=pot_positions,
        pot_contents=pot_contents,
        pot_timer=pot_timer,
    )

    fn = NextToPot.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = next_to_pot_feature(
            agent_pos, idx, object_type_map, pot_type_id,
            pot_positions, pot_contents, pot_timer,
        )
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (16,)
        assert result.dtype == np.int32


def test_closest_obj_parity():
    """ClosestObj ArrayFeature produces output identical to closest_obj_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import closest_obj_feature

    onion_type_id = object_to_idx("onion", scope="overcooked")

    object_type_map = np.zeros((5, 5), dtype=np.int32)
    object_state_map = np.zeros((5, 5), dtype=np.int32)
    # Place onions at known positions
    object_type_map[0, 0] = onion_type_id
    object_type_map[4, 4] = onion_type_id

    agent_pos = np.array([[2, 2], [1, 1]], dtype=np.int32)

    state_dict = _sv(
        agent_pos=agent_pos,
        object_type_map=object_type_map,
        object_state_map=object_state_map,
    )

    # Get the registered ClosestOnion subclass from registry
    metas = get_feature_types(scope="overcooked")
    meta_by_id = {m.feature_id: m for m in metas}
    assert "closest_onion" in meta_by_id, "closest_onion not registered"

    fn = meta_by_id["closest_onion"].cls.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = closest_obj_feature(
            agent_pos, idx, object_type_map, object_state_map,
            onion_type_id, 4,
        )
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (8,)

    # Verify all 7 ClosestObj variants are registered
    expected_closest_ids = [
        "closest_onion", "closest_plate", "closest_plate_stack",
        "closest_onion_stack", "closest_onion_soup", "closest_delivery_zone",
        "closest_counter",
    ]
    for feat_id in expected_closest_ids:
        assert feat_id in meta_by_id, f"{feat_id} not registered in overcooked scope"


def test_ordered_pot_features_parity():
    """OrderedPotFeatures ArrayFeature produces output identical to ordered_pot_features."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import (
        OrderedPotFeatures,
        ordered_pot_features,
    )

    onion_id = object_to_idx("onion", scope="overcooked")
    tomato_id = object_to_idx("tomato", scope="overcooked")

    agent_pos = np.array([[2, 2], [0, 0]], dtype=np.int32)
    pot_positions = np.array([[1, 1], [3, 3]], dtype=np.int32)
    pot_contents = np.zeros((2, 3), dtype=np.int32)
    pot_contents[0, 0] = onion_id  # first pot has one onion
    pot_timer = np.array([0, 0], dtype=np.int32)

    state_dict = _sv(
        agent_pos=agent_pos,
        pot_positions=pot_positions,
        pot_contents=pot_contents,
        pot_timer=pot_timer,
    )

    fn = OrderedPotFeatures.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = ordered_pot_features(
            agent_pos, idx, pot_positions, pot_contents, pot_timer,
            max_num_pots=2, onion_id=onion_id, tomato_id=tomato_id,
        )
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (24,)


def test_dist_to_other_players_parity():
    """DistToOtherPlayers ArrayFeature produces output identical to dist_to_other_players_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.overcooked_array_features import (
        DistToOtherPlayers,
        dist_to_other_players_feature,
    )

    agent_pos = np.array([[2, 3], [4, 1]], dtype=np.int32)
    state_dict = _sv(agent_pos=agent_pos)

    fn = DistToOtherPlayers.build_feature_fn("overcooked")

    for idx in (0, 1):
        result = fn(state_dict, idx)
        expected = dist_to_other_players_feature(agent_pos, idx, n_agents=2)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (2,)
        assert result.dtype == np.int32


def test_all_overcooked_per_agent_registered():
    """All per-agent Overcooked features are discoverable via get_feature_types."""
    import cogrid.envs  # noqa: F401
    import cogrid.envs.overcooked.overcooked_array_features  # noqa: F401

    metas = get_feature_types(scope="overcooked")
    meta_by_id = {m.feature_id: m for m in metas}

    expected_feature_ids = [
        "overcooked_inventory",
        "next_to_counter",
        "next_to_pot",
        "closest_onion",
        "closest_plate",
        "closest_plate_stack",
        "closest_onion_stack",
        "closest_onion_soup",
        "closest_delivery_zone",
        "closest_counter",
        "ordered_pot_features",
        "dist_to_other_players",
    ]

    for feat_id in expected_feature_ids:
        assert feat_id in meta_by_id, f"{feat_id} not registered in overcooked scope"
        assert meta_by_id[feat_id].per_agent is True, f"{feat_id} should be per_agent=True"


# ===================================================================
# Overcooked global ArrayFeature subclass parity tests
# ===================================================================


def test_layout_id_parity():
    """LayoutID ArrayFeature produces output identical to layout_id_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.envs.overcooked.overcooked_array_features import (
        LayoutID,
        layout_id_feature,
    )

    # Test with default _layout_idx = 0
    fn = LayoutID.build_feature_fn("overcooked")
    result = fn({})
    expected = layout_id_feature(0)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (5,)
    assert result.dtype == np.int32

    # Test with overridden _layout_idx = 3
    LayoutID._layout_idx = 3
    fn = LayoutID.build_feature_fn("overcooked")
    result = fn({})
    expected = layout_id_feature(3)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (5,)
    assert result.dtype == np.int32

    # Reset to default
    LayoutID._layout_idx = 0


def test_environment_layout_parity():
    """EnvironmentLayout ArrayFeature produces output identical to environment_layout_feature."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.grid_object import object_to_idx
    from cogrid.envs.overcooked.overcooked_array_features import (
        EnvironmentLayout,
        environment_layout_feature,
    )

    # Create a mock 5x5 object_type_map with known type IDs
    layout_type_names = ["counter", "pot", "onion", "plate", "onion_stack", "plate_stack"]
    layout_type_ids = [object_to_idx(name, scope="overcooked") for name in layout_type_names]

    object_type_map = np.zeros((5, 5), dtype=np.int32)
    # Place known objects
    object_type_map[0, 0] = layout_type_ids[0]  # counter
    object_type_map[1, 1] = layout_type_ids[1]  # pot
    object_type_map[2, 3] = layout_type_ids[2]  # onion
    object_type_map[3, 4] = layout_type_ids[5]  # plate_stack

    state_dict = _sv(object_type_map=object_type_map)

    fn = EnvironmentLayout.build_feature_fn("overcooked")
    result = fn(state_dict)
    expected = environment_layout_feature(object_type_map, layout_type_ids, (11, 7))
    np.testing.assert_array_equal(result, expected)
    # 6 types * 11 * 7 = 462
    assert result.shape == (462,)
    assert result.dtype == np.int32


def test_all_overcooked_features_registered():
    """All 14 Overcooked features (12 per-agent + 2 global) are registered."""
    import cogrid.envs  # noqa: F401
    import cogrid.envs.overcooked.overcooked_array_features  # noqa: F401

    metas = get_feature_types(scope="overcooked")
    meta_by_id = {m.feature_id: m for m in metas}

    # 12 per-agent features
    per_agent_features = {
        "overcooked_inventory": 5,
        "next_to_counter": 4,
        "next_to_pot": 16,
        "closest_onion": 8,
        "closest_plate": 8,
        "closest_plate_stack": 4,
        "closest_onion_stack": 4,
        "closest_onion_soup": 8,
        "closest_delivery_zone": 4,
        "closest_counter": 8,
        "ordered_pot_features": 24,
        "dist_to_other_players": 2,
    }

    # 2 global features
    global_features = {
        "layout_id": 5,
        "environment_layout": 462,
    }

    for feat_id, expected_dim in per_agent_features.items():
        assert feat_id in meta_by_id, f"{feat_id} not registered in overcooked scope"
        assert meta_by_id[feat_id].per_agent is True, f"{feat_id} should be per_agent=True"
        assert meta_by_id[feat_id].obs_dim == expected_dim, (
            f"{feat_id} obs_dim={meta_by_id[feat_id].obs_dim}, expected {expected_dim}"
        )

    for feat_id, expected_dim in global_features.items():
        assert feat_id in meta_by_id, f"{feat_id} not registered in overcooked scope"
        assert meta_by_id[feat_id].per_agent is False, f"{feat_id} should be per_agent=False"
        assert meta_by_id[feat_id].obs_dim == expected_dim, (
            f"{feat_id} obs_dim={meta_by_id[feat_id].obs_dim}, expected {expected_dim}"
        )

    # Total: 14 features
    assert len(metas) == 14, f"Expected 14 overcooked features, got {len(metas)}"


# ===================================================================
# Multi-scope and preserve_order composition tests
# ===================================================================


def test_compose_preserve_order():
    """preserve_order=True keeps caller-specified order (not alphabetical)."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("po_charlie", scope="test_af_preserve_order")
    class _Charlie(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([30], dtype=np.float32)
            return fn

    @register_feature_type("po_alpha", scope="test_af_preserve_order")
    class _Alpha(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([10], dtype=np.float32)
            return fn

    @register_feature_type("po_bravo", scope="test_af_preserve_order")
    class _Bravo(ArrayFeature):
        per_agent = True
        obs_dim = 1

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([20], dtype=np.float32)
            return fn

    # With preserve_order=True, the order should be charlie, alpha, bravo
    composed = compose_feature_fns(
        ["po_charlie", "po_alpha", "po_bravo"],
        "test_af_preserve_order",
        n_agents=1,
        preserve_order=True,
    )
    result = composed({}, agent_idx=0)
    expected = np.array([30, 10, 20], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

    # Without preserve_order (default), order is alphabetical: alpha, bravo, charlie
    composed_default = compose_feature_fns(
        ["po_charlie", "po_alpha", "po_bravo"],
        "test_af_preserve_order",
        n_agents=1,
    )
    result_default = composed_default({}, agent_idx=0)
    expected_default = np.array([10, 20, 30], dtype=np.float32)
    np.testing.assert_array_equal(result_default, expected_default)


def test_compose_multi_scope():
    """scopes parameter merges features from multiple scopes."""
    from cogrid.core.array_features import compose_feature_fns

    @register_feature_type("ms_feat_a", scope="test_scope_a")
    class _FeatA(ArrayFeature):
        per_agent = True
        obs_dim = 2

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([1, 2], dtype=np.float32)
            return fn

    @register_feature_type("ms_feat_b", scope="test_scope_b")
    class _FeatB(ArrayFeature):
        per_agent = True
        obs_dim = 3

        @classmethod
        def build_feature_fn(cls, scope):
            def fn(state_dict, agent_idx):
                return np.array([3, 4, 5], dtype=np.float32)
            return fn

    # Use scopes to merge both
    composed = compose_feature_fns(
        ["ms_feat_a", "ms_feat_b"],
        "test_scope_a",
        n_agents=1,
        scopes=["test_scope_a", "test_scope_b"],
    )
    result = composed({}, agent_idx=0)
    # Alphabetical order: ms_feat_a (2,) then ms_feat_b (3,)
    expected = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.shape == (5,)
