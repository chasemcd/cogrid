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
