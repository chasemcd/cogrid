"""Comprehensive tests for the Phase 10 component registration infrastructure.

Covers:
- ComponentMetadata and RewardMetadata frozen dataclasses
- Classmethod discovery via @register_object_type decorator
- Signature validation (rejects instance methods and wrong params)
- Duplicate char detection within scope
- Reward registration via @register_reward_type decorator
- Query API (all components, tickable, extra_state, rewards)
- Backward compatibility with existing objects and old-style register_object()
"""

import pytest

from cogrid.core.component_registry import (
    ComponentMetadata,
    RewardMetadata,
    get_all_components,
    get_component_metadata,
    get_components_with_extra_state,
    get_reward_types,
    get_tickable_components,
    register_reward_type,
)
from cogrid.core.grid_object import GridObj, register_object_type
from cogrid.core.array_rewards import ArrayReward


# ---------------------------------------------------------------------------
# 1. ComponentMetadata dataclass
# ---------------------------------------------------------------------------


def test_component_metadata_dataclass():
    """ComponentMetadata is frozen and convenience properties work."""
    meta_empty = ComponentMetadata(
        scope="test_cm_1",
        object_id="obj_a",
        cls=GridObj,
        char="1",
        properties={},
        methods={},
    )
    # Frozen -- cannot mutate
    with pytest.raises(AttributeError):
        meta_empty.scope = "other"

    assert meta_empty.has_tick is False
    assert meta_empty.has_extra_state is False

    meta_full = ComponentMetadata(
        scope="test_cm_1",
        object_id="obj_b",
        cls=GridObj,
        char="2",
        properties={},
        methods={
            "build_tick_fn": lambda: None,
            "extra_state_schema": lambda: None,
        },
    )
    assert meta_full.has_tick is True
    assert meta_full.has_extra_state is True


# ---------------------------------------------------------------------------
# 2. RewardMetadata dataclass
# ---------------------------------------------------------------------------


def test_reward_metadata_dataclass():
    """RewardMetadata is frozen and all fields accessible."""
    meta = RewardMetadata(
        scope="test_rm_1",
        reward_id="rew_a",
        cls=object,
    )
    with pytest.raises(AttributeError):
        meta.scope = "other"

    assert meta.scope == "test_rm_1"
    assert meta.reward_id == "rew_a"
    assert meta.cls is object


# ---------------------------------------------------------------------------
# 3. register_object_type discovers classmethods
# ---------------------------------------------------------------------------


def test_register_object_type_discovers_classmethods():
    """Decorator discovers build_tick_fn and extra_state_schema classmethods."""

    @register_object_type("test_obj_tick", scope="test_phase10_discover")
    class _TestTickObj(GridObj):
        char = "\x80"  # unique char

        @classmethod
        def build_tick_fn(cls):
            pass

        @classmethod
        def extra_state_schema(cls):
            pass

    meta = get_component_metadata("test_obj_tick", scope="test_phase10_discover")
    assert meta is not None
    assert "build_tick_fn" in meta.methods
    assert "extra_state_schema" in meta.methods
    assert meta.has_tick is True
    assert meta.has_extra_state is True


# ---------------------------------------------------------------------------
# 4. register_object_type -- no classmethods
# ---------------------------------------------------------------------------


def test_register_object_type_no_classmethods():
    """Bare GridObj subclass has empty methods dict."""

    @register_object_type("test_obj_bare", scope="test_phase10_nomethod")
    class _TestBareObj(GridObj):
        char = "\x81"

    meta = get_component_metadata("test_obj_bare", scope="test_phase10_nomethod")
    assert meta is not None
    assert meta.methods == {}
    assert meta.has_tick is False
    assert meta.has_extra_state is False


# ---------------------------------------------------------------------------
# 5. register_object_type -- all classmethods
# ---------------------------------------------------------------------------


def test_register_object_type_all_classmethods():
    """All classmethods discovered when present."""

    @register_object_type("test_obj_all", scope="test_phase10_allmethod")
    class _TestAllObj(GridObj):
        char = "\x82"

        @classmethod
        def build_tick_fn(cls):
            pass

        @classmethod
        def extra_state_schema(cls):
            pass

        @classmethod
        def extra_state_builder(cls):
            pass

    meta = get_component_metadata("test_obj_all", scope="test_phase10_allmethod")
    assert len(meta.methods) == 3
    for name in ("build_tick_fn", "extra_state_schema", "extra_state_builder"):
        assert name in meta.methods


# ---------------------------------------------------------------------------
# 6. Signature validation rejects instance method
# ---------------------------------------------------------------------------


def test_signature_validation_rejects_instance_method():
    """Instance method (not @classmethod) raises TypeError mentioning 'self'."""
    with pytest.raises(TypeError, match="self"):

        @register_object_type("test_obj_inst", scope="test_phase10_inst")
        class _TestInstObj(GridObj):
            char = "\x83"

            def build_tick_fn(self):  # wrong -- should be @classmethod
                pass


# ---------------------------------------------------------------------------
# 7. Signature validation rejects extra params
# ---------------------------------------------------------------------------


def test_signature_validation_rejects_extra_params():
    """Classmethod with unexpected parameter raises TypeError."""
    with pytest.raises(TypeError, match="some_arg"):

        @register_object_type("test_obj_extra", scope="test_phase10_extra")
        class _TestExtraObj(GridObj):
            char = "\x84"

            @classmethod
            def build_tick_fn(cls, some_arg):
                pass


# ---------------------------------------------------------------------------
# 8. Duplicate char -- same scope raises
# ---------------------------------------------------------------------------


def test_duplicate_char_same_scope_raises():
    """Two classes with same char in same scope raises ValueError."""

    @register_object_type("test_obj_dup1", scope="test_phase10_dup")
    class _TestDup1(GridObj):
        char = "\x85"

    with pytest.raises(ValueError, match="Duplicate char"):

        @register_object_type("test_obj_dup2", scope="test_phase10_dup")
        class _TestDup2(GridObj):
            char = "\x85"  # same char as _TestDup1


# ---------------------------------------------------------------------------
# 9. Duplicate char -- different scope is OK
# ---------------------------------------------------------------------------


def test_duplicate_char_different_scope_ok():
    """Same char in different scopes is fine."""

    @register_object_type("test_obj_scope_a", scope="test_phase10_scope_a")
    class _TestScopeA(GridObj):
        char = "\x86"

    @register_object_type("test_obj_scope_b", scope="test_phase10_scope_b")
    class _TestScopeB(GridObj):
        char = "\x86"  # same char, different scope

    meta_a = get_component_metadata("test_obj_scope_a", scope="test_phase10_scope_a")
    meta_b = get_component_metadata("test_obj_scope_b", scope="test_phase10_scope_b")
    assert meta_a is not None
    assert meta_b is not None


# ---------------------------------------------------------------------------
# 10. get_all_components sorted by object_id
# ---------------------------------------------------------------------------


def test_get_all_components_sorted():
    """get_all_components returns results sorted by object_id."""

    @register_object_type("zzz_last", scope="test_phase10_sorted")
    class _Last(GridObj):
        char = "\x87"

    @register_object_type("aaa_first", scope="test_phase10_sorted")
    class _First(GridObj):
        char = "\x88"

    components = get_all_components(scope="test_phase10_sorted")
    assert len(components) == 2
    assert components[0].object_id == "aaa_first"
    assert components[1].object_id == "zzz_last"


# ---------------------------------------------------------------------------
# 11. get_component_metadata returns None for missing
# ---------------------------------------------------------------------------


def test_get_component_metadata_returns_none_for_missing():
    """Querying nonexistent scope/object returns None."""
    assert get_component_metadata("nonexistent", scope="nonexistent") is None


# ---------------------------------------------------------------------------
# 12. get_tickable_components
# ---------------------------------------------------------------------------


def test_get_tickable_components():
    """Only tickable objects returned by get_tickable_components."""

    @register_object_type("tick_yes", scope="test_phase10_tickable")
    class _TickYes(GridObj):
        char = "\x89"

        @classmethod
        def build_tick_fn(cls):
            pass

    @register_object_type("tick_no", scope="test_phase10_tickable")
    class _TickNo(GridObj):
        char = "\x8a"

    tickable = get_tickable_components(scope="test_phase10_tickable")
    assert len(tickable) == 1
    assert tickable[0].object_id == "tick_yes"


# ---------------------------------------------------------------------------
# 13. get_components_with_extra_state
# ---------------------------------------------------------------------------


def test_get_components_with_extra_state():
    """Only extra_state objects returned by get_components_with_extra_state."""

    @register_object_type("es_yes", scope="test_phase10_extra_state")
    class _EsYes(GridObj):
        char = "\x8b"

        @classmethod
        def extra_state_schema(cls):
            pass

    @register_object_type("es_no", scope="test_phase10_extra_state")
    class _EsNo(GridObj):
        char = "\x8c"

    with_es = get_components_with_extra_state(scope="test_phase10_extra_state")
    assert len(with_es) == 1
    assert with_es[0].object_id == "es_yes"


# ---------------------------------------------------------------------------
# 14. ArrayReward base class
# ---------------------------------------------------------------------------


def test_array_reward_base_class():
    """ArrayReward.compute() raises NotImplementedError."""
    r = ArrayReward()

    with pytest.raises(NotImplementedError, match="ArrayReward.compute"):
        r.compute(None, None, None, None)


# ---------------------------------------------------------------------------
# 15. register_reward_type basic
# ---------------------------------------------------------------------------


def test_register_reward_type_basic():
    """Registering an ArrayReward subclass makes it retrievable via get_reward_types."""

    @register_reward_type("test_rew_basic", scope="test_phase10_rew_basic")
    class _BasicReward(ArrayReward):
        def compute(self, prev_state, state, actions, reward_config):
            return None

    rewards = get_reward_types(scope="test_phase10_rew_basic")
    assert len(rewards) == 1
    assert rewards[0].reward_id == "test_rew_basic"
    assert rewards[0].cls is _BasicReward


# ---------------------------------------------------------------------------
# 16. register_reward_type missing compute raises
# ---------------------------------------------------------------------------


def test_register_reward_type_missing_compute_raises():
    """Class without compute() method cannot be registered as reward type."""
    with pytest.raises(TypeError, match="compute"):

        @register_reward_type("test_rew_nocompute", scope="test_phase10_rew_nocompute")
        class _NoComputeReward:
            pass


# ---------------------------------------------------------------------------
# 18. register_reward_type wrong compute signature raises
# ---------------------------------------------------------------------------


def test_register_reward_type_wrong_compute_signature_raises():
    """Class with wrong compute() signature raises TypeError."""
    with pytest.raises(TypeError, match="compute"):

        @register_reward_type("test_rew_wrongsig", scope="test_phase10_rew_wrongsig")
        class _WrongSigReward(ArrayReward):
            def compute(self, wrong_params):
                return None


# ---------------------------------------------------------------------------
# 19. register_reward_type duplicate raises
# ---------------------------------------------------------------------------


def test_register_reward_type_duplicate_raises():
    """Duplicate (scope, reward_id) raises ValueError."""

    @register_reward_type("test_rew_dup", scope="test_phase10_rew_dup")
    class _Dup1(ArrayReward):
        def compute(self, prev_state, state, actions, reward_config):
            return None

    with pytest.raises(ValueError, match="Duplicate reward type"):

        @register_reward_type("test_rew_dup", scope="test_phase10_rew_dup")
        class _Dup2(ArrayReward):
            def compute(self, prev_state, state, actions, reward_config):
                return None


# ---------------------------------------------------------------------------
# 20. Existing objects backward compat
# ---------------------------------------------------------------------------


def test_existing_objects_backward_compat():
    """Wall, Counter (global) and Pot (overcooked) have ComponentMetadata."""
    from cogrid.core.grid_object import Wall, Counter
    # Ensure overcooked objects are imported / registered
    from cogrid.envs.overcooked import overcooked_grid_objects  # noqa: F401

    wall_meta = get_component_metadata("wall", scope="global")
    assert wall_meta is not None
    assert wall_meta.cls is Wall

    counter_meta = get_component_metadata("counter", scope="global")
    assert counter_meta is not None
    assert counter_meta.cls is Counter

    pot_meta = get_component_metadata("pot", scope="overcooked")
    assert pot_meta is not None
    assert pot_meta.cls is overcooked_grid_objects.Pot


# ---------------------------------------------------------------------------
# 21. Old register_object -- no metadata
# ---------------------------------------------------------------------------


def test_old_register_object_no_metadata():
    """Search rescue objects (old-style register_object) return None from metadata query."""
    # Ensure search_rescue objects are imported / registered
    from cogrid.envs.search_rescue import search_rescue_grid_objects  # noqa: F401

    meta = get_component_metadata("medkit", scope="search_rescue")
    assert meta is None, (
        "Old-style register_object() classes should not appear in ComponentMetadata"
    )

    meta2 = get_component_metadata("rubble", scope="search_rescue")
    assert meta2 is None
