"""Tests for build_scope_config_from_components and build_reward_config.

Covers:
- Returned dict has all required scope_config keys
- type_ids includes global and scope-specific objects
- symbol_table auto-populated from GridObject.char
- symbol_table includes spawn marker and empty space
- symbol_table includes global objects for non-global scopes
- extra_state_schema merged from multiple components, sorted, scope-prefixed
- static_tables built from build_lookup_tables
- tick_handler, interaction_tables default to None
- tick_handler accepts overrides
- reward_config has required keys and callable compute_fn
- reward_config compute_fn sums all reward instances
- reward_config with no rewards returns zeros
- reward_config passes reward_config through to each reward's compute()
"""

import numpy as np

from cogrid.core.autowire import (
    build_reward_config,
    build_scope_config_from_components,
)
from cogrid.core.grid_object import GridObj, register_object_type
from cogrid.core.rewards import Reward

# -----------------------------------------------------------------------
# Test 1: scope_config has all required keys
# -----------------------------------------------------------------------


def test_scope_config_has_all_required_keys():
    """build_scope_config_from_components returns a dict with all required keys."""
    # Ensure overcooked objects are registered
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401

    config = build_scope_config_from_components("overcooked")

    required_keys = {
        "scope",
        "interaction_tables",
        "type_ids",
        "state_extractor",
        "tick_handler",
        "static_tables",
        "symbol_table",
        "extra_state_schema",
        "extra_state_builder",
        "render_sync",
        "interaction_fn",
    }
    assert required_keys == set(config.keys()), (
        f"Missing keys: {required_keys - set(config.keys())}, "
        f"Extra keys: {set(config.keys()) - required_keys}"
    )


# -----------------------------------------------------------------------
# Test 2: type_ids includes global and scope objects
# -----------------------------------------------------------------------


def test_type_ids_includes_global_and_scope_objects():
    """type_ids maps object names to non-negative ints, includes global and scope objects."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401

    config = build_scope_config_from_components("overcooked")
    type_ids = config["type_ids"]

    assert isinstance(type_ids, dict)

    # Must include at least one global object
    assert "wall" in type_ids, "type_ids missing global object 'wall'"

    # Must include at least one overcooked object
    assert "pot" in type_ids, "type_ids missing overcooked object 'pot'"

    # All values must be non-negative ints
    for name, idx in type_ids.items():
        assert isinstance(idx, int), f"type_ids['{name}'] is {type(idx)}, expected int"
        assert idx >= 0, f"type_ids['{name}'] = {idx}, expected non-negative"


# -----------------------------------------------------------------------
# Test 3: symbol_table auto-populated from char
# -----------------------------------------------------------------------


@register_object_type("test_autowire_sym_obj", scope="test_sym_01")
class _TestSymObj(GridObj):
    color = (0, 0, 0)
    char = "Z"


def test_symbol_table_auto_populated_from_char():
    """Registering a GridObject with char='Z' makes it appear in the symbol_table."""
    config = build_scope_config_from_components("test_sym_01")
    symbol_table = config["symbol_table"]

    assert "Z" in symbol_table, f"char 'Z' not in symbol_table: {symbol_table.keys()}"
    assert symbol_table["Z"]["object_id"] == "test_autowire_sym_obj"


# -----------------------------------------------------------------------
# Test 4: symbol_table includes spawn and empty
# -----------------------------------------------------------------------


def test_symbol_table_includes_spawn_and_empty():
    """symbol_table always includes '+' (spawn) and ' ' (empty) special entries."""
    config = build_scope_config_from_components("test_sym_01")
    symbol_table = config["symbol_table"]

    assert "+" in symbol_table, "'+' spawn marker not in symbol_table"
    assert symbol_table["+"].get("is_spawn") is True, "'+' entry missing is_spawn=True"

    assert " " in symbol_table, "' ' empty space not in symbol_table"
    assert symbol_table[" "]["object_id"] is None, "' ' entry should have object_id=None"


# -----------------------------------------------------------------------
# Test 5: symbol_table includes global objects
# -----------------------------------------------------------------------


def test_symbol_table_includes_global_objects():
    """For a non-global scope, symbol_table includes global-scope GridObject chars."""
    config = build_scope_config_from_components("test_sym_01")
    symbol_table = config["symbol_table"]

    # Wall is registered globally with char="#"
    assert "#" in symbol_table, "'#' (Wall) not in symbol_table for non-global scope"
    assert symbol_table["#"]["object_id"] == "wall"


# -----------------------------------------------------------------------
# Test 6: extra_state_schema merged and sorted
# -----------------------------------------------------------------------


@register_object_type("test_aw_schema_b", scope="test_schema_01")
class _TestSchemaObjB(GridObj):
    color = (0, 0, 0)
    char = "7"

    @classmethod
    def extra_state_schema(cls):
        return {
            "zeta_field": {"shape": (4,), "dtype": "float32"},
        }


@register_object_type("test_aw_schema_a", scope="test_schema_01")
class _TestSchemaObjA(GridObj):
    color = (0, 0, 0)
    char = "8"

    @classmethod
    def extra_state_schema(cls):
        return {
            "alpha_field": {"shape": (2, 3), "dtype": "int32"},
        }


def test_extra_state_schema_merged_and_sorted():
    """Merged extra_state_schema has keys from both objects, sorted alphabetically."""
    config = build_scope_config_from_components("test_schema_01")
    schema = config["extra_state_schema"]

    # Both objects contribute keys (scope-prefixed)
    schema_keys = list(schema.keys())
    assert len(schema_keys) >= 2, f"Expected at least 2 schema keys, got {schema_keys}"

    # Keys must be sorted
    assert schema_keys == sorted(schema_keys), f"Schema keys not sorted: {schema_keys}"

    # Each value must have "shape" and "dtype"
    for key, val in schema.items():
        assert "shape" in val, f"Schema key '{key}' missing 'shape'"
        assert "dtype" in val, f"Schema key '{key}' missing 'dtype'"


# -----------------------------------------------------------------------
# Test 7: extra_state_schema scope-prefixed
# -----------------------------------------------------------------------


def test_extra_state_schema_scope_prefixed():
    """Keys in extra_state_schema are prefixed with the scope name."""
    config = build_scope_config_from_components("test_schema_01")
    schema = config["extra_state_schema"]

    for key in schema:
        assert key.startswith("test_schema_01."), (
            f"Schema key '{key}' not prefixed with 'test_schema_01.'"
        )


# -----------------------------------------------------------------------
# Test 8: static_tables built from lookup
# -----------------------------------------------------------------------


def test_static_tables_built_from_lookup():
    """static_tables contains CAN_PICKUP, CAN_PICKUP_FROM, CAN_PLACE_ON keys with arrays."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401

    config = build_scope_config_from_components("overcooked")
    static_tables = config["static_tables"]

    assert isinstance(static_tables, dict)
    for key in ("CAN_PICKUP", "CAN_PICKUP_FROM", "CAN_PLACE_ON"):
        assert key in static_tables, f"'{key}' not in static_tables"
        arr = static_tables[key]
        assert hasattr(arr, "shape"), f"static_tables['{key}'] has no shape attribute"
        assert len(arr.shape) == 1, f"static_tables['{key}'] should be 1-D"


# -----------------------------------------------------------------------
# Test 9: tick_handler default None
# -----------------------------------------------------------------------


def test_tick_handler_default_none():
    """tick_handler is None when called without override."""
    config = build_scope_config_from_components("test_sym_01")
    assert config["tick_handler"] is None


# -----------------------------------------------------------------------
# Test 10: tick_handler accepts override
# -----------------------------------------------------------------------


def test_tick_handler_accepts_override():
    """Passing tick_handler=fn stores it in the config."""

    def my_tick(state, scope_config):
        pass

    config = build_scope_config_from_components("test_sym_01", tick_handler=my_tick)
    assert config["tick_handler"] is my_tick


# -----------------------------------------------------------------------
# Test 11: interaction_tables default None
# -----------------------------------------------------------------------


def test_interaction_tables_default_none():
    """interaction_tables is None by default."""
    config = build_scope_config_from_components("test_sym_01")
    assert config["interaction_tables"] is None


# =======================================================================
# Reward config auto-wiring tests
# =======================================================================


# -----------------------------------------------------------------------
# Test 13: reward_config has required keys
# -----------------------------------------------------------------------


def test_reward_config_has_required_keys():
    """build_reward_config returns a dict with required keys."""
    config = build_reward_config([], n_agents=2, type_ids={}, action_pickup_drop_idx=4)
    required_keys = {
        "compute_fn",
        "type_ids",
        "n_agents",
        "action_pickup_drop_idx",
        "action_toggle_idx",
    }
    assert required_keys == set(config.keys()), (
        f"Missing keys: {required_keys - set(config.keys())}, "
        f"Extra keys: {set(config.keys()) - required_keys}"
    )


# -----------------------------------------------------------------------
# Test 14: compute_fn is callable
# -----------------------------------------------------------------------


def test_reward_config_compute_fn_callable():
    """reward_config['compute_fn'] is callable."""
    config = build_reward_config([], n_agents=2, type_ids={}, action_pickup_drop_idx=4)
    assert callable(config["compute_fn"])


# -----------------------------------------------------------------------
# Test 15: no rewards returns zeros
# -----------------------------------------------------------------------


def test_reward_config_no_rewards_returns_zeros():
    """With no reward instances, compute_fn returns zeros."""
    config = build_reward_config([], n_agents=2, type_ids={}, action_pickup_drop_idx=4)
    result = config["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config
    )
    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"
    np.testing.assert_array_equal(result, np.zeros(2, dtype=np.float32))


# -----------------------------------------------------------------------
# Test 16: single reward
# -----------------------------------------------------------------------


class _TestRewardSingle(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        n = reward_config["n_agents"]
        return np.ones(n, dtype=np.float32) * 2.0


def test_reward_config_single_reward():
    """Single reward instance returning [2.0, 2.0]."""
    config = build_reward_config(
        [_TestRewardSingle()], n_agents=2, type_ids={}, action_pickup_drop_idx=4
    )
    result = config["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config
    )
    np.testing.assert_array_almost_equal(result, np.array([2.0, 2.0], dtype=np.float32))


# -----------------------------------------------------------------------
# Test 17: reward with broadcasting (handled inside compute)
# -----------------------------------------------------------------------


class _TestRewardCommon(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        # Agent 0 earns 1.0, agent 1 earns 0.0, but shared reward -> both get sum
        raw = np.array([1.0, 0.0], dtype=np.float32)
        return np.full(2, np.sum(raw), dtype=np.float32)


def test_reward_config_broadcasting_inside_compute():
    """Reward broadcasting is handled inside compute(), not by the composition layer."""
    config = build_reward_config(
        [_TestRewardCommon()], n_agents=2, type_ids={}, action_pickup_drop_idx=4
    )
    result = config["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config
    )
    # sum([1.0, 0.0]) = 1.0 broadcast to both agents -> [1.0, 1.0]
    np.testing.assert_array_almost_equal(result, np.array([1.0, 1.0], dtype=np.float32))


# -----------------------------------------------------------------------
# Test 18: multiple rewards sum
# -----------------------------------------------------------------------


class _TestRewardMultiA(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        n = reward_config["n_agents"]
        return np.ones(n, dtype=np.float32)  # [1, 1]


class _TestRewardMultiB(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        n = reward_config["n_agents"]
        return np.ones(n, dtype=np.float32) * 6.0  # [6, 6]


def test_reward_config_multiple_rewards_sum():
    """Multiple reward instances are summed."""
    config = build_reward_config(
        [_TestRewardMultiA(), _TestRewardMultiB()],
        n_agents=2,
        type_ids={},
        action_pickup_drop_idx=4,
    )
    result = config["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config
    )
    # r3a: [1, 1]
    # r3b: [6, 6]
    # total: [7, 7]
    np.testing.assert_array_almost_equal(result, np.array([7.0, 7.0], dtype=np.float32))


# -----------------------------------------------------------------------
# Test 19: compute_fn passes reward_config through to compute()
# -----------------------------------------------------------------------


class _TestRewardPassthrough(Reward):
    def compute(self, prev_state, state, actions, reward_config):
        n = reward_config["n_agents"]
        # Return 1.0 per agent if "marker" key is present, else 0.0
        if "marker" in reward_config.get("type_ids", {}):
            return np.ones(n, dtype=np.float32)
        return np.zeros(n, dtype=np.float32)


def test_reward_config_passes_reward_config_to_compute():
    """Composed compute_fn correctly passes reward_config through to each reward."""
    # With "marker" in type_ids -> should return [1, 1]
    config_with = build_reward_config(
        [_TestRewardPassthrough()],
        n_agents=2,
        type_ids={"marker": 99},
        action_pickup_drop_idx=4,
    )
    result_with = config_with["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config_with
    )
    np.testing.assert_array_almost_equal(result_with, np.array([1.0, 1.0], dtype=np.float32))

    # Without "marker" in type_ids -> should return [0, 0]
    config_without = build_reward_config(
        [_TestRewardPassthrough()],
        n_agents=2,
        type_ids={},
        action_pickup_drop_idx=4,
    )
    result_without = config_without["compute_fn"](
        prev_state={}, state={}, actions=np.zeros(2, dtype=np.int32), reward_config=config_without
    )
    np.testing.assert_array_almost_equal(result_without, np.array([0.0, 0.0], dtype=np.float32))


# =======================================================================
# Feature config auto-wiring tests
# =======================================================================


_OVERCOOKED_FEATURES = [
    "agent_dir",
    "overcooked_inventory",
    "next_to_counter",
    "next_to_pot",
    "object_type_masks",
    "ordered_pot_features",
    "dist_to_other_players",
    "agent_position",
    "can_move_direction",
    "layout_id",
    "environment_layout",
]


def test_build_feature_config_overcooked():
    """build_feature_config_from_components returns correct structure for overcooked."""
    import cogrid.envs  # noqa: F401 -- triggers registration
    from cogrid.core.autowire import build_feature_config_from_components

    config = build_feature_config_from_components(
        "overcooked", _OVERCOOKED_FEATURES, n_agents=2, layout_idx=0
    )

    # Required keys
    assert "feature_fn" in config
    assert "obs_dim" in config
    assert "feature_names" in config

    # obs_dim: 8 per-agent(61) * 2 agents = 122, + 3 global(539+5+462) = 1128
    assert config["obs_dim"] == 1128, f"Expected obs_dim=1128, got {config['obs_dim']}"

    # 8 per-agent + 3 global = 11 features
    assert len(config["feature_names"]) == 11, (
        f"Expected 11 feature names, got {len(config['feature_names'])}"
    )


def test_build_feature_config_sets_layout_idx():
    """build_feature_config_from_components sets LayoutID._layout_idx."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_feature_config_from_components
    from cogrid.envs.overcooked.features import LayoutID

    build_feature_config_from_components(
        "overcooked", _OVERCOOKED_FEATURES, n_agents=2, layout_idx=3
    )
    assert LayoutID._layout_idx == 3, f"Expected _layout_idx=3, got {LayoutID._layout_idx}"

    # Reset to default
    LayoutID._layout_idx = 0


def test_build_feature_config_returns_callable():
    """Returned feature_fn produces (677,) float32 output."""
    import cogrid.envs  # noqa: F401
    from cogrid.core.autowire import build_feature_config_from_components
    from cogrid.core.step_pipeline import envstate_to_dict
    from cogrid.envs.overcooked.features import LayoutID

    config = build_feature_config_from_components(
        "overcooked", _OVERCOOKED_FEATURES, n_agents=2, layout_idx=0
    )
    feature_fn = config["feature_fn"]

    # Build a state from a real Overcooked environment
    from cogrid.cogrid_env import CoGridEnv

    env = CoGridEnv(
        config={
            "name": "overcooked",
            "scope": "overcooked",
            "num_agents": 2,
            "max_steps": 10,
            "action_set": "cardinal_actions",
            "features": _OVERCOOKED_FEATURES,
            "grid": {"layout": "overcooked_cramped_room_v0"},
        }
    )
    env.reset()
    state_view = envstate_to_dict(env._env_state)

    result = feature_fn(state_view, agent_idx=0)
    assert result.shape == (1128,), f"Expected shape (1128,), got {result.shape}"
    assert result.dtype == np.float32, f"Expected dtype float32, got {result.dtype}"

    # Reset LayoutID
    LayoutID._layout_idx = 0
