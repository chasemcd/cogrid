"""Parity tests: auto-wired scope_config vs manual build_overcooked_scope_config.

Proves that build_scope_config_from_components("overcooked") produces
a scope_config structurally and behaviorally identical to the manually
assembled build_overcooked_scope_config(), validating the component
classmethod approach for the most complex existing scope.
"""

import numpy as np
import pytest

# Trigger registration of Overcooked grid objects and rewards
import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
import cogrid.envs.overcooked.array_rewards  # noqa: F401

from cogrid.envs.overcooked.array_config import build_overcooked_scope_config
from cogrid.core.autowire import (
    build_reward_config_from_components,
    build_scope_config_from_components,
)


# Build both configs once at module level for all tests
_manual = build_overcooked_scope_config()
_auto = build_scope_config_from_components("overcooked")


# -----------------------------------------------------------------------
# Test 1: type_ids match
# -----------------------------------------------------------------------


def test_type_ids_match():
    """Auto-wired type_ids contains all manual type_ids with identical values."""
    manual_ids = _manual["type_ids"]
    auto_ids = _auto["type_ids"]

    # Auto may have MORE entries (all registered objects) but must contain
    # all entries from manual with the same integer values.
    for name, idx in manual_ids.items():
        assert name in auto_ids, f"type_ids missing '{name}'"
        assert auto_ids[name] == idx, (
            f"type_ids['{name}']: manual={idx}, auto={auto_ids[name]}"
        )


# -----------------------------------------------------------------------
# Test 2: extra_state_schema match
# -----------------------------------------------------------------------


def test_extra_state_schema_match():
    """Auto-wired extra_state_schema has same keys, shapes, and dtypes as manual."""
    manual_schema = _manual["extra_state_schema"]
    auto_schema = _auto["extra_state_schema"]

    assert set(manual_schema.keys()) == set(auto_schema.keys()), (
        f"Schema key mismatch: "
        f"manual={sorted(manual_schema.keys())}, auto={sorted(auto_schema.keys())}"
    )

    for key in manual_schema:
        m_val = manual_schema[key]
        a_val = auto_schema[key]
        assert m_val["shape"] == a_val["shape"], (
            f"Schema['{key}'].shape: manual={m_val['shape']}, auto={a_val['shape']}"
        )
        assert m_val["dtype"] == a_val["dtype"], (
            f"Schema['{key}'].dtype: manual={m_val['dtype']}, auto={a_val['dtype']}"
        )


# -----------------------------------------------------------------------
# Test 3: static_tables match
# -----------------------------------------------------------------------


def test_static_tables_match():
    """Auto-wired static_tables contains all manual static_tables keys with matching values."""
    manual_st = _manual["static_tables"]
    auto_st = _auto["static_tables"]

    # Auto must contain ALL keys from manual
    for key in manual_st:
        assert key in auto_st, f"static_tables missing '{key}'"

    # Compare array-valued tables
    for key in ["pickup_from_produces", "legal_pot_ingredients"]:
        np.testing.assert_array_equal(
            manual_st[key], auto_st[key],
            err_msg=f"static_tables['{key}'] mismatch",
        )

    # Compare scalar-valued tables
    for key in [
        "pot_id", "plate_id", "tomato_id", "onion_soup_id",
        "tomato_soup_id", "delivery_zone_id", "cooking_time",
    ]:
        assert manual_st[key] == auto_st[key], (
            f"static_tables['{key}']: manual={manual_st[key]}, auto={auto_st[key]}"
        )


# -----------------------------------------------------------------------
# Test 4: symbol_table superset
# -----------------------------------------------------------------------


def test_symbol_table_superset():
    """Auto-wired symbol_table contains all entries from manual symbol_table."""
    manual_sym = _manual["symbol_table"]
    auto_sym = _auto["symbol_table"]

    for char, entry in manual_sym.items():
        assert char in auto_sym, f"symbol_table missing char '{char}'"
        assert auto_sym[char]["object_id"] == entry["object_id"], (
            f"symbol_table['{char}'].object_id: "
            f"manual={entry['object_id']}, auto={auto_sym[char]['object_id']}"
        )


# -----------------------------------------------------------------------
# Test 5: tick_handler composed
# -----------------------------------------------------------------------


def test_tick_handler_composed():
    """Both configs have non-None, callable tick_handler."""
    assert _manual["tick_handler"] is not None, "Manual tick_handler is None"
    assert _auto["tick_handler"] is not None, "Auto tick_handler is None"
    assert callable(_manual["tick_handler"]), "Manual tick_handler not callable"
    assert callable(_auto["tick_handler"]), "Auto tick_handler not callable"


# -----------------------------------------------------------------------
# Test 6: interaction_body composed
# -----------------------------------------------------------------------


def test_interaction_body_composed():
    """Both configs have non-None, callable interaction_body."""
    assert _manual["interaction_body"] is not None, "Manual interaction_body is None"
    assert _auto["interaction_body"] is not None, "Auto interaction_body is None"
    assert callable(_manual["interaction_body"]), "Manual interaction_body not callable"
    assert callable(_auto["interaction_body"]), "Auto interaction_body not callable"


# -----------------------------------------------------------------------
# Test 7: extra_state_builder composed
# -----------------------------------------------------------------------


def test_extra_state_builder_composed():
    """Both configs have non-None, callable extra_state_builder that produce matching output."""
    assert _manual["extra_state_builder"] is not None, "Manual extra_state_builder is None"
    assert _auto["extra_state_builder"] is not None, "Auto extra_state_builder is None"
    assert callable(_manual["extra_state_builder"]), "Manual builder not callable"
    assert callable(_auto["extra_state_builder"]), "Auto builder not callable"

    # Build a simple parsed_arrays with a pot to test builder output
    pot_type_id = _auto["type_ids"]["pot"]
    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 2] = pot_type_id  # one pot at (1, 2)

    parsed_arrays = {"object_type_map": otm}

    manual_result = _manual["extra_state_builder"](parsed_arrays, "overcooked")
    auto_result = _auto["extra_state_builder"](parsed_arrays, "overcooked")

    assert set(manual_result.keys()) == set(auto_result.keys()), (
        f"Builder output key mismatch: "
        f"manual={sorted(manual_result.keys())}, auto={sorted(auto_result.keys())}"
    )

    for key in manual_result:
        np.testing.assert_array_equal(
            manual_result[key], auto_result[key],
            err_msg=f"extra_state_builder output['{key}'] mismatch",
        )


# -----------------------------------------------------------------------
# Test 8: reward_config composed
# -----------------------------------------------------------------------


def test_reward_config_composed():
    """Auto-wired reward_config has compute_fn that is callable."""
    reward_config = build_reward_config_from_components(
        "overcooked", n_agents=2, type_ids=_auto["type_ids"]
    )
    assert "compute_fn" in reward_config, "reward_config missing 'compute_fn'"
    assert callable(reward_config["compute_fn"]), "compute_fn not callable"
    assert reward_config["n_agents"] == 2
    assert reward_config["type_ids"] is _auto["type_ids"]
