"""Tests for build_scope_config_from_components auto-wiring.

Covers:
- Returned dict has all required scope_config keys
- type_ids includes global and scope-specific objects
- symbol_table auto-populated from GridObject.char
- symbol_table includes spawn marker and empty space
- symbol_table includes global objects for non-global scopes
- extra_state_schema merged from multiple components, sorted, scope-prefixed
- static_tables built from build_lookup_tables
- tick_handler, interaction_body, interaction_tables default to None
- tick_handler, interaction_body accept overrides
"""

import numpy as np
import pytest

from cogrid.core.autowire import build_scope_config_from_components
from cogrid.core.grid_object import GridObj, register_object_type


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
        "interaction_body",
        "static_tables",
        "symbol_table",
        "extra_state_schema",
        "extra_state_builder",
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
    object_id = "test_autowire_sym_obj"
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
    object_id = "test_aw_schema_b"
    color = (0, 0, 0)
    char = "7"

    @classmethod
    def extra_state_schema(cls):
        return {
            "zeta_field": {"shape": (4,), "dtype": "float32"},
        }


@register_object_type("test_aw_schema_a", scope="test_schema_01")
class _TestSchemaObjA(GridObj):
    object_id = "test_aw_schema_a"
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
    assert schema_keys == sorted(schema_keys), (
        f"Schema keys not sorted: {schema_keys}"
    )

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
# Test 11: interaction_body accepts override
# -----------------------------------------------------------------------


def test_interaction_body_accepts_override():
    """Passing interaction_body=fn stores it in the config."""
    def my_body(*args):
        pass

    config = build_scope_config_from_components("test_sym_01", interaction_body=my_body)
    assert config["interaction_body"] is my_body


# -----------------------------------------------------------------------
# Test 12: interaction_tables default None
# -----------------------------------------------------------------------


def test_interaction_tables_default_none():
    """interaction_tables is None by default."""
    config = build_scope_config_from_components("test_sym_01")
    assert config["interaction_tables"] is None
