"""Tests for guard table compilation from when() conditions.

Verifies that build_guard_tables produces correct (n_types, n_types+1) 2D
arrays from When descriptors on grid object classes.
"""

import numpy as np
import pytest

import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
from cogrid.core.grid_object import build_guard_tables, get_object_names, object_to_idx
from cogrid.core.when import when

# -----------------------------------------------------------------------
# when() descriptor tests
# -----------------------------------------------------------------------


def test_bare_when_has_no_conditions():
    w = when()
    assert bool(w) is True
    assert w.has_conditions is False
    assert w.conditions == {}


def test_when_agent_holding_str_normalizes_to_list():
    w = when(agent_holding="plate")
    assert w.has_conditions is True
    assert w.conditions["agent_holding"] == ["plate"]


def test_when_agent_holding_list_kept():
    w = when(agent_holding=["onion", "tomato"])
    assert w.has_conditions is True
    assert w.conditions["agent_holding"] == ["onion", "tomato"]


def test_when_unsupported_condition_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        when(some_unknown="value")


def test_when_bad_agent_holding_type_raises():
    with pytest.raises(TypeError, match="agent_holding must be"):
        when(agent_holding=123)


# -----------------------------------------------------------------------
# Guard table shape and defaults
# -----------------------------------------------------------------------


def test_guard_table_shape():
    """Guard tables must be (n_types, n_types+1)."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    n_types = len(get_object_names(scope=scope))

    assert tables["PICKUP_FROM_GUARD"].shape == (n_types, n_types + 1)
    assert tables["PLACE_ON_GUARD"].shape == (n_types, n_types + 1)


def test_bare_when_pickup_from_defaults_to_empty_hands():
    """Bare when() on can_pickup_from sets only column 0 (empty hands)."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    pf_guard = tables["PICKUP_FROM_GUARD"]

    # OnionStack has can_pickup_from = when() (bare, no conditions)
    onion_stack_idx = object_to_idx("onion_stack", scope=scope)
    assert pf_guard[onion_stack_idx, 0] == 1  # empty hands OK
    assert np.sum(pf_guard[onion_stack_idx, 1:]) == 0  # no held-item columns


def test_bare_when_place_on_defaults_to_any_held():
    """Bare when() on can_place_on sets columns 1..n (any held item)."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    po_guard = tables["PLACE_ON_GUARD"]

    # Counter has can_place_on = when() (bare, no conditions)
    counter_idx = object_to_idx("counter", scope=scope)
    assert po_guard[counter_idx, 0] == 0  # empty hands NOT allowed
    assert np.all(po_guard[counter_idx, 1:] == 1)  # any held item OK


# -----------------------------------------------------------------------
# Conditional guard tables (Overcooked-specific)
# -----------------------------------------------------------------------


def test_pot_pickup_from_requires_plate():
    """Pot with when(agent_holding='plate') sets only the plate column."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    pf_guard = tables["PICKUP_FROM_GUARD"]

    pot_idx = object_to_idx("pot", scope=scope)
    plate_idx = object_to_idx("plate", scope=scope)

    # Only the plate column should be set
    assert pf_guard[pot_idx, 0] == 0  # empty hands NOT allowed
    assert pf_guard[pot_idx, plate_idx + 1] == 1  # plate OK

    # All other held-item columns should be 0
    for col in range(1, pf_guard.shape[1]):
        if col != plate_idx + 1:
            assert pf_guard[pot_idx, col] == 0, f"unexpected column {col} set for pot pickup_from"


def test_pot_place_on_accepts_onion_and_tomato():
    """Pot with when(agent_holding=['onion','tomato']) sets those columns."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    po_guard = tables["PLACE_ON_GUARD"]

    pot_idx = object_to_idx("pot", scope=scope)
    onion_idx = object_to_idx("onion", scope=scope)
    tomato_idx = object_to_idx("tomato", scope=scope)

    assert po_guard[pot_idx, 0] == 0  # empty hands NOT allowed
    assert po_guard[pot_idx, onion_idx + 1] == 1
    assert po_guard[pot_idx, tomato_idx + 1] == 1

    # Other columns should be 0
    for col in range(1, po_guard.shape[1]):
        if col not in (onion_idx + 1, tomato_idx + 1):
            assert po_guard[pot_idx, col] == 0


def test_delivery_zone_accepts_soups():
    """DeliveryZone with when(agent_holding=['onion_soup','tomato_soup']) sets those columns."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)
    po_guard = tables["PLACE_ON_GUARD"]

    dz_idx = object_to_idx("delivery_zone", scope=scope)
    osoup_idx = object_to_idx("onion_soup", scope=scope)
    tsoup_idx = object_to_idx("tomato_soup", scope=scope)

    assert po_guard[dz_idx, 0] == 0
    assert po_guard[dz_idx, osoup_idx + 1] == 1
    assert po_guard[dz_idx, tsoup_idx + 1] == 1

    # Other columns should be 0
    for col in range(1, po_guard.shape[1]):
        if col not in (osoup_idx + 1, tsoup_idx + 1):
            assert po_guard[dz_idx, col] == 0


def test_non_interactive_objects_have_zero_rows():
    """Objects without can_pickup_from or can_place_on should have all-zero rows."""
    scope = "overcooked"
    tables = build_guard_tables(scope=scope)

    # Onion is pickupable but has no can_pickup_from or can_place_on
    onion_idx = object_to_idx("onion", scope=scope)
    assert np.sum(tables["PICKUP_FROM_GUARD"][onion_idx]) == 0
    assert np.sum(tables["PLACE_ON_GUARD"][onion_idx]) == 0
