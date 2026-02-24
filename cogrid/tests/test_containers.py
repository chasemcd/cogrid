"""Tests for the Container + Recipe declarative system.

Covers:
- Container and Recipe dataclass creation
- Auto-generated when() from container/recipes
- Recipe compilation
- Auto-generated extra_state_schema, builder, tick, render_sync, static_tables
- Auto-generated interaction_fn from container + consume components
- Overcooked Pot uses Container descriptor instead of classmethods
"""

import numpy as np

# -----------------------------------------------------------------------
# Test 1: Container and Recipe are frozen dataclasses
# -----------------------------------------------------------------------


def test_container_dataclass():
    """Container is a frozen dataclass with expected fields."""
    from cogrid.core.containers import Container

    c = Container(capacity=3, pickup_requires="plate")
    assert c.capacity == 3
    assert c.pickup_requires == "plate"

    c2 = Container(capacity=5)
    assert c2.capacity == 5
    assert c2.pickup_requires is None


def test_recipe_dataclass():
    """Recipe is a frozen dataclass with expected fields."""
    from cogrid.core.containers import Recipe

    r = Recipe(["onion", "onion", "onion"], result="onion_soup", cook_time=30, reward=20.0)
    assert r.ingredients == ["onion", "onion", "onion"]
    assert r.result == "onion_soup"
    assert r.cook_time == 30
    assert r.reward == 20.0

    r2 = Recipe(["a"], result="b")
    assert r2.cook_time == 0
    assert r2.reward == 0.0


# -----------------------------------------------------------------------
# Test 2: Pot uses Container descriptor
# -----------------------------------------------------------------------


def test_pot_has_container_descriptor():
    """Overcooked Pot declares container and recipes class attributes."""
    import cogrid.envs.overcooked.overcooked_grid_objects as oc
    from cogrid.core.containers import Container, Recipe

    assert isinstance(oc.Pot.container, Container)
    assert oc.Pot.container.capacity == 3
    assert oc.Pot.container.pickup_requires == "plate"
    assert isinstance(oc.Pot.recipes, list)
    assert all(isinstance(r, Recipe) for r in oc.Pot.recipes)
    assert len(oc.Pot.recipes) == 2


# -----------------------------------------------------------------------
# Test 3: Auto-generated when() from container/recipes
# -----------------------------------------------------------------------


def test_pot_auto_generated_when():
    """Pot's can_place_on and can_pickup_from are When instances."""
    import cogrid.envs.overcooked.overcooked_grid_objects as oc
    from cogrid.core.when import When

    assert isinstance(oc.Pot.can_place_on, When)
    assert isinstance(oc.Pot.can_pickup_from, When)

    # can_place_on should allow onion and tomato (from recipe ingredients)
    assert oc.Pot.can_place_on.has_conditions
    ingredients = oc.Pot.can_place_on.conditions["agent_holding"]
    assert "onion" in ingredients
    assert "tomato" in ingredients

    # can_pickup_from should require plate
    assert oc.Pot.can_pickup_from.has_conditions
    assert oc.Pot.can_pickup_from.conditions["agent_holding"] == ["plate"]


# -----------------------------------------------------------------------
# Test 4: DeliveryZone has consumes_on_place
# -----------------------------------------------------------------------


def test_delivery_zone_consumes_on_place():
    """DeliveryZone has consumes_on_place=True."""
    import cogrid.envs.overcooked.overcooked_grid_objects as oc

    assert oc.DeliveryZone.consumes_on_place is True


# -----------------------------------------------------------------------
# Test 5: Container metadata in component registry
# -----------------------------------------------------------------------


def test_container_metadata_stored():
    """ComponentMetadata stores container_meta for Pot."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.component_registry import get_component_metadata

    meta = get_component_metadata("pot", scope="overcooked")
    assert meta is not None
    assert meta.has_container
    assert meta.container_meta is not None
    assert "container" in meta.container_meta
    assert "recipes" in meta.container_meta


def test_non_container_has_no_container_meta():
    """Non-container components have container_meta=None."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.component_registry import get_component_metadata

    meta = get_component_metadata("onion", scope="overcooked")
    assert meta is not None
    assert not meta.has_container
    assert meta.container_meta is None


# -----------------------------------------------------------------------
# Test 6: Recipe compilation
# -----------------------------------------------------------------------


def test_compile_recipes():
    """compile_recipes produces correct arrays from Recipe objects."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.containers import Recipe, compile_recipes
    from cogrid.core.grid_object_registry import object_to_idx

    scope = "overcooked"
    recipes = [
        Recipe(["onion", "onion", "onion"], result="onion_soup", cook_time=30, reward=20.0),
        Recipe(["tomato", "tomato", "tomato"], result="tomato_soup", cook_time=25, reward=15.0),
    ]
    tables = compile_recipes(recipes, scope=scope)

    assert tables["max_ingredients"] == 3
    assert tables["recipe_ingredients"].shape == (2, 3)
    assert tables["recipe_result"].shape == (2,)
    assert tables["recipe_cooking_time"].shape == (2,)
    assert tables["recipe_reward"].shape == (2,)

    onion_id = object_to_idx("onion", scope=scope)
    onion_soup_id = object_to_idx("onion_soup", scope=scope)
    np.testing.assert_array_equal(tables["recipe_ingredients"][0], [onion_id] * 3)
    assert int(tables["recipe_result"][0]) == onion_soup_id
    assert int(tables["recipe_cooking_time"][0]) == 30
    assert float(tables["recipe_reward"][0]) == 20.0


# -----------------------------------------------------------------------
# Test 7: Auto-generated extra_state_schema
# -----------------------------------------------------------------------


def test_auto_generated_extra_state_schema():
    """Container auto-generates extra_state_schema in scope_config."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components

    config = build_scope_config_from_components("overcooked")
    schema = config["extra_state_schema"]

    assert "overcooked.pot_contents" in schema
    assert "overcooked.pot_timer" in schema
    assert "overcooked.pot_positions" in schema


# -----------------------------------------------------------------------
# Test 8: Auto-generated static_tables
# -----------------------------------------------------------------------


def test_auto_generated_static_tables():
    """Container auto-generates static_tables with recipe arrays."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components

    config = build_scope_config_from_components("overcooked")
    st = config["static_tables"]

    assert "pot_id" in st
    assert "cooking_time" in st
    assert "recipe_ingredients" in st
    assert "recipe_result" in st
    assert "recipe_cooking_time" in st
    assert "max_ingredients" in st
    assert "IS_DELIVERABLE" in st
    assert "delivery_zone_id" in st  # from consumes_on_place


# -----------------------------------------------------------------------
# Test 9: Auto-generated interaction_fn
# -----------------------------------------------------------------------


def test_auto_generated_interaction_fn():
    """Container + consumes_on_place auto-generates interaction_fn."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components

    config = build_scope_config_from_components("overcooked")
    assert config["interaction_fn"] is not None
    assert callable(config["interaction_fn"])


# -----------------------------------------------------------------------
# Test 10: Auto-generated tick_handler
# -----------------------------------------------------------------------


def test_auto_generated_tick_handler():
    """Container auto-generates tick_handler."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components

    config = build_scope_config_from_components("overcooked")
    assert config["tick_handler"] is not None
    assert callable(config["tick_handler"])


# -----------------------------------------------------------------------
# Test 11: Auto-generated render_sync
# -----------------------------------------------------------------------


def test_auto_generated_render_sync():
    """Container auto-generates render_sync."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components

    config = build_scope_config_from_components("overcooked")
    assert config["render_sync"] is not None
    assert callable(config["render_sync"])


# -----------------------------------------------------------------------
# Test 12: Auto-generated extra_state_builder
# -----------------------------------------------------------------------


def test_auto_generated_extra_state_builder():
    """Container auto-generates extra_state_builder that builds pot arrays."""
    import cogrid.envs.overcooked.overcooked_grid_objects  # noqa: F401
    from cogrid.core.autowire import build_scope_config_from_components
    from cogrid.core.grid_object_registry import object_to_idx

    config = build_scope_config_from_components("overcooked")
    builder = config["extra_state_builder"]
    assert builder is not None

    pot_id = object_to_idx("pot", scope="overcooked")

    # Build a fake layout with one pot
    otm = np.zeros((5, 5), dtype=np.int32)
    otm[1, 2] = pot_id
    parsed = {"object_type_map": otm}

    result = builder(parsed, "overcooked")
    assert "overcooked.pot_contents" in result
    assert "overcooked.pot_timer" in result
    assert "overcooked.pot_positions" in result

    assert result["overcooked.pot_contents"].shape == (1, 3)
    assert result["overcooked.pot_timer"].shape == (1,)
    assert result["overcooked.pot_positions"].shape == (1, 2)

    # Contents should be -1 (empty)
    np.testing.assert_array_equal(result["overcooked.pot_contents"], [[-1, -1, -1]])
    # Timer should be max cook time (30)
    np.testing.assert_array_equal(result["overcooked.pot_timer"], [30])
    # Position should be (1, 2)
    np.testing.assert_array_equal(result["overcooked.pot_positions"], [[1, 2]])


# -----------------------------------------------------------------------
# Test 13: Container re-exports from grid_object
# -----------------------------------------------------------------------


def test_container_recipe_reexported():
    """Container and Recipe are re-exported from cogrid.core.grid_object."""
    from cogrid.core.grid_object import Container, Recipe

    assert Container is not None
    assert Recipe is not None
