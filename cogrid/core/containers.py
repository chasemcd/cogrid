"""Declarative container and recipe system for stateful grid objects.

Provides ``Container`` and ``Recipe`` dataclasses that replace hundreds of
lines of imperative array code with class-attribute declarations::

    @register_object_type("pot", scope="overcooked")
    class Pot(GridObj):
        container = Container(capacity=3, pickup_requires="plate")
        recipes = [
            Recipe(["onion", "onion", "onion"], result="onion_soup", cook_time=30),
        ]

The autowire system reads these descriptors and auto-generates extra_state
schemas, builders, tick functions, render syncs, static tables, and
interaction branches.
"""

from __future__ import annotations

from dataclasses import dataclass

# ======================================================================
# Declarative data classes
# ======================================================================


@dataclass(frozen=True)
class Recipe:
    """A recipe that a Container can cook.

    Parameters
    ----------
    ingredients : list[str]
        Object IDs of the required ingredients (order does not matter;
        sorted internally for matching).
    result : str
        Object ID of the output item.
    cook_time : int
        Number of env steps to cook once the container is full.
    reward : float
        Reward granted on delivery of the result item.
    """

    ingredients: list[str]
    result: str
    cook_time: int = 0
    reward: float = 0.0


@dataclass(frozen=True)
class Container:
    """Declares that a GridObj is a stateful container (e.g. a cooking pot).

    Parameters
    ----------
    capacity : int
        Maximum number of items the container can hold.
    pickup_requires : str | list[str] | None
        Object type(s) the agent must be holding to pick up from this
        container.  ``None`` means empty hands.
    """

    capacity: int
    pickup_requires: str | list[str] | None = None


# ======================================================================
# Recipe compilation (init-time, not step-path)
# ======================================================================


def compile_recipes(recipes: list[Recipe], scope: str) -> dict:
    """Compile a list of Recipe objects into fixed-shape lookup arrays.

    Produces sentinel-padded arrays suitable for storage in
    ``static_tables``.

    Parameters
    ----------
    recipes : list[Recipe]
        Recipe objects to compile.
    scope : str
        Object registry scope for resolving type names to IDs.

    Returns:
    -------
    dict with keys:
        recipe_ingredients : (n_recipes, max_ingredients) int32
        recipe_result : (n_recipes,) int32
        recipe_cooking_time : (n_recipes,) int32
        recipe_reward : (n_recipes,) float32
        max_ingredients : int
    """
    import numpy as _np

    from cogrid.core.grid_object_registry import get_object_names, object_to_idx

    if not recipes:
        raise ValueError("recipes must be a non-empty list of Recipe objects.")

    names = get_object_names(scope=scope)

    # Validate
    for i, recipe in enumerate(recipes):
        if not recipe.ingredients:
            raise ValueError(f"Recipe {i}: ingredients must be non-empty")
        for ing in recipe.ingredients:
            if ing not in names:
                raise ValueError(
                    f"Recipe {i}: ingredient '{ing}' not registered in scope '{scope}'"
                )
        if recipe.result not in names:
            raise ValueError(
                f"Recipe {i}: result '{recipe.result}' not registered in scope '{scope}'"
            )

    max_ing = max(len(r.ingredients) for r in recipes)
    n_recipes = len(recipes)

    recipe_ingredients = _np.full((n_recipes, max_ing), -1, dtype=_np.int32)
    recipe_result = _np.zeros(n_recipes, dtype=_np.int32)
    recipe_cooking_time = _np.zeros(n_recipes, dtype=_np.int32)
    recipe_reward = _np.zeros(n_recipes, dtype=_np.float32)

    seen_combos = {}
    for i, recipe in enumerate(recipes):
        ing_ids = sorted([object_to_idx(name, scope=scope) for name in recipe.ingredients])
        combo_key = tuple(ing_ids)
        if combo_key in seen_combos:
            raise ValueError(
                f"Recipe {i} has same sorted ingredients as recipe {seen_combos[combo_key]}."
            )
        seen_combos[combo_key] = i

        for j, tid in enumerate(ing_ids):
            recipe_ingredients[i, j] = tid

        recipe_result[i] = object_to_idx(recipe.result, scope=scope)
        recipe_cooking_time[i] = recipe.cook_time
        recipe_reward[i] = recipe.reward

    return {
        "recipe_ingredients": recipe_ingredients,
        "recipe_result": recipe_result,
        "recipe_cooking_time": recipe_cooking_time,
        "recipe_reward": recipe_reward,
        "max_ingredients": max_ing,
    }


# ======================================================================
# Auto-generation helpers (called by autowire)
# ======================================================================


def build_container_extra_state_schema(object_id: str, container: Container) -> dict:
    """Return extra_state schema entries for a container type."""
    return {
        f"{object_id}_contents": {
            "shape": (f"n_{object_id}s", container.capacity),
            "dtype": "int32",
        },
        f"{object_id}_timer": {"shape": (f"n_{object_id}s",), "dtype": "int32"},
        f"{object_id}_positions": {"shape": (f"n_{object_id}s", 2), "dtype": "int32"},
    }


def build_container_extra_state_builder(
    object_id: str,
    container: Container,
    recipes: list[Recipe],
    scope: str,
) -> callable:
    """Return an extra_state builder closure for a container type."""
    default_timer = max((r.cook_time for r in recipes), default=0) if recipes else 0

    def builder(parsed_arrays, scope=scope):
        import numpy as _np

        from cogrid.core.grid_object_registry import object_to_idx

        type_id = object_to_idx(object_id, scope=scope)
        otm = parsed_arrays["object_type_map"]
        mask = otm == type_id
        positions_list = list(zip(*_np.where(mask)))
        n_instances = len(positions_list)

        prefix = f"{scope}."
        if n_instances > 0:
            positions = _np.array(positions_list, dtype=_np.int32)
            contents = _np.full((n_instances, container.capacity), -1, dtype=_np.int32)
            timer = _np.full((n_instances,), default_timer, dtype=_np.int32)
        else:
            positions = _np.zeros((0, 2), dtype=_np.int32)
            contents = _np.full((0, container.capacity), -1, dtype=_np.int32)
            timer = _np.zeros((0,), dtype=_np.int32)

        return {
            f"{prefix}{object_id}_contents": contents,
            f"{prefix}{object_id}_timer": timer,
            f"{prefix}{object_id}_positions": positions,
        }

    return builder


def build_container_tick_fn(
    object_id: str,
    container: Container,
    scope: str,
) -> callable:
    """Return a tick function that decrements cooking timers."""

    def tick_fn(state, scope_config):
        import dataclasses

        from cogrid.backend import xp
        from cogrid.backend.array_ops import set_at_2d

        prefix = f"{scope}."
        contents = state.extra_state[f"{prefix}{object_id}_contents"]
        timer = state.extra_state[f"{prefix}{object_id}_timer"]
        positions = state.extra_state[f"{prefix}{object_id}_positions"]
        n_instances = positions.shape[0]
        capacity = container.capacity

        # Tick: decrement timer when full and timer > 0
        n_items = xp.sum(contents != -1, axis=1).astype(xp.int32)
        is_cooking = (n_items == capacity) & (timer > 0)
        new_timer = xp.where(is_cooking, timer - 1, timer)

        # Compute state encoding for object_state_map
        pot_state = (n_items + n_items * new_timer).astype(xp.int32)

        osm = state.object_state_map
        for p in range(n_instances):
            osm = set_at_2d(osm, positions[p, 0], positions[p, 1], pot_state[p])

        new_extra = {
            **state.extra_state,
            f"{prefix}{object_id}_contents": contents,
            f"{prefix}{object_id}_timer": new_timer,
        }

        return dataclasses.replace(state, object_state_map=osm, extra_state=new_extra)

    return tick_fn


def build_container_render_sync(object_id: str, scope: str) -> callable:
    """Return a render_sync callback for a container type."""

    def render_sync(grid, env_state, scope=scope):
        import numpy as np

        from cogrid.core.grid_object_registry import idx_to_object, make_object

        extra = env_state.extra_state
        prefix = f"{scope}."
        contents_key = f"{prefix}{object_id}_contents"
        timer_key = f"{prefix}{object_id}_timer"
        positions_key = f"{prefix}{object_id}_positions"

        if not all(k in extra for k in (contents_key, timer_key, positions_key)):
            return

        contents = np.array(extra[contents_key])
        timer = np.array(extra[timer_key])
        positions = np.array(extra[positions_key])

        for p in range(len(positions)):
            pr, pc = int(positions[p, 0]), int(positions[p, 1])
            obj = grid.get(pr, pc)
            if obj is not None and obj.object_id == object_id:
                # Sync contents list
                obj.objects_in_pot = []
                for slot in range(contents.shape[1]):
                    item_id = int(contents[p, slot])
                    if item_id > 0:
                        item_name = idx_to_object(item_id, scope=scope)
                        if item_name:
                            obj.objects_in_pot.append(make_object(item_name, scope=scope))
                # Sync timer
                obj.cooking_timer = int(timer[p])

    return render_sync


def build_container_static_tables(
    object_id: str,
    container: Container,
    recipes: list[Recipe],
    scope: str,
) -> dict:
    """Return static lookup tables for a container type's recipes."""
    import numpy as _np

    from cogrid.core.grid_object_registry import get_object_names, object_to_idx

    tables = {}

    # Container type ID
    tables[f"{object_id}_id"] = object_to_idx(object_id, scope=scope)

    # Default cooking time (max across recipes)
    tables["cooking_time"] = max((r.cook_time for r in recipes), default=0)

    if recipes:
        recipe_tables = compile_recipes(recipes, scope=scope)
        tables["recipe_ingredients"] = recipe_tables["recipe_ingredients"]
        tables["recipe_result"] = recipe_tables["recipe_result"]
        tables["recipe_cooking_time"] = recipe_tables["recipe_cooking_time"]
        tables["recipe_reward"] = recipe_tables["recipe_reward"]
        tables["max_ingredients"] = recipe_tables["max_ingredients"]

        # IS_DELIVERABLE: 1 for any type that is a recipe output
        n_types = len(get_object_names(scope=scope))
        is_deliverable = _np.zeros(n_types, dtype=_np.int32)
        for result_id in recipe_tables["recipe_result"]:
            is_deliverable[int(result_id)] = 1
        tables["IS_DELIVERABLE"] = is_deliverable

    return tables
