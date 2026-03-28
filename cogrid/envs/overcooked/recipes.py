"""Overcooked recipe definitions and compilation.

The ``Recipe`` dataclass declares how ingredients combine in a container
to produce a result item. ``compile_recipes`` converts Recipe objects into
fixed-shape lookup arrays for vectorized step-time matching.

``build_recipe_static_tables`` produces the full set of static tables
(recipe arrays, IS_DELIVERABLE, container ID, cooking time) consumed by
the interaction and reward systems.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def build_recipe_static_tables(
    object_id: str,
    recipes: list[Recipe],
    scope: str,
) -> dict:
    """Build static lookup tables for a container's recipes.

    Returns a dict with recipe arrays, IS_DELIVERABLE table,
    container type ID, and cooking time -- everything the interaction
    and reward systems need at step time.
    """
    import numpy as _np

    from cogrid.core.grid_object_registry import get_object_names, object_to_idx

    tables = {}

    tables[f"{object_id}_id"] = object_to_idx(object_id, scope=scope)
    tables["cooking_time"] = max((r.cook_time for r in recipes), default=0)

    if recipes:
        recipe_tables = compile_recipes(recipes, scope=scope)
        tables["recipe_ingredients"] = recipe_tables["recipe_ingredients"]
        tables["recipe_result"] = recipe_tables["recipe_result"]
        tables["recipe_cooking_time"] = recipe_tables["recipe_cooking_time"]
        tables["recipe_reward"] = recipe_tables["recipe_reward"]
        tables["max_ingredients"] = recipe_tables["max_ingredients"]

        n_types = len(get_object_names(scope=scope))
        is_deliverable = _np.zeros(n_types, dtype=_np.int32)
        for result_id in recipe_tables["recipe_result"]:
            is_deliverable[int(result_id)] = 1
        tables["IS_DELIVERABLE"] = is_deliverable

    return tables
