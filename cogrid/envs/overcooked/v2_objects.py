"""OvercookedV2 object types: OpenPot, indicators, new ingredients, and mixed soups.

This module is imported ONLY by V2 environment configs (not by standard
Overcooked environments) to avoid polluting the shared overcooked scope
with V2-specific types that would interfere with the standard Pot's
recipe matching.
"""

from itertools import combinations_with_replacement

from cogrid.core import constants
from cogrid.core import objects as grid_object
from cogrid.core.objects import register_object_type
from cogrid.core.objects.containers import Container
from cogrid.core.objects.when import when
from cogrid.envs.overcooked.overcooked_grid_objects import (
    Plate,
    Tomato,
    make_ingredient_and_stack,
)
from cogrid.envs.overcooked.recipes import Recipe
from cogrid.visualization.rendering import (
    add_text_to_image,
    fill_coords,
    point_in_circle,
)

# ---------------------------------------------------------------------------
# Soup type generation
# ---------------------------------------------------------------------------

_SOUP_CHAR_POOL = iter(list("acdefghijklnpqrsvwxyz"))


def make_soup(
    soup_name: str,
    soup_color: tuple,
    scope: str = "overcooked",
) -> type:
    """Create and register a new soup result type at runtime.

    The soup is pickupable and renders as a plate with colored soup inside.
    Characters are auto-assigned from a pool of unused ASCII characters.
    """
    char = next(_SOUP_CHAR_POOL)

    SoupCls = type(
        soup_name.title().replace("_", ""),
        (grid_object.GridObj,),
        {
            "color": soup_color,
            "char": char,
            "can_pickup": when(),
        },
    )

    def _soup_init(self, *args, **kwargs):
        grid_object.GridObj.__init__(self, state=0)

    def _soup_render(self, tile_img):
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), Plate.color)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color)

    SoupCls.__init__ = _soup_init
    SoupCls.render = _soup_render

    register_object_type(soup_name, scope=scope)(SoupCls)
    return SoupCls


# ---------------------------------------------------------------------------
# Recipe generation
# ---------------------------------------------------------------------------

_INGREDIENT_COLORS = {
    "onion": constants.Colors.Yellow,
    "tomato": constants.Colors.Red,
    "broccoli": (34, 139, 34),
    "mushroom": (139, 90, 43),
}


def build_open_pot_recipes(
    ingredients: list[str],
    cook_time: int = 20,
) -> list[Recipe]:
    """Generate Recipe objects for all 3-ingredient combos from *ingredients*.

    Reuses existing soup types (onion_soup, tomato_soup) for pure combos.
    Creates new soup types via make_soup() for mixed combos.
    """
    recipes = []
    for combo in combinations_with_replacement(sorted(ingredients), 3):
        sorted_combo = sorted(combo)

        if sorted_combo == ["onion", "onion", "onion"]:
            result_name = "onion_soup"
        elif sorted_combo == ["tomato", "tomato", "tomato"]:
            result_name = "tomato_soup"
        else:
            result_name = "soup_" + "_".join(sorted_combo)
            colors = [_INGREDIENT_COLORS[ing] for ing in sorted_combo]
            avg_color = tuple(int(sum(c[i] for c in colors) / len(colors)) for i in range(3))
            make_soup(result_name, avg_color)

        recipes.append(Recipe(sorted_combo, result=result_name, cook_time=cook_time))
    return recipes


# ---------------------------------------------------------------------------
# New ingredients
# ---------------------------------------------------------------------------

Broccoli, BroccoliStack = make_ingredient_and_stack(
    "broccoli",
    "b",
    (34, 139, 34),
    "broccoli_stack",
    "B",
)

Mushroom, MushroomStack = make_ingredient_and_stack(
    "mushroom",
    "m",
    (139, 90, 43),
    "mushroom_stack",
    "M",
)


# ---------------------------------------------------------------------------
# Open-pot recipes (must come after ingredient registration)
# ---------------------------------------------------------------------------

_OPEN_POT_INGREDIENTS = ["onion", "tomato", "broccoli", "mushroom"]
OPEN_POT_RECIPES = build_open_pot_recipes(_OPEN_POT_INGREDIENTS, cook_time=20)
OPEN_POT_SOUP_NAMES = sorted({r.result for r in OPEN_POT_RECIPES})


# ---------------------------------------------------------------------------
# OpenPot
# ---------------------------------------------------------------------------


@register_object_type("open_pot", scope="overcooked")
class OpenPot(grid_object.GridObj):
    """A pot that accepts any ingredient combination.

    Unlike the standard Pot which only accepts recipe-matching ingredients,
    the OpenPot cooks whatever is placed inside. Correctness is determined
    at delivery time by the reward system, not the pot.
    """

    color = constants.Colors.Grey
    char = "u"
    container = Container(capacity=3, pickup_requires="plate")
    recipes = OPEN_POT_RECIPES
    capacity: int = 3

    @classmethod
    def build_static_tables(cls):
        """Build recipe lookup tables for the interaction and reward systems."""
        from cogrid.envs.overcooked.recipes import build_recipe_static_tables

        return build_recipe_static_tables("open_pot", cls.recipes, scope="overcooked")

    def __init__(self, state: int = 0, **kwargs):
        """Initialize pot with empty contents and default timer."""
        super().__init__(state=state)
        self.objects_in_pot: list[grid_object.GridObj] = []
        self.cooking_timer: int = 20

    def render(self, tile_img):
        """Draw pot circle with ingredient dots and timer text."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), self.color)

        for i, grid_obj in enumerate(self.objects_in_pot):
            fill_coords(
                tile_img,
                point_in_circle(cx=0.25 * (i + 1), cy=0.2, r=0.2),
                grid_obj.color,
            )

        if len(self.objects_in_pot) == self.capacity:
            add_text_to_image(tile_img, text=str(self.cooking_timer), position=(50, 75))

    def encode(self, encode_char: bool = True, scope: str = "global"):
        """Allow encoding to account for the type of soup in the pot."""
        char, _, state = super().encode(encode_char=encode_char, scope=scope)
        extra_state_encoding = int(any(isinstance(obj, Tomato) for obj in self.objects_in_pot))
        return (char, extra_state_encoding, state)


# ---------------------------------------------------------------------------
# Recipe and button indicators
# ---------------------------------------------------------------------------


@register_object_type("recipe_indicator", scope="overcooked")
class RecipeIndicator(grid_object.GridObj):
    """Displays the current target recipe. Visible within agent view radius."""

    color = (100, 100, 255)
    char = "R"
    is_wall = True

    def __init__(self, *args, **kwargs):
        """Initialize with default state."""
        super().__init__(state=0)


@register_object_type("button_indicator", scope="overcooked")
class ButtonIndicator(grid_object.GridObj):
    """Button that reveals the target recipe when activated via Toggle.

    Activation costs reward and lasts for a fixed number of timesteps.
    """

    color = (200, 100, 255)
    char = "L"
    is_wall = True

    def __init__(self, *args, **kwargs):
        """Initialize with default state."""
        super().__init__(state=0)


# ---------------------------------------------------------------------------
# Open delivery zone
# ---------------------------------------------------------------------------


@register_object_type("open_delivery_zone", scope="overcooked")
class OpenDeliveryZone(grid_object.GridObj):
    """Delivery zone that accepts any soup type."""

    color = constants.Colors.Green
    char = "X"
    can_place_on = when(agent_holding=OPEN_POT_SOUP_NAMES)
    consumes_on_place = True

    def __init__(self, *args, **kwargs):
        """Initialize with default state."""
        super().__init__(state=0)
