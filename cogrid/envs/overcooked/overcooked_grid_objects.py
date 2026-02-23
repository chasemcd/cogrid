"""Overcooked grid object types (food, pots, plates, delivery zones)."""

import numpy as np

from cogrid.core import constants, grid_object
from cogrid.core.grid_object import register_object_type
from cogrid.core.when import when
from cogrid.visualization.rendering import (
    add_text_to_image,
    fill_coords,
    point_in_circle,
)


@register_object_type("onion", scope="overcooked")
class Onion(grid_object.GridObj):
    """A single onion ingredient."""

    color = constants.Colors.Yellow
    char = "o"
    can_pickup = when()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def render(self, tile_img):
        """Draw a yellow circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color)


@register_object_type("tomato", scope="overcooked")
class Tomato(grid_object.GridObj):
    """A single tomato ingredient."""

    color = constants.Colors.Red
    char = "t"
    can_pickup = when()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def render(self, tile_img):
        """Draw a red circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color)


class _BaseStack(grid_object.GridObj):
    """Base class for infinite dispenser stacks.

    Subclasses set class attributes only: object_id, color, char, produces.
    All behavior (pick_up_from, render) is inherited.
    """

    produces: str = None
    scope: str = "overcooked"
    can_pickup_from = when()

    def pick_up_from(self, agent) -> grid_object.GridObj:
        """Dispense a fresh instance of the produced item."""
        from cogrid.core.grid_object import make_object

        return make_object(self.produces, scope=self.scope)

    def render(self, tile_img):
        """Draw three stacked circles using self.color."""
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color)


@register_object_type("onion_stack", scope="overcooked")
class OnionStack(_BaseStack):
    """An infinite pile of onions."""

    color = constants.Colors.Yellow
    char = "O"
    produces = "onion"


@register_object_type("tomato_stack", scope="overcooked")
class TomatoStack(_BaseStack):
    """An infinite pile of tomatoes."""

    color = constants.Colors.Red
    char = "T"
    produces = "tomato"


@register_object_type("pot", scope="overcooked")
class Pot(grid_object.GridObj):
    """A cooking pot that accepts ingredients and produces soup."""

    color = constants.Colors.Grey
    char = "U"
    cooking_time: int = 30  # env steps to cook a soup
    _recipes_config = None  # set by pre-compose hook from env_config
    _orders_config = None  # set by pre-compose hook from env_config

    def __init__(
        self,
        state: int = 0,
        capacity: int = 3,
        legal_contents: list[grid_object.GridObj] = [Onion, Tomato],
        *args,
        **kwargs,
    ):
        """Initialize pot with capacity and legal ingredient types."""
        super().__init__(state=state, picked_up_from_value=0.0, placed_on_value=0.0)

        self.objects_in_pot: list[grid_object.GridObj] = []
        self.capacity: int = capacity
        self.cooking_timer: int = self.cooking_time
        self.legal_contents: list[grid_object.GridObj] = legal_contents

    def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
        """Remove soup from pot, consume agent's plate, return soup object."""
        # if all ingredients are tomatoes, return TomatoSoup
        soup = OnionSoup()
        if all([isinstance(grid_obj, Tomato) for grid_obj in self.objects_in_pot]):
            soup = TomatoSoup()

        self.objects_in_pot = []
        self.cooking_timer = self.cooking_time
        agent.inventory.pop(0)
        return soup

    def place_on(self, agent: grid_object.GridAgent, cell: grid_object.GridObj) -> None:
        """Add an ingredient to the pot."""
        self.objects_in_pot.append(cell)

    @property
    def is_cooking(self) -> None:
        """True when pot is full and timer has not reached zero."""
        return len(self.objects_in_pot) == self.capacity and self.cooking_timer > 0

    def tick(self) -> None:
        """Update cooking time if the pot is full."""
        if len(self.objects_in_pot) == self.capacity and self.cooking_timer > 0:
            self.cooking_timer -= 1
            self.state += 100

        self.state = len(self.objects_in_pot) + len(self.objects_in_pot) * self.cooking_timer

    @property
    def dish_ready(self) -> bool:
        """True when cooking timer has reached zero."""
        return self.cooking_timer == 0

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

    @classmethod
    def build_tick_fn(cls):
        """Return the overcooked tick-state function."""
        from cogrid.envs.overcooked.config import overcooked_tick_state

        return overcooked_tick_state

    @classmethod
    def extra_state_schema(cls):
        """Return schema for pot-related and order queue extra state arrays."""
        return {
            "pot_contents": {"shape": ("n_pots", 3), "dtype": "int32"},
            "pot_timer": {"shape": ("n_pots",), "dtype": "int32"},
            "pot_positions": {"shape": ("n_pots", 2), "dtype": "int32"},
            "order_recipe": {"shape": ("max_active",), "dtype": "int32"},
            "order_timer": {"shape": ("max_active",), "dtype": "int32"},
            "order_spawn_counter": {"shape": (), "dtype": "int32"},
            "order_recipe_counter": {"shape": (), "dtype": "int32"},
            "order_n_expired": {"shape": (), "dtype": "int32"},
        }

    @classmethod
    def extra_state_builder(cls):
        """Return the function that builds overcooked extra state."""
        from cogrid.envs.overcooked.config import build_overcooked_extra_state

        return build_overcooked_extra_state

    @classmethod
    def build_static_tables(cls):
        """Return pre-computed static tables for the overcooked scope."""
        from cogrid.envs.overcooked.config import (
            DEFAULT_RECIPES,
            _build_interaction_tables,
            _build_order_tables,
            _build_static_tables,
            _build_type_ids,
            compile_recipes,
        )

        scope = "overcooked"
        recipes = cls._recipes_config if cls._recipes_config is not None else DEFAULT_RECIPES
        itables = _build_interaction_tables(scope)
        type_ids = _build_type_ids(scope)
        recipe_tables = compile_recipes(recipes, scope=scope)
        order_tables = _build_order_tables(cls._orders_config, n_recipes=len(recipes))
        return _build_static_tables(
            scope,
            itables,
            type_ids,
            recipe_tables=recipe_tables,
            order_tables=order_tables,
        )

    @classmethod
    def build_render_sync_fn(cls):
        """Return a render-sync callback that updates pot visuals from state."""

        def pot_render_sync(grid, env_state, scope):
            """Sync pot contents and cooking timer from extra_state."""
            from cogrid.core.grid_object import idx_to_object, make_object

            extra = env_state.extra_state
            prefix = f"{scope}."
            pc_key = f"{prefix}pot_contents"
            pt_key = f"{prefix}pot_timer"
            pp_key = f"{prefix}pot_positions"

            if not all(k in extra for k in (pc_key, pt_key, pp_key)):
                return

            pot_contents = np.array(extra[pc_key])
            pot_timer = np.array(extra[pt_key])
            pot_positions = np.array(extra[pp_key])

            for p in range(len(pot_positions)):
                pr, pc = int(pot_positions[p, 0]), int(pot_positions[p, 1])
                pot_obj = grid.get(pr, pc)
                if pot_obj is not None and pot_obj.object_id == "pot":
                    pot_obj.objects_in_pot = []
                    for slot in range(pot_contents.shape[1]):
                        item_id = int(pot_contents[p, slot])
                        if item_id > 0:
                            item_name = idx_to_object(item_id, scope=scope)
                            if item_name:
                                pot_obj.objects_in_pot.append(make_object(item_name, scope=scope))
                    pot_obj.cooking_timer = int(pot_timer[p])

        return pot_render_sync


@register_object_type("plate_stack", scope="overcooked")
class PlateStack(_BaseStack):
    """An infinite stack of plates for picking up soup."""

    color = constants.Colors.White
    char = "="
    produces = "plate"


def make_ingredient_and_stack(
    ingredient_name: str,
    ingredient_char: str,
    ingredient_color,
    stack_name: str,
    stack_char: str,
    scope: str = "overcooked",
) -> tuple:
    """Create and register a new ingredient + stack pair at runtime.

    Must be called BEFORE ``_build_interaction_tables`` runs (i.e. before
    env autowire). Typical call sites: at module-import time in a custom
    env module, or during env config setup before ``build_overcooked_extra_state``.

    Parameters
    ----------
    ingredient_name : str
        Object ID for the ingredient (e.g. "mushroom").
    ingredient_char : str
        Single character for grid encoding. Must be unique within *scope*.
    ingredient_color : tuple or list
        RGB color for rendering.
    stack_name : str
        Object ID for the stack (e.g. "mushroom_stack").
    stack_char : str
        Single character for grid encoding. Must be unique within *scope*.
    scope : str
        Registry scope (default "overcooked").

    Returns:
    -------
    tuple
        (IngredientCls, StackCls) -- the two newly created and registered classes.
    """
    # --- Create ingredient class ---
    IngredientCls = type(
        ingredient_name.title().replace("_", ""),
        (grid_object.GridObj,),
        {
            "color": ingredient_color,
            "char": ingredient_char,
            "can_pickup": when(),
        },
    )

    def _ingredient_init(self, *args, **kwargs):
        grid_object.GridObj.__init__(self, state=0, inventory_value=0.0)

    def _ingredient_render(self, tile_img):
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color)

    IngredientCls.__init__ = _ingredient_init
    IngredientCls.render = _ingredient_render

    register_object_type(ingredient_name, scope=scope)(IngredientCls)

    # --- Create stack class (inherits can_pickup_from from _BaseStack) ---
    StackCls = type(
        stack_name.title().replace("_", ""),
        (_BaseStack,),
        {
            "color": ingredient_color,
            "char": stack_char,
            "produces": ingredient_name,
        },
    )

    register_object_type(stack_name, scope=scope)(StackCls)

    return (IngredientCls, StackCls)


@register_object_type("plate", scope="overcooked")
class Plate(grid_object.GridObj):
    """A plate used to serve completed soups."""

    color = constants.Colors.White
    char = "P"
    can_pickup = when()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(
            state=0,
            toggle_value=0,
            inventory_value=0.0,
            overlap_value=0,
        )

    def render(self, tile_img):
        """Draw a white circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), self.color)


@register_object_type("delivery_zone", scope="overcooked")
class DeliveryZone(grid_object.GridObj):
    """A zone where agents deliver completed soups for reward."""

    color = constants.Colors.Green
    char = "@"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(state=0, toggle_value=0.0, placed_on_value=0.0)

    def place_on(self, agent: grid_object.GridAgent, cell: grid_object.GridObj) -> None:
        """Accept delivery (no-op; reward handled by reward system)."""
        del cell


@register_object_type("onion_soup", scope="overcooked")
class OnionSoup(grid_object.GridObj):
    """A completed onion soup, ready for delivery."""

    color = constants.Colors.LightBrown
    char = "S"
    can_pickup = when()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def render(self, tile_img):
        """Draw a plate with soup inside."""
        # Draw plate
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), Plate.color)

        # draw soup inside plate
        fill_coords(
            tile_img,
            point_in_circle(cx=0.5, cy=0.5, r=0.3),
            self.color,
        )


@register_object_type("tomato_soup", scope="overcooked")
class TomatoSoup(grid_object.GridObj):
    """A completed tomato soup, ready for delivery."""

    color = constants.Colors.Red
    char = "!"
    can_pickup = when()

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize with default state."""
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def render(self, tile_img):
        """Draw a plate with soup inside."""
        # Draw plate
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), Plate.color)

        # draw soup inside plate
        fill_coords(
            tile_img,
            point_in_circle(cx=0.5, cy=0.5, r=0.3),
            self.color,
        )
