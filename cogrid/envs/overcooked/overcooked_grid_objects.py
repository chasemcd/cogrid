import numpy as np

from cogrid.core import constants
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    add_text_to_image,
)

from cogrid.core import grid_object
from cogrid.core.grid_object import register_object_type


@register_object_type("onion", scope="overcooked", can_pickup=True)
class Onion(grid_object.GridObj):
    object_id = "onion"
    color = constants.Colors.Yellow
    char = "o"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def can_pickup(self, agent: grid_object.GridAgent) -> bool:
        return True

    def render(self, tile_img):
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color
        )


@register_object_type("tomato", scope="overcooked", can_pickup=True)
class Tomato(grid_object.GridObj):
    object_id = "tomato"
    color = constants.Colors.Red
    char = "t"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def can_pickup(self, agent: grid_object.GridAgent) -> bool:
        return True

    def render(self, tile_img):
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), self.color
        )


@register_object_type("onion_stack", scope="overcooked", can_pickup_from=True)
class OnionStack(grid_object.GridObj):
    """An OnionStack is just an (infinite) pile of onions."""

    object_id = "onion_stack"
    color = constants.Colors.Yellow
    char = "O"

    def can_pickup_from(self, agent: grid_object.GridAgent) -> bool:
        return True

    def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
        return Onion()

    def render(self, tile_img: np.ndarray):
        fill_coords(
            tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color
        )


@register_object_type("tomato_stack", scope="overcooked", can_pickup_from=True)
class TomatoStack(grid_object.GridObj):
    """A TomatoStack is just an (infinite) pile of tomatoes."""

    object_id = "tomato_stack"
    color = constants.Colors.Red
    char = "T"

    def can_pickup_from(self, agent: grid_object.GridAgent) -> bool:
        return True

    def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
        return Tomato()

    def render(self, tile_img: np.ndarray):
        fill_coords(
            tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color
        )


@register_object_type("pot", scope="overcooked", can_place_on=True, can_pickup_from=True)
class Pot(grid_object.GridObj):
    object_id = "pot"
    color = constants.Colors.Grey
    char = "U"
    cooking_time: int = 30  # env steps to cook a soup

    def __init__(
        self,
        state: int = 0,
        capacity: int = 3,
        legal_contents: list[grid_object.GridObj] = [Onion, Tomato],
        *args,
        **kwargs,
    ):

        super().__init__(
            state=state, picked_up_from_value=0.0, placed_on_value=0.0
        )

        self.objects_in_pot: list[grid_object.GridObj] = []
        self.capacity: int = capacity
        self.cooking_timer: int = self.cooking_time
        self.legal_contents: list[grid_object.GridObj] = legal_contents

    def can_pickup_from(self, agent: grid_object.GridAgent) -> bool:
        return self.dish_ready and any(
            [isinstance(grid_obj, Plate) for grid_obj in agent.inventory]
        )

    def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
        # if all ingredients are tomatoes, return TomatoSoup
        soup = OnionSoup()
        if all(
            [isinstance(grid_obj, Tomato) for grid_obj in self.objects_in_pot]
        ):
            soup = TomatoSoup()

        self.objects_in_pot = []
        self.cooking_timer = self.cooking_time
        agent.inventory.pop(0)
        return soup

    def can_place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> bool:

        is_legal_ingredient = any(
            [isinstance(cell, grid_obj) for grid_obj in self.legal_contents]
        )

        # return true if cell is the same ingredient type as other ingredients in the pot
        is_same_type = all(
            [
                isinstance(cell, type(grid_obj))
                for grid_obj in self.objects_in_pot
            ]
        )  # return true even if pot is empty

        return (
            len(self.objects_in_pot) < self.capacity
            and is_legal_ingredient
            and is_same_type
        )

    def place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> None:
        self.objects_in_pot.append(cell)

    @property
    def is_cooking(self) -> None:
        return (
            len(self.objects_in_pot) == self.capacity and self.cooking_timer > 0
        )

    def tick(self) -> None:
        """Update cooking time if the pot is full"""
        if len(self.objects_in_pot) == self.capacity and self.cooking_timer > 0:
            self.cooking_timer -= 1
            self.state += 100

        self.state = (
            len(self.objects_in_pot)
            + len(self.objects_in_pot) * self.cooking_timer
        )

    @property
    def dish_ready(self) -> bool:
        return self.cooking_timer == 0

    def render(self, tile_img):
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), self.color
        )

        for i, grid_obj in enumerate(self.objects_in_pot):
            fill_coords(
                tile_img,
                point_in_circle(cx=0.25 * (i + 1), cy=0.2, r=0.2),
                grid_obj.color,
            )

        if len(self.objects_in_pot) == self.capacity:
            add_text_to_image(
                tile_img, text=str(self.cooking_timer), position=(50, 75)
            )

    def encode(self, encode_char: bool = True, scope: str = "global"):
        """Allow encoding to account for the type of soup in the pot"""
        char, _, state = super().encode(encode_char=encode_char, scope=scope)
        extra_state_encoding = int(
            any(isinstance(obj, Tomato) for obj in self.objects_in_pot)
        )
        return (char, extra_state_encoding, state)

    @classmethod
    def build_tick_fn(cls):
        from cogrid.envs.overcooked.config import overcooked_tick_state
        return overcooked_tick_state

    @classmethod
    def extra_state_schema(cls):
        return {
            "pot_contents": {"shape": ("n_pots", 3), "dtype": "int32"},
            "pot_timer": {"shape": ("n_pots",), "dtype": "int32"},
            "pot_positions": {"shape": ("n_pots", 2), "dtype": "int32"},
        }

    @classmethod
    def extra_state_builder(cls):
        from cogrid.envs.overcooked.config import build_overcooked_extra_state
        return build_overcooked_extra_state

    @classmethod
    def build_static_tables(cls):
        from cogrid.envs.overcooked.config import (
            _build_interaction_tables,
            _build_type_ids,
            _build_static_tables,
        )
        scope = "overcooked"
        itables = _build_interaction_tables(scope)
        type_ids = _build_type_ids(scope)
        return _build_static_tables(scope, itables, type_ids)

    @classmethod
    def build_render_sync_fn(cls):
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
                                pot_obj.objects_in_pot.append(
                                    make_object(item_name, scope=scope)
                                )
                    pot_obj.cooking_timer = int(pot_timer[p])
        return pot_render_sync


@register_object_type("plate_stack", scope="overcooked", can_pickup_from=True)
class PlateStack(grid_object.GridObj):
    object_id = "plate_stack"
    color = constants.Colors.White
    char = "="

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            toggle_value=0,
            inventory_value=0,
            overlap_value=0,
        )

    def can_pickup_from(self, agent: grid_object.GridAgent) -> bool:
        return True

    def pick_up_from(self, agent: grid_object.GridAgent) -> grid_object.GridObj:
        return Plate()

    def render(self, tile_img):
        fill_coords(
            tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color
        )
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color
        )


@register_object_type("plate", scope="overcooked", can_pickup=True)
class Plate(grid_object.GridObj):
    object_id = "plate"
    color = constants.Colors.White
    char = "P"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            toggle_value=0,
            inventory_value=0.0,
            overlap_value=0,
        )

    def can_pickup(self, agent: grid_object.GridAgent) -> bool:
        return True

    def render(self, tile_img):
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), self.color
        )


@register_object_type("delivery_zone", scope="overcooked", can_place_on=True)
class DeliveryZone(grid_object.GridObj):
    object_id = "delivery_zone"
    color = constants.Colors.Green
    char = "@"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(state=0, toggle_value=0.0, placed_on_value=0.0)

    def can_place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> bool:
        """Delivery can be toggled by an agent with Soup"""
        toggling_agent_has_soup = any(
            [
                isinstance(grid_obj, (OnionSoup, TomatoSoup))
                for grid_obj in agent.inventory
            ]
        )

        if toggling_agent_has_soup:
            return True

        return False

    def place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> None:
        del cell


@register_object_type("onion_soup", scope="overcooked", can_pickup=True)
class OnionSoup(grid_object.GridObj):
    object_id = "onion_soup"
    color = constants.Colors.LightBrown
    char = "S"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def can_pickup(self, agent: grid_object.GridAgent) -> bool:
        return True

    def render(self, tile_img):
        # Draw plate
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), Plate.color
        )

        # draw soup inside plate
        fill_coords(
            tile_img,
            point_in_circle(cx=0.5, cy=0.5, r=0.3),
            self.color,
        )


@register_object_type("tomato_soup", scope="overcooked", can_pickup=True)
class TomatoSoup(grid_object.GridObj):
    object_id = "tomato_soup"
    color = constants.Colors.Red
    char = "!"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            state=0,
            inventory_value=0.0,
        )

    def can_pickup(self, agent: grid_object.GridAgent) -> bool:
        return True

    def render(self, tile_img):
        # Draw plate
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), Plate.color
        )

        # draw soup inside plate
        fill_coords(
            tile_img,
            point_in_circle(cx=0.5, cy=0.5, r=0.3),
            self.color,
        )
