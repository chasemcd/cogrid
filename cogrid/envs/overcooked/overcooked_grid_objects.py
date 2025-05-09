from __future__ import annotations
from copy import deepcopy
import math

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core.constants import COLORS
from cogrid.core import constants
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
    add_text_to_image,
)

from cogrid.core import grid_object


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


grid_object.register_object(Onion.object_id, Onion, scope="overcooked")


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

grid_object.register_object(Tomato.object_id, Tomato, scope="overcooked")


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


grid_object.register_object(
    OnionStack.object_id, OnionStack, scope="overcooked"
)


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


grid_object.register_object(
    TomatoStack.object_id, TomatoStack, scope="overcooked"
)


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
        self.objects_in_pot = []
        self.cooking_timer = self.cooking_time
        agent.inventory.pop(0)  # TODO(chase): assumes size 1 inventory
        return OnionSoup()

    def can_place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> bool:
        """Can only place onions in the soup!"""
        if not any(
            [isinstance(cell, grid_obj) for grid_obj in self.legal_contents]
        ):
            return False

        return len(self.objects_in_pot) < self.capacity

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


grid_object.register_object(Pot.object_id, Pot, scope="overcooked")


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


grid_object.register_object(
    PlateStack.object_id, PlateStack, scope="overcooked"
)


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


grid_object.register_object(Plate.object_id, Plate, scope="overcooked")


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
            [isinstance(grid_obj, (OnionSoup, TomatoSoup)) for grid_obj in agent.inventory]
        )

        if toggling_agent_has_soup:
            return True

        return False

    def place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> None:
        del cell


grid_object.register_object(
    DeliveryZone.object_id, DeliveryZone, scope="overcooked"
)


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


grid_object.register_object(OnionSoup.object_id, OnionSoup, scope="overcooked")


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