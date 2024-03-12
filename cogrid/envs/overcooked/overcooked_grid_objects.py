from __future__ import annotations
from copy import deepcopy
import math

import numpy as np

from cogrid.constants import GridConstants
from cogrid.core.constants import COLORS
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
    color = "yellow"
    char = "0"

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
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), c)


grid_object.register_object(Onion.object_id, Onion)


class OnionStack(grid_object.GridObj):
    """An OnionStack is just an (infinite) pile of onions."""

    object_id = "onion_stack"
    color = "yellow"
    char = "+"

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
        return Onion()

    def render(self, tile_img: np.ndarray):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), c)


grid_object.register_object(OnionStack.object_id, OnionStack)


class Pot(grid_object.GridObj):
    object_id = "pot"
    color = "grey"
    char = GridConstants.Pot
    cooking_time: int = 30  # env steps to cook a soup

    def __init__(
        self,
        state: int = 0,
        capacity: int = 3,
        legal_contents: list[grid_object.GridObj] = [Onion],
        *args,
        **kwargs,
    ):

        super().__init__(state=state, picked_up_from_value=0.0, placed_on_value=0.0)

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
        if not any([isinstance(cell, grid_obj) for grid_obj in self.legal_contents]):
            return False

        return len(self.objects_in_pot) < self.capacity

    def place_on(self, agent: grid_object.GridAgent, cell: grid_object.GridObj) -> None:
        self.objects_in_pot.append(cell)

    @property
    def is_cooking(self) -> None:
        return len(self.objects_in_pot) == self.capacity

    def tick(self) -> None:
        """Update cooking time if the pot is full"""
        if len(self.objects_in_pot) == self.capacity and self.cooking_timer > 0:
            self.cooking_timer -= 1
            self.state += 100

        self.state = (
            len(self.objects_in_pot) + len(self.objects_in_pot) * self.cooking_timer
        )

    @property
    def dish_ready(self) -> bool:
        return self.cooking_timer == 0

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), c)

        for i, grid_obj in enumerate(self.objects_in_pot):
            fill_coords(
                tile_img,
                point_in_circle(cx=0.25 * (i + 1), cy=0.2, r=0.2),
                COLORS[grid_obj.color],
            )

        if len(self.objects_in_pot) == self.capacity:
            add_text_to_image(tile_img, text=str(self.cooking_timer), position=(50, 75))


grid_object.register_object(Pot.object_id, Pot)


class PlateStack(grid_object.GridObj):
    object_id = "plate_stack"
    color = "white"
    char = GridConstants.PlateStack

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
        c = COLORS[self.color]
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), c)


grid_object.register_object(PlateStack.object_id, PlateStack)


class Plate(grid_object.GridObj):
    object_id = "plate"
    color = "white"
    char = GridConstants.Plate

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
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), c)


grid_object.register_object(Plate.object_id, Plate)


class DeliveryZone(grid_object.GridObj):
    object_id = "delivery_zone"
    color = "green"
    char = GridConstants.DeliveryZone

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(state=0, toggle_value=0.0, placed_on_value=1.0)

    def can_place_on(
        self, agent: grid_object.GridAgent, cell: grid_object.GridObj
    ) -> bool:
        """Delivery can be toggled by an agent with Soup"""
        toggling_agent_has_soup = any(
            [isinstance(grid_obj, OnionSoup) for grid_obj in agent.inventory]
        )

        if toggling_agent_has_soup:
            return True

        return False

    def place_on(self, agent: grid_object.GridAgent, cell: grid_object.GridObj) -> None:
        del cell


grid_object.register_object(DeliveryZone.object_id, DeliveryZone)


class OnionSoup(grid_object.GridObj):
    object_id = "onion_soup"
    color = "brown"
    char = GridConstants.OnionSoup

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
        c = COLORS[self.color]
        # draw white plate
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.5), COLORS["white"])

        # draw soup inside plate
        fill_coords(
            tile_img, point_in_circle(cx=0.5, cy=0.5, r=0.3), COLORS["light_brown"]
        )


grid_object.register_object(OnionSoup.object_id, OnionSoup)
