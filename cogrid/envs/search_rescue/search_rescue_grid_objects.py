from cogrid.core import grid_object

from cogrid.constants import GridConstants
from cogrid.core.roles import Roles
from cogrid.core.constants import ObjectColors, COLORS
from cogrid.core.grid_utils import adjacent_positions
from cogrid.core import typing
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
)


class MedKit(grid_object.GridObj):
    object_id = "medkit"
    color = ObjectColors.MedKit
    char = GridConstants.MedKit

    def __init__(self, state=0):
        super().__init__(
            state=state,
        )

    def can_pickup(self, agent: grid_object.GridAgent):
        return True

    def render(self, tile_img):
        # red background with white cross
        fill_coords(
            tile_img, point_in_rect(0.1, 0.9, 0.1, 0.9), (255, 0, 0)
        )  # red background
        fill_coords(
            tile_img, point_in_rect(0.4, 0.6, 0.2, 0.8), (255, 255, 255)
        )  # vertical bar
        fill_coords(
            tile_img, point_in_rect(0.2, 0.8, 0.4, 0.6), (255, 255, 255)
        )  # horizontal bar


grid_object.register_object(MedKit.object_id, MedKit)


class Pickaxe(grid_object.GridObj):
    object_id = "pickaxe"
    color = ObjectColors.Pickaxe
    char = GridConstants.Pickaxe

    def __init__(self, state=0):
        super().__init__(
            state=state,
        )

    def can_pickup(self, agent: grid_object.GridAgent):
        return True

    def render(self, tile_img):
        # red background with white cross
        fill_coords(
            tile_img, point_in_rect(0.45, 0.55, 0.15, 0.9), COLORS["brown"]
        )  # brown handle

        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.9, 0.35),
        )
        fill_coords(tile_img, tri_fn, COLORS["grey"])

        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.1, 0.35),
        )
        fill_coords(tile_img, tri_fn, COLORS["grey"])


grid_object.register_object(Pickaxe.object_id, Pickaxe)


class Rubble(grid_object.GridObj):
    object_id = "plate"
    color = ObjectColors.Rubble
    char = GridConstants.Rubble

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.05,  # reward for clearing rubble
        )

    def see_behind(self) -> bool:
        return False

    def toggle(self, env, agent=None) -> bool:
        """Rubble can be toggled by an Engineer/agent with Pickaxe"""
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        toggling_agent_is_engineer = (
            any([isinstance(obj, Pickaxe) for obj in agent.inventory])
            or agent.role == Roles.Engineer
        )

        assert toggling_agent_is_adjacent, "Rubble toggled by non-adjacent agent."

        toggle_success = toggling_agent_is_engineer

        if toggle_success:
            self._remove_from_grid(env.grid)

        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), c)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), c)


grid_object.register_object(Rubble.object_id, Rubble)


class GreenVictim(grid_object.GridObj):
    object_id = "green_victim"
    color = ObjectColors.GreenVictim
    char = GridConstants.GreenVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.1,  # 0.1 reward for rescuing
        )

    def toggle(self, env, agent=None) -> bool:
        """Toggling a victim rescues them. A GreenVictim can be rescued if any agent is adjacent to it"""
        # Toggle should only be triggered if the GreenVictim is directly in front of it. For debugging purposes,
        # we'll just check to make sure that's true (if this isn't triggered it can be removed).
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(GreenVictim.object_id, GreenVictim)


class PurpleVictim(grid_object.GridObj):
    object_id = "purple_victim"
    color = ObjectColors.PurpleVictim
    char = None

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, agent=None) -> bool:
        """Toggling a victim rescues them. A PurpleVictim can be rescued if any agent is adjacent to it"""
        # Toggle should only be triggered if the GreenVictim is directly in front of it. For debugging purposes,
        # we'll just check to make sure that's true (if this isn't triggered it can be removed).
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(PurpleVictim.object_id, PurpleVictim)


class YellowVictim(grid_object.GridObj):
    object_id = "yellow_victim"
    color = ObjectColors.YellowVictim
    char = GridConstants.YellowVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, agent=None) -> bool:
        """
        Toggling a victim rescues them. A YellowVictim can be rescued if a Medic is adjcent to it or the agent
        is carrying a MedKit
        """
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in agent.inventory])
            or agent.role == Roles.Medic
        )

        assert toggling_agent_is_adjacent, "YellowVictim toggled by non-adjacent agent."

        toggle_success = toggling_agent_is_medic

        if toggle_success:
            self._remove_from_grid(env.grid)
        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(YellowVictim.object_id, YellowVictim)


class RedVictim(grid_object.GridObj):
    object_id = "red_victim"
    color = ObjectColors.RedVictim
    char = GridConstants.RedVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
        )
        self.toggle_countdown = 0
        self.first_toggle_agent_id: typing.AgentID = None

    def tick(self):
        """At each timestep, decrement toggle countdown and set the count as the state."""
        if self.toggle_countdown > 0:
            self.toggle_countdown -= 1
        self.state = self.toggle_countdown

    def toggle(self, env, agent) -> bool:
        """A RedVictim can be rescued if a Medic (or agent carrying MedKit) is the adjacent toggling agent
        and then another agent toggles within 30 timesteps."""

        if self.toggle_countdown == 0:

            toggling_agent_has_medkit = any(
                [isinstance(obj, MedKit) for obj in agent.inventory]
            )

            if toggling_agent_has_medkit:
                self.first_toggle_agent = agent.agent_id
                self.toggle_countdown = 30

            return True

        if self.toggle_countdown > 0 and agent.agent_id != self.first_toggle_agent:
            self._remove_from_grid(env.grid)
            return True

        return False

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(RedVictim.object_id, RedVictim)
