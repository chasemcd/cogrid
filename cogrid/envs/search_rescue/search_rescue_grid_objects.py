from cogrid.core import grid_object

from cogrid.constants import GridConstants
from cogrid.core.roles import Roles
from cogrid.core.constants import ObjectColors, COLORS
from cogrid.core.grid_utils import adjacent_positions
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
)


class MedKit(grid_object.GridObj):
    name = "medkit"
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
    name = "pickaxe"
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
    name = "rubble"
    color = ObjectColors.Rubble
    char = GridConstants.Rubble

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.05,  # reward for clearing rubble
        )

    def see_behind(self) -> bool:
        return False

    def toggle(self, env, toggling_agent=None) -> bool:
        """Rubble can be toggled by an Engineer/agent with Pickaxe"""
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_engineer = (
            any([isinstance(obj, Pickaxe) for obj in toggling_agent.inventory])
            or toggling_agent.role == Roles.Engineer
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
    name = "green_victim"
    color = ObjectColors.GreenVictim
    char = GridConstants.GreenVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.1,  # 0.1 reward for rescuing
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """Toggling a victim rescues them. A GreenVictim can be rescued if any agent is adjacent to it"""
        # Toggle should only be triggered if the GreenVictim is directly in front of it. For debugging purposes,
        # we'll just check to make sure that's true (if this isn't triggered it can be removed).
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(GreenVictim.object_id, GreenVictim)


class PurpleVictim(grid_object.GridObj):
    name = "purple_victim"
    color = ObjectColors.PurpleVictim
    char = None

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """Toggling a victim rescues them. A PurpleVictim can be rescued if any agent is adjacent to it"""
        # Toggle should only be triggered if the GreenVictim is directly in front of it. For debugging purposes,
        # we'll just check to make sure that's true (if this isn't triggered it can be removed).
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(PurpleVictim.object_id, PurpleVictim)


class YellowVictim(grid_object.GridObj):
    name = "yellow_victim"
    color = ObjectColors.YellowVictim
    char = GridConstants.YellowVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """
        Toggling a victim rescues them. A YellowVictim can be rescued if a Medic is adjcent to it or the agent
        is carrying a MedKit
        """
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in toggling_agent.inventory])
            or toggling_agent.role == Roles.Medic
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
    name = "red_victim"
    color = ObjectColors.RedVictim
    char = GridConstants.RedVictim

    def __init__(self, state=0):
        super().__init__(
            state=state,
            toggle_value=0.3,
        )

    def toggle(self, env, toggling_agent=None) -> bool:
        """A RedVictim can be rescued if a Medic (or agent carrying MedKit) is the adjacent toggling agent
        and there is another agent adjacent."""
        assert toggling_agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(toggling_agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in toggling_agent.inventory])
            or toggling_agent.role == Roles.Medic
        )

        assert toggling_agent_is_adjacent, "RedVictim toggled by non-adjacent agent."

        other_adjacent_agent = None
        for agent in env.grid_agents.values():
            if agent is toggling_agent or tuple(agent.pos) not in adj_positions:
                continue
            other_adjacent_agent = agent
            break

        toggle_success = toggling_agent_is_medic and other_adjacent_agent is not None

        if toggle_success:
            self._remove_from_grid(env.grid)

            # If we're using common reward both will automatically receive. If not, both acted in saving,
            # so we reward both for the red victim.
            if not env.common_reward:
                other_adjacent_agent.reward += self.toggle_value

        return toggle_success

    def render(self, tile_img):
        c = COLORS[self.color]
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), c)


grid_object.register_object(RedVictim.object_id, RedVictim)
