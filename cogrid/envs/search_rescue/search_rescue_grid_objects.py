"""Search-and-rescue grid object types (items, obstacles, victims)."""

from cogrid.core import constants, grid_object, typing
from cogrid.core.grid_utils import adjacent_positions
from cogrid.core.roles import Roles
from cogrid.visualization.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
)


class MedKit(grid_object.GridObj):
    """A medical kit that enables rescuing yellow victims."""

    object_id = "medkit"
    color = constants.Colors.LightPink
    char = "M"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(
            state=state,
        )

    def can_pickup(self, agent: grid_object.GridAgent):
        """Return True; medkits are always pickable."""
        return True

    def render(self, tile_img):
        """Draw a red cross icon."""
        # red background with white cross
        fill_coords(tile_img, point_in_rect(0.1, 0.9, 0.1, 0.9), (255, 0, 0))
        fill_coords(tile_img, point_in_rect(0.4, 0.6, 0.2, 0.8), (255, 255, 255))
        fill_coords(tile_img, point_in_rect(0.2, 0.8, 0.4, 0.6), (255, 255, 255))


grid_object.register_object(MedKit.object_id, MedKit, scope="search_rescue")


class Pickaxe(grid_object.GridObj):
    """A tool that enables clearing rubble obstacles."""

    object_id = "pickaxe"
    color = constants.Colors.Grey
    char = "T"

    def __init__(self, state=0):
        """Initialize with default state."""
        super().__init__(
            state=state,
        )

    def can_pickup(self, agent: grid_object.GridAgent):
        """Return True; pickaxes are always pickable."""
        return True

    def render(self, tile_img):
        """Draw a pickaxe with brown handle and grey head."""
        # Brown Handle
        fill_coords(tile_img, point_in_rect(0.45, 0.55, 0.15, 0.9), constants.Colors.Brown)

        # Use two triangles to make the pickaxe head
        # These are of the specified color
        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.9, 0.35),
        )
        fill_coords(tile_img, tri_fn, self.color)

        tri_fn = point_in_triangle(
            (0.5, 0.1),
            (0.5, 0.3),
            (0.1, 0.35),
        )
        fill_coords(tile_img, tri_fn, self.color)


grid_object.register_object(Pickaxe.object_id, Pickaxe, scope="search_rescue")


class Rubble(grid_object.GridObj):
    """An obstacle that can be cleared by an Engineer or agent with Pickaxe."""

    object_id = "rubble"
    color = constants.Colors.Brown
    char = "X"

    def __init__(self, state=0):
        """Initialize with toggle reward for clearing."""
        super().__init__(
            state=state,
            toggle_value=0.05,  # reward for clearing rubble
        )

    def see_behind(self) -> bool:
        """Return False; rubble blocks visibility."""
        return False

    def toggle(self, env, agent=None) -> bool:
        """Clear rubble if the toggling agent is an Engineer or holds a Pickaxe."""
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
        """Draw three brown circles representing rubble pile."""
        fill_coords(tile_img, point_in_circle(cx=0.25, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.75, cy=0.3, r=0.2), self.color)
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.7, r=0.2), self.color)


grid_object.register_object(Rubble.object_id, Rubble, scope="search_rescue")


class GreenVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent."""

    object_id = "green_victim"
    color = constants.Colors.Green
    char = "G"

    def __init__(self, state=0):
        """Initialize with toggle reward for rescuing."""
        super().__init__(
            state=state,
            toggle_value=0.1,  # 0.1 reward for rescuing
        )

    def toggle(self, env, agent=None) -> bool:
        """Rescue the victim if any agent is adjacent."""
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "GreenVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        """Draw a green circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


grid_object.register_object(GreenVictim.object_id, GreenVictim, scope="search_rescue")


class PurpleVictim(grid_object.GridObj):
    """A victim rescuable by any adjacent agent (higher reward)."""

    object_id = "purple_victim"
    color = constants.Colors.Purple
    char = "P"

    def __init__(self, state=0):
        """Initialize with toggle reward for rescuing."""
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, agent=None) -> bool:
        """Rescue the victim if any agent is adjacent."""
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        assert toggling_agent_is_adjacent, "PurpleVictim toggled by non-adjacent agent."

        self._remove_from_grid(env.grid)
        return toggling_agent_is_adjacent

    def render(self, tile_img):
        """Draw a purple circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


grid_object.register_object(PurpleVictim.object_id, PurpleVictim, scope="search_rescue")


class YellowVictim(grid_object.GridObj):
    """A victim rescuable only by a Medic or agent carrying a MedKit."""

    object_id = "yellow_victim"
    color = constants.Colors.Yellow
    char = "Y"

    def __init__(self, state=0):
        """Initialize with toggle reward for rescuing."""
        super().__init__(
            state=state,
            toggle_value=0.2,
        )

    def toggle(self, env, agent=None) -> bool:
        """Rescue the victim if a Medic or MedKit-holding agent is adjacent."""
        assert agent
        adj_positions = [*adjacent_positions(*self.pos)]
        toggling_agent_is_adjacent = tuple(agent.pos) in adj_positions
        toggling_agent_is_medic = (
            any([isinstance(obj, MedKit) for obj in agent.inventory]) or agent.role == Roles.Medic
        )

        assert toggling_agent_is_adjacent, "YellowVictim toggled by non-adjacent agent."

        toggle_success = toggling_agent_is_medic

        if toggle_success:
            self._remove_from_grid(env.grid)
        return toggle_success

    def render(self, tile_img):
        """Draw a yellow circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


grid_object.register_object(YellowVictim.object_id, YellowVictim, scope="search_rescue")


class RedVictim(grid_object.GridObj):
    """A victim requiring two-agent cooperative rescue within a time window."""

    object_id = "red_victim"
    color = constants.Colors.Red
    char = "R"

    def __init__(self, state=0):
        """Initialize with countdown timer for cooperative rescue."""
        super().__init__(
            state=state,
        )
        self.toggle_countdown = 0
        self.first_toggle_agent_id: typing.AgentID = None

    def tick(self):
        """Decrement toggle countdown each timestep and update state."""
        if self.toggle_countdown > 0:
            self.toggle_countdown -= 1
        self.state = self.toggle_countdown

    def toggle(self, env, agent) -> bool:
        """Start or complete a cooperative rescue.

        First toggle by a MedKit-holder starts a 30-step countdown.
        A second toggle by a different agent within the window completes rescue.
        """
        if self.toggle_countdown == 0:
            toggling_agent_has_medkit = any([isinstance(obj, MedKit) for obj in agent.inventory])

            if toggling_agent_has_medkit:
                self.first_toggle_agent = agent.agent_id
                self.toggle_countdown = 30

            return True

        if self.toggle_countdown > 0 and agent.agent_id != self.first_toggle_agent:
            self._remove_from_grid(env.grid)
            return True

        return False

    def render(self, tile_img):
        """Draw a red circle."""
        fill_coords(tile_img, point_in_circle(cx=0.5, cy=0.47, r=0.4), self.color)


grid_object.register_object(RedVictim.object_id, RedVictim, scope="search_rescue")
