"""Agent class and array-based agent utilities."""

import numpy as np

from cogrid.backend import xp
from cogrid.core.directions import Directions
from cogrid.core.grid_object import GridObj


class Agent:
    """A single agent in the environment with position, direction, and inventory."""

    def __init__(self, agent_id, start_position, start_direction, **kwargs):
        """Initialize agent with ID, spawn position, and direction."""
        self.id: str = agent_id
        self.pos: tuple[int, int] = start_position
        self.dir: Directions = start_direction
        self.role: str = None
        self.role_idx: int = None
        self.inventory_capacity: int = kwargs.get("inventory_capacity", 1)

        self.terminated: bool = False

        self.collision: bool = (
            False  # Some envs track if an agent crashed into another agent/object
        )

        self.orientation: str = "down"
        self.inventory: list[GridObj] = []
        self.cell_toggled: GridObj | None = None
        self.cell_placed_on: GridObj | None = None
        self.cell_picked_up_from: GridObj | None = None
        self.cell_overlapped: GridObj | None = None

    def rotate_left(self):
        """Rotate the agent's direction counter-clockwise."""
        self.dir -= 1
        if self.dir < 0:
            self.dir += 4

    def rotate_right(self):
        """Rotate the agent's direction clockwise."""
        self.dir = (self.dir + 1) % 4

    @property
    def front_pos(self):
        """Return the position directly in front of the agent."""
        return self.pos + self.dir_vec

    @property
    def dir_vec(self):
        """Return the (delta_row, delta_col) vector for the current direction."""
        dir_to_vec = {
            Directions.Right: np.array((0, 1)),
            Directions.Down: np.array((1, 0)),
            Directions.Left: np.array((0, -1)),
            Directions.Up: np.array((-1, 0)),
        }
        return dir_to_vec[self.dir]

    @property
    def right_vec(self):
        """Return the vector perpendicular to the right of the agent."""
        dy, dx = self.dir_vec
        return np.array((dx, -dy))

    def set_orientation(self):
        """Set the orientation string from the current direction."""
        self.orientation = {
            Directions.Up: "up",
            Directions.Down: "down",
            Directions.Left: "left",
            Directions.Right: "right",
        }[self.dir]

    @property
    def agent_number(self) -> int:
        """Convert agent id to integer, beginning with 1.

        For example, agent-0 -> 1, agent-1 -> 2, etc.
        """
        return int(self.id[-1]) + 1 if isinstance(self.id, str) else self.id + 1


# Direction vectors as an array for vectorized lookups.
# Indexed by direction enum: Right=0, Down=1, Left=2, Up=3
# Each row is [delta_row, delta_col].
DIR_VEC_TABLE = None  # Initialized lazily after backend is set


def get_dir_vec_table():
    """Return the (4, 2) direction vector lookup table, creating it lazily.

    The table is indexed by the direction integer (Right=0, Down=1, Left=2,
    Up=3). Each row is ``[delta_row, delta_col]``, matching the existing
    ``Agent.dir_vec`` property.
    """
    global DIR_VEC_TABLE
    if DIR_VEC_TABLE is None:
        DIR_VEC_TABLE = xp.array(
            [
                [0, 1],  # Right (0) -- increase col
                [1, 0],  # Down  (1) -- increase row
                [0, -1],  # Left  (2) -- decrease col
                [-1, 0],  # Up    (3) -- decrease row
            ],
            dtype=xp.int32,
        )
    return DIR_VEC_TABLE


def create_agent_arrays(env_agents: dict, scope: str = "global") -> dict:
    """Convert Agent objects to parallel arrays (pos, dir, inv).

    Returns dict with ``agent_pos`` (n_agents, 2), ``agent_dir`` (n_agents,),
    ``agent_inv`` (n_agents, 1) with -1 sentinel for empty, ``agent_ids``,
    and ``n_agents``. Agents are sorted by ID for deterministic ordering.
    """
    import numpy as _np

    from cogrid.core.grid_object import object_to_idx

    # Sort by agent_id for deterministic array ordering
    sorted_items = sorted(env_agents.items(), key=lambda x: x[0])
    n_agents = len(sorted_items)

    # Always use numpy for mutable agent array construction.
    # Callers convert to JAX arrays when needed.
    agent_pos = _np.zeros((n_agents, 2), dtype=_np.int32)
    agent_dir = _np.zeros((n_agents,), dtype=_np.int32)
    agent_inv = _np.full((n_agents, 1), -1, dtype=_np.int32)
    agent_ids = []

    for i, (a_id, agent) in enumerate(sorted_items):
        agent_ids.append(a_id)
        agent_pos[i, 0] = agent.pos[0]
        agent_pos[i, 1] = agent.pos[1]
        agent_dir[i] = int(agent.dir)

        if len(agent.inventory) > 0:
            agent_inv[i, 0] = object_to_idx(agent.inventory[0].object_id, scope=scope)

    return {
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "agent_inv": agent_inv,
        "agent_ids": agent_ids,
        "n_agents": n_agents,
    }


def sync_arrays_to_agents(agent_arrays: dict, env_agents: dict) -> None:
    """Write array-state pos/dir back to Agent objects (inverse of create_agent_arrays)."""
    sorted_items = sorted(env_agents.items(), key=lambda x: x[0])

    for i, (a_id, agent) in enumerate(sorted_items):
        agent.pos = (
            int(agent_arrays["agent_pos"][i, 0]),
            int(agent_arrays["agent_pos"][i, 1]),
        )
        agent.dir = int(agent_arrays["agent_dir"][i])
