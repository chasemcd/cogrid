import numpy as np

from cogrid.constants import GridConstants


def ascii_to_numpy(ascii_list):
    rows, cols = len(ascii_list), len(ascii_list[0])
    for row in range(0, rows):
        assert len(ascii_list[row]) == cols, print("The ascii map is not rectangular!")
    arr = np.full((rows, cols), GridConstants.FreeSpace)
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            arr[row, col] = ascii_list[row][col]
    return arr


def adjacent_positions(row, col):
    for rdelta, cdelta in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        yield row + rdelta, col + cdelta


# PHASE2: This function will operate on EnvState pytree instead of Grid object
def layout_to_array_state(grid, scope: str = "global", scope_config=None) -> dict:
    """Convert a Grid object into the array-based state representation.

    Takes an existing Grid instance (already created from an ASCII layout via
    the _gen_grid() path) and produces parallel array representations for use
    by vectorized operations.

    Args:
        grid: A Grid instance populated from an ASCII layout.
        scope: Object registry scope for type ID lookups.

    Returns:
        Dict containing:
        - ``"object_type_map"``: int32 array of shape ``(height, width)`` with type IDs.
          Empty cells = 0 (matching ``object_to_idx(None) == 0``).
        - ``"object_state_map"``: int32 array of shape ``(height, width)`` with object state values.
        - ``"wall_map"``: int32 array of shape ``(height, width)``, 1 where cell is a Wall.
        - ``"pot_positions"``: list of ``(row, col)`` tuples for all pots.
        - ``"pot_contents"``: int32 array of shape ``(n_pots, 3)`` with ingredient type IDs, -1 sentinel for empty slots.
        - ``"pot_timer"``: int32 array of shape ``(n_pots,)`` with cooking timer values.
        - ``"spawn_points"``: list of ``(row, col)`` tuples where spawn markers exist.
    """
    from cogrid.backend import xp
    from cogrid.core.grid_object import object_to_idx, Wall

    height, width = grid.height, grid.width

    object_type_map = xp.zeros((height, width), dtype=xp.int32)
    object_state_map = xp.zeros((height, width), dtype=xp.int32)
    wall_map = xp.zeros((height, width), dtype=xp.int32)

    pot_positions = []
    pots = []  # collect pot objects for content extraction

    for r in range(height):
        for c in range(width):
            cell = grid.get(r, c)

            if cell is None:
                # Empty cell: type_id = 0, state = 0 (already initialized)
                continue

            type_id = object_to_idx(cell, scope=scope)
            object_type_map[r, c] = type_id
            object_state_map[r, c] = int(cell.state)

            if isinstance(cell, Wall):
                wall_map[r, c] = 1

            if cell.object_id == "pot":
                pot_positions.append((r, c))
                pots.append(cell)

    # Build pot arrays
    n_pots = len(pots)
    if n_pots > 0:
        pot_contents = xp.full((n_pots, 3), -1, dtype=xp.int32)
        pot_timer = xp.zeros((n_pots,), dtype=xp.int32)

        for i, pot in enumerate(pots):
            pot_timer[i] = int(pot.cooking_timer)
            for j, ingredient in enumerate(pot.objects_in_pot):
                if j >= 3:
                    break
                pot_contents[i, j] = object_to_idx(ingredient, scope=scope)
    else:
        pot_contents = xp.full((0, 3), -1, dtype=xp.int32)
        pot_timer = xp.zeros((0,), dtype=xp.int32)

    # Spawn points are handled by _gen_grid before Grid.decode is called,
    # so they won't appear in the grid. Return empty list here -- callers
    # should use env.spawn_points from the CoGridEnv instead.
    spawn_points = []

    return {
        "object_type_map": object_type_map,
        "object_state_map": object_state_map,
        "wall_map": wall_map,
        "pot_positions": pot_positions,
        "pot_contents": pot_contents,
        "pot_timer": pot_timer,
        "spawn_points": spawn_points,
    }


def grid_to_array_state(grid, env_agents, scope: str = "global") -> dict:
    """Convenience wrapper that converts both grid and agents to array state.

    Calls :func:`layout_to_array_state` for the grid and
    :func:`~cogrid.core.agent.create_agent_arrays` for the agents,
    returning a combined dict.

    Args:
        grid: A Grid instance.
        env_agents: Dict mapping AgentID -> Agent.
        scope: Object registry scope for type ID lookups.

    Returns:
        Dict containing all keys from ``layout_to_array_state()`` plus all
        keys from ``create_agent_arrays()``.
    """
    from cogrid.core.agent import create_agent_arrays

    result = layout_to_array_state(grid, scope=scope)
    agent_arrays = create_agent_arrays(env_agents, scope=scope)
    result.update(agent_arrays)
    return result
