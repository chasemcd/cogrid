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


def layout_to_state(grid, scope: str = "global", scope_config=None) -> dict:
    """Convert a Grid instance into array-based state (type map, state map, wall map).

    Scope-specific state is extracted via scope_config's ``state_extractor``
    if provided.
    """
    from cogrid.core.grid_object import object_to_idx, Wall

    height, width = grid.height, grid.width

    # Always use numpy for layout construction (requires mutable in-place
    # assignment).  Callers convert to JAX arrays when needed (e.g. in
    # CoGridEnv.reset() JAX path).
    object_type_map = np.zeros((height, width), dtype=np.int32)
    object_state_map = np.zeros((height, width), dtype=np.int32)
    wall_map = np.zeros((height, width), dtype=np.int32)

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

    # Spawn points are handled by _gen_grid before Grid.decode is called,
    # so they won't appear in the grid. Return empty list here -- callers
    # should use env.spawn_points from the CoGridEnv instead.
    spawn_points = []

    result = {
        "object_type_map": object_type_map,
        "object_state_map": object_state_map,
        "wall_map": wall_map,
        "spawn_points": spawn_points,
    }

    # Extract scope-specific container state (e.g. scope-specific container arrays)
    state_extractor = scope_config.get("state_extractor") if scope_config else None
    if state_extractor is not None:
        extra_state = state_extractor(grid, scope)
        result.update(extra_state)

    return result


def grid_to_state(grid, env_agents, scope: str = "global", scope_config=None) -> dict:
    """Convenience wrapper: layout_to_state + create_agent_arrays."""
    from cogrid.core.agent import create_agent_arrays

    result = layout_to_state(grid, scope=scope, scope_config=scope_config)
    agent_arrays = create_agent_arrays(env_agents, scope=scope)
    result.update(agent_arrays)
    return result
