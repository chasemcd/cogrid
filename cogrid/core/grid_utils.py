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


def layout_to_array_state(grid, scope: str = "global", scope_config=None) -> dict:
    """Convert a Grid object into the array-based state representation.

    Takes an existing Grid instance (already created from an ASCII layout via
    the _gen_grid() path) and produces parallel array representations for use
    by vectorized operations.

    Scope-specific container state (e.g. scope-specific container arrays) is
    extracted by the scope config's ``state_extractor`` callback if provided.

    Args:
        grid: A Grid instance populated from an ASCII layout.
        scope: Object registry scope for type ID lookups.
        scope_config: Optional scope config dict from ``build_scope_config_from_components()``.
            If provided and contains a ``state_extractor``, it is called to
            extract scope-specific state (e.g. container positions/contents/timer).

    Returns:
        Dict containing:
        - ``"object_type_map"``: int32 array of shape ``(height, width)`` with type IDs.
          Empty cells = 0 (matching ``object_to_idx(None) == 0``).
        - ``"object_state_map"``: int32 array of shape ``(height, width)`` with object state values.
        - ``"wall_map"``: int32 array of shape ``(height, width)``, 1 where cell is a Wall.
        - ``"spawn_points"``: list of ``(row, col)`` tuples where spawn markers exist.
        - Plus any scope-specific keys from the state_extractor.
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


def grid_to_array_state(grid, env_agents, scope: str = "global", scope_config=None) -> dict:
    """Convenience wrapper that converts both grid and agents to array state.

    Calls :func:`layout_to_array_state` for the grid and
    :func:`~cogrid.core.agent.create_agent_arrays` for the agents,
    returning a combined dict.

    Args:
        grid: A Grid instance.
        env_agents: Dict mapping AgentID -> Agent.
        scope: Object registry scope for type ID lookups.
        scope_config: Optional scope config dict from ``build_scope_config_from_components()``.

    Returns:
        Dict containing all keys from ``layout_to_array_state()`` plus all
        keys from ``create_agent_arrays()``.
    """
    from cogrid.core.agent import create_agent_arrays

    result = layout_to_array_state(grid, scope=scope, scope_config=scope_config)
    agent_arrays = create_agent_arrays(env_agents, scope=scope)
    result.update(agent_arrays)
    return result
