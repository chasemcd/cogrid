"""Array-based layout parser with symbol registry.

Parses ASCII layout strings directly into a fully initialized
:class:`~cogrid.backend.env_state.EnvState` without creating any
Grid, GridObj, or Agent objects.

Symbol mappings can be registered per-scope via :func:`register_symbols`
or included in the scope config's ``symbol_table`` key. The parser
checks both sources (explicit registry first, scope config as fallback).

Parsing always uses numpy (layouts are small, init-time only). Arrays
are converted to JAX arrays when the JAX backend is active.

Usage::

    from cogrid.core.layout_parser import register_symbols, parse_layout
    from cogrid.core.autowire import build_scope_config_from_components

    register_symbols("overcooked", {
        "#": {"object_id": "wall", "is_wall": True},
        "U": {"object_id": "pot"},
        ...
    })

    scope_config = build_scope_config_from_components("overcooked")
    state = parse_layout(layout_strings, "overcooked", scope_config)
"""

from __future__ import annotations

import numpy as np

from cogrid.core.grid_object import object_to_idx


# Module-level registry: scope -> char -> properties dict.
SYMBOL_REGISTRY: dict[str, dict[str, dict]] = {}


def register_symbols(scope: str, symbols: dict[str, dict]) -> None:
    """Register symbol mappings for a scope.

    Each entry maps a single character to a properties dict, e.g.
    ``{"object_id": "wall", "is_wall": True}``.

    Args:
        scope: Scope name (e.g. "overcooked").
        symbols: Dict mapping characters to property dicts.
    """
    SYMBOL_REGISTRY[scope] = symbols


def get_symbols(scope: str, scope_config: dict | None = None) -> dict[str, dict]:
    """Return symbol mappings for a scope.

    Checks the explicit :data:`SYMBOL_REGISTRY` first. Falls back to
    the ``symbol_table`` key in *scope_config* if provided.

    Args:
        scope: Scope name.
        scope_config: Optional scope config dict with a ``symbol_table`` key.

    Returns:
        Dict mapping characters to property dicts.

    Raises:
        ValueError: If no symbols are found for the scope.
    """
    if scope in SYMBOL_REGISTRY:
        return SYMBOL_REGISTRY[scope]
    if scope_config and scope_config.get("symbol_table"):
        return scope_config["symbol_table"]
    raise ValueError(
        f"No symbol mappings found for scope '{scope}'. "
        "Call register_symbols() or include 'symbol_table' in scope config."
    )


def parse_layout(
    layout_strings: list[str],
    scope: str,
    scope_config: dict,
    n_agents: int = 2,
    action_set: str = "cardinal",
    rng_key=None,
) -> "EnvState":
    """Parse ASCII layout strings into a fully initialized EnvState.

    Iterates over the layout character-by-character, building numpy
    arrays for the grid state. No Grid, GridObj, or Agent objects are
    created.

    Special characters:
        - ``' '`` (space): free/empty cell (type 0).
        - ``'#'``: wall (wall_map=1, object_type_map=wall type ID).
        - ``'+'``: spawn position (cell stays empty, position collected).
        - All others: looked up in the symbol table.

    After grid parsing, calls the scope config's ``extra_state_builder``
    (if present) to produce environment-specific extra_state arrays,
    then validates against ``extra_state_schema``.

    All arrays are converted to JAX arrays when the JAX backend is active.

    Args:
        layout_strings: List of strings, each string is one row of the grid.
        scope: Scope name for type ID lookups and symbol resolution.
        scope_config: Scope config dict (from ``build_scope_config_from_components``).
        n_agents: Number of agents (default 2).
        action_set: Action set name (default "cardinal").
        rng_key: JAX PRNG key, or None for numpy backend.

    Returns:
        A fully initialized :class:`~cogrid.backend.env_state.EnvState`.
    """
    from cogrid.backend import get_backend
    from cogrid.backend.env_state import create_env_state, validate_extra_state

    symbols = get_symbols(scope, scope_config)

    height = len(layout_strings)
    width = max(len(row) for row in layout_strings)

    # Build grid arrays with numpy (init-time only).
    wall_map = np.zeros((height, width), dtype=np.int32)
    object_type_map = np.zeros((height, width), dtype=np.int32)
    object_state_map = np.zeros((height, width), dtype=np.int32)
    spawn_positions: list[tuple[int, int]] = []

    wall_type_id = object_to_idx("wall", scope=scope)

    for r, row in enumerate(layout_strings):
        for c, char in enumerate(row):
            if char == " ":
                # Free space -- cell stays 0.
                continue

            if char == "+":
                # Spawn position -- collect but leave cell empty.
                spawn_positions.append((r, c))
                continue

            sym = symbols.get(char)
            if sym is None:
                raise ValueError(
                    f"Unknown symbol '{char}' at row {r}, col {c} "
                    f"for scope '{scope}'. "
                    f"Available symbols: {list(symbols.keys())}"
                )

            object_id = sym.get("object_id")
            is_wall = sym.get("is_wall", False)
            is_spawn = sym.get("is_spawn", False)

            if is_spawn:
                spawn_positions.append((r, c))
                continue

            if object_id is not None:
                type_id = object_to_idx(object_id, scope=scope)
                object_type_map[r, c] = type_id

            if is_wall:
                wall_map[r, c] = 1
                # Ensure wall type ID is set even if object_id wasn't
                # explicitly "wall" (defensive).
                if object_id is None:
                    object_type_map[r, c] = wall_type_id

    # Build extra_state via scope config builder.
    parsed_arrays = {
        "object_type_map": object_type_map,
        "wall_map": wall_map,
        "object_state_map": object_state_map,
    }

    extra_state_builder = scope_config.get("extra_state_builder")
    if extra_state_builder is not None:
        extra_state = extra_state_builder(parsed_arrays, scope)
    else:
        extra_state = {}

    # Validate extra_state against schema.
    schema = scope_config.get("extra_state_schema", {})
    if schema:
        validate_extra_state(extra_state, schema)

    # Build agent arrays from spawn positions.
    agent_pos = np.zeros((n_agents, 2), dtype=np.int32)
    for i in range(min(n_agents, len(spawn_positions))):
        agent_pos[i] = spawn_positions[i]

    agent_dir = np.zeros(n_agents, dtype=np.int32)
    agent_inv = np.full((n_agents, 1), -1, dtype=np.int32)
    done = np.zeros(n_agents, dtype=np.bool_)
    time_step = np.int32(0)

    # Convert to JAX arrays if backend is active.
    if get_backend() == "jax":
        import jax.numpy as jnp

        wall_map = jnp.array(wall_map)
        object_type_map = jnp.array(object_type_map)
        object_state_map = jnp.array(object_state_map)
        agent_pos = jnp.array(agent_pos)
        agent_dir = jnp.array(agent_dir)
        agent_inv = jnp.array(agent_inv)
        done = jnp.array(done)
        time_step = jnp.int32(0)
        extra_state = {k: jnp.array(v) for k, v in extra_state.items()}

    return create_env_state(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        agent_inv=agent_inv,
        wall_map=wall_map,
        object_type_map=object_type_map,
        object_state_map=object_state_map,
        extra_state=extra_state,
        rng_key=rng_key,
        time=time_step,
        done=done,
        n_agents=n_agents,
        height=height,
        width=width,
        action_set=action_set,
    )
