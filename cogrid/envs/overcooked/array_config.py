"""Overcooked-specific array configuration for the scope config registry.

Provides all environment-specific array logic that was previously embedded
in core modules (interactions.py, cogrid_env.py, grid_utils.py). This
module is registered via ``register_scope_config("overcooked", ...)`` in
``cogrid/envs/overcooked/__init__.py``.

Functions:
    - ``build_overcooked_scope_config()`` -- top-level config builder
    - ``_build_interaction_tables()`` -- pickup_from_produces, legal_pot_ingredients
    - ``_build_type_ids()`` -- name -> type_id mapping
    - ``_extract_overcooked_state()`` -- pot state extraction from Grid
    - ``overcooked_tick()`` -- unified pot cooking timer state machine
    - ``overcooked_interaction_body()`` -- unified per-agent interaction body
"""

from __future__ import annotations

from cogrid.backend.array_ops import set_at, set_at_2d
from cogrid.core.grid_object import object_to_idx, get_object_names


def build_overcooked_extra_state(parsed_arrays, scope="overcooked"):
    """Build extra_state dict for Overcooked from parsed layout arrays.

    Called by the layout parser when scope='overcooked'. Finds pot
    positions from object_type_map and creates the pot state arrays.

    Args:
        parsed_arrays: Dict with "object_type_map" and other grid arrays.
        scope: Scope name for type ID lookups.

    Returns:
        Dict with scope-prefixed keys: overcooked.pot_contents,
        overcooked.pot_timer, overcooked.pot_positions.
    """
    import numpy as _np

    pot_type_id = object_to_idx("pot", scope=scope)
    otm = parsed_arrays["object_type_map"]

    # Find pot positions from object_type_map.
    pot_mask = (otm == pot_type_id)
    pot_positions_list = list(zip(*_np.where(pot_mask)))  # list of (row, col)
    n_pots = len(pot_positions_list)

    if n_pots > 0:
        pot_positions = _np.array(pot_positions_list, dtype=_np.int32)
        pot_contents = _np.full((n_pots, 3), -1, dtype=_np.int32)
        pot_timer = _np.full((n_pots,), 30, dtype=_np.int32)
    else:
        pot_positions = _np.zeros((0, 2), dtype=_np.int32)
        pot_contents = _np.full((0, 3), -1, dtype=_np.int32)
        pot_timer = _np.zeros((0,), dtype=_np.int32)

    return {
        "overcooked.pot_contents": pot_contents,
        "overcooked.pot_timer": pot_timer,
        "overcooked.pot_positions": pot_positions,
    }


def _wrap_overcooked_interaction_body(original_fn):
    """Wrap positional-arg interaction_body for generic extra_state dict protocol.

    Unpacks pot arrays from the ``extra_state`` dict, calls the original
    Overcooked interaction body with positional args, and repacks the
    returned pot arrays back into a new dict.  ``pot_positions`` is
    read-only and preserved via the ``{**extra_state, ...}`` spread.

    Args:
        original_fn: The positional-arg ``overcooked_interaction_body``.

    Returns:
        Wrapped function matching the generic interaction_body protocol:
        ``(i, agent_inv, otm, osm, fwd_r, fwd_c, fwd_type, inv_item,
        base_ok, extra_state, static_tables) -> (agent_inv, otm, osm,
        extra_state)``.
    """
    def wrapped(i, agent_inv, otm, osm, fwd_r, fwd_c, fwd_type, inv_item,
                base_ok, extra_state, static_tables):
        pot_contents = extra_state["pot_contents"]
        pot_timer = extra_state["pot_timer"]
        pot_positions = extra_state["pot_positions"]

        agent_inv, otm, osm, pot_contents, pot_timer = original_fn(
            i, agent_inv, otm, osm, fwd_r, fwd_c, fwd_type, inv_item,
            base_ok, pot_contents, pot_timer, pot_positions, static_tables,
        )

        new_extra = {**extra_state, "pot_contents": pot_contents, "pot_timer": pot_timer}
        return agent_inv, otm, osm, new_extra

    return wrapped


def build_overcooked_scope_config() -> dict:
    """Build the complete Overcooked scope configuration.

    Returns a dict consumed by core modules for all Overcooked-specific
    array-based interaction, tick, and state extraction logic.

    Returns:
        Dict with keys: ``interaction_tables``, ``type_ids``,
        ``state_extractor``, ``tick_handler``, ``interaction_body``,
        ``static_tables``, ``symbol_table``, ``extra_state_schema``,
        ``extra_state_builder``.
    """
    scope = "overcooked"
    itables = _build_interaction_tables(scope)
    type_ids = _build_type_ids(scope)

    # Build static tables for the interaction body.
    # These are Python ints and numpy/jax arrays closed over at trace time,
    # NOT traced values -- so they can be used as compile-time constants.
    static_tables = _build_static_tables(scope, itables, type_ids)

    return {
        "scope": scope,
        "interaction_tables": itables,
        "type_ids": type_ids,
        "state_extractor": _extract_overcooked_state,
        "tick_handler": overcooked_tick_state,
        "interaction_body": _wrap_overcooked_interaction_body(overcooked_interaction_body),
        "static_tables": static_tables,
        # v1.1: layout parser support
        "symbol_table": {
            "#": {"object_id": "wall", "is_wall": True},
            "C": {"object_id": "counter"},
            "U": {"object_id": "pot"},
            "O": {"object_id": "onion_stack"},
            "=": {"object_id": "plate_stack"},
            "@": {"object_id": "delivery_zone"},
            "+": {"object_id": None, "is_spawn": True},
            " ": {"object_id": None},
        },
        "extra_state_schema": {
            "overcooked.pot_contents": {"shape": ("n_pots", 3), "dtype": "int32"},
            "overcooked.pot_timer": {"shape": ("n_pots",), "dtype": "int32"},
            "overcooked.pot_positions": {"shape": ("n_pots", 2), "dtype": "int32"},
        },
        "extra_state_builder": build_overcooked_extra_state,
    }


def _build_interaction_tables(scope: str = "overcooked") -> dict:
    """Build auxiliary lookup tables for interaction processing.

    Moved from ``cogrid.core.interactions.build_interaction_tables()``.

    Returns a dict containing:
    - ``"pickup_from_produces"``: int32 array indexed by type_id, giving the
      type_id of the item produced when picking up from that object. 0 = N/A.
    - ``"legal_pot_ingredients"``: int32 array indexed by type_id, 1 if the
      type is a legal pot ingredient.
    - ``"type_ids"``: dict mapping human-readable names to type_id integers.

    Args:
        scope: Object registry scope.
    """
    from cogrid.backend import xp

    names = get_object_names(scope=scope)
    n_types = len(names)

    pickup_from_produces = xp.zeros(n_types, dtype=xp.int32)
    legal_pot_ingredients = xp.zeros(n_types, dtype=xp.int32)

    onion_id = object_to_idx("onion", scope=scope)
    tomato_id = object_to_idx("tomato", scope=scope)
    plate_id = object_to_idx("plate", scope=scope)
    pot_id = object_to_idx("pot", scope=scope)
    onion_soup_id = object_to_idx("onion_soup", scope=scope)
    tomato_soup_id = object_to_idx("tomato_soup", scope=scope)
    onion_stack_id = object_to_idx("onion_stack", scope=scope)
    tomato_stack_id = object_to_idx("tomato_stack", scope=scope)
    plate_stack_id = object_to_idx("plate_stack", scope=scope)
    counter_id = object_to_idx("counter", scope=scope)
    delivery_zone_id = object_to_idx("delivery_zone", scope=scope)

    # What each pickup-from source produces (stacks only; pot is special-cased)
    pickup_from_produces = set_at(pickup_from_produces, onion_stack_id, onion_id)
    pickup_from_produces = set_at(pickup_from_produces, tomato_stack_id, tomato_id)
    pickup_from_produces = set_at(pickup_from_produces, plate_stack_id, plate_id)

    # Legal pot ingredients
    legal_pot_ingredients = set_at(legal_pot_ingredients, onion_id, 1)
    legal_pot_ingredients = set_at(legal_pot_ingredients, tomato_id, 1)

    type_ids = {
        "onion": onion_id,
        "tomato": tomato_id,
        "plate": plate_id,
        "pot": pot_id,
        "onion_soup": onion_soup_id,
        "tomato_soup": tomato_soup_id,
        "onion_stack": onion_stack_id,
        "tomato_stack": tomato_stack_id,
        "plate_stack": plate_stack_id,
        "counter": counter_id,
        "delivery_zone": delivery_zone_id,
    }

    return {
        "pickup_from_produces": pickup_from_produces,
        "legal_pot_ingredients": legal_pot_ingredients,
        "type_ids": type_ids,
    }


def _build_type_ids(scope: str = "overcooked") -> dict:
    """Build a mapping of type name -> type_id for the Overcooked scope.

    Moved from ``CoGridEnv._build_type_ids()``.

    Args:
        scope: Object registry scope.

    Returns:
        Dict mapping type name strings to integer type IDs.
        Returns -1 for types that do not exist in the scope.
    """
    names = get_object_names(scope=scope)
    type_ids = {}
    type_names_needed = [
        "pot",
        "onion",
        "tomato",
        "plate",
        "onion_soup",
        "tomato_soup",
        "onion_stack",
        "tomato_stack",
        "plate_stack",
        "counter",
        "delivery_zone",
    ]
    for name in type_names_needed:
        if name in names:
            type_ids[name] = object_to_idx(name, scope=scope)
        else:
            type_ids[name] = -1
    return type_ids


def _extract_overcooked_state(grid, scope: str = "overcooked") -> dict:
    """Extract Overcooked-specific state arrays from a Grid object.

    Moved from the pot-specific extraction in ``layout_to_array_state()``.

    Given a Grid object, iterates cells to find pots and extracts their
    positions, contents, and timer values into parallel arrays.

    Always uses numpy for mutable array construction (called during reset's
    layout parsing phase). Callers convert to JAX arrays when needed.

    Args:
        grid: A Grid instance.
        scope: Object registry scope for type ID lookups.

    Returns:
        Dict containing:
        - ``"pot_positions"``: list of ``(row, col)`` tuples for all pots.
        - ``"pot_contents"``: int32 array of shape ``(n_pots, 3)`` with
          ingredient type IDs, -1 sentinel for empty slots.
        - ``"pot_timer"``: int32 array of shape ``(n_pots,)`` with cooking
          timer values.
    """
    import numpy as _np

    pot_positions = []
    pots = []

    for r in range(grid.height):
        for c in range(grid.width):
            cell = grid.get(r, c)
            if cell is not None and cell.object_id == "pot":
                pot_positions.append((r, c))
                pots.append(cell)

    n_pots = len(pots)
    if n_pots > 0:
        pot_contents = _np.full((n_pots, 3), -1, dtype=_np.int32)
        pot_timer = _np.zeros((n_pots,), dtype=_np.int32)

        for i, pot in enumerate(pots):
            pot_timer[i] = int(pot.cooking_timer)
            for j, ingredient in enumerate(pot.objects_in_pot):
                if j >= 3:
                    break
                pot_contents[i, j] = object_to_idx(ingredient, scope=scope)
    else:
        pot_contents = _np.full((0, 3), -1, dtype=_np.int32)
        pot_timer = _np.zeros((0,), dtype=_np.int32)

    return {
        "pot_positions": pot_positions,
        "pot_contents": pot_contents,
        "pot_timer": pot_timer,
    }


def overcooked_tick(pot_contents, pot_timer, capacity=3, cooking_time=30):
    """Unified pot cooking timer update using xp.

    Matches existing ``Pot.tick()`` behavior: count non-sentinel items
    per pot; if pot is full and timer > 0, decrement; compute pot state
    encoding as ``n_items + n_items * timer``.

    Works on both numpy and JAX backends via xp.

    Args:
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``.
        capacity: Maximum items per pot (default 3).
        cooking_time: Initial timer value (default 30).

    Returns:
        Tuple of ``(pot_contents, new_timer, pot_state)``.
    """
    from cogrid.backend import xp

    n_items = xp.sum(pot_contents != -1, axis=1).astype(xp.int32)
    is_cooking = (n_items == capacity) & (pot_timer > 0)
    new_timer = xp.where(is_cooking, pot_timer - 1, pot_timer)
    pot_state = (n_items + n_items * new_timer).astype(xp.int32)
    return pot_contents, new_timer, pot_state


def overcooked_tick_state(state, scope_config):
    """Tick handler with the generic (state, scope_config) signature.

    Wraps :func:`overcooked_tick` to conform to the scope-generic
    tick handler interface expected by ``step_pipeline.step()``.
    Extracts pot arrays from ``state.extra_state``, calls
    ``overcooked_tick()``, writes ``pot_state`` into
    ``object_state_map``, and returns the updated ``EnvState``.

    Args:
        state: Current :class:`EnvState`.
        scope_config: Scope config dict (unused beyond convention).

    Returns:
        Updated :class:`EnvState` with ticked pot timers and
        updated object_state_map.
    """
    import dataclasses

    pot_contents = state.extra_state["overcooked.pot_contents"]
    pot_timer = state.extra_state["overcooked.pot_timer"]
    pot_positions = state.extra_state["overcooked.pot_positions"]
    n_pots = pot_positions.shape[0]

    pot_contents, pot_timer, pot_state = overcooked_tick(
        pot_contents, pot_timer
    )

    # Write pot_state into object_state_map at pot positions
    osm = state.object_state_map
    for p in range(n_pots):
        osm = set_at_2d(osm, pot_positions[p, 0], pot_positions[p, 1], pot_state[p])

    new_extra = {
        **state.extra_state,
        "overcooked.pot_contents": pot_contents,
        "overcooked.pot_timer": pot_timer,
    }
    return dataclasses.replace(
        state, object_state_map=osm, extra_state=new_extra
    )


def overcooked_interaction_body(
    i,                    # agent index
    agent_inv,            # (n_agents, 1) int32
    object_type_map,      # (H, W) int32
    object_state_map,     # (H, W) int32
    fwd_r, fwd_c,         # forward cell coordinates (scalar arrays)
    fwd_type,             # type at forward cell (scalar array)
    inv_item,             # inventory item (scalar array)
    base_ok,              # bool scalar -- is this agent interacting and no agent ahead
    pot_contents,         # (n_pots, 3) int32
    pot_timer,            # (n_pots,) int32
    pot_positions,        # (n_pots, 2) int32
    static_tables,        # dict of static lookup arrays
):
    """Unified per-agent Overcooked interaction body using xp.

    Computes ALL branch conditions using xp.where cascading (no Python
    control flow on traced values). Uses ``array_ops.set_at`` /
    ``set_at_2d`` for mutations instead of ``.at[].set()`` or in-place
    assignment.

    Processes one agent at a time. The caller (process_interactions)
    calls this sequentially for each agent, re-reading state between
    calls so that agent 0's mutations are visible to agent 1.

    Args:
        i: Agent index (Python int or scalar array).
        agent_inv: Agent inventories, shape ``(n_agents, 1)``.
        object_type_map: Grid object type IDs, shape ``(H, W)``.
        object_state_map: Grid object states, shape ``(H, W)``.
        fwd_r: Forward cell row (scalar array, not Python int).
        fwd_c: Forward cell column (scalar array, not Python int).
        fwd_type: Type ID at forward cell (scalar array).
        inv_item: Type ID in agent inventory (scalar array).
        base_ok: Whether this agent is interacting and no agent ahead.
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``.
        pot_positions: Pot positions, shape ``(n_pots, 2)``.
        static_tables: Dict of static lookup arrays and int constants.

    Returns:
        Tuple of ``(agent_inv, object_type_map, object_state_map,
        pot_contents, pot_timer)``.
    """
    from cogrid.backend import xp

    # Unpack static tables (Python-level, not traced)
    CAN_PICKUP = static_tables["CAN_PICKUP"]
    CAN_PICKUP_FROM = static_tables["CAN_PICKUP_FROM"]
    CAN_PLACE_ON = static_tables["CAN_PLACE_ON"]
    pickup_from_produces = static_tables["pickup_from_produces"]
    legal_pot_ingredients = static_tables["legal_pot_ingredients"]
    pot_id = static_tables["pot_id"]
    plate_id = static_tables["plate_id"]
    tomato_id = static_tables["tomato_id"]
    onion_soup_id = static_tables["onion_soup_id"]
    tomato_soup_id = static_tables["tomato_soup_id"]
    delivery_zone_id = static_tables["delivery_zone_id"]
    cooking_time = static_tables["cooking_time"]

    # --- Branch 1: can_pickup ---
    b1_cond = base_ok & (fwd_type > 0) & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1)

    b1_inv = set_at(agent_inv, (i, 0), fwd_type)
    b1_otm = set_at_2d(object_type_map, fwd_r, fwd_c, 0)
    b1_osm = set_at_2d(object_state_map, fwd_r, fwd_c, 0)

    # --- Branch 2: can_pickup_from (Overcooked-specific) ---
    # Sub-case A: pot pickup (soup ready + agent has plate)
    is_pot = (fwd_type == pot_id)

    # Find pot index via array matching
    fwd_pos_2d = xp.stack([fwd_r, fwd_c])
    pot_match = xp.all(pot_positions == fwd_pos_2d[None, :], axis=1)
    pot_idx = xp.argmax(pot_match)
    has_pot_match = xp.any(pot_match)

    # Check pot state
    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = pot_timer[pot_idx] == 0

    # Determine soup type
    all_tomato = xp.all(
        (pot_contents[pot_idx] == -1) | (pot_contents[pot_idx] == tomato_id)
    )
    soup_type = xp.where(all_tomato, tomato_soup_id, onion_soup_id)

    b2_pot_cond = (base_ok & ~b1_cond & is_pot & has_pot_match
                   & has_contents & is_ready & (inv_item == plate_id))

    # Branch 2A results: pick up soup from pot
    b2_pot_inv = set_at(agent_inv, (i, 0), soup_type)
    b2_pot_pc = set_at(pot_contents, (pot_idx, slice(None)), -1)
    b2_pot_pt = set_at(pot_timer, pot_idx, cooking_time)

    # Sub-case B: stack pickup
    is_stack = ~is_pot & (CAN_PICKUP_FROM[fwd_type] == 1)
    produced = pickup_from_produces[fwd_type]
    b2_stack_cond = base_ok & ~b1_cond & is_stack & (inv_item == -1) & (produced > 0)

    # Branch 2B results: pick up produced item from stack
    b2_stack_inv = set_at(agent_inv, (i, 0), produced)

    # --- Branch 3: drop on empty ---
    b3_cond = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond
               & (fwd_type == 0) & (inv_item != -1))

    b3_otm = set_at_2d(object_type_map, fwd_r, fwd_c, inv_item)
    b3_osm = set_at_2d(object_state_map, fwd_r, fwd_c, 0)
    b3_inv = set_at(agent_inv, (i, 0), -1)

    # --- Branch 4: place_on ---
    b4_base = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond
               & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1) & (inv_item != -1))

    # Sub-case A: place on pot
    is_legal = legal_pot_ingredients[inv_item] == 1
    n_items_in_pot = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items_in_pot < 3

    # Same-type constraint: first non-sentinel slot must match inv_item
    first_slot = pot_contents[pot_idx, 0]
    same_type = (n_items_in_pot == 0) | (first_slot == inv_item)

    # Find first empty slot index (0, 1, or 2)
    slot_empty = (pot_contents[pot_idx] == -1)
    first_empty_slot = xp.argmax(slot_empty)

    b4_pot_cond = (b4_base & is_pot & has_pot_match
                   & is_legal & has_capacity & same_type)

    b4_pot_pc = set_at(pot_contents, (pot_idx, first_empty_slot), inv_item)
    b4_pot_inv = set_at(agent_inv, (i, 0), -1)

    # Sub-case B: place on delivery zone
    is_dz = (fwd_type == delivery_zone_id)
    is_soup = (inv_item == onion_soup_id) | (inv_item == tomato_soup_id)
    b4_dz_cond = b4_base & is_dz & is_soup

    b4_dz_inv = set_at(agent_inv, (i, 0), -1)

    # Sub-case C: generic counter place_on
    is_generic = ~is_pot & ~is_dz
    counter_empty = (object_state_map[fwd_r, fwd_c] == 0)
    b4_gen_cond = b4_base & is_generic & counter_empty

    b4_gen_osm = set_at_2d(object_state_map, fwd_r, fwd_c, inv_item)
    b4_gen_inv = set_at(agent_inv, (i, 0), -1)

    # --- Apply all updates with cascading xp.where ---
    # agent_inv
    new_inv = xp.where(b1_cond, b1_inv, agent_inv)
    new_inv = xp.where(b2_pot_cond, b2_pot_inv, new_inv)
    new_inv = xp.where(b2_stack_cond, b2_stack_inv, new_inv)
    new_inv = xp.where(b3_cond, b3_inv, new_inv)
    new_inv = xp.where(b4_pot_cond, b4_pot_inv, new_inv)
    new_inv = xp.where(b4_dz_cond, b4_dz_inv, new_inv)
    new_inv = xp.where(b4_gen_cond, b4_gen_inv, new_inv)

    # object_type_map
    new_otm = xp.where(b1_cond, b1_otm, object_type_map)
    new_otm = xp.where(b3_cond, b3_otm, new_otm)

    # object_state_map
    new_osm = xp.where(b1_cond, b1_osm, object_state_map)
    new_osm = xp.where(b3_cond, b3_osm, new_osm)
    new_osm = xp.where(b4_gen_cond, b4_gen_osm, new_osm)

    # pot_contents
    new_pc = xp.where(b2_pot_cond, b2_pot_pc, pot_contents)
    new_pc = xp.where(b4_pot_cond, b4_pot_pc, new_pc)

    # pot_timer
    new_pt = xp.where(b2_pot_cond, b2_pot_pt, pot_timer)

    return (new_inv, new_otm, new_osm, new_pc, new_pt)


# ======================================================================
# Init-time helpers (not step-path, int() casts here are acceptable)
# ======================================================================


def _build_static_tables(scope, itables, type_ids):
    """Build the static tables dict consumed by the interaction body.

    All values are Python ints or arrays created at config-build time.
    They are closed over by the interaction body at trace time,
    not traced themselves.

    Args:
        scope: Object registry scope.
        itables: Interaction tables dict from ``_build_interaction_tables``.
        type_ids: Type IDs dict from ``_build_type_ids``.

    Returns:
        Dict with CAN_PICKUP, CAN_PICKUP_FROM, CAN_PLACE_ON arrays,
        pickup_from_produces, legal_pot_ingredients arrays, and all
        Overcooked-specific type ID constants.
    """
    from cogrid.core.grid_object import build_lookup_tables

    lookup = build_lookup_tables(scope=scope)

    return {
        "CAN_PICKUP": lookup["CAN_PICKUP"],
        "CAN_PICKUP_FROM": lookup["CAN_PICKUP_FROM"],
        "CAN_PLACE_ON": lookup["CAN_PLACE_ON"],
        "pickup_from_produces": itables["pickup_from_produces"],
        "legal_pot_ingredients": itables["legal_pot_ingredients"],
        "pot_id": int(type_ids.get("pot", -1)),
        "plate_id": int(type_ids.get("plate", -1)),
        "tomato_id": int(type_ids.get("tomato", -1)),
        "onion_soup_id": int(type_ids.get("onion_soup", -1)),
        "tomato_soup_id": int(type_ids.get("tomato_soup", -1)),
        "delivery_zone_id": int(type_ids.get("delivery_zone", -1)),
        "cooking_time": 30,
    }
