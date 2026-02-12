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
    - ``_overcooked_tick_handler()`` -- pot cooking timer state machine
    - ``_place_on_pot()`` -- ingredient placement into pots
    - ``_place_on_delivery_zone()`` -- soup delivery logic
    - ``_overcooked_interaction_handler()`` -- scope-specific interaction sub-cases
    - ``overcooked_interaction_body_jax()`` -- JAX-path interaction loop body
    - ``overcooked_tick_jax()`` -- JAX-path pot cooking tick
"""

from __future__ import annotations

from cogrid.backend.array_ops import set_at
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


def build_overcooked_scope_config() -> dict:
    """Build the complete Overcooked scope configuration.

    Returns a dict consumed by core modules for all Overcooked-specific
    array-based interaction, tick, and state extraction logic.

    Returns:
        Dict with keys: ``interaction_tables``, ``type_ids``,
        ``state_extractor``, ``interaction_handler``, ``tick_handler``,
        ``place_on_handlers``, ``symbol_table``, ``extra_state_schema``,
        ``extra_state_builder``, plus JAX-specific entries.
    """
    scope = "overcooked"
    itables = _build_interaction_tables(scope)
    type_ids = _build_type_ids(scope)

    # Build static tables for the JAX interaction body.
    # These are Python ints and numpy/jax arrays closed over at trace time,
    # NOT traced values -- so they can be used as compile-time constants.
    static_tables = _build_static_tables(scope, itables, type_ids)

    return {
        "interaction_tables": itables,
        "type_ids": type_ids,
        "state_extractor": _extract_overcooked_state,
        "interaction_handler": _overcooked_interaction_handler,
        "tick_handler": _overcooked_tick_handler,
        "place_on_handlers": {},  # handled inside interaction_handler
        # JAX-specific entries
        "interaction_body_jax": overcooked_interaction_body_jax,
        "tick_handler_jax": overcooked_tick_jax,
        "static_tables": static_tables,
        "toggle_branches_jax": [],  # Overcooked has no toggle types
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


def _overcooked_tick_handler(
    pot_contents,     # (n_pots, 3) int32
    pot_timer,        # (n_pots,) int32
    capacity=3,       # max items per pot
    cooking_time=30,  # initial timer value
):
    """Vectorized pot cooking timer update.

    Moved from ``cogrid.core.interactions.tick_objects_array()``.

    Matches existing ``Pot.tick()`` behavior (overcooked_grid_objects.py
    lines 201-210):

    - Count non-sentinel items per pot.
    - If pot is full (``n_items == capacity``) AND ``timer > 0``: decrement.
    - Compute pot state encoding: ``n_items + n_items * timer``.

    This is fully vectorized across all pots.

    Args:
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``.
        capacity: Maximum items per pot (default 3).
        cooking_time: Initial timer value (default 30).

    Returns:
        Tuple of ``(pot_contents, pot_timer, pot_state)``:

        - pot_contents: unchanged (ingredients don't change during tick).
        - pot_timer: updated timers with decremented cooking pots.
        - pot_state: int32 array of shape ``(n_pots,)`` with state encoding
          values to write into ``object_state_map`` at pot positions.
    """
    from cogrid.backend import xp

    if pot_contents.shape[0] == 0:
        return pot_contents, pot_timer.copy(), xp.zeros((0,), dtype=xp.int32)

    pot_timer = pot_timer.copy()  # PHASE2: convert to .at[].set()

    # Count non-sentinel items per pot: (n_pots,)
    n_items = xp.sum(pot_contents != -1, axis=1).astype(xp.int32)

    # Determine which pots are cooking (full and timer > 0)
    is_cooking = (n_items == capacity) & (pot_timer > 0)

    # Decrement timer for cooking pots
    pot_timer = xp.where(is_cooking, pot_timer - 1, pot_timer)  # PHASE2: use .at[].set()

    # Compute pot state encoding: n_items + n_items * timer
    # Matches existing Pot.tick() state calculation at line 207-210:
    #   self.state = len(self.objects_in_pot) + len(self.objects_in_pot) * self.cooking_timer
    pot_state = (n_items + n_items * pot_timer).astype(xp.int32)

    return pot_contents, pot_timer, pot_state


def _place_on_pot(
    agent_inv, agent_idx, inv_item, pot_contents, pot_timer,
    pot_idx, legal_pot_ingredients, cooking_time,
):
    """Handle placing an ingredient into a pot.

    Moved from ``cogrid.core.interactions._place_on_pot()``.

    Checks match existing ``Pot.can_place_on()`` logic:

    1. Ingredient is legal (onion or tomato per ``legal_contents``)
    2. Pot has capacity (fewer than 3 items)
    3. Ingredient matches type of existing contents (same-type constraint)

    If all checks pass: add ingredient to first empty slot, clear inventory.
    """
    from cogrid.backend import xp

    # Check legal ingredient
    if legal_pot_ingredients[inv_item] != 1:
        return

    # Count items in pot
    n_items = int(xp.sum(pot_contents[pot_idx] != -1))

    # Check capacity
    if n_items >= 3:
        return

    # Check same-type constraint: if pot has items, new item must match
    if n_items > 0:
        existing_type = -1
        for s in range(3):
            if int(pot_contents[pot_idx, s]) != -1:
                existing_type = int(pot_contents[pot_idx, s])
                break
        if existing_type != inv_item:
            return

    # Find first empty slot and place ingredient
    for s in range(3):
        if int(pot_contents[pot_idx, s]) == -1:
            pot_contents[pot_idx, s] = inv_item  # PHASE2: convert to .at[].set()
            agent_inv[agent_idx, 0] = -1  # PHASE2: convert to .at[].set()
            break


def _place_on_delivery_zone(agent_inv, agent_idx, inv_item, type_ids):
    """Handle placing an item on a delivery zone.

    Extracted from ``cogrid.core.interactions._place_on_non_pot()``.
    Only accepts soup (OnionSoup or TomatoSoup). Consumes soup from
    inventory on delivery.

    Args:
        agent_inv: Agent inventories array, shape ``(n_agents, 1)``.
        agent_idx: Index of the acting agent.
        inv_item: Type ID of item in agent's inventory.
        type_ids: Dict mapping type name strings to integer type IDs.
    """
    onion_soup_id = type_ids.get("onion_soup", -1)
    tomato_soup_id = type_ids.get("tomato_soup", -1)

    if inv_item == onion_soup_id or inv_item == tomato_soup_id:
        agent_inv[agent_idx, 0] = -1  # PHASE2: convert to .at[].set()
        # Soup is consumed -- delivery zone doesn't visibly store it


def _overcooked_interaction_handler(
    action_type,       # str: "pickup_from" or "place_on"
    agent_idx,         # int: index of the acting agent
    agent_inv,         # (n_agents, 1) int32
    fwd_r,             # int: forward cell row
    fwd_c,             # int: forward cell column
    fwd_type,          # int: type ID of forward cell
    inv_item,          # int: type ID in agent inventory (-1 = empty)
    object_type_map,   # (H, W) int32
    object_state_map,  # (H, W) int32
    extra_state,       # dict with pot_contents, pot_timer, pot_positions, type_ids, etc.
):
    """Handle Overcooked-specific interaction sub-cases.

    This function is called by the generic ``process_interactions_array``
    (after Plan 03 refactoring) to handle scope-specific sub-cases within
    the pickup_from and place_on priority branches.

    Args:
        action_type: Either ``"pickup_from"`` or ``"place_on"``, indicating
            which priority branch is delegating.
        agent_idx: Index of the agent performing the action.
        agent_inv: Agent inventories array, shape ``(n_agents, 1)``.
        fwd_r: Row of the cell the agent is facing.
        fwd_c: Column of the cell the agent is facing.
        fwd_type: Type ID of the object in the forward cell.
        inv_item: Type ID of the item in the agent's inventory (-1 = empty).
        object_type_map: Grid object type IDs, shape ``(H, W)``.
        object_state_map: Grid object states, shape ``(H, W)``.
        extra_state: Dict containing scope-specific state:
            - ``pot_contents``: ``(n_pots, 3)`` int32
            - ``pot_timer``: ``(n_pots,)`` int32
            - ``pot_positions``: ``(n_pots, 2)`` int32
            - ``pot_pos_to_idx``: dict mapping ``(r, c)`` -> pot index
            - ``type_ids``: dict mapping name -> type_id
            - ``pickup_from_produces``: ``(n_types,)`` int32
            - ``legal_pot_ingredients``: ``(n_types,)`` int32
            - ``cooking_time``: int

    Returns:
        bool: True if the interaction was handled, False otherwise.
    """
    from cogrid.backend import xp

    type_ids = extra_state["type_ids"]
    pot_id = type_ids["pot"]
    plate_id = type_ids["plate"]
    tomato_id = type_ids["tomato"]
    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]
    delivery_zone_id = type_ids["delivery_zone"]

    pot_contents = extra_state["pot_contents"]
    pot_timer = extra_state["pot_timer"]
    pot_pos_to_idx = extra_state["pot_pos_to_idx"]
    cooking_time = extra_state.get("cooking_time", 30)

    if action_type == "pickup_from":
        if fwd_type == pot_id:
            # Pot pickup-from: check dish_ready AND agent has plate
            pot_idx = pot_pos_to_idx.get((fwd_r, fwd_c))
            if pot_idx is None:
                return False

            has_contents = int(xp.sum(pot_contents[pot_idx] != -1)) > 0
            is_ready = int(pot_timer[pot_idx]) == 0

            if not (has_contents and is_ready and inv_item == plate_id):
                return False

            # Determine soup type: all tomato -> tomato_soup, else onion_soup
            all_tomato = True
            for s in range(3):
                slot = int(pot_contents[pot_idx, s])
                if slot != -1 and slot != tomato_id:
                    all_tomato = False
                    break
            soup_type = tomato_soup_id if all_tomato else onion_soup_id

            agent_inv[agent_idx, 0] = soup_type  # PHASE2: convert to .at[].set()
            pot_contents[pot_idx, :] = -1  # PHASE2: convert to .at[].set()
            pot_timer[pot_idx] = cooking_time  # PHASE2: convert to .at[].set()
            return True
        else:
            # Stacks: use pickup_from_produces table
            pickup_from_produces = extra_state.get("pickup_from_produces")
            if pickup_from_produces is not None and inv_item == -1:
                produced = int(pickup_from_produces[fwd_type])
                if produced > 0:
                    agent_inv[agent_idx, 0] = produced  # PHASE2: convert to .at[].set()
                    return True
            return False

    elif action_type == "place_on":
        if fwd_type == pot_id:
            # Place ingredient in pot
            pot_idx = pot_pos_to_idx.get((fwd_r, fwd_c))
            if pot_idx is not None:
                legal_pot_ingredients = extra_state.get("legal_pot_ingredients")
                if legal_pot_ingredients is not None:
                    _place_on_pot(
                        agent_inv, agent_idx, inv_item, pot_contents, pot_timer,
                        pot_idx, legal_pot_ingredients, cooking_time,
                    )
                    return True
            return False

        elif fwd_type == delivery_zone_id:
            _place_on_delivery_zone(agent_inv, agent_idx, inv_item, type_ids)
            return True

        else:
            # Not an Overcooked-specific place_on target
            return False

    return False


# ======================================================================
# JAX-path functions (JIT-compatible)
# ======================================================================


def _build_static_tables(scope, itables, type_ids):
    """Build the static tables dict consumed by the JAX interaction body.

    All values are Python ints or arrays created at config-build time.
    They are closed over by the JAX interaction body at trace time,
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


def overcooked_interaction_body_jax(
    i, carry, *,
    actions, agent_pos, agent_dir, dir_vec_table,
    action_pickup_drop_idx, static_tables,
):
    """JAX-compatible loop body for Overcooked interaction processing.

    Handles a single agent's interaction for use inside
    ``jax.lax.fori_loop``. All branches are computed unconditionally
    and selected via ``jnp.where`` masking -- no Python control flow
    on traced values.

    The ``carry`` tuple contains all mutable state arrays. Read-only
    arrays (actions, agent_pos, agent_dir) and static tables are
    closed over from the enclosing scope.

    Args:
        i: Loop index (agent index).
        carry: Tuple of mutable state:
            ``(agent_inv, object_type_map, object_state_map,
              pot_contents, pot_timer, pot_positions)``.
        actions: (n_agents,) int32 action indices (closed over).
        agent_pos: (n_agents, 2) int32 positions (closed over).
        agent_dir: (n_agents,) int32 directions (closed over).
        dir_vec_table: (4, 2) int32 direction vectors (closed over).
        action_pickup_drop_idx: int, PickupDrop action index (closed over).
        static_tables: Dict of static lookup arrays and int constants
            (closed over).

    Returns:
        Updated carry tuple with same structure.
    """
    import jax.numpy as jnp

    (agent_inv, object_type_map, object_state_map,
     pot_contents, pot_timer, pot_positions) = carry

    n_agents = agent_pos.shape[0]
    H, W = object_type_map.shape

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

    # --- Common setup ---
    is_interact = (actions[i] == action_pickup_drop_idx)

    # Compute forward position, clip to bounds
    fwd_pos = agent_pos[i] + dir_vec_table[agent_dir[i]]
    fwd_r = jnp.clip(fwd_pos[0], 0, H - 1)
    fwd_c = jnp.clip(fwd_pos[1], 0, W - 1)

    # Check agent ahead: any other agent at fwd_pos
    other_mask = jnp.arange(n_agents) != i
    pos_match = jnp.all(agent_pos == jnp.array([fwd_r, fwd_c])[None, :], axis=1)
    agent_ahead = jnp.any(pos_match & other_mask)

    fwd_type = object_type_map[fwd_r, fwd_c]
    inv_item = agent_inv[i, 0]

    # Common guard: must be interacting and no agent ahead
    base_ok = is_interact & ~agent_ahead

    # --- Branch 1: can_pickup ---
    b1_cond = base_ok & (fwd_type > 0) & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1)

    # Branch 1 results
    b1_inv = agent_inv.at[i, 0].set(fwd_type)
    b1_otm = object_type_map.at[fwd_r, fwd_c].set(0)
    b1_osm = object_state_map.at[fwd_r, fwd_c].set(0)

    # --- Branch 2: can_pickup_from (Overcooked-specific) ---
    # Sub-case A: pot pickup (soup ready + agent has plate)
    is_pot = (fwd_type == pot_id)

    # Find pot index via array matching
    pot_match = jnp.all(pot_positions == jnp.array([fwd_r, fwd_c])[None, :], axis=1)
    pot_idx = jnp.argmax(pot_match)
    has_pot_match = jnp.any(pot_match)

    # Check pot state
    has_contents = jnp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = pot_timer[pot_idx] == 0

    # Determine soup type
    all_tomato = jnp.all(
        (pot_contents[pot_idx] == -1) | (pot_contents[pot_idx] == tomato_id)
    )
    soup_type = jnp.where(all_tomato, tomato_soup_id, onion_soup_id)

    b2_pot_cond = (base_ok & ~b1_cond & is_pot & has_pot_match
                   & has_contents & is_ready & (inv_item == plate_id))

    # Branch 2A results: pick up soup from pot
    b2_pot_inv = agent_inv.at[i, 0].set(soup_type)
    b2_pot_pc = pot_contents.at[pot_idx, :].set(-1)
    b2_pot_pt = pot_timer.at[pot_idx].set(cooking_time)

    # Sub-case B: stack pickup
    is_stack = ~is_pot & (CAN_PICKUP_FROM[fwd_type] == 1)
    produced = pickup_from_produces[fwd_type]
    b2_stack_cond = base_ok & ~b1_cond & is_stack & (inv_item == -1) & (produced > 0)

    # Branch 2B results: pick up produced item from stack
    b2_stack_inv = agent_inv.at[i, 0].set(produced)

    # --- Branch 3: drop on empty ---
    b3_cond = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond
               & (fwd_type == 0) & (inv_item != -1))

    b3_otm = object_type_map.at[fwd_r, fwd_c].set(inv_item)
    b3_osm = object_state_map.at[fwd_r, fwd_c].set(0)
    b3_inv = agent_inv.at[i, 0].set(-1)

    # --- Branch 4: place_on ---
    b4_base = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond
               & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1) & (inv_item != -1))

    # Sub-case A: place on pot
    is_legal = legal_pot_ingredients[inv_item] == 1
    n_items_in_pot = jnp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items_in_pot < 3

    # Same-type constraint: first non-sentinel slot must match inv_item
    # Use a safe lookup: if pot is empty, existing_type=-1 and constraint is vacuously true
    first_slot = pot_contents[pot_idx, 0]
    same_type = (n_items_in_pot == 0) | (first_slot == inv_item)

    # Find first empty slot index (0, 1, or 2)
    slot_empty = (pot_contents[pot_idx] == -1)
    # If all full, first_empty_slot = 3 but we guard with has_capacity
    first_empty_slot = jnp.argmax(slot_empty)

    b4_pot_cond = (b4_base & is_pot & has_pot_match
                   & is_legal & has_capacity & same_type)

    b4_pot_pc = pot_contents.at[pot_idx, first_empty_slot].set(inv_item)
    b4_pot_inv = agent_inv.at[i, 0].set(-1)

    # Sub-case B: place on delivery zone
    is_dz = (fwd_type == delivery_zone_id)
    is_soup = (inv_item == onion_soup_id) | (inv_item == tomato_soup_id)
    b4_dz_cond = b4_base & is_dz & is_soup

    b4_dz_inv = agent_inv.at[i, 0].set(-1)

    # Sub-case C: generic counter place_on
    is_generic = ~is_pot & ~is_dz
    counter_empty = (object_state_map[fwd_r, fwd_c] == 0)
    b4_gen_cond = b4_base & is_generic & counter_empty

    b4_gen_osm = object_state_map.at[fwd_r, fwd_c].set(inv_item)
    b4_gen_inv = agent_inv.at[i, 0].set(-1)

    # --- Apply all updates with cascading jnp.where ---
    # agent_inv
    new_inv = jnp.where(b1_cond, b1_inv, agent_inv)
    new_inv = jnp.where(b2_pot_cond, b2_pot_inv, new_inv)
    new_inv = jnp.where(b2_stack_cond, b2_stack_inv, new_inv)
    new_inv = jnp.where(b3_cond, b3_inv, new_inv)
    new_inv = jnp.where(b4_pot_cond, b4_pot_inv, new_inv)
    new_inv = jnp.where(b4_dz_cond, b4_dz_inv, new_inv)
    new_inv = jnp.where(b4_gen_cond, b4_gen_inv, new_inv)

    # object_type_map
    new_otm = jnp.where(b1_cond, b1_otm, object_type_map)
    new_otm = jnp.where(b3_cond, b3_otm, new_otm)

    # object_state_map
    new_osm = jnp.where(b1_cond, b1_osm, object_state_map)
    new_osm = jnp.where(b3_cond, b3_osm, new_osm)
    new_osm = jnp.where(b4_gen_cond, b4_gen_osm, new_osm)

    # pot_contents
    new_pc = jnp.where(b2_pot_cond, b2_pot_pc, pot_contents)
    new_pc = jnp.where(b4_pot_cond, b4_pot_pc, new_pc)

    # pot_timer
    new_pt = jnp.where(b2_pot_cond, b2_pot_pt, pot_timer)

    return (new_inv, new_otm, new_osm, new_pc, new_pt, pot_positions)


def overcooked_tick_jax(
    pot_contents,     # (n_pots, 3) int32
    pot_timer,        # (n_pots,) int32
    capacity=3,       # max items per pot (static)
    cooking_time=30,  # initial timer value (static)
):
    """JAX-compatible pot cooking timer update.

    Functionally identical to :func:`_overcooked_tick_handler` but uses
    ``jnp`` operations and avoids ``.copy()`` (JAX arrays are immutable).

    Safe to call via ``jax.jit(overcooked_tick_jax,
    static_argnames=['capacity', 'cooking_time'])``.

    Args:
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``.
        capacity: Maximum items per pot (default 3, static).
        cooking_time: Initial timer value (default 30, static).

    Returns:
        Tuple of ``(pot_contents, pot_timer, pot_state)``:

        - pot_contents: unchanged (ingredients don't change during tick).
        - pot_timer: updated timers with decremented cooking pots.
        - pot_state: int32 array of shape ``(n_pots,)`` with state encoding.
    """
    import jax.numpy as jnp

    # Count non-sentinel items per pot: (n_pots,)
    n_items = jnp.sum(pot_contents != -1, axis=1).astype(jnp.int32)

    # Determine which pots are cooking (full and timer > 0)
    is_cooking = (n_items == capacity) & (pot_timer > 0)

    # Decrement timer for cooking pots
    new_timer = jnp.where(is_cooking, pot_timer - 1, pot_timer)

    # Compute pot state encoding: n_items + n_items * timer
    pot_state = (n_items + n_items * new_timer).astype(jnp.int32)

    return pot_contents, new_timer, pot_state
