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
"""

from __future__ import annotations

from cogrid.core.grid_object import object_to_idx, get_object_names


def build_overcooked_scope_config() -> dict:
    """Build the complete Overcooked scope configuration.

    Returns a dict consumed by core modules for all Overcooked-specific
    array-based interaction, tick, and state extraction logic.

    Returns:
        Dict with keys: ``interaction_tables``, ``type_ids``,
        ``state_extractor``, ``interaction_handler``, ``tick_handler``,
        ``place_on_handlers``.
    """
    scope = "overcooked"
    itables = _build_interaction_tables(scope)
    type_ids = _build_type_ids(scope)
    return {
        "interaction_tables": itables,
        "type_ids": type_ids,
        "state_extractor": _extract_overcooked_state,
        "interaction_handler": _overcooked_interaction_handler,
        "tick_handler": _overcooked_tick_handler,
        "place_on_handlers": {},  # handled inside interaction_handler
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
    pickup_from_produces[onion_stack_id] = onion_id
    pickup_from_produces[tomato_stack_id] = tomato_id
    pickup_from_produces[plate_stack_id] = plate_id

    # Legal pot ingredients
    legal_pot_ingredients[onion_id] = 1
    legal_pot_ingredients[tomato_id] = 1

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
    from cogrid.backend import xp

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
