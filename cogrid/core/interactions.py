"""Vectorized interaction processing using integer lookup tables.

Replaces isinstance()-based dispatch in ``CoGridEnv.interact()`` with
integer type code lookups on array state. All operations work on
parallel arrays and avoid Python object manipulation.

Key functions:

- ``process_interactions_array()`` -- handles pickup, pickup_from, drop,
  place_on interactions for all agents using lookup tables.
- ``tick_objects_array()`` -- vectorized pot cooking timer.
- ``build_interaction_tables()`` -- helper to build the auxiliary tables
  (PICKUP_FROM_PRODUCES, LEGAL_POT_INGREDIENTS, type_ids dict) needed by
  the interaction processor.
"""

from __future__ import annotations

from cogrid.backend import xp
from cogrid.core.grid_object import object_to_idx, get_object_names


def build_interaction_tables(scope: str = "overcooked") -> dict:
    """Build auxiliary lookup tables for interaction processing.

    Returns a dict containing:
    - ``"pickup_from_produces"``: int32 array indexed by type_id, giving the
      type_id of the item produced when picking up from that object. 0 = N/A.
    - ``"legal_pot_ingredients"``: int32 array indexed by type_id, 1 if the
      type is a legal pot ingredient.
    - ``"type_ids"``: dict mapping human-readable names to type_id integers.

    Args:
        scope: Object registry scope.
    """
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


def process_interactions_array(
    agent_pos,                # (n_agents, 2) int32
    agent_dir,                # (n_agents,) int32
    agent_inv,                # (n_agents, 1) int32, -1 = empty
    actions,                  # (n_agents,) int32
    object_type_map,          # (H, W) int32
    object_state_map,         # (H, W) int32
    pot_contents,             # (n_pots, 3) int32, -1 = empty slot
    pot_timer,                # (n_pots,) int32
    pot_positions,            # (n_pots, 2) int32
    lookup_tables,            # dict with CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON, CAN_PICKUP_FROM
    type_ids,                 # dict mapping name -> type_id integer
    scope,                    # str -- for object_to_idx lookups
    dir_vec_table,            # (4, 2) int32 -- direction vectors
    action_pickup_drop_idx,   # int -- index of PickupDrop action
    action_toggle_idx,        # int -- index of Toggle action
    pickup_from_produces=None,  # (n_types,) int32 -- what each pickup-from source produces
    legal_pot_ingredients=None, # (n_types,) int32 -- 1 if legal pot ingredient
    cooking_time=30,          # int -- initial timer value for pots
):
    """Process interactions for all agents on array state.

    Implements the same priority order as ``CoGridEnv.interact()``
    (cogrid_env.py lines 573-643):

    1. can_pickup -- pick up a pickupable object (empty inventory required)
    2. can_pickup_from -- pick up from a dispenser/stack (empty inventory) or
       pick up soup from a ready pot (plate required in inventory, special case)
    3. drop on empty -- drop held item onto empty grid cell
    4. place_on -- place held item onto a container (counter, pot, delivery zone)

    Phase 1: loops over agents since interactions can affect shared grid state.

    Args:
        agent_pos: Agent positions, shape ``(n_agents, 2)``.
        agent_dir: Agent directions, shape ``(n_agents,)``.
        agent_inv: Agent inventories, shape ``(n_agents, 1)``. -1 = empty.
        actions: Action indices, shape ``(n_agents,)``.
        object_type_map: Grid object type IDs, shape ``(H, W)``.
        object_state_map: Grid object states, shape ``(H, W)``.
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``. -1 = empty.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``.
        pot_positions: Pot grid positions, shape ``(n_pots, 2)``.
        lookup_tables: Dict of property arrays from ``build_lookup_tables()``.
        type_ids: Dict mapping type name strings to integer type IDs.
        scope: Object registry scope string.
        dir_vec_table: Direction vector lookup, shape ``(4, 2)``.
        action_pickup_drop_idx: Integer index of the PickupDrop action.
        action_toggle_idx: Integer index of the Toggle action.
        pickup_from_produces: What each pickup-from source dispenses.
        legal_pot_ingredients: Which type IDs are legal pot ingredients.
        cooking_time: Initial cooking timer value for pots.

    Returns:
        Tuple of ``(agent_inv, object_type_map, object_state_map,
        pot_contents, pot_timer)``.

    .. note::
        This function does NOT wire into CoGridEnv.step() yet. That happens
        in Plan 07 (integration). These are standalone functions that take
        array state and return updated array state.
    """
    # Make copies so we don't mutate the caller's arrays
    agent_inv = agent_inv.copy()  # PHASE2: convert to .at[].set()
    object_type_map = object_type_map.copy()  # PHASE2: convert to .at[].set()
    object_state_map = object_state_map.copy()  # PHASE2: convert to .at[].set()
    pot_contents = pot_contents.copy()  # PHASE2: convert to .at[].set()
    pot_timer = pot_timer.copy()  # PHASE2: convert to .at[].set()

    CAN_PICKUP = lookup_tables["CAN_PICKUP"]
    CAN_PICKUP_FROM = lookup_tables["CAN_PICKUP_FROM"]
    CAN_PLACE_ON = lookup_tables["CAN_PLACE_ON"]

    pot_id = type_ids["pot"]
    plate_id = type_ids["plate"]
    tomato_id = type_ids["tomato"]
    onion_soup_id = type_ids["onion_soup"]
    tomato_soup_id = type_ids["tomato_soup"]

    n_agents = agent_pos.shape[0]
    n_pots = pot_positions.shape[0] if pot_positions.ndim > 0 and pot_positions.shape[0] > 0 else 0

    # Build pot position lookup: (row, col) -> pot array index
    pot_pos_to_idx = {}
    for p in range(n_pots):
        r, c = int(pot_positions[p, 0]), int(pot_positions[p, 1])
        pot_pos_to_idx[(r, c)] = p

    H, W = object_type_map.shape

    # Phase 1: loop over agents (shared grid state mutations)
    # PHASE2: vectorize with careful ordering
    for i in range(n_agents):
        action = int(actions[i])

        # Skip non-interaction actions
        if action != action_pickup_drop_idx:
            continue

        # Compute forward position
        fwd_pos = agent_pos[i] + dir_vec_table[int(agent_dir[i])]
        fwd_r, fwd_c = int(fwd_pos[0]), int(fwd_pos[1])

        # Bounds check
        if fwd_r < 0 or fwd_r >= H or fwd_c < 0 or fwd_c >= W:
            continue

        # Check agent ahead: if any other agent is at fwd_pos, skip
        # Matches cogrid_env.py line 589-594
        agent_ahead = False
        for j in range(n_agents):
            if j != i and int(agent_pos[j, 0]) == fwd_r and int(agent_pos[j, 1]) == fwd_c:
                agent_ahead = True
                break

        if agent_ahead:
            continue

        fwd_type = int(object_type_map[fwd_r, fwd_c])
        inv_item = int(agent_inv[i, 0])

        # Evaluate dynamic can_pickup_from condition.
        # The static CAN_PICKUP_FROM[fwd_type] flag says the *type* supports
        # pickup-from, but the original code calls the *instance* method which
        # has dynamic conditions:
        #   - Stacks: always True (unconditional)
        #   - Pots: True only when dish_ready AND agent has plate
        #   - Base GridObj: True when obj_placed_on is not None
        # We compute the effective dynamic condition here so the elif chain
        # mirrors the original priority order exactly.
        can_pickup_from_dynamic = False
        can_agent_pickup = False  # mirrors OvercookedAgent.can_pickup(fwd_cell)
        if fwd_type > 0 and CAN_PICKUP_FROM[fwd_type] == 1:
            if fwd_type == pot_id:
                # Pot: dish_ready (timer==0 with contents) AND agent has plate
                pot_idx_for_b2 = pot_pos_to_idx.get((fwd_r, fwd_c))
                if pot_idx_for_b2 is not None:
                    has_contents = int(xp.sum(pot_contents[pot_idx_for_b2] != -1)) > 0
                    is_ready = int(pot_timer[pot_idx_for_b2]) == 0
                    if has_contents and is_ready and inv_item == plate_id:
                        can_pickup_from_dynamic = True
                        can_agent_pickup = True  # special override
            else:
                # Stacks: always True
                can_pickup_from_dynamic = True
                # agent.can_pickup for stacks: len(inv) < capacity (need empty)
                can_agent_pickup = (inv_item == -1)

        # ---- Priority branch 1: can_pickup (lines 598-606) ----
        # Condition: object in front is pickupable AND agent has empty inventory
        if fwd_type > 0 and CAN_PICKUP[fwd_type] == 1 and inv_item == -1:
            agent_inv[i, 0] = fwd_type  # PHASE2: convert to .at[].set()
            object_type_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()
            object_state_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()

        # ---- Priority branch 2: can_pickup_from (lines 607-615) ----
        # Uses the dynamic condition computed above so that non-ready pots
        # and full-inventory stacks correctly fall through to later branches.
        elif can_pickup_from_dynamic and can_agent_pickup:
            if fwd_type == pot_id:
                # Pot pickup-from: replace plate with soup, clear pot
                pot_idx = pot_pos_to_idx[(fwd_r, fwd_c)]
                # Determine soup type: all tomato -> tomato_soup, else onion_soup
                all_tomato = True
                for s in range(3):
                    slot = int(pot_contents[pot_idx, s])
                    if slot != -1 and slot != tomato_id:
                        all_tomato = False
                        break
                soup_type = tomato_soup_id if all_tomato else onion_soup_id

                agent_inv[i, 0] = soup_type  # PHASE2: convert to .at[].set()
                pot_contents[pot_idx, :] = -1  # PHASE2: convert to .at[].set()
                pot_timer[pot_idx] = cooking_time  # PHASE2: convert to .at[].set()
            else:
                # Stack pickup-from: dispense item into empty inventory
                if pickup_from_produces is not None:
                    produced = int(pickup_from_produces[fwd_type])
                    if produced > 0:
                        agent_inv[i, 0] = produced  # PHASE2: convert to .at[].set()

        # ---- Priority branch 3: drop on empty (lines 616-619) ----
        # Condition: empty cell ahead AND agent has item in inventory
        elif fwd_type == 0 and inv_item != -1:
            object_type_map[fwd_r, fwd_c] = inv_item  # PHASE2: convert to .at[].set()
            object_state_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()
            agent_inv[i, 0] = -1  # PHASE2: convert to .at[].set()

        # ---- Priority branch 4: place_on (lines 620-631) ----
        # Condition: object ahead supports place_on AND agent has item
        elif fwd_type > 0 and CAN_PLACE_ON[fwd_type] == 1 and inv_item != -1:
            if fwd_type == pot_id:
                # Place ingredient in pot
                pot_idx = pot_pos_to_idx.get((fwd_r, fwd_c))
                if pot_idx is not None and legal_pot_ingredients is not None:
                    _place_on_pot(
                        agent_inv, i, inv_item, pot_contents, pot_timer,
                        pot_idx, legal_pot_ingredients, cooking_time,
                    )
            else:
                # Place on counter / delivery zone / other
                _place_on_non_pot(
                    agent_inv, i, inv_item, fwd_r, fwd_c,
                    fwd_type, object_state_map, type_ids,
                )

    return agent_inv, object_type_map, object_state_map, pot_contents, pot_timer


def _place_on_pot(
    agent_inv, agent_idx, inv_item, pot_contents, pot_timer,
    pot_idx, legal_pot_ingredients, cooking_time,
):
    """Handle placing an ingredient into a pot.

    Checks match existing ``Pot.can_place_on()`` logic:

    1. Ingredient is legal (onion or tomato per ``legal_contents``)
    2. Pot has capacity (fewer than 3 items)
    3. Ingredient matches type of existing contents (same-type constraint)

    If all checks pass: add ingredient to first empty slot, clear inventory.
    """
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


def _place_on_non_pot(
    agent_inv, agent_idx, inv_item, fwd_r, fwd_c,
    fwd_type, object_state_map, type_ids,
):
    """Handle placing an item on a counter or delivery zone.

    For counters: places item only if nothing already placed
    (``object_state_map[r, c] == 0``). Uses ``object_state_map``
    to store the placed item's type ID.

    For delivery zones: only accepts soup (OnionSoup or TomatoSoup).
    The soup is consumed on delivery.
    """
    counter_id = type_ids.get("counter", -1)
    delivery_zone_id = type_ids.get("delivery_zone", -1)
    onion_soup_id = type_ids.get("onion_soup", -1)
    tomato_soup_id = type_ids.get("tomato_soup", -1)

    if fwd_type == counter_id:
        # Counter: only place if nothing already on it
        if int(object_state_map[fwd_r, fwd_c]) == 0:
            object_state_map[fwd_r, fwd_c] = inv_item  # PHASE2: convert to .at[].set()
            agent_inv[agent_idx, 0] = -1  # PHASE2: convert to .at[].set()
    elif fwd_type == delivery_zone_id:
        # Delivery zone: only accept soup
        if inv_item == onion_soup_id or inv_item == tomato_soup_id:
            agent_inv[agent_idx, 0] = -1  # PHASE2: convert to .at[].set()
            # Soup is consumed -- delivery zone doesn't visibly store it
    else:
        # Generic can_place_on: store in object_state_map
        if int(object_state_map[fwd_r, fwd_c]) == 0:
            object_state_map[fwd_r, fwd_c] = inv_item  # PHASE2: convert to .at[].set()
            agent_inv[agent_idx, 0] = -1  # PHASE2: convert to .at[].set()


def tick_objects_array(
    pot_contents,    # (n_pots, 3) int32
    pot_timer,       # (n_pots,) int32
    capacity=3,      # max items per pot
    cooking_time=30, # initial timer value
):
    """Vectorized pot cooking timer update.

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
