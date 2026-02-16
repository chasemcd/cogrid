"""Overcooked-specific array configuration.

Provides environment-specific array logic for Overcooked: tick handlers,
interaction function, extra state building, and static table construction.

Functions:
    - ``build_overcooked_extra_state()`` -- extra_state builder for pot arrays
    - ``overcooked_interaction_fn()`` -- per-agent interaction with (state, ...) -> state signature
    - ``overcooked_interaction_body()`` -- low-level per-agent dispatch to branch handlers
    - ``overcooked_tick()`` -- unified pot cooking timer state machine
    - ``overcooked_tick_state()`` -- tick handler with generic signature
"""

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d
from cogrid.core.grid_object import object_to_idx, get_object_names


def build_overcooked_extra_state(parsed_arrays, scope="overcooked"):
    """Build pot state arrays (contents, timer, positions) from the layout."""
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


def overcooked_interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
    """Per-agent Overcooked interaction: takes state, returns state.

    Extracts arrays, delegates to ``overcooked_interaction_body``, repacks.
    """
    import dataclasses

    static_tables = scope_config.get("static_tables", {})
    fwd_type = state.object_type_map[fwd_r, fwd_c]
    inv_item = state.agent_inv[agent_idx, 0]

    agent_inv, otm, osm, pot_contents, pot_timer = overcooked_interaction_body(
        agent_idx, state.agent_inv, state.object_type_map, state.object_state_map,
        fwd_r, fwd_c, fwd_type, inv_item, base_ok,
        state.extra_state["overcooked.pot_contents"],
        state.extra_state["overcooked.pot_timer"],
        state.extra_state["overcooked.pot_positions"],
        static_tables,
    )

    new_extra = {
        **state.extra_state,
        "overcooked.pot_contents": pot_contents,
        "overcooked.pot_timer": pot_timer,
    }
    return dataclasses.replace(
        state,
        agent_inv=agent_inv,
        object_type_map=otm,
        object_state_map=osm,
        extra_state=new_extra,
    )


def _build_interaction_tables(scope: str = "overcooked") -> dict:
    """Build pickup_from_produces and legal_pot_ingredients lookup arrays."""
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
    """Map Overcooked type names to integer type IDs (-1 if missing)."""
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
    """Extract pot positions, contents, and timer arrays from a Grid object.

    Uses numpy (init-time only). Returns pot_positions, pot_contents, pot_timer.
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
    """Decrement cooking timer for full pots and compute pot state encoding.

    ``pot_state = n_items + n_items * timer``. Returns
    ``(pot_contents, new_timer, pot_state)``.
    """
    n_items = xp.sum(pot_contents != -1, axis=1).astype(xp.int32)
    is_cooking = (n_items == capacity) & (pot_timer > 0)
    new_timer = xp.where(is_cooking, pot_timer - 1, pot_timer)
    pot_state = (n_items + n_items * new_timer).astype(xp.int32)
    return pot_contents, new_timer, pot_state


def overcooked_tick_state(state, scope_config):
    """Generic tick handler: extract pot arrays, tick, write back to EnvState."""
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


def _interact_pickup(base_ok, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                     agent_inv, object_type_map, object_state_map, CAN_PICKUP):
    """Pick up a loose object from the forward cell."""
    b1_cond = base_ok & (fwd_type > 0) & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1)
    b1_inv = set_at(agent_inv, (agent_idx, 0), fwd_type)
    b1_otm = set_at_2d(object_type_map, fwd_r, fwd_c, 0)
    b1_osm = set_at_2d(object_state_map, fwd_r, fwd_c, 0)
    return b1_cond, b1_inv, b1_otm, b1_osm


def _interact_pickup_from_pot(base_ok, b1_cond, fwd_type, inv_item, agent_idx,
                              agent_inv, pot_contents, pot_timer,
                              pot_idx, has_pot_match, pot_id, plate_id,
                              tomato_id, onion_soup_id, tomato_soup_id,
                              cooking_time):
    """Pick up cooked soup from a ready pot (requires plate in hand)."""
    is_pot = (fwd_type == pot_id)

    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = pot_timer[pot_idx] == 0

    all_tomato = xp.all(
        (pot_contents[pot_idx] == -1) | (pot_contents[pot_idx] == tomato_id)
    )
    soup_type = xp.where(all_tomato, tomato_soup_id, onion_soup_id)

    b2_pot_cond = (base_ok & ~b1_cond & is_pot & has_pot_match
                   & has_contents & is_ready & (inv_item == plate_id))

    b2_pot_inv = set_at(agent_inv, (agent_idx, 0), soup_type)
    b2_pot_pc = set_at(pot_contents, (pot_idx, slice(None)), -1)
    b2_pot_pt = set_at(pot_timer, pot_idx, cooking_time)
    return b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt


def _interact_pickup_from_stack(base_ok, b1_cond, fwd_type, inv_item, agent_idx,
                                agent_inv, pot_id, CAN_PICKUP_FROM,
                                pickup_from_produces):
    """Pick up a produced item from a dispenser stack."""
    is_stack = ~(fwd_type == pot_id) & (CAN_PICKUP_FROM[fwd_type] == 1)
    produced = pickup_from_produces[fwd_type]
    b2_stack_cond = base_ok & ~b1_cond & is_stack & (inv_item == -1) & (produced > 0)
    b2_stack_inv = set_at(agent_inv, (agent_idx, 0), produced)
    return b2_stack_cond, b2_stack_inv


def _interact_drop_on_empty(base_ok, b1_cond, b2_pot_cond, b2_stack_cond,
                            fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                            agent_inv, object_type_map, object_state_map):
    """Drop held item onto an empty cell."""
    b3_cond = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond
               & (fwd_type == 0) & (inv_item != -1))
    b3_otm = set_at_2d(object_type_map, fwd_r, fwd_c, inv_item)
    b3_osm = set_at_2d(object_state_map, fwd_r, fwd_c, 0)
    b3_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b3_cond, b3_inv, b3_otm, b3_osm


def _interact_place_on_pot(b4_base, fwd_type, inv_item, agent_idx, agent_inv,
                           pot_contents, pot_idx, has_pot_match, pot_id,
                           legal_pot_ingredients):
    """Place an ingredient into a pot with capacity and same-type checks."""
    is_pot = (fwd_type == pot_id)
    is_legal = legal_pot_ingredients[inv_item] == 1
    n_items_in_pot = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items_in_pot < 3

    first_slot = pot_contents[pot_idx, 0]
    same_type = (n_items_in_pot == 0) | (first_slot == inv_item)

    slot_empty = (pot_contents[pot_idx] == -1)
    first_empty_slot = xp.argmax(slot_empty)

    b4_pot_cond = (b4_base & is_pot & has_pot_match
                   & is_legal & has_capacity & same_type)
    b4_pot_pc = set_at(pot_contents, (pot_idx, first_empty_slot), inv_item)
    b4_pot_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b4_pot_cond, b4_pot_inv, b4_pot_pc


def _interact_place_on_delivery(b4_base, fwd_type, inv_item, agent_idx, agent_inv,
                                delivery_zone_id, onion_soup_id,
                                tomato_soup_id):
    """Place a soup on the delivery zone."""
    is_dz = (fwd_type == delivery_zone_id)
    is_soup = (inv_item == onion_soup_id) | (inv_item == tomato_soup_id)
    b4_dz_cond = b4_base & is_dz & is_soup
    b4_dz_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b4_dz_cond, b4_dz_inv


def _interact_place_on_counter(b4_base, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                               agent_inv, object_state_map, pot_id,
                               delivery_zone_id):
    """Place a held item on a generic counter cell."""
    is_generic = ~(fwd_type == pot_id) & ~(fwd_type == delivery_zone_id)
    counter_empty = (object_state_map[fwd_r, fwd_c] == 0)
    b4_gen_cond = b4_base & is_generic & counter_empty
    b4_gen_osm = set_at_2d(object_state_map, fwd_r, fwd_c, inv_item)
    b4_gen_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b4_gen_cond, b4_gen_inv, b4_gen_osm


def _apply_interaction_updates(
    b1_cond, b1_inv, b1_otm, b1_osm,
    b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt,
    b2_stack_cond, b2_stack_inv,
    b3_cond, b3_inv, b3_otm, b3_osm,
    b4_pot_cond, b4_pot_inv, b4_pot_pc,
    b4_dz_cond, b4_dz_inv,
    b4_gen_cond, b4_gen_inv, b4_gen_osm,
    agent_inv, object_type_map, object_state_map, pot_contents, pot_timer,
):
    """Merge all branch results with cascading xp.where."""
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

    return new_inv, new_otm, new_osm, new_pc, new_pt


def overcooked_interaction_body(
    agent_idx,            # agent index
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
    """Dispatch per-agent interaction to branch handlers and merge results.

    Branches: (1) pickup, (2a) pickup from pot, (2b) pickup from stack,
    (3) drop on empty, (4a) place on pot, (4b) place on delivery zone,
    (4c) place on counter. All results merged via cascading ``xp.where``.
    """
    # Unpack static tables
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

    # Shared pot matching (used by pickup_from_pot and place_on_pot)
    fwd_pos_2d = xp.stack([fwd_r, fwd_c])
    pot_match = xp.all(pot_positions == fwd_pos_2d[None, :], axis=1)
    pot_idx = xp.argmax(pot_match)
    has_pot_match = xp.any(pot_match)

    # Branch 1: pickup
    b1_cond, b1_inv, b1_otm, b1_osm = _interact_pickup(
        base_ok, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_type_map, object_state_map, CAN_PICKUP)

    # Branch 2A: pickup from pot
    b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt = _interact_pickup_from_pot(
        base_ok, b1_cond, fwd_type, inv_item, agent_idx,
        agent_inv, pot_contents, pot_timer,
        pot_idx, has_pot_match, pot_id, plate_id,
        tomato_id, onion_soup_id, tomato_soup_id, cooking_time)

    # Branch 2B: pickup from stack
    b2_stack_cond, b2_stack_inv = _interact_pickup_from_stack(
        base_ok, b1_cond, fwd_type, inv_item, agent_idx,
        agent_inv, pot_id, CAN_PICKUP_FROM, pickup_from_produces)

    # Branch 3: drop on empty
    b3_cond, b3_inv, b3_otm, b3_osm = _interact_drop_on_empty(
        base_ok, b1_cond, b2_pot_cond, b2_stack_cond,
        fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_type_map, object_state_map)

    # Shared base condition for all place-on branches
    b4_base = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond
               & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1) & (inv_item != -1))

    # Branch 4A: place on pot
    b4_pot_cond, b4_pot_inv, b4_pot_pc = _interact_place_on_pot(
        b4_base, fwd_type, inv_item, agent_idx, agent_inv,
        pot_contents, pot_idx, has_pot_match, pot_id, legal_pot_ingredients)

    # Branch 4B: place on delivery zone
    b4_dz_cond, b4_dz_inv = _interact_place_on_delivery(
        b4_base, fwd_type, inv_item, agent_idx, agent_inv,
        delivery_zone_id, onion_soup_id, tomato_soup_id)

    # Branch 4C: place on counter
    b4_gen_cond, b4_gen_inv, b4_gen_osm = _interact_place_on_counter(
        b4_base, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_state_map, pot_id, delivery_zone_id)

    # Merge all branch results
    return _apply_interaction_updates(
        b1_cond, b1_inv, b1_otm, b1_osm,
        b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt,
        b2_stack_cond, b2_stack_inv,
        b3_cond, b3_inv, b3_otm, b3_osm,
        b4_pot_cond, b4_pot_inv, b4_pot_pc,
        b4_dz_cond, b4_dz_inv,
        b4_gen_cond, b4_gen_inv, b4_gen_osm,
        agent_inv, object_type_map, object_state_map, pot_contents, pot_timer)


# ======================================================================
# Init-time helpers (not step-path, int() casts here are acceptable)
# ======================================================================


def _build_static_tables(scope, itables, type_ids):
    """Build the static tables dict closed over by the interaction body.

    Includes property arrays, interaction tables, and type ID constants.
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
