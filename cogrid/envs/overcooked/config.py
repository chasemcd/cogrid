"""Overcooked-specific array configuration.

Provides environment-specific array logic for Overcooked: tick handlers,
interaction function, extra state building, and static table construction.

Functions:
    - ``build_overcooked_extra_state()`` -- extra_state builder for pot arrays
    - ``overcooked_interaction_fn()`` -- per-agent interaction with (state, ...) -> state signature
    - ``overcooked_interaction_body()`` -- low-level per-agent dispatch to branch handlers
    - ``overcooked_tick()`` -- unified pot cooking timer state machine
    - ``overcooked_tick_state()`` -- tick handler with generic signature


Overcooked Interaction Overview
===============================

When an agent issues a PickupDrop action, ``process_interactions`` (in
``cogrid.core.interactions``) determines the cell the agent is facing and
whether another agent is blocking it. It then calls
``overcooked_interaction_fn`` once per agent (lower index = higher priority).

The interaction resolves to exactly one of seven mutually exclusive branches,
evaluated in strict priority order. The first branch whose condition is True
wins; all later branches are suppressed via cascading ``~earlier_cond`` guards.

Decision tree (evaluated per agent per step):

    base_ok?  (agent issued PickupDrop AND no other agent in the forward cell)
        |
        +-- No  --> no-op (all branches short-circuit)
        |
        +-- Yes
             |
             +-- Branch 1: PICKUP loose object
             |     condition: forward cell has a pickupable object, hand is empty
             |     effect:    object moves from grid to agent inventory
             |
             +-- Branch 2A: PICKUP FROM POT (cooked soup)
             |     condition: forward cell is a pot, pot is done cooking,
             |                agent holds a plate
             |     effect:    plate replaced with soup in inventory,
             |                pot contents/timer reset
             |
             +-- Branch 2B: PICKUP FROM STACK (dispenser)
             |     condition: forward cell is a stack (onion/tomato/plate),
             |                hand is empty
             |     effect:    produced item placed in agent inventory,
             |                stack remains on grid (infinite supply)
             |
             +-- Branch 3: DROP on empty cell
             |     condition: forward cell is empty (type 0), hand is not empty
             |     effect:    held item placed on grid, inventory cleared
             |
             +-- Branch 4 (place-on): agent holds an item, forward cell is
             |   a "place-on" target (pot, delivery zone, or counter)
             |     |
             |     +-- 4A: PLACE ON POT
             |     |     condition: forward cell is pot, held item is a legal
             |     |                ingredient, pot has capacity, same ingredient type
             |     |     effect:    item added to pot_contents, inventory cleared
             |     |
             |     +-- 4B: PLACE ON DELIVERY ZONE
             |     |     condition: forward cell is delivery zone, held item is soup
             |     |     effect:    soup removed from inventory (delivery scored
             |     |                by the reward function, not here)
             |     |
             |     +-- 4C: PLACE ON COUNTER
             |           condition: forward cell is a counter/generic place-on
             |                      target (not pot, not delivery zone),
             |                      counter cell is empty (state == 0)
             |           effect:    item stored in object_state_map, inventory cleared


Array layout (extra_state)
--------------------------

Each pot on the grid is tracked by index in three parallel arrays stored
in ``state.extra_state``:

    pot_contents:  (n_pots, 3) int32   -- type IDs of ingredients, -1 = empty slot
    pot_timer:     (n_pots,)   int32   -- cooking countdown, 30 = not started, 0 = done
    pot_positions: (n_pots, 2) int32   -- (row, col) of each pot on the grid

Example with 2 pots, pot 0 has two onions cooking, pot 1 is empty:

    pot_contents = [[onion_id, onion_id, -1],    pot_timer = [25, 30]
                    [-1,       -1,       -1]]

To find which pot the agent is facing, we compare the agent's forward
position against all entries in pot_positions and take argmax of the
boolean match vector.


Branchless xp.where pattern
----------------------------

Every branch function computes BOTH the condition (bool scalar) AND the
would-be result arrays unconditionally. No Python if/else gates the
computation -- this is required for JAX tracing where all code paths
must execute. The final ``_apply_interaction_updates`` merges results
using cascading ``xp.where(cond, branch_result, previous)``:

    result = xp.where(b1_cond, b1_result, original)
    result = xp.where(b2_cond, b2_result, result)
    ...

Because conditions are mutually exclusive (each guards against all
earlier conditions), exactly zero or one condition is True, so the
final value is either the matching branch's result or the original.
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


# ======================================================================
# Top-level interaction entry point (called by process_interactions)
# ======================================================================


def overcooked_interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
    """Per-agent Overcooked interaction: state in, state out.

    Called once per agent by ``process_interactions`` in priority order
    (agent 0 first). Mutations from earlier agents are visible to later
    agents because the updated state is threaded through sequentially.

    This function is a thin adapter between the generic ``process_interactions``
    contract and the Overcooked-specific ``overcooked_interaction_body``:

        1. Extract mutable arrays from the immutable EnvState
        2. Call overcooked_interaction_body (pure array logic)
        3. Pack the (possibly mutated) arrays back into a new EnvState

    Parameters
    ----------
    state : EnvState
        Current environment state. Immutable -- a new state is returned.
    agent_idx : int or scalar array
        Index of the agent being processed (0-based).
    fwd_r, fwd_c : scalar int arrays
        Row and column of the cell directly in front of this agent,
        already clipped to grid bounds by process_interactions.
    base_ok : bool scalar array
        True if this agent issued PickupDrop AND no other agent occupies
        the forward cell. When False, all branches no-op.
    scope_config : dict
        Scope configuration containing ``"static_tables"`` -- a dict of
        pre-built lookup arrays (CAN_PICKUP, type IDs, etc.) that the
        branch functions use to resolve interactions without isinstance().

    Returns
    -------
    EnvState
        New state with agent_inv, object_type_map, object_state_map, and
        extra_state potentially updated.
    """
    import dataclasses

    static_tables = scope_config.get("static_tables", {})

    # Look up what's in front of the agent and what they're holding.
    fwd_type = state.object_type_map[fwd_r, fwd_c]
    inv_item = state.agent_inv[agent_idx, 0]

    # Delegate to the pure-array interaction body which evaluates all
    # seven branches and returns the (possibly updated) arrays.
    agent_inv, otm, osm, pot_contents, pot_timer = overcooked_interaction_body(
        agent_idx, state.agent_inv, state.object_type_map, state.object_state_map,
        fwd_r, fwd_c, fwd_type, inv_item, base_ok,
        state.extra_state["overcooked.pot_contents"],
        state.extra_state["overcooked.pot_timer"],
        state.extra_state["overcooked.pot_positions"],
        static_tables,
    )

    # Repack into a new immutable EnvState. pot_positions never changes
    # (pots don't move), so we only update contents and timer.
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


# ======================================================================
# Init-time table builders
# ======================================================================


def _build_interaction_tables(scope: str = "overcooked") -> dict:
    """Build pickup_from_produces and legal_pot_ingredients lookup arrays.

    These are (n_types,) int32 arrays indexed by object type ID:

    - ``pickup_from_produces[type_id]``: what picking up from this type
      produces (e.g. onion_stack -> onion). 0 means "not a pickup-from source".
    - ``legal_pot_ingredients[type_id]``: 1 if this type can go in a pot, else 0.
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

    # Stacks are infinite dispensers: picking up from a stack produces
    # the corresponding loose item, but the stack itself stays on the grid.
    #   onion_stack -> onion,  tomato_stack -> tomato,  plate_stack -> plate
    pickup_from_produces = set_at(pickup_from_produces, onion_stack_id, onion_id)
    pickup_from_produces = set_at(pickup_from_produces, tomato_stack_id, tomato_id)
    pickup_from_produces = set_at(pickup_from_produces, plate_stack_id, plate_id)

    # Only onions and tomatoes can go into a pot.
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


# ======================================================================
# Tick (cooking timer) logic
# ======================================================================


def overcooked_tick(pot_contents, pot_timer, capacity=3, cooking_time=30):
    """Advance the cooking timer for all pots by one step.

    Pot state machine:
        - Empty or partially filled (n_items < capacity): timer unchanged.
        - Full and timer > 0 (cooking): timer decrements by 1.
        - Full and timer == 0 (done): timer stays at 0 until soup is picked up.

    Also computes ``pot_state``, an integer encoding written into
    ``object_state_map`` at each pot's position so that features and
    rendering can inspect pot status without accessing extra_state:

        pot_state = n_items + n_items * timer

    Examples:
        Empty pot:                 0 + 0*30 = 0
        2 items, not cooking:      2 + 2*30 = 62
        3 items, timer=29:         3 + 3*29 = 90
        3 items, done (timer=0):   3 + 3*0  = 3

    Returns (pot_contents, new_timer, pot_state).
    """
    n_items = xp.sum(pot_contents != -1, axis=1).astype(xp.int32)
    is_cooking = (n_items == capacity) & (pot_timer > 0)
    new_timer = xp.where(is_cooking, pot_timer - 1, pot_timer)
    pot_state = (n_items + n_items * new_timer).astype(xp.int32)
    return pot_contents, new_timer, pot_state


def overcooked_tick_state(state, scope_config):
    """Tick handler with generic (state, scope_config) -> state signature.

    Extracts pot arrays from extra_state, runs ``overcooked_tick``, writes
    the updated timer back into extra_state and the pot_state encoding
    into ``object_state_map`` at each pot's grid position.
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


# ======================================================================
# Interaction branch functions
# ======================================================================
#
# Each function evaluates ONE branch of the interaction decision tree.
# All functions are pure: they take arrays, return arrays. No mutations.
#
# Naming convention:
#   b<N>_cond  -- bool scalar, True if this branch fires
#   b<N>_inv   -- would-be agent_inv if this branch fires
#   b<N>_otm   -- would-be object_type_map if this branch fires
#   b<N>_osm   -- would-be object_state_map if this branch fires
#   b<N>_pc    -- would-be pot_contents if this branch fires
#   b<N>_pt    -- would-be pot_timer if this branch fires
#
# Every function computes results unconditionally (no Python if/else)
# so that JAX can trace through all branches. The condition is returned
# alongside the results; the caller uses xp.where to select.
# ======================================================================


def _interact_pickup(base_ok, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                     agent_inv, object_type_map, object_state_map, CAN_PICKUP):
    """Branch 1: Pick up a loose object from the forward cell.

    Preconditions (all must be True):
        - base_ok: agent is interacting and no agent ahead
        - fwd_type > 0: forward cell is not empty
        - CAN_PICKUP[fwd_type] == 1: the object type is pickupable
        - inv_item == -1: agent's hand is empty

    Effects when condition is True:
        - agent_inv[agent_idx] = fwd_type  (object moves to inventory)
        - object_type_map[fwd_r, fwd_c] = 0  (cell becomes empty)
        - object_state_map[fwd_r, fwd_c] = 0  (cell state cleared)

    Example: Agent faces a loose onion on the floor.

        BEFORE                        AFTER
        +-------+-------+            +-------+-------+
        | Agent |  (o)  |            | Agent |       |
        | inv=- | onion |   --->     | inv=o | empty |
        +-------+-------+            +-------+-------+
    """
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
    """Branch 2A: Pick up cooked soup from a pot that has finished cooking.

    Preconditions (all must be True):
        - base_ok AND Branch 1 did NOT fire (~b1_cond)
        - fwd_type == pot_id: forward cell is a pot
        - has_pot_match: the pot was found in pot_positions
        - pot has contents (at least one slot != -1)
        - pot_timer[pot_idx] == 0: cooking is complete
        - inv_item == plate_id: agent is holding a plate

    Effects when condition is True:
        - agent_inv[agent_idx] = soup_type  (plate replaced with soup)
        - pot_contents[pot_idx, :] = -1  (pot emptied)
        - pot_timer[pot_idx] = cooking_time  (timer reset to 30)

    Soup type determination:
        If ALL non-empty slots contain tomato -> tomato_soup
        Otherwise -> onion_soup  (mixed or all-onion)

    Example: Agent holds a plate, pot has 3 cooked onions.

        BEFORE                            AFTER
        +-------+-------+               +-------+-------+
        | Agent |  Pot  |               | Agent |  Pot  |
        | inv=P | ooo,0 |   --->        | inv=S | ---,30|
        +-------+-------+               +-------+-------+
          P=plate  ooo=3 onions            S=onion_soup  ---=empty, timer reset
    """
    is_pot = (fwd_type == pot_id)

    # Check pot state: does it have contents and is cooking complete?
    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = pot_timer[pot_idx] == 0

    # Determine soup type: check if all non-empty slots are tomato.
    # Slots that are -1 (empty) are masked out with the OR condition.
    all_tomato = xp.all(
        (pot_contents[pot_idx] == -1) | (pot_contents[pot_idx] == tomato_id)
    )
    soup_type = xp.where(all_tomato, tomato_soup_id, onion_soup_id)

    b2_pot_cond = (base_ok & ~b1_cond & is_pot & has_pot_match
                   & has_contents & is_ready & (inv_item == plate_id))

    # If condition fires: replace plate in inventory with soup, clear pot.
    b2_pot_inv = set_at(agent_inv, (agent_idx, 0), soup_type)
    b2_pot_pc = set_at(pot_contents, (pot_idx, slice(None)), -1)  # clear all 3 slots
    b2_pot_pt = set_at(pot_timer, pot_idx, cooking_time)          # reset to 30
    return b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt


def _interact_pickup_from_stack(base_ok, b1_cond, fwd_type, inv_item, agent_idx,
                                agent_inv, pot_id, CAN_PICKUP_FROM,
                                pickup_from_produces):
    """Branch 2B: Pick up a produced item from an infinite dispenser stack.

    Stacks are objects with CAN_PICKUP_FROM=1 that produce a different
    item type when picked from (e.g. onion_stack produces onion). The
    stack itself remains on the grid (infinite supply).

    Pots also have CAN_PICKUP_FROM=1 but are handled separately in
    Branch 2A, so we explicitly exclude pots here.

    Preconditions (all must be True):
        - base_ok AND Branch 1 did NOT fire (~b1_cond)
        - fwd_type is NOT a pot
        - CAN_PICKUP_FROM[fwd_type] == 1: forward cell is a pickup-from source
        - inv_item == -1: agent's hand is empty
        - pickup_from_produces[fwd_type] > 0: stack actually produces something

    Effects when condition is True:
        - agent_inv[agent_idx] = produced item type ID
        - Grid is unchanged (stack stays)

    Lookup:
        pickup_from_produces[onion_stack_id]  = onion_id
        pickup_from_produces[tomato_stack_id] = tomato_id
        pickup_from_produces[plate_stack_id]  = plate_id

    Example: Agent faces an onion stack with empty hands.

        BEFORE                        AFTER
        +-------+-------+            +-------+-------+
        | Agent | Stack |            | Agent | Stack |
        | inv=- |  OOO  |   --->     | inv=o |  OOO  |  (stack unchanged)
        +-------+-------+            +-------+-------+
    """
    # Exclude pots (they have CAN_PICKUP_FROM=1 but use Branch 2A)
    is_stack = ~(fwd_type == pot_id) & (CAN_PICKUP_FROM[fwd_type] == 1)
    produced = pickup_from_produces[fwd_type]
    b2_stack_cond = base_ok & ~b1_cond & is_stack & (inv_item == -1) & (produced > 0)
    b2_stack_inv = set_at(agent_inv, (agent_idx, 0), produced)
    return b2_stack_cond, b2_stack_inv


def _interact_drop_on_empty(base_ok, b1_cond, b2_pot_cond, b2_stack_cond,
                            fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                            agent_inv, object_type_map, object_state_map):
    """Branch 3: Drop held item onto an empty floor cell.

    Preconditions (all must be True):
        - base_ok AND Branches 1, 2A, 2B did NOT fire
        - fwd_type == 0: forward cell is empty (no object)
        - inv_item != -1: agent is holding something

    Effects when condition is True:
        - object_type_map[fwd_r, fwd_c] = inv_item  (item appears on grid)
        - object_state_map[fwd_r, fwd_c] = 0  (fresh state)
        - agent_inv[agent_idx] = -1  (hand emptied)

    Example: Agent holds an onion, faces an empty cell.

        BEFORE                        AFTER
        +-------+-------+            +-------+-------+
        | Agent | empty |            | Agent |  (o)  |
        | inv=o |       |   --->     | inv=- | onion |
        +-------+-------+            +-------+-------+
    """
    b3_cond = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond
               & (fwd_type == 0) & (inv_item != -1))
    b3_otm = set_at_2d(object_type_map, fwd_r, fwd_c, inv_item)
    b3_osm = set_at_2d(object_state_map, fwd_r, fwd_c, 0)
    b3_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b3_cond, b3_inv, b3_otm, b3_osm


def _interact_place_on_pot(b4_base, fwd_type, inv_item, agent_idx, agent_inv,
                           pot_contents, pot_idx, has_pot_match, pot_id,
                           legal_pot_ingredients):
    """Branch 4A: Place an ingredient into a pot.

    Preconditions (all must be True, b4_base encodes the shared ones):
        - b4_base: base_ok, no earlier branch fired, forward cell is a
          place-on target, agent holds an item
        - fwd_type == pot_id: forward cell is specifically a pot
        - has_pot_match: pot found in pot_positions array
        - legal_pot_ingredients[inv_item] == 1: held item is onion or tomato
        - n_items_in_pot < 3: pot has capacity (max 3 ingredients)
        - same_type: pot is empty OR first slot matches held item type
          (prevents mixing onions and tomatoes in the same pot)

    Effects when condition is True:
        - pot_contents[pot_idx, first_empty_slot] = inv_item  (ingredient added)
        - agent_inv[agent_idx] = -1  (hand emptied)

    Slot assignment: ``first_empty_slot = argmax(pot_contents[pot_idx] == -1)``
    fills left-to-right (slot 0, then 1, then 2).

    Example: Agent holds an onion, pot already has one onion.

        BEFORE                                  AFTER
        +-------+-----------+                  +-------+-----------+
        | Agent |    Pot    |                  | Agent |    Pot    |
        | inv=o | [o, -, -] |     --->         | inv=- | [o, o, -] |
        +-------+-----------+                  +-------+-----------+
    """
    is_pot = (fwd_type == pot_id)
    is_legal = legal_pot_ingredients[inv_item] == 1
    n_items_in_pot = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items_in_pot < 3

    # Same-type check: pot must be empty OR existing ingredients must
    # match the held item. We check by comparing the first slot only
    # (all filled slots are guaranteed to be the same type).
    first_slot = pot_contents[pot_idx, 0]
    same_type = (n_items_in_pot == 0) | (first_slot == inv_item)

    # Find the first empty slot to place the ingredient into.
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
    """Branch 4B: Place a soup on the delivery zone.

    Only soups (onion_soup or tomato_soup) can be delivered. Attempting
    to deliver any other item (raw ingredients, plates, etc.) is rejected.

    The delivery zone acts as a sink -- the soup disappears from the
    agent's inventory. Scoring is handled by the reward function, not here.

    Preconditions (all must be True):
        - b4_base: shared place-on preconditions
        - fwd_type == delivery_zone_id: forward cell is a delivery zone
        - inv_item is onion_soup or tomato_soup

    Effects when condition is True:
        - agent_inv[agent_idx] = -1  (soup consumed / delivered)

    Example: Agent delivers onion soup.

        BEFORE                        AFTER
        +-------+-------+            +-------+-------+
        | Agent |   @   |            | Agent |   @   |
        | inv=S |  D.Z. |   --->     | inv=- |  D.Z. |
        +-------+-------+            +-------+-------+
    """
    is_dz = (fwd_type == delivery_zone_id)
    is_soup = (inv_item == onion_soup_id) | (inv_item == tomato_soup_id)
    b4_dz_cond = b4_base & is_dz & is_soup
    b4_dz_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b4_dz_cond, b4_dz_inv


def _interact_place_on_counter(b4_base, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
                               agent_inv, object_state_map, pot_id,
                               delivery_zone_id):
    """Branch 4C: Place a held item on a counter (generic place-on target).

    Counters store placed items in ``object_state_map`` rather than
    ``object_type_map`` -- the counter object itself stays in the type
    map, and the placed item's type ID is written into the state map.
    This means a counter can hold at most one item (state != 0 means
    occupied).

    This branch is the catch-all for place-on targets that are NOT
    pots and NOT delivery zones.

    Preconditions (all must be True):
        - b4_base: shared place-on preconditions
        - fwd_type is NOT pot_id and NOT delivery_zone_id
        - object_state_map[fwd_r, fwd_c] == 0: counter is empty

    Effects when condition is True:
        - object_state_map[fwd_r, fwd_c] = inv_item  (item stored on counter)
        - agent_inv[agent_idx] = -1  (hand emptied)

    Note: Picking up an item from a counter is handled by Branch 1 only
    if the item is in object_type_map. Items stored in object_state_map
    (on counters) have a separate pickup path via the rendering sync
    and tick logic. In practice, counters use the state map as temporary
    storage visible to features and rendering.

    Example: Agent places an onion on an empty counter.

        BEFORE                            AFTER
        +-------+---------+              +-------+---------+
        | Agent | Counter |              | Agent | Counter |
        | inv=o | state=0 |   --->       | inv=- | state=o |
        +-------+---------+              +-------+---------+
    """
    # Exclude pots and delivery zones (they have their own branches)
    is_generic = ~(fwd_type == pot_id) & ~(fwd_type == delivery_zone_id)
    counter_empty = (object_state_map[fwd_r, fwd_c] == 0)
    b4_gen_cond = b4_base & is_generic & counter_empty
    b4_gen_osm = set_at_2d(object_state_map, fwd_r, fwd_c, inv_item)
    b4_gen_inv = set_at(agent_inv, (agent_idx, 0), -1)
    return b4_gen_cond, b4_gen_inv, b4_gen_osm


# ======================================================================
# Branch merge
# ======================================================================


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
    """Merge all branch results into final arrays using cascading xp.where.

    Each array is updated by every branch that could modify it. Because
    branch conditions are mutually exclusive (enforced by the ~earlier_cond
    guards), at most one condition is True for each array.

    The cascade order matches the priority order:

        Branch 1   (pickup)            -- highest priority
        Branch 2A  (pickup from pot)
        Branch 2B  (pickup from stack)
        Branch 3   (drop on empty)
        Branch 4A  (place on pot)
        Branch 4B  (place on delivery)
        Branch 4C  (place on counter)  -- lowest priority

    Which arrays each branch modifies:

        Branch  | agent_inv | obj_type_map | obj_state_map | pot_contents | pot_timer
        --------|-----------|--------------|---------------|--------------|----------
        1       |     X     |      X       |       X       |              |
        2A      |     X     |              |               |      X       |     X
        2B      |     X     |              |               |              |
        3       |     X     |      X       |       X       |              |
        4A      |     X     |              |               |      X       |
        4B      |     X     |              |               |              |
        4C      |     X     |              |       X       |              |

    Returns (agent_inv, object_type_map, object_state_map, pot_contents, pot_timer).
    """
    # --- agent_inv: modified by every branch ---
    new_inv = xp.where(b1_cond, b1_inv, agent_inv)
    new_inv = xp.where(b2_pot_cond, b2_pot_inv, new_inv)
    new_inv = xp.where(b2_stack_cond, b2_stack_inv, new_inv)
    new_inv = xp.where(b3_cond, b3_inv, new_inv)
    new_inv = xp.where(b4_pot_cond, b4_pot_inv, new_inv)
    new_inv = xp.where(b4_dz_cond, b4_dz_inv, new_inv)
    new_inv = xp.where(b4_gen_cond, b4_gen_inv, new_inv)

    # --- object_type_map: only Branch 1 (pickup removes object) and
    #     Branch 3 (drop places object) ---
    new_otm = xp.where(b1_cond, b1_otm, object_type_map)
    new_otm = xp.where(b3_cond, b3_otm, new_otm)

    # --- object_state_map: Branch 1 (clear cell), Branch 3 (clear cell),
    #     Branch 4C (store item on counter) ---
    new_osm = xp.where(b1_cond, b1_osm, object_state_map)
    new_osm = xp.where(b3_cond, b3_osm, new_osm)
    new_osm = xp.where(b4_gen_cond, b4_gen_osm, new_osm)

    # --- pot_contents: Branch 2A (clear pot) and Branch 4A (add ingredient) ---
    new_pc = xp.where(b2_pot_cond, b2_pot_pc, pot_contents)
    new_pc = xp.where(b4_pot_cond, b4_pot_pc, new_pc)

    # --- pot_timer: Branch 2A only (reset timer after soup pickup) ---
    new_pt = xp.where(b2_pot_cond, b2_pot_pt, pot_timer)

    return new_inv, new_otm, new_osm, new_pc, new_pt


# ======================================================================
# Interaction body (orchestrator)
# ======================================================================


def overcooked_interaction_body(
    agent_idx,            # int: which agent is interacting (0-based)
    agent_inv,            # (n_agents, 1) int32: all agent inventories
    object_type_map,      # (H, W) int32: object type IDs on the grid
    object_state_map,     # (H, W) int32: object state values on the grid
    fwd_r, fwd_c,         # scalar int arrays: forward cell coordinates
    fwd_type,             # scalar int array: object type at forward cell
    inv_item,             # scalar int array: what this agent is holding (-1 = empty)
    base_ok,              # bool scalar: True if agent can interact (PickupDrop + no agent ahead)
    pot_contents,         # (n_pots, 3) int32: ingredient type IDs per pot slot
    pot_timer,            # (n_pots,) int32: cooking countdown per pot
    pot_positions,        # (n_pots, 2) int32: (row, col) of each pot
    static_tables,        # dict: pre-built lookup arrays (see _build_static_tables)
):
    """Evaluate all seven interaction branches for one agent and merge results.

    This is the core dispatch function. It:
        1. Unpacks static lookup tables (type IDs, property arrays)
        2. Resolves which pot (if any) the agent is facing
        3. Calls each branch function to compute conditions and would-be results
        4. Calls _apply_interaction_updates to merge via cascading xp.where

    The branch evaluation order and mutual exclusion are critical:

        Branch 1  -> b1_cond
        Branch 2A -> requires ~b1_cond
        Branch 2B -> requires ~b1_cond
        Branch 3  -> requires ~b1_cond & ~b2_pot_cond & ~b2_stack_cond
        Branch 4* -> requires ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond

    This ensures that if an agent could both pick up and place on (ambiguous),
    pickup always wins.

    Parameters
    ----------
    static_tables : dict
        Pre-built at init time by ``_build_static_tables``. Contains:

        Property arrays (indexed by type ID):
            CAN_PICKUP[type_id]          -> 1 if pickupable
            CAN_PICKUP_FROM[type_id]     -> 1 if can pick from (stacks + pot)
            CAN_PLACE_ON[type_id]        -> 1 if can place items on
            pickup_from_produces[type_id] -> produced type ID (0 if N/A)
            legal_pot_ingredients[type_id] -> 1 if can go in pot

        Scalar type IDs:
            pot_id, plate_id, tomato_id, onion_soup_id,
            tomato_soup_id, delivery_zone_id

        Constants:
            cooking_time = 30  (ticks to cook a full pot)

    Returns
    -------
    (agent_inv, object_type_map, object_state_map, pot_contents, pot_timer)
        Updated arrays. Unchanged if base_ok is False or no branch fires.
    """
    # --- Unpack static tables ---
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

    # --- Pot matching ---
    # If the forward cell is a pot, find which pot index it corresponds to
    # in the pot_positions array. pot_idx is the argmax of the boolean
    # match vector; has_pot_match is False if no pot is at this position
    # (in which case pot_idx is 0 but guarded by has_pot_match in conditions).
    fwd_pos_2d = xp.stack([fwd_r, fwd_c])
    pot_match = xp.all(pot_positions == fwd_pos_2d[None, :], axis=1)
    pot_idx = xp.argmax(pot_match)
    has_pot_match = xp.any(pot_match)

    # --- Branch 1: pickup loose object ---
    b1_cond, b1_inv, b1_otm, b1_osm = _interact_pickup(
        base_ok, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_type_map, object_state_map, CAN_PICKUP)

    # --- Branch 2A: pickup cooked soup from pot ---
    b2_pot_cond, b2_pot_inv, b2_pot_pc, b2_pot_pt = _interact_pickup_from_pot(
        base_ok, b1_cond, fwd_type, inv_item, agent_idx,
        agent_inv, pot_contents, pot_timer,
        pot_idx, has_pot_match, pot_id, plate_id,
        tomato_id, onion_soup_id, tomato_soup_id, cooking_time)

    # --- Branch 2B: pickup from stack (onion/tomato/plate dispenser) ---
    b2_stack_cond, b2_stack_inv = _interact_pickup_from_stack(
        base_ok, b1_cond, fwd_type, inv_item, agent_idx,
        agent_inv, pot_id, CAN_PICKUP_FROM, pickup_from_produces)

    # --- Branch 3: drop on empty cell ---
    b3_cond, b3_inv, b3_otm, b3_osm = _interact_drop_on_empty(
        base_ok, b1_cond, b2_pot_cond, b2_stack_cond,
        fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_type_map, object_state_map)

    # --- Shared base condition for all place-on branches (4A, 4B, 4C) ---
    # All three require: base_ok, no earlier branch fired, forward cell
    # has a place-on target, and agent is holding something.
    b4_base = (base_ok & ~b1_cond & ~b2_pot_cond & ~b2_stack_cond & ~b3_cond
               & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1) & (inv_item != -1))

    # --- Branch 4A: place ingredient on pot ---
    b4_pot_cond, b4_pot_inv, b4_pot_pc = _interact_place_on_pot(
        b4_base, fwd_type, inv_item, agent_idx, agent_inv,
        pot_contents, pot_idx, has_pot_match, pot_id, legal_pot_ingredients)

    # --- Branch 4B: deliver soup to delivery zone ---
    b4_dz_cond, b4_dz_inv = _interact_place_on_delivery(
        b4_base, fwd_type, inv_item, agent_idx, agent_inv,
        delivery_zone_id, onion_soup_id, tomato_soup_id)

    # --- Branch 4C: place item on counter ---
    b4_gen_cond, b4_gen_inv, b4_gen_osm = _interact_place_on_counter(
        b4_base, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
        agent_inv, object_state_map, pot_id, delivery_zone_id)

    # --- Merge all branch results using cascading xp.where ---
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
    """Build the static tables dict used by overcooked_interaction_body.

    Called once at environment init time. The resulting dict is stored in
    ``scope_config["static_tables"]`` and closed over by the step pipeline.

    Contains three categories of data:

    1. Property arrays -- (n_types,) int32 arrays indexed by type ID.
       Built by ``build_lookup_tables`` from @register_object_type metadata.
         - CAN_PICKUP: 1 for onion, tomato, plate, onion_soup, tomato_soup
         - CAN_PICKUP_FROM: 1 for onion_stack, tomato_stack, plate_stack, pot
         - CAN_PLACE_ON: 1 for pot, delivery_zone, counter

    2. Interaction tables -- (n_types,) int32 arrays for Overcooked-specific
       rules. Built by ``_build_interaction_tables``.
         - pickup_from_produces: maps stack type -> produced type
         - legal_pot_ingredients: 1 for onion and tomato

    3. Scalar type IDs -- int constants for direct comparison.
         - pot_id, plate_id, tomato_id, onion_soup_id, tomato_soup_id,
           delivery_zone_id
         - cooking_time: 30 (number of ticks to cook a full pot)
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
