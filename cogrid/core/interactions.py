"""Vectorized interaction processing using integer lookup tables.

Replaces isinstance()-based dispatch in ``CoGridEnv.interact()`` with
integer type code lookups on array state. All operations work on
parallel arrays and avoid Python object manipulation.

Core modules contain only generic infrastructure. Environment-specific
interaction logic (e.g. Overcooked pot/delivery zone handling) is
delegated to scope config handlers registered via
``cogrid.core.scope_config``.

Key functions:

- ``process_interactions_array()`` -- handles pickup, pickup_from, drop,
  place_on interactions for all agents using lookup tables. Delegates
  environment-specific sub-cases to scope config interaction_handler.
"""

from __future__ import annotations


def process_interactions_array(
    agent_pos,                # (n_agents, 2) int32
    agent_dir,                # (n_agents,) int32
    agent_inv,                # (n_agents, 1) int32, -1 = empty
    actions,                  # (n_agents,) int32
    object_type_map,          # (H, W) int32
    object_state_map,         # (H, W) int32
    lookup_tables,            # dict with CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON, CAN_PICKUP_FROM
    scope_config,             # scope config dict from get_scope_config()
    dir_vec_table,            # (4, 2) int32 -- direction vectors
    action_pickup_drop_idx,   # int -- index of PickupDrop action
    action_toggle_idx,        # int -- index of Toggle action
    **extra_state,            # scope-specific state arrays (e.g. pot_contents, pot_timer, pot_positions)
):
    """Process interactions for all agents on array state.

    Implements the same priority order as ``CoGridEnv.interact()``:

    1. can_pickup -- pick up a pickupable object (empty inventory required)
    2. can_pickup_from -- delegated to scope config interaction_handler
    3. drop on empty -- drop held item onto empty grid cell
    4. place_on -- generic structure with scope-specific delegation

    Phase 1: loops over agents since interactions can affect shared grid state.

    Args:
        agent_pos: Agent positions, shape ``(n_agents, 2)``.
        agent_dir: Agent directions, shape ``(n_agents,)``.
        agent_inv: Agent inventories, shape ``(n_agents, 1)``. -1 = empty.
        actions: Action indices, shape ``(n_agents,)``.
        object_type_map: Grid object type IDs, shape ``(H, W)``.
        object_state_map: Grid object states, shape ``(H, W)``.
        lookup_tables: Dict of property arrays from ``build_lookup_tables()``.
        scope_config: Scope config dict from ``get_scope_config()``.
        dir_vec_table: Direction vector lookup, shape ``(4, 2)``.
        action_pickup_drop_idx: Integer index of the PickupDrop action.
        action_toggle_idx: Integer index of the Toggle action.
        **extra_state: Scope-specific state arrays passed through to
            the interaction_handler (e.g. pot_contents, pot_timer,
            pot_positions for Overcooked).

    Returns:
        Tuple of ``(agent_inv, object_type_map, object_state_map, extra_state)``
        where extra_state is the dict of scope-specific arrays (potentially
        mutated by the interaction_handler).
    """
    # Make copies so we don't mutate the caller's arrays
    agent_inv = agent_inv.copy()  # PHASE2: convert to .at[].set()
    object_type_map = object_type_map.copy()  # PHASE2: convert to .at[].set()
    object_state_map = object_state_map.copy()  # PHASE2: convert to .at[].set()

    # Copy scope-specific arrays in extra_state
    extra_state = {
        k: v.copy() if hasattr(v, 'copy') else v
        for k, v in extra_state.items()
    }

    CAN_PICKUP = lookup_tables["CAN_PICKUP"]
    CAN_PICKUP_FROM = lookup_tables["CAN_PICKUP_FROM"]
    CAN_PLACE_ON = lookup_tables["CAN_PLACE_ON"]

    # Get scope-specific interaction handler (may be None for generic scopes)
    interaction_handler = scope_config.get("interaction_handler") if scope_config else None

    n_agents = agent_pos.shape[0]
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
        agent_ahead = False
        for j in range(n_agents):
            if j != i and int(agent_pos[j, 0]) == fwd_r and int(agent_pos[j, 1]) == fwd_c:
                agent_ahead = True
                break

        if agent_ahead:
            continue

        fwd_type = int(object_type_map[fwd_r, fwd_c])
        inv_item = int(agent_inv[i, 0])
        handled = False

        # ---- Priority branch 1: can_pickup (GENERIC) ----
        # Condition: object in front is pickupable AND agent has empty inventory
        if fwd_type > 0 and CAN_PICKUP[fwd_type] == 1 and inv_item == -1:
            agent_inv[i, 0] = fwd_type  # PHASE2: convert to .at[].set()
            object_type_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()
            object_state_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()
            handled = True

        # ---- Priority branch 2: can_pickup_from (DELEGATED if handler exists) ----
        # The handler returns True if it handled the interaction, False otherwise.
        # If the handler returns False (e.g. pot not ready), we fall through to
        # branches 3-4, matching the original dynamic-condition behavior.
        if not handled and fwd_type > 0 and CAN_PICKUP_FROM[fwd_type] == 1:
            if interaction_handler is not None:
                handled = interaction_handler(
                    "pickup_from", i, agent_inv, fwd_r, fwd_c, fwd_type,
                    inv_item, object_type_map, object_state_map, extra_state,
                )
            # No handler: skip (nothing generic to do for pickup_from)

        # ---- Priority branch 3: drop on empty (GENERIC) ----
        # Condition: empty cell ahead AND agent has item in inventory
        if not handled and fwd_type == 0 and inv_item != -1:
            object_type_map[fwd_r, fwd_c] = inv_item  # PHASE2: convert to .at[].set()
            object_state_map[fwd_r, fwd_c] = 0  # PHASE2: convert to .at[].set()
            agent_inv[i, 0] = -1  # PHASE2: convert to .at[].set()
            handled = True

        # ---- Priority branch 4: place_on (GENERIC structure, DELEGATED specifics) ----
        # Condition: object ahead supports place_on AND agent has item
        if not handled and fwd_type > 0 and CAN_PLACE_ON[fwd_type] == 1 and inv_item != -1:
            if interaction_handler is not None:
                handled = interaction_handler(
                    "place_on", i, agent_inv, fwd_r, fwd_c, fwd_type,
                    inv_item, object_type_map, object_state_map, extra_state,
                )
            if not handled:
                # Generic place_on: store in object_state_map if empty
                if int(object_state_map[fwd_r, fwd_c]) == 0:
                    object_state_map[fwd_r, fwd_c] = inv_item
                    agent_inv[i, 0] = -1

    return agent_inv, object_type_map, object_state_map, extra_state
