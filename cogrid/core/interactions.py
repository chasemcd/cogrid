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
- ``process_interactions_jax()`` -- JIT-compatible interaction processing
  using ``lax.fori_loop``, ``jnp.where``, and ``lax.switch`` for toggle.
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


# ======================================================================
# JAX-path interaction processing (JIT-compatible)
# ======================================================================


def process_interactions_jax(
    agent_pos,          # (n_agents, 2) int32
    agent_dir,          # (n_agents,) int32
    agent_inv,          # (n_agents, 1) int32
    actions,            # (n_agents,) int32
    object_type_map,    # (H, W) int32
    object_state_map,   # (H, W) int32
    lookup_tables,      # dict -- static, closed over
    scope_config,       # dict -- static, closed over
    dir_vec_table,      # (4, 2) int32
    action_pickup_drop_idx,  # int -- static
    action_toggle_idx,       # int -- static
    pot_contents=None,  # (n_pots, 3) int32 or None
    pot_timer=None,     # (n_pots,) int32 or None
    pot_positions=None, # (n_pots, 2) int32 or None
):
    """JIT-compatible interaction processing using JAX primitives.

    Functionally equivalent to :func:`process_interactions_array` but
    replaces Python ``for`` loops with ``lax.fori_loop``, Python
    ``if/else`` with ``jnp.where`` masking, and Python dict lookups
    with array-based matching. Toggle actions dispatch via
    ``jax.lax.switch`` on object type ID.

    This function is designed to be called from within a JIT context
    (e.g., a jitted ``step()`` function). The ``lookup_tables``,
    ``scope_config``, ``action_pickup_drop_idx``, and
    ``action_toggle_idx`` parameters are used at Python/trace time to
    build closures and are not traced. All internal control flow uses
    ``lax.fori_loop`` and ``lax.switch`` for JIT compatibility.

    To JIT this function standalone, use :func:`functools.partial` to
    bind the non-array arguments, then JIT the resulting closure::

        import functools
        fn = functools.partial(
            process_interactions_jax,
            lookup_tables=tables, scope_config=config,
            action_pickup_drop_idx=4, action_toggle_idx=5,
        )
        jitted_fn = jax.jit(fn)

    Args:
        agent_pos: Agent positions, shape ``(n_agents, 2)``, int32.
        agent_dir: Agent directions, shape ``(n_agents,)``, int32.
        agent_inv: Agent inventories, shape ``(n_agents, 1)``, int32.
        actions: Action indices, shape ``(n_agents,)``, int32.
        object_type_map: Grid object type IDs, shape ``(H, W)``, int32.
        object_state_map: Grid object states, shape ``(H, W)``, int32.
        lookup_tables: Dict of property arrays from ``build_lookup_tables()``.
        scope_config: Scope config dict from ``get_scope_config()``.
        dir_vec_table: Direction vector lookup, shape ``(4, 2)``, int32.
        action_pickup_drop_idx: Integer index of the PickupDrop action.
        action_toggle_idx: Integer index of the Toggle action.
        pot_contents: Pot ingredient arrays, shape ``(n_pots, 3)``, int32.
            Pass ``None`` for non-Overcooked scopes.
        pot_timer: Pot cooking timers, shape ``(n_pots,)``, int32.
            Pass ``None`` for non-Overcooked scopes.
        pot_positions: Pot positions, shape ``(n_pots, 2)``, int32.
            Pass ``None`` for non-Overcooked scopes.

    Returns:
        Tuple of ``(agent_inv, object_type_map, object_state_map,
        pot_contents, pot_timer)``.
    """
    import functools
    import jax
    import jax.numpy as jnp
    import jax.lax as lax

    n_agents = agent_pos.shape[0]
    H, W = object_type_map.shape

    # Get JAX interaction body from scope config (Overcooked has one)
    interaction_body_jax = (
        scope_config.get("interaction_body_jax") if scope_config else None
    )

    # Provide dummy pot arrays for non-Overcooked scopes
    if pot_contents is None:
        pot_contents = jnp.full((1, 3), -1, dtype=jnp.int32)
    if pot_timer is None:
        pot_timer = jnp.zeros(1, dtype=jnp.int32)
    if pot_positions is None:
        pot_positions = jnp.full((1, 2), -1, dtype=jnp.int32)

    # ---- Pickup/Drop interactions via lax.fori_loop ----
    if interaction_body_jax is not None:
        # Overcooked scope: use the scope-specific interaction body
        static_tables = scope_config.get("static_tables", {})
        body = functools.partial(
            interaction_body_jax,
            actions=actions, agent_pos=agent_pos, agent_dir=agent_dir,
            dir_vec_table=dir_vec_table,
            action_pickup_drop_idx=action_pickup_drop_idx,
            static_tables=static_tables,
        )
    else:
        # Generic scope: handle branches 1 (pickup), 3 (drop), 4 (place_on)
        CAN_PICKUP = lookup_tables["CAN_PICKUP"]
        CAN_PLACE_ON = lookup_tables["CAN_PLACE_ON"]

        def body(i, carry):
            (agent_inv_c, otm_c, osm_c, pc_c, pt_c, pp_c) = carry

            is_interact = (actions[i] == action_pickup_drop_idx)

            fwd_pos = agent_pos[i] + dir_vec_table[agent_dir[i]]
            fwd_r = jnp.clip(fwd_pos[0], 0, H - 1)
            fwd_c = jnp.clip(fwd_pos[1], 0, W - 1)

            other_mask = jnp.arange(n_agents) != i
            pos_match = jnp.all(
                agent_pos == jnp.array([fwd_r, fwd_c])[None, :], axis=1
            )
            agent_ahead = jnp.any(pos_match & other_mask)

            fwd_type = otm_c[fwd_r, fwd_c]
            inv_item = agent_inv_c[i, 0]
            base_ok = is_interact & ~agent_ahead

            # Branch 1: can_pickup
            b1_cond = (base_ok & (fwd_type > 0)
                       & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1))
            b1_inv = agent_inv_c.at[i, 0].set(fwd_type)
            b1_otm = otm_c.at[fwd_r, fwd_c].set(0)
            b1_osm = osm_c.at[fwd_r, fwd_c].set(0)

            # Branch 3: drop on empty
            b3_cond = (base_ok & ~b1_cond
                       & (fwd_type == 0) & (inv_item != -1))
            b3_otm = otm_c.at[fwd_r, fwd_c].set(inv_item)
            b3_osm = osm_c.at[fwd_r, fwd_c].set(0)
            b3_inv = agent_inv_c.at[i, 0].set(-1)

            # Branch 4: generic place_on
            b4_cond = (base_ok & ~b1_cond & ~b3_cond
                       & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1)
                       & (inv_item != -1) & (osm_c[fwd_r, fwd_c] == 0))
            b4_osm = osm_c.at[fwd_r, fwd_c].set(inv_item)
            b4_inv = agent_inv_c.at[i, 0].set(-1)

            # Apply updates
            new_inv = jnp.where(b1_cond, b1_inv, agent_inv_c)
            new_inv = jnp.where(b3_cond, b3_inv, new_inv)
            new_inv = jnp.where(b4_cond, b4_inv, new_inv)

            new_otm = jnp.where(b1_cond, b1_otm, otm_c)
            new_otm = jnp.where(b3_cond, b3_otm, new_otm)

            new_osm = jnp.where(b1_cond, b1_osm, osm_c)
            new_osm = jnp.where(b3_cond, b3_osm, new_osm)
            new_osm = jnp.where(b4_cond, b4_osm, new_osm)

            return (new_inv, new_otm, new_osm, pc_c, pt_c, pp_c)

    init_carry = (agent_inv, object_type_map, object_state_map,
                  pot_contents, pot_timer, pot_positions)
    (agent_inv, object_type_map, object_state_map,
     pot_contents, pot_timer, _) = lax.fori_loop(
        0, n_agents, body, init_carry
    )

    # ---- Toggle actions via lax.switch on object type ID ----
    # Build toggle branch table: indexed by type ID, default to no-op.
    # Scope config may provide "toggle_branches_jax" with
    # (type_id, handler_fn) pairs.
    n_types = lookup_tables["CAN_PICKUP"].shape[0]

    def toggle_noop(carry, agent_idx):
        """No-op toggle branch."""
        return carry

    def toggle_door(carry, agent_idx):
        """Toggle door: flip object_state_map between 0 and 1."""
        inv_c, otm_c, osm_c, pc_c, pt_c = carry
        fwd_pos = agent_pos[agent_idx] + dir_vec_table[agent_dir[agent_idx]]
        fwd_r = jnp.clip(fwd_pos[0], 0, H - 1)
        fwd_c = jnp.clip(fwd_pos[1], 0, W - 1)
        current = osm_c[fwd_r, fwd_c]
        new_state = 1 - current  # toggle between 0 and 1
        osm_c = osm_c.at[fwd_r, fwd_c].set(new_state)
        return (inv_c, otm_c, osm_c, pc_c, pt_c)

    # Default all branches to no-op
    toggle_branches = [toggle_noop] * n_types

    # Insert door toggle if Door type exists
    from cogrid.core.grid_object import get_object_names, object_to_idx
    obj_names = get_object_names(scope="overcooked" if scope_config else "global")
    if "door" in obj_names:
        door_id = object_to_idx("door", scope="overcooked" if scope_config else "global")
        if door_id < n_types:
            toggle_branches[door_id] = toggle_door

    # Insert any scope-provided toggle handlers
    if scope_config and "toggle_branches_jax" in scope_config:
        for type_id, handler_fn in scope_config["toggle_branches_jax"]:
            if type_id < n_types:
                toggle_branches[type_id] = handler_fn

    # Process toggles in a second fori_loop pass
    def toggle_body(i, carry):
        inv_c, otm_c, osm_c, pc_c, pt_c = carry
        is_toggle = (actions[i] == action_toggle_idx)

        fwd_pos = agent_pos[i] + dir_vec_table[agent_dir[i]]
        fwd_r = jnp.clip(fwd_pos[0], 0, H - 1)
        fwd_c = jnp.clip(fwd_pos[1], 0, W - 1)
        fwd_type = otm_c[fwd_r, fwd_c]

        # Clamp index to valid range for lax.switch
        branch_idx = jnp.clip(fwd_type, 0, n_types - 1)

        # Dispatch via lax.switch -- all branches traced, selected at runtime
        toggle_carry = lax.switch(
            branch_idx, toggle_branches,
            (inv_c, otm_c, osm_c, pc_c, pt_c), i,
        )

        # Apply only if this agent is actually toggling
        result = jax.tree.map(
            lambda old, new: jnp.where(is_toggle, new, old),
            (inv_c, otm_c, osm_c, pc_c, pt_c),
            toggle_carry,
        )
        return result

    toggle_init = (agent_inv, object_type_map, object_state_map,
                   pot_contents, pot_timer)
    (agent_inv, object_type_map, object_state_map,
     pot_contents, pot_timer) = lax.fori_loop(
        0, n_agents, toggle_body, toggle_init
    )

    return agent_inv, object_type_map, object_state_map, pot_contents, pot_timer
