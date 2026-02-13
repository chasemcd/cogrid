"""Vectorized interaction processing using integer lookup tables.

Replaces isinstance()-based dispatch in ``CoGridEnv.interact()`` with
integer type code lookups on array state. All operations work on
parallel arrays and avoid Python object manipulation.

Core modules contain only generic infrastructure. Environment-specific
interaction logic (e.g. Overcooked pot/delivery zone handling) is
delegated to scope config handlers composed by the auto-wiring layer
in ``cogrid.core.autowire``.

Key function:

- ``process_interactions()`` -- handles pickup, pickup_from, drop,
  place_on interactions for all agents using xp array operations and
  scope config ``interaction_body`` delegation. Processes agents
  sequentially (2 unrolled calls for Overcooked) with vectorized
  condition computation per agent.
"""

from __future__ import annotations

from cogrid.backend.array_ops import set_at, set_at_2d


def process_interactions(
    agent_pos,                # (n_agents, 2) int32
    agent_dir,                # (n_agents,) int32
    agent_inv,                # (n_agents, 1) int32, -1 = empty
    actions,                  # (n_agents,) int32
    object_type_map,          # (H, W) int32
    object_state_map,         # (H, W) int32
    lookup_tables,            # dict with CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON, CAN_PICKUP_FROM
    scope_config,             # scope config dict
    dir_vec_table,            # (4, 2) int32
    action_pickup_drop_idx,   # int -- index of PickupDrop action
    action_toggle_idx,        # int -- index of Toggle action
    extra_state=None,         # dict of scope-specific state arrays
    **extra_state_kwargs,     # backward compat with **kwargs callers
):
    """Process interactions for all agents using xp array operations.

    Implements the same priority order as ``CoGridEnv.interact()``:

    1. can_pickup -- pick up a pickupable object (empty inventory required)
    2. can_pickup_from -- delegated to scope config interaction_body
    3. drop on empty -- drop held item onto empty grid cell
    4. place_on -- generic structure with scope-specific delegation

    For Overcooked (2 agents), interactions are unrolled as 2 sequential
    calls to the scope config's ``interaction_body``. Agent 0 has priority
    (processes first), and agent 1 sees the updated state. All per-agent
    condition computation uses xp.where cascading with zero int() casts.

    Args:
        agent_pos: Agent positions, shape ``(n_agents, 2)``.
        agent_dir: Agent directions, shape ``(n_agents,)``.
        agent_inv: Agent inventories, shape ``(n_agents, 1)``. -1 = empty.
        actions: Action indices, shape ``(n_agents,)``.
        object_type_map: Grid object type IDs, shape ``(H, W)``.
        object_state_map: Grid object states, shape ``(H, W)``.
        lookup_tables: Dict of property arrays from ``build_lookup_tables()``.
        scope_config: Scope config dict from ``build_scope_config_from_components()``.
        dir_vec_table: Direction vector lookup, shape ``(4, 2)``.
        action_pickup_drop_idx: Integer index of the PickupDrop action.
        action_toggle_idx: Integer index of the Toggle action.
        extra_state: Dict of scope-specific state arrays passed through
            to the interaction_body. The interaction_body reads/writes
            what it needs; core code never inspects the contents.
        **extra_state_kwargs: Backward-compatible kwargs form. If
            ``extra_state`` is None, kwargs are collected into a dict.

    Returns:
        Tuple of ``(agent_inv, object_type_map, object_state_map, extra_state)``
        where extra_state is the dict of scope-specific arrays (potentially
        mutated by the interaction_body).
    """
    if extra_state is None:
        extra_state = extra_state_kwargs if extra_state_kwargs else {}

    from cogrid.backend import xp

    n_agents = agent_pos.shape[0]
    H, W = object_type_map.shape

    CAN_PICKUP = lookup_tables["CAN_PICKUP"]
    CAN_PLACE_ON = lookup_tables["CAN_PLACE_ON"]

    # Get scope-specific interaction body (may be None for generic scopes)
    interaction_body = scope_config.get("interaction_body") if scope_config else None

    # Compute forward positions for ALL agents
    fwd_pos = agent_pos + dir_vec_table[agent_dir]  # (n_agents, 2)
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)

    # Which agents are interacting
    is_interact = (actions == action_pickup_drop_idx)

    # Agent-ahead check for each agent (vectorized pairwise)
    # fwd_matches_pos[i,j] = True iff agent i's forward pos == agent j's position
    fwd_rc = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2)
    fwd_matches_pos = xp.all(fwd_rc[:, None, :] == agent_pos[None, :, :], axis=2)
    not_self = ~xp.eye(n_agents, dtype=xp.bool_)
    agent_ahead = xp.any(fwd_matches_pos & not_self, axis=1)

    base_ok = is_interact & ~agent_ahead  # (n_agents,) bool

    if interaction_body is not None:
        # Scope with interaction_body: process agents sequentially.
        # Lower-index agents have priority (process first, mutations visible
        # to later agents). n_agents is a static shape dim, so this Python
        # range loop unrolls at trace time -- no lax.fori_loop needed.
        static_tables = scope_config.get("static_tables", {})

        for i in range(n_agents):
            fwd_type_i = object_type_map[fwd_r[i], fwd_c[i]]
            inv_item_i = agent_inv[i, 0]

            result = interaction_body(
                i, agent_inv, object_type_map, object_state_map,
                fwd_r[i], fwd_c[i], fwd_type_i, inv_item_i, base_ok[i],
                extra_state, static_tables,
            )
            agent_inv, object_type_map, object_state_map, extra_state = result
    else:
        # Generic scope: handle branches 1, 3, 4 without scope-specific handler.
        # Unroll 2 agent calls with generic xp.where logic.
        for agent_idx in range(n_agents):
            fwd_type = object_type_map[fwd_r[agent_idx], fwd_c[agent_idx]]
            inv_item = agent_inv[agent_idx, 0]
            ok = base_ok[agent_idx]

            # Branch 1: can_pickup
            b1_cond = ok & (fwd_type > 0) & (CAN_PICKUP[fwd_type] == 1) & (inv_item == -1)

            b1_inv = set_at(agent_inv, (agent_idx, 0), fwd_type)
            b1_otm = set_at_2d(object_type_map, fwd_r[agent_idx], fwd_c[agent_idx], 0)
            b1_osm = set_at_2d(object_state_map, fwd_r[agent_idx], fwd_c[agent_idx], 0)

            # Branch 3: drop on empty
            b3_cond = ok & ~b1_cond & (fwd_type == 0) & (inv_item != -1)

            b3_otm = set_at_2d(object_type_map, fwd_r[agent_idx], fwd_c[agent_idx], inv_item)
            b3_osm = set_at_2d(object_state_map, fwd_r[agent_idx], fwd_c[agent_idx], 0)
            b3_inv = set_at(agent_inv, (agent_idx, 0), -1)

            # Branch 4: generic place_on
            b4_cond = (ok & ~b1_cond & ~b3_cond
                       & (fwd_type > 0) & (CAN_PLACE_ON[fwd_type] == 1)
                       & (inv_item != -1) & (object_state_map[fwd_r[agent_idx], fwd_c[agent_idx]] == 0))

            b4_osm = set_at_2d(object_state_map, fwd_r[agent_idx], fwd_c[agent_idx], inv_item)
            b4_inv = set_at(agent_inv, (agent_idx, 0), -1)

            # Apply updates with cascading xp.where
            agent_inv = xp.where(b1_cond, b1_inv, agent_inv)
            agent_inv = xp.where(b3_cond, b3_inv, agent_inv)
            agent_inv = xp.where(b4_cond, b4_inv, agent_inv)

            object_type_map = xp.where(b1_cond, b1_otm, object_type_map)
            object_type_map = xp.where(b3_cond, b3_otm, object_type_map)

            object_state_map = xp.where(b1_cond, b1_osm, object_state_map)
            object_state_map = xp.where(b3_cond, b3_osm, object_state_map)
            object_state_map = xp.where(b4_cond, b4_osm, object_state_map)

    return agent_inv, object_type_map, object_state_map, extra_state
