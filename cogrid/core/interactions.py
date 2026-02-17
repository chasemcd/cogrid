"""Vectorized interaction processing using integer lookup tables.

Core modules contain only generic infrastructure. Environment-specific
interaction logic is delegated via an explicit ``interaction_fn`` parameter.

Key function:

- ``process_interactions()`` -- handles pickup, drop, place_on interactions
  for all agents using xp array operations. Processes agents sequentially
  (lower index = higher priority) with vectorized condition computation.
"""

import dataclasses

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d


def process_interactions(
    state,  # EnvState
    actions,  # (n_agents,) int32
    interaction_fn,  # callable or None
    lookup_tables,  # dict with CAN_PICKUP, CAN_OVERLAP, CAN_PLACE_ON, CAN_PICKUP_FROM
    scope_config,  # scope config dict (passed through to interaction_fn)
    dir_vec_table,  # (4, 2) int32
    action_pickup_drop_idx,  # int -- index of PickupDrop action
    action_toggle_idx,  # int -- index of Toggle action
):
    """Process pickup/drop/place_on interactions for all agents.

    Priority order: (1) pickup, (2) pickup_from (scope-specific),
    (3) drop on empty, (4) place_on. Agents are processed sequentially
    (lower index = higher priority).

    Returns updated ``state``.
    """
    n_agents = state.agent_pos.shape[0]
    H, W = state.object_type_map.shape

    CAN_PICKUP = lookup_tables["CAN_PICKUP"]
    CAN_PLACE_ON = lookup_tables["CAN_PLACE_ON"]

    # Compute forward positions for ALL agents
    fwd_pos = state.agent_pos + dir_vec_table[state.agent_dir]  # (n_agents, 2)
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)

    # Which agents are interacting
    is_interact = actions == action_pickup_drop_idx

    # Agent-ahead check for each agent (vectorized pairwise)
    # fwd_matches_pos[i,j] = True iff agent i's forward pos == agent j's position
    fwd_rc = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2)
    fwd_matches_pos = xp.all(fwd_rc[:, None, :] == state.agent_pos[None, :, :], axis=2)
    not_self = ~xp.eye(n_agents, dtype=xp.bool_)
    agent_ahead = xp.any(fwd_matches_pos & not_self, axis=1)

    base_ok = is_interact & ~agent_ahead  # (n_agents,) bool

    if interaction_fn is not None:
        # Scope with interaction_fn: process agents sequentially.
        # Lower-index agents have priority (process first, mutations visible
        # to later agents). n_agents is a static shape dim, so this Python
        # range loop unrolls at trace time -- no lax.fori_loop needed.
        for i in range(n_agents):
            state = interaction_fn(state, i, fwd_r[i], fwd_c[i], base_ok[i], scope_config)
    else:
        # Generic scope: handle branches 1, 3, 4 without scope-specific handler.
        agent_inv = state.agent_inv
        object_type_map = state.object_type_map
        object_state_map = state.object_state_map

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
            b4_cond = (
                ok
                & ~b1_cond
                & ~b3_cond
                & (fwd_type > 0)
                & (CAN_PLACE_ON[fwd_type] == 1)
                & (inv_item != -1)
                & (object_state_map[fwd_r[agent_idx], fwd_c[agent_idx]] == 0)
            )

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

        state = dataclasses.replace(
            state,
            agent_inv=agent_inv,
            object_type_map=object_type_map,
            object_state_map=object_state_map,
        )

    return state
