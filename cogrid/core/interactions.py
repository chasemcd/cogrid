"""Vectorized interaction processing using integer lookup tables.

Core modules contain only generic infrastructure. Environment-specific
interaction logic is delegated via an explicit ``interaction_fn`` parameter.

Key functions:

- ``process_interactions()`` -- handles pickup, drop, place_on interactions
  for all agents using xp array operations. Processes agents sequentially
  (lower index = higher priority) with vectorized condition computation.

Generic branch functions (shared signature ``(handled, ctx) -> (cond, updates, new_handled)``):

- ``branch_pickup`` -- pick up a loose object from the grid
- ``branch_pickup_from`` -- unified pickup from stacks and counters (PICKUP_FROM_GUARD)
- ``branch_drop_on_empty`` -- drop held item on an empty cell
- ``branch_place_on`` -- place held item on a PLACE_ON_GUARD surface

- ``merge_branch_results`` -- apply accumulated branch results via sparse xp.where
"""

import dataclasses

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d

# ======================================================================
# Generic branch functions
# ======================================================================
#
# Each function evaluates ONE branch of the interaction decision tree.
# All functions are pure: they take arrays, return arrays. No mutations.
#
# Uniform interface:
#   (handled, ctx) -> (cond, updates, new_handled)
#
# Standard ctx keys used by core branches:
#   base_ok, fwd_type, fwd_r, fwd_c, inv_item, agent_idx,
#   agent_inv, object_type_map, object_state_map,
#   CAN_PICKUP, PICKUP_FROM_GUARD, PLACE_ON_GUARD, pickup_from_produces
# ======================================================================


def _held_col(inv_item):
    """Map inventory item to guard table column index.

    Column 0 = empty hands (inv_item == -1).
    Columns 1..n = held type ID + 1.
    """
    return xp.where(inv_item == -1, 0, inv_item + 1)


def branch_pickup(handled, ctx):
    """Pick up a loose object from the forward cell.

    Conditions: forward cell has a pickupable object, hand is empty.
    Effects: object moves from grid to agent inventory, cell cleared.
    """
    cond = (
        ~handled
        & ctx["base_ok"]
        & (ctx["fwd_type"] > 0)
        & (ctx["CAN_PICKUP"][ctx["fwd_type"]] == 1)
        & (ctx["inv_item"] == -1)
    )
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), ctx["fwd_type"])
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    updates = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return cond, updates, handled | cond


def branch_pickup_from(handled, ctx):
    """Unified pickup from stacks and counters (PICKUP_FROM_GUARD objects).

    For stacks (pickup_from_produces[fwd_type] > 0): dispense produced item,
    grid unchanged (infinite supply).
    For counters (pickup_from_produces[fwd_type] == 0): pick up item stored
    in object_state_map, clear counter state.

    Conditions: forward cell passes PICKUP_FROM_GUARD for the agent's held
    item, and either stack produces something or counter has an item.
    """
    fwd_type = ctx["fwd_type"]
    pickup_from_produces = ctx["pickup_from_produces"]
    produced = pickup_from_produces[fwd_type]
    state_item = ctx["object_state_map"][ctx["fwd_r"], ctx["fwd_c"]]

    # Stack: produced > 0 -> use produced. Counter: produced == 0 -> use state_item.
    item = xp.where(produced > 0, produced, state_item)

    has_item = item > 0
    held_c = _held_col(ctx["inv_item"])
    cond = (
        ~handled
        & ctx["base_ok"]
        & (fwd_type > 0)
        & (ctx["PICKUP_FROM_GUARD"][fwd_type, held_c] == 1)
        & has_item
    )

    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), item)
    # Clear state_map only for counters (produced == 0); stacks keep their state.
    is_counter = produced == 0
    new_osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    osm = xp.where(is_counter, new_osm, ctx["object_state_map"])

    updates = {"agent_inv": inv, "object_state_map": osm}
    return cond, updates, handled | cond


def branch_drop_on_empty(handled, ctx):
    """Drop held item onto an empty floor cell.

    Conditions: forward cell is empty (type 0), hand is not empty.
    Effects: item placed on grid, inventory cleared.
    """
    cond = ~handled & ctx["base_ok"] & (ctx["fwd_type"] == 0) & (ctx["inv_item"] != -1)
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], ctx["inv_item"])
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return cond, updates, handled | cond


def branch_place_on(handled, ctx):
    """Place a held item on a PLACE_ON_GUARD surface (counter, etc.).

    Items are stored in object_state_map (the surface stays in object_type_map).
    Conditions: forward cell passes PLACE_ON_GUARD for agent's held item,
    surface is empty (state == 0).
    Effects: item stored in object_state_map, inventory cleared.
    """
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]
    counter_empty = ctx["object_state_map"][ctx["fwd_r"], ctx["fwd_c"]] == 0
    held_c = _held_col(inv_item)
    cond = (
        ~handled
        & ctx["base_ok"]
        & (fwd_type > 0)
        & (ctx["PLACE_ON_GUARD"][fwd_type, held_c] == 1)
        & counter_empty
    )
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], inv_item)
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv, "object_state_map": osm}
    return cond, updates, handled | cond


# ======================================================================
# Merge helper
# ======================================================================

# Standard array keys that core branches may update.
_STANDARD_KEYS = ("agent_inv", "object_type_map", "object_state_map")


def merge_branch_results(branch_results, arrays, extra_keys=()):
    """Merge accumulated branch results via sparse xp.where.

    Parameters
    ----------
    branch_results : list[(cond, updates)]
        Each entry is (bool scalar, dict of updated arrays).
    arrays : dict
        Initial arrays keyed by the same keys used in updates.
    extra_keys : tuple[str]
        Additional keys beyond the standard three to merge (e.g., "pot_contents").

    Returns:
    -------
    dict with the same keys as *arrays*, after applying all branch updates.
    """
    all_keys = _STANDARD_KEYS + tuple(extra_keys)
    result = dict(arrays)
    for cond, updates in branch_results:
        for key in all_keys:
            if key in updates:
                result[key] = xp.where(cond, updates[key], result[key])
    return result


# ======================================================================
# Default generic branch list (used when no interaction_fn is provided)
# ======================================================================

_GENERIC_BRANCHES = [
    branch_pickup,
    branch_pickup_from,
    branch_drop_on_empty,
    branch_place_on,
]


# ======================================================================
# Fallback guard table helper
# ======================================================================


def _default_guard_tables(lookup_tables):
    """Synthesize 2D guard tables from 1D arrays for backward compatibility.

    Used by envs that go through ``process_interactions`` without the
    autowire path (which calls ``build_guard_tables`` automatically).

    For ``PICKUP_FROM_GUARD``: sets column 0 (empty hands) wherever
    ``CAN_PICKUP_FROM`` is 1.
    For ``PLACE_ON_GUARD``: sets columns 1..n (any held item) wherever
    ``CAN_PLACE_ON`` is 1.
    """
    import numpy as np

    can_pf = lookup_tables["CAN_PICKUP_FROM"]
    can_po = lookup_tables["CAN_PLACE_ON"]
    n_types = can_pf.shape[0]
    n_cols = n_types + 1

    pf_guard = np.zeros((n_types, n_cols), dtype=np.int32)
    po_guard = np.zeros((n_types, n_cols), dtype=np.int32)

    for i in range(n_types):
        if int(can_pf[i]) == 1:
            pf_guard[i, 0] = 1  # empty hands
        if int(can_po[i]) == 1:
            po_guard[i, 1:] = 1  # any held item

    return {"PICKUP_FROM_GUARD": pf_guard, "PLACE_ON_GUARD": po_guard}


# ======================================================================
# Top-level interaction processor
# ======================================================================


def process_interactions(
    state,  # EnvState
    actions,  # (n_agents,) int32
    interaction_fn,  # callable or None
    lookup_tables,  # dict with CAN_PICKUP, CAN_OVERLAP, etc.
    scope_config,  # scope config dict (passed through to interaction_fn)
    dir_vec_table,  # (4, 2) int32
    action_pickup_drop_idx,  # int -- index of PickupDrop action
    action_toggle_idx,  # int -- index of Toggle action
):
    """Process pickup/drop/place_on interactions for all agents.

    Priority order: (1) pickup, (2) pickup_from,
    (3) drop on empty, (4) place_on. Agents are processed sequentially
    (lower index = higher priority).

    Returns updated ``state``.
    """
    n_agents = state.agent_pos.shape[0]
    H, W = state.object_type_map.shape

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
        for i in range(n_agents):
            state = interaction_fn(state, i, fwd_r[i], fwd_c[i], base_ok[i], scope_config)
    else:
        # Generic scope: use core branch functions.
        # Resolve guard tables: prefer scope_config, fall back to 1D synthesis.
        st = scope_config.get("static_tables", {}) if scope_config else {}
        if "PICKUP_FROM_GUARD" in st:
            pf_guard = st["PICKUP_FROM_GUARD"]
            po_guard = st["PLACE_ON_GUARD"]
        else:
            fallback = _default_guard_tables(lookup_tables)
            pf_guard = fallback["PICKUP_FROM_GUARD"]
            po_guard = fallback["PLACE_ON_GUARD"]

        agent_inv = state.agent_inv
        object_type_map = state.object_type_map
        object_state_map = state.object_state_map

        for agent_idx in range(n_agents):
            fwd_type = object_type_map[fwd_r[agent_idx], fwd_c[agent_idx]]
            inv_item = agent_inv[agent_idx, 0]

            ctx = {
                "base_ok": base_ok[agent_idx],
                "fwd_type": fwd_type,
                "fwd_r": fwd_r[agent_idx],
                "fwd_c": fwd_c[agent_idx],
                "inv_item": inv_item,
                "agent_idx": agent_idx,
                "agent_inv": agent_inv,
                "object_type_map": object_type_map,
                "object_state_map": object_state_map,
                "CAN_PICKUP": lookup_tables["CAN_PICKUP"],
                "PICKUP_FROM_GUARD": pf_guard,
                "PLACE_ON_GUARD": po_guard,
                "pickup_from_produces": lookup_tables["pickup_from_produces"],
            }

            handled = xp.bool_(False)
            branch_results = []
            for branch_fn in _GENERIC_BRANCHES:
                cond, updates, handled = branch_fn(handled, ctx)
                branch_results.append((cond, updates))

            merged = merge_branch_results(
                branch_results,
                {
                    "agent_inv": agent_inv,
                    "object_type_map": object_type_map,
                    "object_state_map": object_state_map,
                },
            )
            agent_inv = merged["agent_inv"]
            object_type_map = merged["object_type_map"]
            object_state_map = merged["object_state_map"]

        state = dataclasses.replace(
            state,
            agent_inv=agent_inv,
            object_type_map=object_type_map,
            object_state_map=object_state_map,
        )

    return state
