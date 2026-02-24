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
# Container-aware branch functions
# ======================================================================


def branch_pickup_from_container(handled, ctx):
    """Pick up a cooked recipe result from a container.

    Conditions: forward cell is the container type, agent holds the
    required pickup item (e.g. plate), container has contents, timer
    is done (== 0), and contents match a known recipe.

    Effects: agent inventory gets the recipe result, container is emptied,
    timer is reset.
    """
    container_id = ctx["container_id"]
    is_container = ctx["fwd_type"] == container_id

    pot_idx = ctx["pot_idx"]
    pot_contents = ctx["pot_contents"]

    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = ctx["pot_timer"][pot_idx] == 0

    # Recipe matching: sort contents and compare against all recipes
    _SORT_HIGH = xp.int32(2147483647)
    raw = pot_contents[pot_idx]
    sort_buf = xp.where(raw == -1, _SORT_HIGH, raw)
    sorted_contents = xp.where(
        xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf)
    )
    matches = xp.all(sorted_contents[None, :] == ctx["recipe_ingredients"], axis=1)
    matched_idx = xp.argmax(matches)
    has_recipe_match = xp.any(matches)

    result_type = ctx["recipe_result"][matched_idx]

    held_c = _held_col(ctx["inv_item"])
    guard_ok = ctx["PICKUP_FROM_GUARD"][ctx["fwd_type"], held_c] == 1
    cond = (
        ~handled
        & ctx["base_ok"]
        & is_container
        & ctx["has_pot_match"]
        & has_contents
        & is_ready
        & guard_ok
        & has_recipe_match
    )

    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), result_type)
    pc = set_at(pot_contents, (pot_idx, slice(None)), -1)
    pt = set_at(ctx["pot_timer"], pot_idx, ctx["cooking_time"])
    updates = {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}
    return cond, updates, handled | cond


def branch_place_on_container(handled, ctx):
    """Place an ingredient into a container using recipe prefix matching.

    Conditions: forward cell is the container type, agent holds a legal
    ingredient (passes PLACE_ON_GUARD), container has capacity, and the
    would-be contents match a prefix of at least one recipe.

    Effects: ingredient added to container, inventory cleared. If the
    container is now full, timer is set from the matched recipe.
    """
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]
    container_id = ctx["container_id"]
    pot_idx = ctx["pot_idx"]
    pot_contents = ctx["pot_contents"]
    max_ingredients = ctx["max_ingredients"]
    recipe_ingredients = ctx["recipe_ingredients"]
    recipe_cooking_time = ctx["recipe_cooking_time"]

    is_container = fwd_type == container_id
    held_c = _held_col(inv_item)
    guard_ok = ctx["PLACE_ON_GUARD"][fwd_type, held_c] == 1
    n_items = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items < max_ingredients

    slot_empty = pot_contents[pot_idx] == -1
    first_empty_slot = xp.argmax(slot_empty)

    # Build would-be contents and sort
    _SORT_HIGH = xp.int32(2147483647)
    would_be = set_at(pot_contents[pot_idx], first_empty_slot, inv_item)
    sort_buf = xp.where(would_be == -1, _SORT_HIGH, would_be)
    sorted_would_be = xp.where(
        xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf)
    )
    n_would_be = n_items + 1

    # Prefix match
    slot_mask = xp.arange(max_ingredients) < n_would_be
    slot_matches = (sorted_would_be[None, :] == recipe_ingredients) | ~slot_mask[None, :]
    recipe_compatible = xp.all(slot_matches, axis=1)
    any_recipe_accepts = xp.any(recipe_compatible)

    cond = (
        ~handled
        & ctx["base_ok"]
        & (inv_item != -1)
        & is_container
        & ctx["has_pot_match"]
        & guard_ok
        & has_capacity
        & any_recipe_accepts
    )
    pc = set_at(pot_contents, (pot_idx, first_empty_slot), inv_item)

    # If this fills the container, set cook time from matched recipe
    is_now_full = n_would_be == max_ingredients
    full_matches = xp.all(sorted_would_be[None, :] == recipe_ingredients, axis=1)
    full_match_idx = xp.argmax(full_matches)
    new_cook_time = recipe_cooking_time[full_match_idx]
    pt = xp.where(
        cond & is_now_full,
        set_at(ctx["pot_timer"], pot_idx, new_cook_time),
        ctx["pot_timer"],
    )

    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}
    return cond, updates, handled | cond


def branch_place_on_consume(handled, ctx):
    """Place an item on a surface that consumes it (e.g. delivery zone).

    Conditions: forward cell type has consumes_on_place flag set in
    PLACE_ON_GUARD.

    Effects: agent inventory cleared (item vanishes). No object_state_map
    write. Order consumption (if enabled) is handled here.
    """
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]
    consume_ids = ctx["consume_type_ids"]

    is_consume = xp.bool_(False)
    for cid in consume_ids:
        is_consume = is_consume | (fwd_type == cid)

    held_c = _held_col(inv_item)
    guard_ok = ctx["PLACE_ON_GUARD"][fwd_type, held_c] == 1
    cond = ~handled & ctx["base_ok"] & (inv_item != -1) & is_consume & guard_ok
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv}

    # Order consumption (only when orders are enabled)
    if ctx.get("order_recipe") is not None:
        order_recipe = ctx["order_recipe"]
        order_timer = ctx["order_timer"]
        recipe_result = ctx["recipe_result"]

        delivered_type = inv_item
        recipe_matches = recipe_result == delivered_type
        safe_idx = xp.where(order_recipe >= 0, order_recipe, 0)
        order_is_match = recipe_matches[safe_idx] & (order_recipe >= 0)
        has_match = xp.any(order_is_match)
        first_match = xp.argmax(order_is_match)
        consume = cond & has_match

        new_order_recipe = xp.where(
            consume, set_at(order_recipe, first_match, -1), order_recipe
        )
        new_order_timer = xp.where(
            consume, set_at(order_timer, first_match, 0), order_timer
        )
        updates["order_recipe"] = new_order_recipe
        updates["order_timer"] = new_order_timer

    return cond, updates, handled | cond


# ======================================================================
# Composed interaction function builder
# ======================================================================


def compose_interaction_fn(container_specs, consume_type_ids, scope):
    """Build a composed interaction_fn from container and consume specs.

    Parameters
    ----------
    container_specs : list[dict]
        Each dict has keys: object_id, container (Container), recipes (list[Recipe]).
    consume_type_ids : list[int]
        Type IDs of objects with consumes_on_place=True.
    scope : str
        Registry scope.

    Returns:
    -------
    callable
        interaction_fn with signature (state, agent_idx, fwd_r, fwd_c,
        base_ok, scope_config) -> state.
    """
    import dataclasses

    def interaction_fn(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config):
        static_tables = scope_config.get("static_tables", {})

        fwd_type = state.object_type_map[fwd_r, fwd_c]
        inv_item = state.agent_inv[agent_idx, 0]

        # Extract container arrays from extra_state
        prefix = f"{scope}."
        container_arrays = {}
        for spec in container_specs:
            oid = spec["object_id"]
            container_arrays[oid] = {
                "contents": state.extra_state[f"{prefix}{oid}_contents"],
                "timer": state.extra_state[f"{prefix}{oid}_timer"],
                "positions": state.extra_state[f"{prefix}{oid}_positions"],
            }

        # For now, support single container type (multi-container is a
        # natural extension but not needed for current environments).
        # Use the first container spec for pot_* keys in ctx.
        spec = container_specs[0] if container_specs else None
        if spec:
            oid = spec["object_id"]
            ca = container_arrays[oid]
            pot_contents = ca["contents"]
            pot_timer = ca["timer"]
            pot_positions = ca["positions"]

            # Find which instance the agent is facing
            fwd_pos_2d = xp.stack([fwd_r, fwd_c])
            pot_match = xp.all(pot_positions == fwd_pos_2d[None, :], axis=1)
            pot_idx = xp.argmax(pot_match)
            has_pot_match = xp.any(pot_match)
        else:
            pot_contents = xp.zeros((0, 1), dtype=xp.int32)
            pot_timer = xp.zeros((0,), dtype=xp.int32)
            pot_positions = xp.zeros((0, 2), dtype=xp.int32)
            pot_idx = xp.int32(0)
            has_pot_match = xp.bool_(False)

        # Build ctx
        ctx = {
            "base_ok": base_ok,
            "fwd_type": fwd_type,
            "fwd_r": fwd_r,
            "fwd_c": fwd_c,
            "inv_item": inv_item,
            "agent_idx": agent_idx,
            "agent_inv": state.agent_inv,
            "object_type_map": state.object_type_map,
            "object_state_map": state.object_state_map,
            "pot_contents": pot_contents,
            "pot_timer": pot_timer,
            "pot_idx": pot_idx,
            "has_pot_match": has_pot_match,
            "CAN_PICKUP": static_tables["CAN_PICKUP"],
            "PICKUP_FROM_GUARD": static_tables["PICKUP_FROM_GUARD"],
            "PLACE_ON_GUARD": static_tables["PLACE_ON_GUARD"],
            "pickup_from_produces": static_tables["pickup_from_produces"],
            "container_id": static_tables.get(f"{spec['object_id']}_id", -1) if spec else -1,
            "cooking_time": static_tables.get("cooking_time", 0),
            "recipe_ingredients": static_tables.get(
                "recipe_ingredients", xp.zeros((0, 1), dtype=xp.int32)
            ),
            "recipe_result": static_tables.get(
                "recipe_result", xp.zeros((0,), dtype=xp.int32)
            ),
            "recipe_cooking_time": static_tables.get(
                "recipe_cooking_time", xp.zeros((0,), dtype=xp.int32)
            ),
            "max_ingredients": static_tables.get("max_ingredients", 0),
            "consume_type_ids": consume_type_ids,
        }

        # Add order arrays if present
        or_ = state.extra_state.get(f"{scope}.order_recipe")
        ot_ = state.extra_state.get(f"{scope}.order_timer")
        if or_ is not None:
            ctx["order_recipe"] = or_
            ctx["order_timer"] = ot_

        # Build branch list
        branches = [branch_pickup]
        if container_specs:
            branches.append(branch_pickup_from_container)
        branches.append(branch_pickup_from)
        branches.append(branch_drop_on_empty)
        if container_specs:
            branches.append(branch_place_on_container)
        if consume_type_ids:
            branches.append(branch_place_on_consume)
        branches.append(branch_place_on)

        # Evaluate branches
        handled = xp.bool_(False)
        branch_results = []
        for branch_fn in branches:
            cond, updates, handled = branch_fn(handled, ctx)
            branch_results.append((cond, updates))

        # Merge results
        agent_inv = state.agent_inv
        otm = state.object_type_map
        osm = state.object_state_map
        pc = pot_contents
        pt = pot_timer
        or_out = or_
        ot_out = ot_

        for cond, updates in branch_results:
            if "agent_inv" in updates:
                agent_inv = xp.where(cond, updates["agent_inv"], agent_inv)
            if "object_type_map" in updates:
                otm = xp.where(cond, updates["object_type_map"], otm)
            if "object_state_map" in updates:
                osm = xp.where(cond, updates["object_state_map"], osm)
            if "pot_contents" in updates:
                pc = xp.where(cond, updates["pot_contents"], pc)
            if "pot_timer" in updates:
                pt = xp.where(cond, updates["pot_timer"], pt)
            if "order_recipe" in updates:
                or_out = xp.where(cond, updates["order_recipe"], or_out)
            if "order_timer" in updates:
                ot_out = xp.where(cond, updates["order_timer"], ot_out)

        # Repack into state
        new_extra = dict(state.extra_state)
        if spec:
            oid = spec["object_id"]
            new_extra[f"{prefix}{oid}_contents"] = pc
            new_extra[f"{prefix}{oid}_timer"] = pt
        if or_out is not None:
            new_extra[f"{scope}.order_recipe"] = or_out
            new_extra[f"{scope}.order_timer"] = ot_out

        return dataclasses.replace(
            state,
            agent_inv=agent_inv,
            object_type_map=otm,
            object_state_map=osm,
            extra_state=new_extra,
        )

    return interaction_fn


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
