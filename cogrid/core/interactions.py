"""Vectorized interaction processing using integer lookup tables.

Core modules contain only generic infrastructure. Environment-specific
interaction logic is delegated via branch functions with signature
``(ctx) -> (should_apply, changes)``.

Key functions:

- ``process_interactions()`` -- handles pickup, drop, place_on interactions
  for all agents using xp array operations. Processes agents sequentially
  (lower index = higher priority) with vectorized condition computation.

- ``compose_interactions()`` -- returns ``(extras_fn, interactions_list)``
  for autowire scopes with containers and consume surfaces.

Branch functions (shared signature ``(ctx) -> (should_apply, changes)``):

- ``branch_pickup`` -- pick up a loose object from the grid
- ``branch_pickup_from`` -- unified pickup from stacks and counters
- ``branch_drop_on_empty`` -- drop held item on an empty cell
- ``branch_place_on`` -- place held item on a PLACE_ON_GUARD surface
- ``branch_pickup_from_container`` -- pick up cooked result from container
- ``branch_place_on_container`` -- place ingredient into container
- ``branch_place_on_consume`` -- place item on a consume surface (delivery)

- ``merge_branch_results`` -- apply accumulated branch results via sparse xp.where
"""

import dataclasses

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d
from cogrid.core.actions import ActionID
from cogrid.core.interaction_context import (
    InteractionContext,
    build_context,
    clear_facing_cell,
    empty_hands,
    find_facing_instance,
    give_item,
)

# ======================================================================
# Generic branch functions
# ======================================================================
#
# Each function evaluates ONE branch of the interaction decision tree.
# All functions are pure: they take arrays, return arrays. No mutations.
#
# Uniform interface:
#   (ctx) -> (should_apply, changes)
#
# Standard ctx fields used by core branches:
#   can_interact, facing_type, facing_row, facing_col, held_item,
#   agent_index, agent_inv, object_type_map, object_state_map,
#   CAN_PICKUP, PICKUP_FROM_GUARD, PLACE_ON_GUARD, pickup_from_produces
# ======================================================================


def _held_col(inv_item):
    """Map inventory item to guard table column index.

    Column 0 = empty hands (inv_item == -1).
    Columns 1..n = held type ID + 1.
    """
    return xp.where(inv_item == -1, 0, inv_item + 1)


def branch_pickup(ctx):
    """Pick up a loose object from the forward cell.

    Conditions: forward cell has a pickupable object, hand is empty.
    Effects: object moves from grid to agent inventory, cell cleared.
    """
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact
        & is_pickup_drop
        & (ctx.facing_type > 0)
        & (ctx.CAN_PICKUP[ctx.facing_type] == 1)
        & (ctx.held_item == -1)
    )
    inv = give_item(ctx, ctx.facing_type)
    otm = clear_facing_cell(ctx)
    osm = set_at_2d(ctx.object_state_map, ctx.facing_row, ctx.facing_col, 0)
    changes = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return should_apply, changes


def branch_pickup_from(ctx):
    """Unified pickup from stacks and counters (PICKUP_FROM_GUARD objects).

    For stacks (pickup_from_produces[fwd_type] > 0): dispense produced item,
    grid unchanged (infinite supply).
    For counters (pickup_from_produces[fwd_type] == 0): pick up item stored
    in object_state_map, clear counter state.

    Conditions: forward cell passes PICKUP_FROM_GUARD for the agent's held
    item, and either stack produces something or counter has an item.
    """
    fwd_type = ctx.facing_type
    pickup_from_produces = ctx.pickup_from_produces
    produced = pickup_from_produces[fwd_type]
    state_item = ctx.object_state_map[ctx.facing_row, ctx.facing_col]

    # Stack: produced > 0 -> use produced. Counter: produced == 0 -> use state_item.
    item = xp.where(produced > 0, produced, state_item)

    has_item = item > 0
    held_c = _held_col(ctx.held_item)
    # Exclude container types (e.g., pot) — their pickup is handled by
    # branch_pickup_from_container which checks recipe/timer conditions.
    container_id = ctx.extra.get("container_id", -1)
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact
        & is_pickup_drop
        & (fwd_type > 0)
        & (fwd_type != container_id)
        & (ctx.PICKUP_FROM_GUARD[fwd_type, held_c] == 1)
        & has_item
    )

    inv = give_item(ctx, item)
    # Clear state_map only for counters (produced == 0); stacks keep their state.
    is_counter = produced == 0
    new_osm = set_at_2d(ctx.object_state_map, ctx.facing_row, ctx.facing_col, 0)
    osm = xp.where(is_counter, new_osm, ctx.object_state_map)

    changes = {"agent_inv": inv, "object_state_map": osm}
    return should_apply, changes


def branch_drop_on_empty(ctx):
    """Drop held item onto an empty floor cell.

    Conditions: forward cell is empty (type 0), hand is not empty.
    Effects: item placed on grid, inventory cleared.
    """
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact & is_pickup_drop & (ctx.facing_type == 0) & (ctx.held_item != -1)
    )
    otm = set_at_2d(ctx.object_type_map, ctx.facing_row, ctx.facing_col, ctx.held_item)
    osm = set_at_2d(ctx.object_state_map, ctx.facing_row, ctx.facing_col, 0)
    inv = empty_hands(ctx)
    changes = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return should_apply, changes


def branch_place_on(ctx):
    """Place a held item on a PLACE_ON_GUARD surface (counter, etc.).

    Items are stored in object_state_map (the surface stays in object_type_map).
    Conditions: forward cell passes PLACE_ON_GUARD for agent's held item,
    surface is empty (state == 0).
    Effects: item stored in object_state_map, inventory cleared.
    """
    fwd_type = ctx.facing_type
    inv_item = ctx.held_item
    counter_empty = ctx.object_state_map[ctx.facing_row, ctx.facing_col] == 0
    held_c = _held_col(inv_item)
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact
        & is_pickup_drop
        & (fwd_type > 0)
        & (ctx.PLACE_ON_GUARD[fwd_type, held_c] == 1)
        & counter_empty
    )
    osm = set_at_2d(ctx.object_state_map, ctx.facing_row, ctx.facing_col, inv_item)
    inv = empty_hands(ctx)
    changes = {"agent_inv": inv, "object_state_map": osm}
    return should_apply, changes


# ======================================================================
# Merge helper
# ======================================================================


def merge_branch_results(branch_results, originals):
    """Merge accumulated branch results via sparse xp.where.

    Parameters
    ----------
    branch_results : list[(should_apply, changes)]
        Each entry is (bool scalar, dict of updated arrays).
    originals : dict
        Initial arrays keyed by the same keys used in changes.

    Returns:
    -------
    dict with the same keys as *originals*, after applying all branch updates.
    """
    result = dict(originals)
    for should_apply, changes in branch_results:
        for key in changes:
            if key in result:
                result[key] = xp.where(should_apply, changes[key], result[key])
    return result


# ======================================================================
# Container-aware branch functions
# ======================================================================


def branch_pickup_from_container(ctx):
    """Pick up a cooked recipe result from a container.

    Conditions: forward cell is the container type, agent holds the
    required pickup item (e.g. plate), container has contents, timer
    is done (== 0), and contents match a known recipe.

    Effects: agent inventory gets the recipe result, container is emptied,
    timer is reset.
    """
    container_id = ctx.container_id
    is_container = ctx.facing_type == container_id

    pot_idx = ctx.pot_idx
    pot_contents = ctx.pot_contents

    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = ctx.pot_timer[pot_idx] == 0

    # Recipe matching: sort contents and compare against all recipes
    _SORT_HIGH = xp.int32(2147483647)
    raw = pot_contents[pot_idx]
    sort_buf = xp.where(raw == -1, _SORT_HIGH, raw)
    sorted_contents = xp.where(xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf))
    matches = xp.all(sorted_contents[None, :] == ctx.recipe_ingredients, axis=1)
    matched_idx = xp.argmax(matches)
    has_recipe_match = xp.any(matches)

    result_type = ctx.recipe_result[matched_idx]

    held_c = _held_col(ctx.held_item)
    guard_ok = ctx.PICKUP_FROM_GUARD[ctx.facing_type, held_c] == 1
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact
        & is_pickup_drop
        & is_container
        & ctx.has_pot_match
        & has_contents
        & is_ready
        & guard_ok
        & has_recipe_match
    )

    inv = give_item(ctx, result_type)
    pc = set_at(pot_contents, (pot_idx, slice(None)), -1)
    pt = set_at(ctx.pot_timer, pot_idx, ctx.cooking_time)
    changes = {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}
    return should_apply, changes


def branch_place_on_container(ctx):
    """Place an ingredient into a container using recipe prefix matching.

    Conditions: forward cell is the container type, agent holds a legal
    ingredient (passes PLACE_ON_GUARD), container has capacity, and the
    would-be contents match a prefix of at least one recipe.

    Effects: ingredient added to container, inventory cleared. If the
    container is now full, timer is set from the matched recipe.
    """
    fwd_type = ctx.facing_type
    inv_item = ctx.held_item
    container_id = ctx.container_id
    pot_idx = ctx.pot_idx
    pot_contents = ctx.pot_contents
    max_ingredients = ctx.max_ingredients
    recipe_ingredients = ctx.recipe_ingredients
    recipe_cooking_time = ctx.recipe_cooking_time

    is_container = fwd_type == container_id
    held_c = _held_col(inv_item)
    guard_ok = ctx.PLACE_ON_GUARD[fwd_type, held_c] == 1
    n_items = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items < max_ingredients

    slot_empty = pot_contents[pot_idx] == -1
    first_empty_slot = xp.argmax(slot_empty)

    # Build would-be contents and sort
    _SORT_HIGH = xp.int32(2147483647)
    would_be = set_at(pot_contents[pot_idx], first_empty_slot, inv_item)
    sort_buf = xp.where(would_be == -1, _SORT_HIGH, would_be)
    sorted_would_be = xp.where(xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf))
    n_would_be = n_items + 1

    # Prefix match
    slot_mask = xp.arange(max_ingredients) < n_would_be
    slot_matches = (sorted_would_be[None, :] == recipe_ingredients) | ~slot_mask[None, :]
    recipe_compatible = xp.all(slot_matches, axis=1)
    any_recipe_accepts = xp.any(recipe_compatible)

    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = (
        ctx.can_interact
        & is_pickup_drop
        & (inv_item != -1)
        & is_container
        & ctx.has_pot_match
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
        should_apply & is_now_full,
        set_at(ctx.pot_timer, pot_idx, new_cook_time),
        ctx.pot_timer,
    )

    inv = empty_hands(ctx)
    changes = {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}
    return should_apply, changes


def branch_place_on_consume(ctx):
    """Place an item on a surface that consumes it (e.g. delivery zone).

    Conditions: forward cell type has consumes_on_place flag set in
    PLACE_ON_GUARD.

    Effects: agent inventory cleared (item vanishes). No object_state_map
    write. Order consumption (if enabled) is handled here.
    """
    fwd_type = ctx.facing_type
    inv_item = ctx.held_item
    consume_ids = ctx.consume_type_ids

    is_consume = xp.bool_(False)
    for cid in consume_ids:
        is_consume = is_consume | (fwd_type == cid)

    held_c = _held_col(inv_item)
    guard_ok = ctx.PLACE_ON_GUARD[fwd_type, held_c] == 1
    is_pickup_drop = ctx.action == ctx.action_id.pickup_drop
    should_apply = ctx.can_interact & is_pickup_drop & (inv_item != -1) & is_consume & guard_ok
    inv = empty_hands(ctx)
    changes = {"agent_inv": inv}

    # Order consumption (only when orders are enabled)
    order_recipe = ctx.extra.get("order_recipe")
    if order_recipe is not None:
        order_timer = ctx.order_timer
        recipe_result = ctx.recipe_result

        delivered_type = inv_item
        recipe_matches = recipe_result == delivered_type
        safe_idx = xp.where(order_recipe >= 0, order_recipe, 0)
        order_is_match = recipe_matches[safe_idx] & (order_recipe >= 0)
        has_match = xp.any(order_is_match)
        first_match = xp.argmax(order_is_match)
        consume = should_apply & has_match

        new_order_recipe = xp.where(consume, set_at(order_recipe, first_match, -1), order_recipe)
        new_order_timer = xp.where(consume, set_at(order_timer, first_match, 0), order_timer)
        changes["order_recipe"] = new_order_recipe
        changes["order_timer"] = new_order_timer

    return should_apply, changes


# ======================================================================
# Composed interaction function builder
# ======================================================================


def compose_interactions(container_specs, consume_type_ids, scope):
    """Build an extras_fn and interactions list from container and consume specs.

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
    tuple[callable, list[callable]]
        ``(extras_fn, interactions_list)``.

        ``extras_fn`` has signature ``(state, scope_config) -> dict`` and returns
        SHORT keys (no scope prefix) for container arrays, pot matching, etc.

        ``interactions_list`` is a list of branch functions with signature
        ``(ctx) -> (should_apply, changes)``.
    """

    def extras_fn(state, scope_config):
        """Compute extra context fields from state and scope config.

        Reads container arrays from extra_state and resolves pot matching.
        Returns a dict with short keys (no scope prefix).
        """
        static_tables = scope_config.get("static_tables", {})
        prefix = f"{scope}."

        extras = {}

        # Extract container arrays from extra_state
        spec = container_specs[0] if container_specs else None
        if spec:
            oid = spec["object_id"]
            pot_contents = state.extra_state[f"{prefix}{oid}_contents"]
            pot_timer = state.extra_state[f"{prefix}{oid}_timer"]
            pot_positions = state.extra_state[f"{prefix}{oid}_positions"]

            if pot_positions.shape[0] > 0:
                extras["pot_contents"] = pot_contents
                extras["pot_timer"] = pot_timer

                extras["container_id"] = static_tables.get(f"{oid}_id", -1)
                extras["cooking_time"] = static_tables.get("cooking_time", 0)
                extras["recipe_ingredients"] = static_tables.get(
                    "recipe_ingredients", xp.zeros((0, 1), dtype=xp.int32)
                )
                extras["recipe_result"] = static_tables.get(
                    "recipe_result", xp.zeros((0,), dtype=xp.int32)
                )
                extras["recipe_cooking_time"] = static_tables.get(
                    "recipe_cooking_time", xp.zeros((0,), dtype=xp.int32)
                )
                extras["max_ingredients"] = static_tables.get("max_ingredients", 0)
            else:
                # Dummy arrays so pot_idx=0 is always valid;
                # has_pot_match=False ensures no branch fires.
                extras["pot_contents"] = xp.full((1, 1), -1, dtype=xp.int32)
                extras["pot_timer"] = xp.zeros((1,), dtype=xp.int32)
                extras["container_id"] = -1
                extras["cooking_time"] = 0
                extras["recipe_ingredients"] = xp.zeros((0, 1), dtype=xp.int32)
                extras["recipe_result"] = xp.zeros((0,), dtype=xp.int32)
                extras["recipe_cooking_time"] = xp.zeros((0,), dtype=xp.int32)
                extras["max_ingredients"] = 0
        else:
            extras["container_id"] = -1

        if consume_type_ids:
            extras["consume_type_ids"] = consume_type_ids

        # Add order arrays if present
        or_ = state.extra_state.get(f"{scope}.order_recipe")
        ot_ = state.extra_state.get(f"{scope}.order_timer")
        if or_ is not None:
            extras["order_recipe"] = or_
            extras["order_timer"] = ot_

        return extras

    # Build branch list
    interactions_list = [branch_pickup]
    if container_specs:
        interactions_list.append(branch_pickup_from_container)
    interactions_list.append(branch_pickup_from)
    interactions_list.append(branch_drop_on_empty)
    if container_specs:
        interactions_list.append(branch_place_on_container)
    if consume_type_ids:
        interactions_list.append(branch_place_on_consume)
    interactions_list.append(branch_place_on)

    return extras_fn, interactions_list


# ======================================================================
# Default generic branch list (used when no interactions are provided)
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
# Internal interaction runner
# ======================================================================


def _run_interactions(
    state,
    agent_idx,
    fwd_r,
    fwd_c,
    base_ok,
    scope_config,
    interactions,
    autowire_extras_fn,
    scope,
    *,
    action,
    action_id,
):
    """Run interaction branches for a single agent and merge results.

    1. Build InteractionContext for standard fields.
    2. Auto-read ALL extra_state keys onto ctx.extra (strip scope prefix).
    3. If autowire_extras_fn: merge its output into ctx.extra.
    4. Run each interaction: ``should_apply, changes = fn(ctx)``; collect results.
    5. Auto-merge: standard keys -> state fields; everything else -> extra_state
       with scope prefix.
    6. Return updated state.
    """
    ctx = build_context(
        state, agent_idx, fwd_r, fwd_c, base_ok, scope_config, action=action, action_id=action_id
    )

    # Auto-read extra_state keys (strip scope prefix)
    extra = dict(ctx.extra)
    if scope:
        prefix = f"{scope}."
        for key, val in state.extra_state.items():
            if key.startswith(prefix):
                short_key = key[len(prefix) :]
                extra[short_key] = val

    # Autowire extras (container arrays, recipe config, etc.)
    if autowire_extras_fn is not None:
        autowire_extra = autowire_extras_fn(state, scope_config)
        extra.update(autowire_extra)

    # Find facing instance for containers (pot_idx, has_pot_match)
    if "pot_contents" in extra and scope:
        spec_prefix = f"{scope}."
        # Look for positions array
        for key, val in state.extra_state.items():
            if key.startswith(spec_prefix) and key.endswith("_positions"):
                positions = val
                if positions.shape[0] > 0:
                    pot_idx, has_pot_match = find_facing_instance(positions, fwd_r, fwd_c)
                else:
                    pot_idx = xp.int32(0)
                    has_pot_match = xp.bool_(False)
                extra["pot_idx"] = pot_idx
                extra["has_pot_match"] = has_pot_match
                break
        else:
            extra["pot_idx"] = xp.int32(0)
            extra["has_pot_match"] = xp.bool_(False)

    # Rebuild ctx with populated extra
    ctx = InteractionContext(
        can_interact=ctx.can_interact,
        action=ctx.action,
        action_id=ctx.action_id,
        facing_row=ctx.facing_row,
        facing_col=ctx.facing_col,
        facing_type=ctx.facing_type,
        agent_index=ctx.agent_index,
        held_item=ctx.held_item,
        agent_inv=ctx.agent_inv,
        object_type_map=ctx.object_type_map,
        object_state_map=ctx.object_state_map,
        type_ids=ctx.type_ids,
        extra=extra,
        _tables=ctx._tables,
    )

    # Run branches. If multiple branches fire for the same agent, the
    # later branch's changes overwrite earlier ones (last writer wins).
    branch_results = []
    for fn in interactions:
        should_apply, changes = fn(ctx)
        branch_results.append((should_apply, changes))

    # Collect all keys that appear in any branch's changes
    all_keys = set()
    for _, changes in branch_results:
        all_keys.update(changes.keys())

    # Standard keys that map directly to state fields
    _STD_KEYS = {"agent_inv", "object_type_map", "object_state_map"}

    # Build originals dict for merge
    originals = {}
    for key in all_keys:
        if key in _STD_KEYS:
            originals[key] = getattr(state, key)
        elif key in extra:
            originals[key] = extra[key]

    merged = merge_branch_results(branch_results, originals)

    # Apply standard fields
    state_updates = {}
    for key in _STD_KEYS:
        if key in merged:
            state_updates[key] = merged[key]

    # Apply extra_state fields
    new_extra = dict(state.extra_state)
    prefix = f"{scope}." if scope else ""
    for key in merged:
        if key not in _STD_KEYS:
            new_extra[f"{prefix}{key}"] = merged[key]

    state_updates["extra_state"] = new_extra
    state = dataclasses.replace(state, **state_updates)

    return state


# ======================================================================
# Top-level interaction processor
# ======================================================================


def process_interactions(
    state,  # EnvState
    actions,  # (n_agents,) int32
    interactions,  # list[callable] or None
    lookup_tables,  # dict with CAN_PICKUP, CAN_OVERLAP, etc.
    scope_config,  # scope config dict
    dir_vec_table,  # (4, 2) int32
    action_pickup_drop_idx,  # int -- index of PickupDrop action
    action_toggle_idx,  # int -- index of Toggle action
):
    """Process pickup/drop/place_on interactions for all agents.

    Priority order: (1) pickup, (2) pickup_from,
    (3) drop on empty, (4) place_on. Agents are processed sequentially
    (lower index = higher priority).

    Parameters
    ----------
    interactions : list[callable] or None
        Branch functions with signature ``(ctx) -> (should_apply, changes)``.
        If None, uses the generic fallback branches.
    scope_config : dict
        Carries ``autowire_extras_fn``, ``autowire_interactions``,
        ``user_interactions``, ``scope``, ``static_tables``, etc.

    Returns updated ``state``.
    """
    n_agents = state.agent_pos.shape[0]
    H, W = state.object_type_map.shape

    # Compute forward positions for ALL agents
    fwd_pos = state.agent_pos + dir_vec_table[state.agent_dir]  # (n_agents, 2)
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)

    # Which agents are interacting (PickupDrop OR Toggle)
    is_interact = (actions == action_pickup_drop_idx) | (actions == action_toggle_idx)

    # Agent-ahead check for each agent (vectorized pairwise)
    fwd_rc = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2)
    fwd_matches_pos = xp.all(fwd_rc[:, None, :] == state.agent_pos[None, :, :], axis=2)
    not_self = ~xp.eye(n_agents, dtype=xp.bool_)
    agent_ahead = xp.any(fwd_matches_pos & not_self, axis=1)

    base_ok = is_interact & ~agent_ahead  # (n_agents,) bool

    # Gather scope info
    sc = scope_config if scope_config else {}
    scope = sc.get("scope", "")
    autowire_extras_fn = sc.get("autowire_extras_fn")
    autowire_interactions = sc.get("autowire_interactions") or []
    user_interactions = sc.get("user_interactions") or []

    # Build the full interaction list
    if interactions is not None:
        all_interactions = interactions
    elif user_interactions or autowire_interactions:
        all_interactions = user_interactions + autowire_interactions
    else:
        # Generic fallback: ensure guard tables are available
        st = sc.get("static_tables", {})
        if "PICKUP_FROM_GUARD" not in st:
            fallback = _default_guard_tables(lookup_tables)
            # Inject into scope_config so build_context picks them up
            if "static_tables" not in sc:
                sc["static_tables"] = {}
            sc["static_tables"]["PICKUP_FROM_GUARD"] = fallback["PICKUP_FROM_GUARD"]
            sc["static_tables"]["PLACE_ON_GUARD"] = fallback["PLACE_ON_GUARD"]
            # Also ensure core tables are present
            for tbl_key in ("CAN_PICKUP", "pickup_from_produces"):
                if tbl_key not in sc["static_tables"] and tbl_key in lookup_tables:
                    sc["static_tables"][tbl_key] = lookup_tables[tbl_key]
        all_interactions = _GENERIC_BRANCHES

    action_id = sc.get("action_id")
    if action_id is None:
        action_id = ActionID(pickup_drop=action_pickup_drop_idx, toggle=action_toggle_idx)

    for agent_idx in range(n_agents):
        state = _run_interactions(
            state,
            agent_idx,
            fwd_r[agent_idx],
            fwd_c[agent_idx],
            base_ok[agent_idx],
            sc,
            all_interactions,
            autowire_extras_fn,
            scope,
            action=actions[agent_idx],
            action_id=action_id,
        )

    return state
