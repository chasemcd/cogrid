"""Overcooked-specific array configuration.

Provides environment-specific array logic for Overcooked: tick handlers,
interaction function, extra state building, and static table construction.

Key public functions:

- ``build_overcooked_extra_state`` -- extra_state builder for pot arrays
- ``overcooked_interaction_fn`` -- per-agent interaction with (state, ...) -> state signature
- ``overcooked_interaction_body`` -- low-level per-agent dispatch to branch handlers
- ``overcooked_tick`` -- unified pot cooking timer state machine
- ``overcooked_tick_state`` -- tick handler with generic signature


Overcooked Interaction Overview
===============================

When an agent issues a PickupDrop action, ``process_interactions`` (in
``cogrid.core.interactions``) determines the cell the agent is facing and
whether another agent is blocking it. It then calls
``overcooked_interaction_fn`` once per agent (lower index = higher priority).

The interaction resolves to exactly one of seven mutually exclusive branches,
evaluated in strict priority order via an accumulated ``handled`` mask. Each
branch checks ``~handled`` as its first condition term; if a prior branch
already fired, all subsequent branches are suppressed. Adding a new branch
means writing one ``(handled, ctx) -> (cond, updates, handled)`` function and
appending it to the ``_BRANCHES`` list.

Decision tree (evaluated per agent per step):

    base_ok?  (agent issued PickupDrop AND no other agent in the forward cell)
        |
        +-- No  --> no-op (all branches short-circuit via ~handled or base_ok)
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


Branch list + accumulated-handled pattern
------------------------------------------

Each branch function has the uniform signature:

    (handled, ctx) -> (cond, updates, new_handled)

Where:
    - ``handled`` is a bool scalar, True if a prior branch already fired
    - ``ctx`` is a dict of all shared arrays and static tables (assembled
      once by the orchestrator, never mutated between branches)
    - ``cond`` is a bool scalar, True if this branch fires (always
      includes ``~handled`` as the first term)
    - ``updates`` is a dict with ONLY the arrays this branch modifies,
      using these exact keys: ``"agent_inv"``, ``"object_type_map"``,
      ``"object_state_map"``, ``"pot_contents"``, ``"pot_timer"``
    - ``new_handled`` is ``handled | cond``

The orchestrator (``overcooked_interaction_body``) iterates the
``_BRANCHES`` list in priority order, passing the accumulated ``handled``
through each branch. After all branches run, it merges results using
sparse ``xp.where`` over the 5 output arrays:

    for cond, updates in branch_results:
        if "agent_inv" in updates:
            inv = xp.where(cond, updates["agent_inv"], inv)
        ...

Because conditions are mutually exclusive (each guards with ``~handled``),
at most one condition is True, so the final value is either the matching
branch's result or the original.
"""

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d
from cogrid.core.component_registry import get_all_components
from cogrid.core.grid_object import get_object_names, object_to_idx


def build_overcooked_extra_state(parsed_arrays, scope="overcooked"):
    """Build pot state arrays (contents, timer, positions) from the layout."""
    import numpy as _np

    pot_type_id = object_to_idx("pot", scope=scope)
    otm = parsed_arrays["object_type_map"]

    # Find pot positions from object_type_map.
    pot_mask = otm == pot_type_id
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

    Returns:
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
        agent_idx,
        state.agent_inv,
        state.object_type_map,
        state.object_state_map,
        fwd_r,
        fwd_c,
        fwd_type,
        inv_item,
        base_ok,
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
    counter_id = object_to_idx("counter", scope=scope)
    delivery_zone_id = object_to_idx("delivery_zone", scope=scope)

    # Stacks are infinite dispensers: picking up from a stack produces
    # the corresponding loose item, but the stack itself stays on the grid.
    # Built by scanning all components with a `produces` attribute.
    for component_scope in [scope, "global"]:
        for meta in get_all_components(component_scope):
            produces = getattr(meta.cls, "produces", None)
            if produces is not None and meta.properties.get("can_pickup_from", False):
                stack_id = object_to_idx(meta.object_id, scope=scope)
                produced_id = object_to_idx(produces, scope=scope)
                pickup_from_produces = set_at(pickup_from_produces, stack_id, produced_id)

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
        "onion_stack": object_to_idx("onion_stack", scope=scope),
        "tomato_stack": object_to_idx("tomato_stack", scope=scope),
        "plate_stack": object_to_idx("plate_stack", scope=scope),
        "counter": counter_id,
        "delivery_zone": delivery_zone_id,
    }

    return {
        "pickup_from_produces": pickup_from_produces,
        "legal_pot_ingredients": legal_pot_ingredients,
        "type_ids": type_ids,
    }


DEFAULT_RECIPES = [
    {
        "ingredients": ["onion", "onion", "onion"],
        "result": "onion_soup",
        "cook_time": 30,
        "reward": 20.0,
    },
    {
        "ingredients": ["tomato", "tomato", "tomato"],
        "result": "tomato_soup",
        "cook_time": 30,
        "reward": 20.0,
    },
]


def _validate_recipe(recipe, index, scope):
    """Validate a single recipe dict at init time.

    Checks that ``recipe`` is a dict with exactly the keys
    ``{"ingredients", "result", "cook_time", "reward"}`` and that each
    value has the correct type and refers to registered object names.

    Raises ``ValueError`` with a message including *index* for any failure.
    """
    required_keys = {"ingredients", "result", "cook_time", "reward"}
    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe {index}: expected dict, got {type(recipe).__name__}")

    missing = required_keys - set(recipe.keys())
    if missing:
        raise ValueError(f"Recipe {index}: missing keys {missing}")

    extra = set(recipe.keys()) - required_keys
    if extra:
        raise ValueError(f"Recipe {index}: unexpected keys {extra}")

    ingredients = recipe["ingredients"]
    if not isinstance(ingredients, list) or len(ingredients) == 0:
        raise ValueError(f"Recipe {index}: 'ingredients' must be a non-empty list")

    names = get_object_names(scope=scope)
    for ing_name in ingredients:
        if not isinstance(ing_name, str):
            raise ValueError(f"Recipe {index}: ingredient {ing_name!r} must be a string")
        if ing_name not in names:
            raise ValueError(
                f"Recipe {index}: ingredient '{ing_name}' is not a registered object type "
                f"in scope '{scope}'. Registered types: {[n for n in names if n]}"
            )

    result = recipe["result"]
    if not isinstance(result, str) or result not in names:
        raise ValueError(
            f"Recipe {index}: result '{result}' is not a registered object type "
            f"in scope '{scope}'."
        )

    cook_time = recipe["cook_time"]
    if not isinstance(cook_time, int) or cook_time <= 0:
        raise ValueError(
            f"Recipe {index}: 'cook_time' must be a positive integer, got {cook_time}"
        )

    reward = recipe["reward"]
    if not isinstance(reward, (int, float)):
        raise ValueError(
            f"Recipe {index}: 'reward' must be a number, got {type(reward).__name__}"
        )


def compile_recipes(recipe_config, scope="overcooked"):
    """Compile a list of recipe dicts into fixed-shape lookup arrays.

    Called at init time (not under JIT). Produces sentinel-padded arrays
    suitable for storage in ``static_tables``.

    Parameters
    ----------
    recipe_config : list[dict]
        Each dict has keys: ``"ingredients"`` (list[str]), ``"result"`` (str),
        ``"cook_time"`` (int), ``"reward"`` (float).
    scope : str
        Object registry scope for resolving type names to IDs.

    Returns
    -------
    dict with keys:
        recipe_ingredients : (n_recipes, max_ingredients) int32
            Sorted ingredient type IDs per recipe, -1 sentinel for empty slots.
        recipe_result : (n_recipes,) int32
            Output item type ID per recipe.
        recipe_cooking_time : (n_recipes,) int32
            Cook time per recipe.
        recipe_reward : (n_recipes,) float32
            Reward value per recipe.
        legal_pot_ingredients : (n_types,) int32
            1 for any type that appears as an ingredient in any recipe.
        max_ingredients : int
            Maximum number of ingredients across all recipes.

    Raises
    ------
    ValueError
        If ``recipe_config`` is empty, any recipe dict is malformed,
        contains invalid type names, or has duplicate sorted ingredient
        combinations.
    """
    import numpy as _np

    if not recipe_config:
        raise ValueError("recipe_config must be a non-empty list of recipe dicts.")

    names = get_object_names(scope=scope)
    n_types = len(names)

    # Validate all recipes first
    for i, recipe in enumerate(recipe_config):
        _validate_recipe(recipe, i, scope)

    max_ing = max(len(r["ingredients"]) for r in recipe_config)
    n_recipes = len(recipe_config)

    # Build arrays
    recipe_ingredients = _np.full((n_recipes, max_ing), -1, dtype=_np.int32)
    recipe_result = _np.zeros(n_recipes, dtype=_np.int32)
    recipe_cooking_time = _np.zeros(n_recipes, dtype=_np.int32)
    recipe_reward = _np.zeros(n_recipes, dtype=_np.float32)
    legal_pot_ingredients = _np.zeros(n_types, dtype=_np.int32)

    seen_combos = {}
    for i, recipe in enumerate(recipe_config):
        ing_ids = sorted(
            [object_to_idx(name, scope=scope) for name in recipe["ingredients"]]
        )

        # Duplicate combo check
        combo_key = tuple(ing_ids)
        if combo_key in seen_combos:
            raise ValueError(
                f"Recipe {i} has same sorted ingredients as recipe {seen_combos[combo_key]}. "
                f"Duplicate ingredient combinations are not allowed."
            )
        seen_combos[combo_key] = i

        # Fill ingredient slots (sorted), remaining slots stay -1
        for j, tid in enumerate(ing_ids):
            recipe_ingredients[i, j] = tid
            legal_pot_ingredients[tid] = 1

        recipe_result[i] = object_to_idx(recipe["result"], scope=scope)
        recipe_cooking_time[i] = recipe["cook_time"]
        recipe_reward[i] = recipe["reward"]

    return {
        "recipe_ingredients": recipe_ingredients,
        "recipe_result": recipe_result,
        "recipe_cooking_time": recipe_cooking_time,
        "recipe_reward": recipe_reward,
        "legal_pot_ingredients": legal_pot_ingredients,
        "max_ingredients": max_ing,
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

    static_tables = scope_config.get("static_tables", {})
    capacity = static_tables.get("max_ingredients", 3)

    pot_contents, pot_timer, pot_state = overcooked_tick(
        pot_contents, pot_timer, capacity=capacity
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
    return dataclasses.replace(state, object_state_map=osm, extra_state=new_extra)


# ======================================================================
# Interaction branch functions
# ======================================================================
#
# Each function evaluates ONE branch of the interaction decision tree.
# All functions are pure: they take arrays, return arrays. No mutations.
#
# Uniform interface:
#   (handled, ctx) -> (cond, updates, new_handled)
#
# Where:
#   handled    -- bool scalar, True if a prior branch already fired
#   ctx        -- dict of all shared arrays and static tables
#   cond       -- bool scalar, True if this branch fires
#   updates    -- dict with ONLY the arrays this branch modifies, using
#                 keys: "agent_inv", "object_type_map", "object_state_map",
#                 "pot_contents", "pot_timer"
#   new_handled -- handled | cond
#
# Every function computes results unconditionally (no Python if/else)
# so that JAX can trace through all branches. The condition is returned
# alongside the results; the caller uses xp.where to select.
# ======================================================================


def _branch_pickup(handled, ctx):
    """Branch 1: Pick up a loose object from the forward cell.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
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


def _branch_pickup_from_pot(handled, ctx):
    """Branch 2A: Pick up cooked soup from a pot using recipe table lookup.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
        - fwd_type == pot_id: forward cell is a pot
        - has_pot_match: the pot was found in pot_positions
        - pot has contents (at least one slot != -1)
        - pot_timer[pot_idx] == 0: cooking is complete
        - inv_item == plate_id: agent is holding a plate
        - has_recipe_match: pot contents match a known recipe

    Effects when condition is True:
        - agent_inv[agent_idx] = recipe_result[matched_idx]  (plate replaced with recipe output)
        - pot_contents[pot_idx, :] = -1  (pot emptied)
        - pot_timer[pot_idx] = cooking_time  (timer reset)

    Recipe matching: sort pot contents, compare against all recipe_ingredients
    rows. The matched recipe's result item is placed in the agent's inventory.

    Example: Agent holds a plate, pot has 3 cooked onions.

        BEFORE                            AFTER
        +-------+-------+               +-------+-------+
        | Agent |  Pot  |               | Agent |  Pot  |
        | inv=P | ooo,0 |   --->        | inv=S | ---,30|
        +-------+-------+               +-------+-------+
          P=plate  ooo=3 onions            S=onion_soup  ---=empty, timer reset
    """
    pot_idx = ctx["pot_idx"]
    pot_contents = ctx["pot_contents"]

    is_pot = ctx["fwd_type"] == ctx["pot_id"]

    # Check pot state: does it have contents and is cooking complete?
    has_contents = xp.sum(pot_contents[pot_idx] != -1) > 0
    is_ready = ctx["pot_timer"][pot_idx] == 0

    # Recipe matching: sort pot contents and compare against all recipes.
    # Replace -1 sentinels with max int32 before sorting so they sort to
    # the end (matching recipe_ingredients layout: values first, -1 last).
    _SORT_HIGH = xp.int32(2147483647)
    raw = pot_contents[pot_idx]
    sort_buf = xp.where(raw == -1, _SORT_HIGH, raw)
    sorted_pot = xp.where(xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf))
    matches = xp.all(sorted_pot[None, :] == ctx["recipe_ingredients"], axis=1)
    matched_idx = xp.argmax(matches)
    has_recipe_match = xp.any(matches)

    # Output item comes from the matched recipe.
    soup_type = ctx["recipe_result"][matched_idx]

    cond = (
        ~handled
        & ctx["base_ok"]
        & is_pot
        & ctx["has_pot_match"]
        & has_contents
        & is_ready
        & (ctx["inv_item"] == ctx["plate_id"])
        & has_recipe_match
    )

    # If condition fires: replace plate in inventory with recipe output, clear pot.
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), soup_type)
    pc = set_at(pot_contents, (pot_idx, slice(None)), -1)  # clear all slots
    pt = set_at(ctx["pot_timer"], pot_idx, ctx["cooking_time"])  # reset timer
    updates = {"agent_inv": inv, "pot_contents": pc, "pot_timer": pt}
    return cond, updates, handled | cond


def _branch_pickup_from_stack(handled, ctx):
    """Branch 2B: Pick up a produced item from an infinite dispenser stack.

    Stacks are objects with CAN_PICKUP_FROM=1 that produce a different
    item type when picked from (e.g. onion_stack produces onion). The
    stack itself remains on the grid (infinite supply).

    Pots also have CAN_PICKUP_FROM=1 but are handled separately in
    Branch 2A, so we explicitly exclude pots here.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
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
    fwd_type = ctx["fwd_type"]
    # Exclude pots (they have CAN_PICKUP_FROM=1 but use Branch 2A)
    is_stack = ~(fwd_type == ctx["pot_id"]) & (ctx["CAN_PICKUP_FROM"][fwd_type] == 1)
    produced = ctx["pickup_from_produces"][fwd_type]
    cond = ~handled & ctx["base_ok"] & is_stack & (ctx["inv_item"] == -1) & (produced > 0)
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), produced)
    updates = {"agent_inv": inv}
    return cond, updates, handled | cond


def _branch_drop_on_empty(handled, ctx):
    """Branch 3: Drop held item onto an empty floor cell.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
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
    cond = (
        ~handled
        & ctx["base_ok"]
        & (ctx["fwd_type"] == 0)
        & (ctx["inv_item"] != -1)
    )
    otm = set_at_2d(ctx["object_type_map"], ctx["fwd_r"], ctx["fwd_c"], ctx["inv_item"])
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], 0)
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv, "object_type_map": otm, "object_state_map": osm}
    return cond, updates, handled | cond


def _branch_place_on_pot(handled, ctx):
    """Branch 4A: Place an ingredient into a pot using recipe prefix matching.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
        - fwd_type > 0: forward cell is not empty
        - CAN_PLACE_ON[fwd_type] == 1: forward cell is a place-on target
        - inv_item != -1: agent is holding something
        - fwd_type == pot_id: forward cell is specifically a pot
        - has_pot_match: pot found in pot_positions array
        - legal_pot_ingredients[inv_item] == 1: held item is a legal ingredient
        - n_items_in_pot < max_ingredients: pot has capacity
        - any_recipe_accepts: the would-be pot contents (after adding the
          ingredient) match a prefix of at least one recipe

    Effects when condition is True:
        - pot_contents[pot_idx, first_empty_slot] = inv_item  (ingredient added)
        - agent_inv[agent_idx] = -1  (hand emptied)
        - pot_timer[pot_idx] = recipe cook time  (if pot becomes full)

    When the pot becomes full, an exact recipe match determines the cook
    time. The matched recipe's ``recipe_cooking_time`` is written into
    ``pot_timer`` so that ``overcooked_tick`` uses per-recipe cook times.

    Slot assignment: ``first_empty_slot = argmax(pot_contents[pot_idx] == -1)``
    fills left-to-right (slot 0, then 1, then 2).

    Example: Agent holds an onion, pot already has one onion.

        BEFORE                                  AFTER
        +-------+-----------+                  +-------+-----------+
        | Agent |    Pot    |                  | Agent |    Pot    |
        | inv=o | [o, -, -] |     --->         | inv=- | [o, o, -] |
        +-------+-----------+                  +-------+-----------+
    """
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]
    pot_idx = ctx["pot_idx"]
    pot_contents = ctx["pot_contents"]
    max_ingredients = ctx["max_ingredients"]
    recipe_ingredients = ctx["recipe_ingredients"]
    recipe_cooking_time = ctx["recipe_cooking_time"]

    is_pot = fwd_type == ctx["pot_id"]
    is_legal = ctx["legal_pot_ingredients"][inv_item] == 1
    n_items_in_pot = xp.sum(pot_contents[pot_idx] != -1)
    has_capacity = n_items_in_pot < max_ingredients

    # Find the first empty slot to place the ingredient into.
    slot_empty = pot_contents[pot_idx] == -1
    first_empty_slot = xp.argmax(slot_empty)

    # Build would-be contents: current contents with new item in first empty slot.
    # Replace -1 sentinels with max int32 before sorting so they sort to
    # the end (matching recipe_ingredients layout: values first, -1 last).
    _SORT_HIGH = xp.int32(2147483647)
    would_be = set_at(pot_contents[pot_idx], first_empty_slot, inv_item)
    sort_buf = xp.where(would_be == -1, _SORT_HIGH, would_be)
    sorted_would_be = xp.where(xp.sort(sort_buf) == _SORT_HIGH, xp.int32(-1), xp.sort(sort_buf))
    n_would_be = n_items_in_pot + 1

    # Prefix match: compare filled slots against each recipe.
    # For positions beyond n_would_be, treat as "don't care" (always True).
    slot_mask = xp.arange(max_ingredients) < n_would_be
    slot_matches = (sorted_would_be[None, :] == recipe_ingredients) | ~slot_mask[None, :]
    recipe_compatible = xp.all(slot_matches, axis=1)
    any_recipe_accepts = xp.any(recipe_compatible)

    cond = (
        ~handled
        & ctx["base_ok"]
        & (fwd_type > 0)
        & (ctx["CAN_PLACE_ON"][fwd_type] == 1)
        & (inv_item != -1)
        & is_pot
        & ctx["has_pot_match"]
        & is_legal
        & has_capacity
        & any_recipe_accepts
    )
    pc = set_at(pot_contents, (pot_idx, first_empty_slot), inv_item)

    # If this placement fills the pot, set cook time from matched recipe.
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


def _branch_place_on_delivery(handled, ctx):
    """Branch 4B: Place any recipe output on the delivery zone.

    Any item marked as deliverable (derived from recipe_result at init
    time via IS_DELIVERABLE) can be delivered. Attempting to deliver a
    non-recipe item (raw ingredients, plates, etc.) is rejected.

    The delivery zone acts as a sink -- the item disappears from the
    agent's inventory. Scoring is handled by the reward function, not here.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
        - fwd_type > 0: forward cell is not empty
        - CAN_PLACE_ON[fwd_type] == 1: forward cell is a place-on target
        - inv_item != -1: agent is holding something
        - fwd_type == delivery_zone_id: forward cell is a delivery zone
        - IS_DELIVERABLE[inv_item] == 1: held item is a recipe output

    Effects when condition is True:
        - agent_inv[agent_idx] = -1  (item consumed / delivered)

    Example: Agent delivers onion soup.

        BEFORE                        AFTER
        +-------+-------+            +-------+-------+
        | Agent |   @   |            | Agent |   @   |
        | inv=S |  D.Z. |   --->     | inv=- |  D.Z. |
        +-------+-------+            +-------+-------+
    """
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]

    is_dz = fwd_type == ctx["delivery_zone_id"]
    is_deliverable = ctx["IS_DELIVERABLE"][inv_item] == 1
    cond = (
        ~handled
        & ctx["base_ok"]
        & (fwd_type > 0)
        & (ctx["CAN_PLACE_ON"][fwd_type] == 1)
        & (inv_item != -1)
        & is_dz
        & is_deliverable
    )
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv}
    return cond, updates, handled | cond


def _branch_place_on_counter(handled, ctx):
    """Branch 4C: Place a held item on a counter (generic place-on target).

    Counters store placed items in ``object_state_map`` rather than
    ``object_type_map`` -- the counter object itself stays in the type
    map, and the placed item's type ID is written into the state map.
    This means a counter can hold at most one item (state != 0 means
    occupied).

    This branch is the catch-all for place-on targets that are NOT
    pots and NOT delivery zones.

    Preconditions (all must be True):
        - ~handled: no prior branch has fired
        - base_ok: agent is interacting and no agent ahead
        - fwd_type > 0: forward cell is not empty
        - CAN_PLACE_ON[fwd_type] == 1: forward cell is a place-on target
        - inv_item != -1: agent is holding something
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
    fwd_type = ctx["fwd_type"]
    inv_item = ctx["inv_item"]

    # Exclude pots and delivery zones (they have their own branches)
    is_generic = ~(fwd_type == ctx["pot_id"]) & ~(fwd_type == ctx["delivery_zone_id"])
    counter_empty = ctx["object_state_map"][ctx["fwd_r"], ctx["fwd_c"]] == 0
    cond = (
        ~handled
        & ctx["base_ok"]
        & (fwd_type > 0)
        & (ctx["CAN_PLACE_ON"][fwd_type] == 1)
        & (inv_item != -1)
        & is_generic
        & counter_empty
    )
    osm = set_at_2d(ctx["object_state_map"], ctx["fwd_r"], ctx["fwd_c"], inv_item)
    inv = set_at(ctx["agent_inv"], (ctx["agent_idx"], 0), -1)
    updates = {"agent_inv": inv, "object_state_map": osm}
    return cond, updates, handled | cond


# ======================================================================
# Branch list (priority order)
# ======================================================================

_BRANCHES = [
    _branch_pickup,
    _branch_pickup_from_pot,
    _branch_pickup_from_stack,
    _branch_drop_on_empty,
    _branch_place_on_pot,
    _branch_place_on_delivery,
    _branch_place_on_counter,
]


# ======================================================================
# Interaction body (orchestrator)
# ======================================================================


def overcooked_interaction_body(
    agent_idx,  # int: which agent is interacting (0-based)
    agent_inv,  # (n_agents, 1) int32: all agent inventories
    object_type_map,  # (H, W) int32: object type IDs on the grid
    object_state_map,  # (H, W) int32: object state values on the grid
    fwd_r,
    fwd_c,  # scalar int arrays: forward cell coordinates
    fwd_type,  # scalar int array: object type at forward cell
    inv_item,  # scalar int array: what this agent is holding (-1 = empty)
    base_ok,  # bool scalar: True if agent can interact (PickupDrop + no agent ahead)
    pot_contents,  # (n_pots, 3) int32: ingredient type IDs per pot slot
    pot_timer,  # (n_pots,) int32: cooking countdown per pot
    pot_positions,  # (n_pots, 2) int32: (row, col) of each pot
    static_tables,  # dict: pre-built lookup arrays (see _build_static_tables)
):
    """Evaluate all seven interaction branches for one agent and merge results.

    This is the core dispatch function. It:
        1. Unpacks static lookup tables (type IDs, property arrays)
        2. Resolves which pot (if any) the agent is facing
        3. Assembles a shared context dict (``ctx``) with all arrays and tables
        4. Iterates ``_BRANCHES`` in priority order, accumulating a ``handled``
           scalar that suppresses later branches once one fires
        5. Merges branch results via sparse ``xp.where`` over the 5 output arrays

    Each branch receives ``(handled, ctx)`` and returns
    ``(cond, updates_dict, new_handled)``. The ``~handled`` guard in each
    branch's condition ensures mutual exclusion: at most one branch fires.

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

    Returns:
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

    # --- Assemble shared context ---
    ctx = {
        "base_ok": base_ok,
        "fwd_type": fwd_type,
        "fwd_r": fwd_r,
        "fwd_c": fwd_c,
        "inv_item": inv_item,
        "agent_idx": agent_idx,
        "agent_inv": agent_inv,
        "object_type_map": object_type_map,
        "object_state_map": object_state_map,
        "pot_contents": pot_contents,
        "pot_timer": pot_timer,
        "pot_idx": pot_idx,
        "has_pot_match": has_pot_match,
        "CAN_PICKUP": CAN_PICKUP,
        "CAN_PICKUP_FROM": CAN_PICKUP_FROM,
        "CAN_PLACE_ON": CAN_PLACE_ON,
        "pickup_from_produces": pickup_from_produces,
        "legal_pot_ingredients": legal_pot_ingredients,
        "pot_id": pot_id,
        "plate_id": plate_id,
        "tomato_id": tomato_id,
        "onion_soup_id": onion_soup_id,
        "tomato_soup_id": tomato_soup_id,
        "delivery_zone_id": delivery_zone_id,
        "cooking_time": cooking_time,
        "recipe_ingredients": static_tables["recipe_ingredients"],
        "recipe_result": static_tables["recipe_result"],
        "recipe_cooking_time": static_tables["recipe_cooking_time"],
        "max_ingredients": static_tables["max_ingredients"],
        "IS_DELIVERABLE": static_tables["IS_DELIVERABLE"],
    }

    # --- Evaluate all branches with accumulated handled ---
    handled = xp.bool_(False)
    branch_results = []
    for branch_fn in _BRANCHES:
        cond, updates, handled = branch_fn(handled, ctx)
        branch_results.append((cond, updates))

    # --- Merge results using sparse xp.where ---
    inv = agent_inv
    otm = object_type_map
    osm = object_state_map
    pc = pot_contents
    pt = pot_timer

    for cond, updates in branch_results:
        if "agent_inv" in updates:
            inv = xp.where(cond, updates["agent_inv"], inv)
        if "object_type_map" in updates:
            otm = xp.where(cond, updates["object_type_map"], otm)
        if "object_state_map" in updates:
            osm = xp.where(cond, updates["object_state_map"], osm)
        if "pot_contents" in updates:
            pc = xp.where(cond, updates["pot_contents"], pc)
        if "pot_timer" in updates:
            pt = xp.where(cond, updates["pot_timer"], pt)

    return inv, otm, osm, pc, pt


# ======================================================================
# Init-time helpers (not step-path, int() casts here are acceptable)
# ======================================================================


def _build_static_tables(scope, itables, type_ids, recipe_tables=None):
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

    Parameters
    ----------
    scope : str
        Object registry scope.
    itables : dict
        From ``_build_interaction_tables``.
    type_ids : dict
        From ``_build_type_ids``.
    recipe_tables : dict or None
        If provided (from ``compile_recipes``), recipe arrays are merged
        into the returned dict and ``legal_pot_ingredients`` is taken from
        the recipe compilation instead of the hardcoded interaction tables.
    """
    from cogrid.core.grid_object import build_lookup_tables

    lookup = build_lookup_tables(scope=scope)

    tables = {
        "CAN_PICKUP": lookup["CAN_PICKUP"],
        "CAN_PICKUP_FROM": lookup["CAN_PICKUP_FROM"],
        "CAN_PLACE_ON": lookup["CAN_PLACE_ON"],
        "pickup_from_produces": itables["pickup_from_produces"],
        "legal_pot_ingredients": (
            recipe_tables["legal_pot_ingredients"]
            if recipe_tables is not None
            else itables["legal_pot_ingredients"]
        ),
        "pot_id": int(type_ids.get("pot", -1)),
        "plate_id": int(type_ids.get("plate", -1)),
        "tomato_id": int(type_ids.get("tomato", -1)),
        "onion_soup_id": int(type_ids.get("onion_soup", -1)),
        "tomato_soup_id": int(type_ids.get("tomato_soup", -1)),
        "delivery_zone_id": int(type_ids.get("delivery_zone", -1)),
        "cooking_time": 30,
    }

    # Add recipe tables if provided
    if recipe_tables is not None:
        tables["recipe_ingredients"] = recipe_tables["recipe_ingredients"]
        tables["recipe_result"] = recipe_tables["recipe_result"]
        tables["recipe_cooking_time"] = recipe_tables["recipe_cooking_time"]
        tables["recipe_reward"] = recipe_tables["recipe_reward"]
        tables["max_ingredients"] = recipe_tables["max_ingredients"]

    # Build IS_DELIVERABLE array: 1 for any type that is a recipe output
    import numpy as _np

    n_types = len(get_object_names(scope=scope))
    is_deliverable = _np.zeros(n_types, dtype=_np.int32)
    if recipe_tables is not None:
        for result_id in recipe_tables["recipe_result"]:
            is_deliverable[int(result_id)] = 1
    else:
        # Backward compat fallback: hardcode onion_soup and tomato_soup
        is_deliverable[int(type_ids.get("onion_soup", -1))] = 1
        is_deliverable[int(type_ids.get("tomato_soup", -1))] = 1
        tables["max_ingredients"] = 3
    tables["IS_DELIVERABLE"] = is_deliverable

    return tables
