"""Interaction context and helpers for branch functions.

:class:`InteractionContext` is a frozen dataclass passed to interaction
functions (branches). It contains everything the branch needs to decide
whether to fire and what to change.

Standard fields are auto-populated by the pipeline. Extra-state arrays
are auto-read from ``EnvState.extra_state`` with the scope prefix
stripped (e.g., ``state.extra_state["global.goals_collected"]`` becomes
``ctx.goals_collected``).

Helpers provide intent-level operations for common actions:

- :func:`clear_facing_cell` -- set the faced cell to empty
- :func:`set_facing_cell` -- set the faced cell to a type
- :func:`pickup_from_facing_cell` -- pick up the faced object
- :func:`place_in_facing_cell` -- place held item in the faced cell
- :func:`give_item` -- put an item in the agent's inventory
- :func:`empty_hands` -- clear the agent's inventory
- :func:`increment` -- increment an array element by 1
- :func:`find_facing_instance` -- which instance of a multi-position object
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cogrid.backend import xp
from cogrid.backend.array_ops import set_at, set_at_2d

if TYPE_CHECKING:
    from cogrid.core.typing import ArrayLike

_interaction_context_pytree_registered: bool = False


@dataclass(frozen=True)
class InteractionContext:
    """What the agent sees when it interacts.

    Always available (auto-populated by the pipeline):

        can_interact      bool -- agent performed PickupDrop or Toggle and cell ahead is clear
        action            int  -- raw action index this agent chose this step
        action_id         ActionID -- indices for all actions in this env
        facing_row        int  -- row of the cell this agent faces
        facing_col        int  -- col of the cell this agent faces
        facing_type       int  -- type ID of object in faced cell (0 = empty)
        agent_index       int  -- which agent (0, 1, ...)
        held_item         int  -- type ID of item the agent holds (-1 = empty)
        agent_inv         (n_agents, 1) int32 -- all inventories
        object_type_map   (H, W) int32 -- the grid
        object_state_map  (H, W) int32 -- per-cell state
        type_ids          dict -- {"goal": 5, "wall": 1, ...}

    Extra state (auto-populated from EnvState.extra_state):

        Any extra_state key is accessible by name with the scope prefix
        stripped.  ``state.extra_state["global.goals_collected"]`` becomes
        ``ctx.goals_collected``.
    """

    # --- Per-agent situation ---
    can_interact: object
    action: object
    action_id: object
    facing_row: object
    facing_col: object
    facing_type: object
    agent_index: object
    held_item: object

    # --- Grid state arrays ---
    agent_inv: object
    object_type_map: object
    object_state_map: object

    # --- Type IDs (always available) ---
    type_ids: dict = field(default_factory=dict)

    # --- Environment-specific extras (auto-populated) ---
    extra: dict = field(default_factory=dict)

    # --- Internal lookup tables (for built-in branches) ---
    _tables: dict = field(default_factory=dict, repr=False)

    def __getattr__(self, name):
        """Look up extra-state and internal-table fields by name."""
        try:
            return self.extra[name]
        except KeyError:
            pass
        try:
            return self._tables[name]
        except KeyError:
            pass
        raise AttributeError(
            f"InteractionContext has no field '{name}'. Available extras: {list(self.extra.keys())}"
        )


def build_context(state, agent_idx, fwd_r, fwd_c, base_ok, scope_config, *, action, action_id):
    """Build an InteractionContext from state and per-agent scalars.

    Standard fields are derived automatically.  ``type_ids`` comes from
    ``scope_config["type_ids"]``.  Internal lookup tables come from
    ``scope_config["static_tables"]``.  The ``extra`` dict is left
    empty -- the pipeline populates it from extra_state and autowire.
    """
    return InteractionContext(
        can_interact=base_ok,
        action=action,
        action_id=action_id,
        facing_row=fwd_r,
        facing_col=fwd_c,
        facing_type=state.object_type_map[fwd_r, fwd_c],
        agent_index=agent_idx,
        held_item=state.agent_inv[agent_idx, 0],
        agent_inv=state.agent_inv,
        object_type_map=state.object_type_map,
        object_state_map=state.object_state_map,
        type_ids=scope_config.get("type_ids", {}),
        extra={},
        _tables=scope_config.get("static_tables", {}),
    )


def register_interaction_context_pytree() -> None:
    """Register :class:`InteractionContext` as a JAX pytree (idempotent)."""
    global _interaction_context_pytree_registered
    if _interaction_context_pytree_registered:
        return

    import jax.tree_util

    jax.tree_util.register_dataclass(InteractionContext)
    _interaction_context_pytree_registered = True


# ---------------------------------------------------------------------------
# Helpers -- intent-level operations for common branch actions
# ---------------------------------------------------------------------------


def clear_facing_cell(ctx):
    """Return ``object_type_map`` with the faced cell set to empty (0)."""
    return set_at_2d(ctx.object_type_map, ctx.facing_row, ctx.facing_col, 0)


def set_facing_cell(ctx, type_id):
    """Return ``object_type_map`` with the faced cell set to *type_id*."""
    return set_at_2d(ctx.object_type_map, ctx.facing_row, ctx.facing_col, type_id)


def pickup_from_facing_cell(ctx):
    """Pick up the faced object.

    Returns ``(object_type_map, agent_inv)`` with the cell cleared and
    the object in the agent's inventory.
    """
    new_otm = set_at_2d(ctx.object_type_map, ctx.facing_row, ctx.facing_col, 0)
    new_inv = set_at(ctx.agent_inv, (ctx.agent_index, 0), ctx.facing_type)
    return new_otm, new_inv


def place_in_facing_cell(ctx):
    """Place the held item in the faced cell.

    Returns ``(object_type_map, agent_inv)`` with the item on the grid
    and the agent's hands empty.
    """
    new_otm = set_at_2d(ctx.object_type_map, ctx.facing_row, ctx.facing_col, ctx.held_item)
    new_inv = set_at(ctx.agent_inv, (ctx.agent_index, 0), -1)
    return new_otm, new_inv


def give_item(ctx, type_id):
    """Return ``agent_inv`` with this agent holding *type_id*."""
    return set_at(ctx.agent_inv, (ctx.agent_index, 0), type_id)


def empty_hands(ctx):
    """Return ``agent_inv`` with this agent holding nothing."""
    return set_at(ctx.agent_inv, (ctx.agent_index, 0), -1)


def increment(array, index):
    """Return *array* with ``array[index] += 1``.

    Works on both JAX arrays (``.at[].add``) and numpy arrays.
    """
    if hasattr(array, "at"):
        return array.at[index].add(1)
    import numpy as np

    out = np.array(array)
    out[index] += 1
    return out


def find_facing_instance(positions: ArrayLike, facing_row: int, facing_col: int) -> tuple:
    """Find which instance of a multi-position object the agent faces.

    Args:
        positions: ``(n_instances, 2)`` int32 array of positions.
        facing_row: Row the agent faces.
        facing_col: Column the agent faces.

    Returns:
        ``(index, is_match)`` -- index into *positions*, and whether
            any instance matched.
    """
    fwd = xp.stack([facing_row, facing_col])
    match = xp.all(positions == fwd[None, :], axis=1)
    return xp.argmax(match), xp.any(match)
