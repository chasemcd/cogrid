"""Declarative container system for stateful grid objects.

Provides the ``Container`` dataclass and auto-generation helpers for
extra_state schemas, builders, tick functions, and render syncs.

Environment-specific recipe definitions live in their respective
modules (e.g. ``cogrid.envs.overcooked.recipes``).

Usage::

    @register_object_type("pot", scope="overcooked")
    class Pot(GridObj):
        container = Container(capacity=3, pickup_requires="plate")
        recipes = [...]  # environment-specific Recipe objects

The autowire system reads the ``container`` descriptor and auto-generates
extra_state schemas, builders, tick functions, render syncs, and
interaction branches.
"""

from __future__ import annotations

from dataclasses import dataclass

# ======================================================================
# Declarative data classes
# ======================================================================


@dataclass(frozen=True)
class Container:
    """Declares that a GridObj is a stateful container (e.g. a cooking pot).

    Parameters
    ----------
    capacity : int
        Maximum number of items the container can hold.
    pickup_requires : str | list[str] | None
        Object type(s) the agent must be holding to pick up from this
        container.  ``None`` means empty hands.
    """

    capacity: int
    pickup_requires: str | list[str] | None = None


# ======================================================================
# Auto-generation helpers (called by autowire)
# ======================================================================


def build_container_extra_state_schema(object_id: str, container: Container) -> dict:
    """Return extra_state schema entries for a container type."""
    return {
        f"{object_id}_contents": {
            "shape": (f"n_{object_id}s", container.capacity),
            "dtype": "int32",
        },
        f"{object_id}_timer": {"shape": (f"n_{object_id}s",), "dtype": "int32"},
        f"{object_id}_positions": {"shape": (f"n_{object_id}s", 2), "dtype": "int32"},
    }


def build_container_extra_state_builder(
    object_id: str,
    container: Container,
    scope: str,
    default_timer: int = 0,
) -> callable:
    """Return an extra_state builder closure for a container type."""

    def builder(parsed_arrays, scope=scope):
        import numpy as _np

        from cogrid.core.objects.registry import object_to_idx

        type_id = object_to_idx(object_id, scope=scope)
        otm = parsed_arrays["object_type_map"]
        mask = otm == type_id
        positions_list = list(zip(*_np.where(mask)))
        n_instances = len(positions_list)

        prefix = f"{scope}."
        if n_instances > 0:
            positions = _np.array(positions_list, dtype=_np.int32)
            contents = _np.full((n_instances, container.capacity), -1, dtype=_np.int32)
            timer = _np.full((n_instances,), default_timer, dtype=_np.int32)
        else:
            positions = _np.zeros((0, 2), dtype=_np.int32)
            contents = _np.full((0, container.capacity), -1, dtype=_np.int32)
            timer = _np.zeros((0,), dtype=_np.int32)

        return {
            f"{prefix}{object_id}_contents": contents,
            f"{prefix}{object_id}_timer": timer,
            f"{prefix}{object_id}_positions": positions,
        }

    return builder


def build_container_tick_fn(
    object_id: str,
    container: Container,
    scope: str,
) -> callable:
    """Return a tick function that decrements cooking timers."""

    def tick_fn(state, scope_config):
        import dataclasses

        from cogrid.backend import xp
        from cogrid.backend.array_ops import set_at_2d

        prefix = f"{scope}."
        contents = state.extra_state[f"{prefix}{object_id}_contents"]
        timer = state.extra_state[f"{prefix}{object_id}_timer"]
        positions = state.extra_state[f"{prefix}{object_id}_positions"]
        n_instances = positions.shape[0]
        capacity = container.capacity

        # Tick: decrement timer when full and timer > 0
        n_items = xp.sum(contents != -1, axis=1).astype(xp.int32)
        is_cooking = (n_items == capacity) & (timer > 0)
        new_timer = xp.where(is_cooking, timer - 1, timer)

        # Compute state encoding for object_state_map
        pot_state = (n_items + n_items * new_timer).astype(xp.int32)

        osm = state.object_state_map
        for p in range(n_instances):
            osm = set_at_2d(osm, positions[p, 0], positions[p, 1], pot_state[p])

        new_extra = {
            **state.extra_state,
            f"{prefix}{object_id}_contents": contents,
            f"{prefix}{object_id}_timer": new_timer,
        }

        return dataclasses.replace(state, object_state_map=osm, extra_state=new_extra)

    return tick_fn


def build_container_render_sync(object_id: str, scope: str) -> callable:
    """Return a render_sync callback for a container type."""

    def render_sync(grid, env_state, scope=scope):
        import numpy as np

        from cogrid.core.objects.registry import idx_to_object, make_object

        extra = env_state.extra_state
        prefix = f"{scope}."
        contents_key = f"{prefix}{object_id}_contents"
        timer_key = f"{prefix}{object_id}_timer"
        positions_key = f"{prefix}{object_id}_positions"

        if not all(k in extra for k in (contents_key, timer_key, positions_key)):
            return

        contents = np.array(extra[contents_key])
        timer = np.array(extra[timer_key])
        positions = np.array(extra[positions_key])

        for p in range(len(positions)):
            pr, pc = int(positions[p, 0]), int(positions[p, 1])
            obj = grid.get(pr, pc)
            if obj is not None and obj.object_id == object_id:
                # Sync contents list
                obj.objects_in_pot = []
                for slot in range(contents.shape[1]):
                    item_id = int(contents[p, slot])
                    if item_id > 0:
                        item_name = idx_to_object(item_id, scope=scope)
                        if item_name:
                            obj.objects_in_pot.append(make_object(item_name, scope=scope))
                # Sync timer
                obj.cooking_timer = int(timer[p])

    return render_sync
