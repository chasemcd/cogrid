"""Grid object registration and lookup machinery.

This module contains the OBJECT_REGISTRY, type property metadata, and all
functions for registering, looking up, encoding, and decoding grid objects.
"""

import numpy as np

from cogrid.backend import xp
from cogrid.core.grid_object_base import GridObj

# The OBJECT_REGISTRY holds all registered objects under a "scope"
# (e.g., "global", "search_rescue", "overcooked") which allows re-using
# IDs and characters across environments.
OBJECT_REGISTRY: dict[str, dict[str, GridObj]] = {}

# Maps (scope, object_id) -> dict of static properties for lookup table generation.
# Values are True, False, or When instances. Populated by @register_object_type.
_OBJECT_TYPE_PROPERTIES: dict[tuple[str, str], dict] = {}

# Known classmethod names to scan for during component registration.
_COMPONENT_METHODS = frozenset(
    {
        "build_tick_fn",
        "extra_state_schema",
        "extra_state_builder",
        "build_static_tables",
        "build_render_sync_fn",
    }
)


def make_object(object_id: str | None, scope: str = "global", **kwargs) -> GridObj:
    """Create a GridObj instance by object_id, checking global then scope."""
    if object_id is None:
        return None

    if scope not in OBJECT_REGISTRY:
        raise ValueError(
            f"No objects registered with scope `{scope}`. "
            f"Existing scopes are {list(OBJECT_REGISTRY.keys())}."
        )

    if object_id in OBJECT_REGISTRY["global"]:
        return OBJECT_REGISTRY["global"][object_id](**kwargs)
    elif object_id not in OBJECT_REGISTRY[scope]:
        raise ValueError(
            f"Object with object_id `{object_id}` not registered "
            f"in scope `{scope}`. Call register_object() to add it."
        )

    return OBJECT_REGISTRY[scope][object_id](**kwargs)


def get_object_class(object_id: str, scope: str = "global") -> GridObj:
    """Return the class registered for the given object_id."""
    return OBJECT_REGISTRY[scope][object_id]


def register_object(object_id: str, obj_class: GridObj, scope: str = "global") -> None:
    """Register an object class under the given scope."""
    global_scope_chars = [obj.char for obj in OBJECT_REGISTRY.get("global", {}).values()]

    if obj_class.char in global_scope_chars:
        raise ValueError(
            f"Character `{obj_class.char}` is already in use in the global scope. "
            "Please choose a different character."
        )

    if object_id in OBJECT_REGISTRY.get("global", {}):
        raise ValueError(
            f"Object with object_id `{object_id}` already registered in the global scope. "
            "Please select a different ID."
        )

    if scope not in OBJECT_REGISTRY:
        OBJECT_REGISTRY[scope] = {}

    OBJECT_REGISTRY[scope][object_id] = obj_class


_CAPABILITY_ATTRS = frozenset(
    {"can_pickup", "can_overlap", "can_place_on", "can_pickup_from", "is_wall", "consumes_on_place"}
)


def register_object_type(
    object_id: str,
    scope: str = "global",
):
    """Register a GridObj subclass with static property metadata.

    Stores boolean properties for ``build_lookup_tables()`` and
    auto-discovers component classmethods (tick, interaction, etc.).

    Capabilities are declared as class attributes using :func:`when`
    descriptors or plain ``True``::

        @register_object_type("onion", scope="overcooked")
        class Onion(GridObj):
            can_pickup = when()

        @register_object_type("wall")
        class Wall(GridObj):
            is_wall = True
    """

    def decorator(cls):
        # Lazy import to avoid circular dependency
        from cogrid.core.component_registry import (
            _validate_classmethod_signature,
            get_all_components,
            register_component_metadata,
        )
        from cogrid.core.containers import Container
        from cogrid.core.when import When, when

        # --- Auto-generate when() from Container + Recipe descriptors ---
        container = getattr(cls, "container", None)
        recipes = getattr(cls, "recipes", None)

        if isinstance(container, Container):
            # Auto-generate can_place_on from recipe ingredients (if not explicit)
            if "can_place_on" not in cls.__dict__ and recipes:
                all_ingredients = sorted(
                    {ing for r in recipes for ing in r.ingredients}
                )
                cls.can_place_on = when(agent_holding=all_ingredients)

            # Auto-generate can_pickup_from from pickup_requires (if not explicit)
            if "can_pickup_from" not in cls.__dict__:
                if container.pickup_requires is not None:
                    cls.can_pickup_from = when(agent_holding=container.pickup_requires)
                else:
                    cls.can_pickup_from = when()

        # Scan class for capability attributes (When instances or plain bool True)
        properties = {}
        for attr in _CAPABILITY_ATTRS:
            val = getattr(cls, attr, None)
            if isinstance(val, When):
                properties[attr] = val  # preserve When for guard table generation
            elif val is True:
                properties[attr] = True
            else:
                properties[attr] = False

        # Store static properties for lookup table generation
        _OBJECT_TYPE_PROPERTIES[(scope, object_id)] = properties

        # Duplicate char detection within the same scope
        for existing in get_all_components(scope):
            if existing.char == cls.char:
                raise ValueError(
                    f"Duplicate char '{cls.char}' in scope '{scope}': "
                    f"{existing.cls.__name__} and {cls.__name__}"
                )

        # Set object_id on the class
        cls.object_id = object_id

        # Delegate to existing register_object for backward compatibility
        register_object(object_id, cls, scope=scope)

        # Convention-based classmethod scan
        discovered: dict = {}
        for method_name in _COMPONENT_METHODS:
            method = getattr(cls, method_name, None)
            if method is not None and callable(method):
                _validate_classmethod_signature(cls, method_name, method)
                discovered[method_name] = method

        # Build container metadata for autowire
        container_meta = None
        if isinstance(container, Container):
            container_meta = {
                "container": container,
                "recipes": list(recipes) if recipes else [],
            }

        # Store component metadata in the registry
        register_component_metadata(
            scope=scope,
            object_id=object_id,
            cls=cls,
            properties=properties,
            methods=discovered,
            container_meta=container_meta,
        )

        return cls

    return decorator


def build_lookup_tables(scope: str = "global") -> dict[str, np.ndarray]:
    """Build per-type boolean property arrays (CAN_PICKUP, CAN_OVERLAP, etc.).

    Returns ``(n_types,)`` int32 arrays indexed by the integer encoding
    from ``object_to_idx()``.
    """
    from cogrid.backend.array_ops import set_at

    type_names = get_object_names(scope=scope)
    n_types = len(type_names)

    property_keys = [
        "CAN_PICKUP",
        "CAN_OVERLAP",
        "CAN_PLACE_ON",
        "CAN_PICKUP_FROM",
        "IS_WALL",
    ]

    tables = {key: xp.zeros(n_types, dtype=xp.int32) for key in property_keys}

    for idx, name in enumerate(type_names):
        if name is None:
            # Index 0: empty cell -- overlappable
            tables["CAN_OVERLAP"] = set_at(tables["CAN_OVERLAP"], idx, 1)
            continue

        if name == "free_space":
            # Index 1: free_space -- overlappable (not in OBJECT_REGISTRY, hardcoded)
            tables["CAN_OVERLAP"] = set_at(tables["CAN_OVERLAP"], idx, 1)
            continue

        if name.startswith("agent_"):
            # Agent direction placeholders -- skip, leave all-zero
            continue

        # Look up properties: try (scope, name) first, then ("global", name)
        props = _OBJECT_TYPE_PROPERTIES.get((scope, name))
        if props is None:
            props = _OBJECT_TYPE_PROPERTIES.get(("global", name))

        if props is None:
            # Object registered via old register_object() without decorator.
            # Default to all-False properties.
            import warnings

            warnings.warn(
                f"Object '{name}' in scope '{scope}' has no static properties "
                f"(not registered via @register_object_type). "
                f"Defaulting to all-False in lookup tables.",
                stacklevel=2,
            )
            continue

        prop_map = {
            "can_pickup": "CAN_PICKUP",
            "can_overlap": "CAN_OVERLAP",
            "can_place_on": "CAN_PLACE_ON",
            "can_pickup_from": "CAN_PICKUP_FROM",
            "is_wall": "IS_WALL",
        }

        for prop_name, table_key in prop_map.items():
            if props.get(prop_name, False):
                tables[table_key] = set_at(tables[table_key], idx, 1)

    # Build pickup_from_produces: maps stack/dispenser type -> produced type.
    # Scans all components with a `produces` class attribute AND can_pickup_from=True.
    pickup_from_produces = xp.zeros(n_types, dtype=xp.int32)
    from cogrid.core.component_registry import get_all_components

    for component_scope in [scope, "global"]:
        for meta in get_all_components(component_scope):
            produced_name = getattr(meta.cls, "produces", None)
            if produced_name is not None and meta.properties.get("can_pickup_from", False):
                stack_idx = object_to_idx(meta.object_id, scope=scope)
                if produced_name in type_names:
                    produced_idx = type_names.index(produced_name)
                    pickup_from_produces = set_at(pickup_from_produces, stack_idx, produced_idx)
    tables["pickup_from_produces"] = pickup_from_produces

    return tables


def build_guard_tables(scope: str = "global") -> dict[str, np.ndarray]:
    """Build 2D guard tables for pickup_from and place_on interactions.

    Returns ``(n_types, n_types + 1)`` int32 arrays where:
    - Row = forward cell type ID
    - Column 0 = agent has empty hands (inv_item == -1)
    - Columns 1..n = agent holds type ID ``col - 1``

    A cell value of 1 means the interaction is allowed for that
    (forward_type, held_item) pair.

    When conditions determine which columns are set:
    - Bare ``when()`` (no conditions): pickup_from defaults to column 0
      (empty hands); place_on defaults to columns 1..n (any held item).
    - ``when(agent_holding="plate")``: sets only the plate column.
    - ``when(agent_holding=["onion", "tomato"])``: sets those columns.
    """
    from cogrid.core.when import When

    type_names = get_object_names(scope=scope)
    n_types = len(type_names)
    n_cols = n_types + 1  # col 0 = empty hands, cols 1..n = held type IDs

    pickup_from_guard = np.zeros((n_types, n_cols), dtype=np.int32)
    place_on_guard = np.zeros((n_types, n_cols), dtype=np.int32)

    for idx, name in enumerate(type_names):
        if name is None or name == "free_space" or name.startswith("agent_"):
            continue

        props = _OBJECT_TYPE_PROPERTIES.get((scope, name))
        if props is None:
            props = _OBJECT_TYPE_PROPERTIES.get(("global", name))
        if props is None:
            continue

        # --- can_pickup_from ---
        pf_val = props.get("can_pickup_from", False)
        if pf_val is not False and pf_val is not None:
            if isinstance(pf_val, When) and pf_val.has_conditions:
                for held_name in pf_val.conditions["agent_holding"]:
                    if held_name not in type_names:
                        raise ValueError(
                            f"when(agent_holding={held_name!r}) on {name!r}: "
                            f"{held_name!r} is not a registered type in scope {scope!r}"
                        )
                    col = type_names.index(held_name) + 1
                    pickup_from_guard[idx, col] = 1
            else:
                # Bare when() or True: default = empty hands only (col 0)
                pickup_from_guard[idx, 0] = 1

        # --- can_place_on ---
        po_val = props.get("can_place_on", False)
        if po_val is not False and po_val is not None:
            if isinstance(po_val, When) and po_val.has_conditions:
                for held_name in po_val.conditions["agent_holding"]:
                    if held_name not in type_names:
                        raise ValueError(
                            f"when(agent_holding={held_name!r}) on {name!r}: "
                            f"{held_name!r} is not a registered type in scope {scope!r}"
                        )
                    col = type_names.index(held_name) + 1
                    place_on_guard[idx, col] = 1
            else:
                # Bare when() or True: default = any held item (cols 1..n)
                place_on_guard[idx, 1:] = 1

    return {
        "PICKUP_FROM_GUARD": pickup_from_guard,
        "PLACE_ON_GUARD": place_on_guard,
    }


def get_registered_object_ids(scope: str = "global") -> list[str]:
    """Return a list of the object_ids of available objects in a given scope."""
    return list(OBJECT_REGISTRY[scope].keys())


def get_object_char(object_id: str, scope: str = "global") -> str:
    """Return the character representation of an object."""
    return get_object_class(object_id, scope=scope).char


def get_object_id_from_char(object_char: str, scope: str = "global") -> str:
    """Look up an object_id from its character, checking global first."""
    # First check global scope, no matter what scope was passed (default to global).
    for object_id, object_class in OBJECT_REGISTRY["global"].items():
        if object_class.char == object_char:
            return object_id

    if scope != "global":
        for object_id, object_class in OBJECT_REGISTRY[scope].items():
            if object_class.char == object_char:
                return object_id

    raise ValueError(f"No registered object with char `{object_char}` in scope `{scope}`.")


def get_object_names(scope: str = "global") -> list[str]:
    """Return all registered object IDs in stable encoding order.

    Order: [None, "free_space", sorted globals, sorted scope, agent directions].
    """
    # Start with None and free_space which are special cases
    names = [None, "free_space"]

    # Add all registered global objects in sorted order (except free_space which we already added)
    global_objects = sorted(
        [obj_id for obj_id in OBJECT_REGISTRY.get("global", {}).keys() if obj_id != "free_space"]
    )
    names.extend(global_objects)

    # Add scope-specific objects if a non-global scope is specified
    if scope != "global" and scope in OBJECT_REGISTRY:
        scope_objects = sorted(
            [
                obj_id
                for obj_id in OBJECT_REGISTRY[scope].keys()
                # Skip any objects that might overlap with global scope
                if obj_id not in OBJECT_REGISTRY.get("global", {})
            ]
        )
        names.extend(scope_objects)

    # Add agent directions last
    names.extend([f"agent_{direction}" for direction in "^>v<"])

    return names


def object_to_idx(object: GridObj | str | None, scope: str = "global") -> int:
    """Convert an object or object_id to its integer index."""
    if isinstance(object, GridObj):
        object_id = object.object_id
    else:
        object_id = object

    return get_object_names(scope=scope).index(object_id)


def idx_to_object(idx: int, scope: str = "global") -> str:
    """Convert an integer index back to its object_id."""
    names = get_object_names(scope=scope)
    if idx >= len(names):
        raise ValueError(
            f"Object index {idx} not in object registry (checked global and {scope} scopes)."
        )
    return names[idx]
