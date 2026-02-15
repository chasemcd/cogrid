"""Grid object registration and lookup machinery.

This module contains the OBJECT_REGISTRY, type property metadata, and all
functions for registering, looking up, encoding, and decoding grid objects.
"""

import numpy as np

from cogrid.backend import xp
from cogrid.core.grid_object_base import GridObj


# The OBJECT_REGISTRY holds all registered objects under a "scope" (e.g., "global", "search_rescue", "overcooked")
# which allows us to re-use IDs and character representations across environments (e.g., P is a purple target in
# search_rescue and a plate in overcooked).
OBJECT_REGISTRY: dict[str, dict[str, GridObj]] = {}

# Maps (scope, object_id) -> dict of static boolean properties for lookup table generation.
# Populated by the @register_object_type decorator.
_OBJECT_TYPE_PROPERTIES: dict[tuple[str, str], dict[str, bool]] = {}

# Known classmethod names to scan for during component registration.
_COMPONENT_METHODS = frozenset({
    "build_tick_fn",
    "build_interaction_fn",
    "extra_state_schema",
    "extra_state_builder",
    "build_static_tables",
    "build_render_sync_fn",
})


def make_object(
    object_id: str | None, scope: str = "global", **kwargs
) -> GridObj:
    if object_id is None:
        return None

    if scope not in OBJECT_REGISTRY:
        raise ValueError(
            f"No objects registered with scope `{scope}`. Existing scopes are {list(OBJECT_REGISTRY.keys())}."
        )

    if object_id in OBJECT_REGISTRY["global"]:
        return OBJECT_REGISTRY["global"][object_id](**kwargs)
    elif object_id not in OBJECT_REGISTRY[scope]:
        raise ValueError(
            f"Object with object_id `{object_id}` not registered in scope `{scope}`. "
            f"Call register_object('{object_id}', <class>, scope='{scope}') to add it to the registry."
        )

    return OBJECT_REGISTRY[scope][object_id](**kwargs)


def get_object_class(object_id: str, scope: str = "global") -> GridObj:
    return OBJECT_REGISTRY[scope][object_id]


def register_object(
    object_id: str, obj_class: GridObj, scope: str = "global"
) -> None:
    global_scope_chars = [
        obj.char for obj in OBJECT_REGISTRY.get("global", {}).values()
    ]

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


def register_object_type(
    object_id: str,
    scope: str = "global",
    can_pickup: bool = False,
    can_overlap: bool = False,
    can_place_on: bool = False,
    can_pickup_from: bool = False,
    is_wall: bool = False,
):
    """Register a GridObj subclass with static property metadata.

    Stores boolean properties for ``build_lookup_tables()`` and
    auto-discovers component classmethods (tick, interaction, etc.).

    Usage::

        @register_object_type("wall", is_wall=True)
        class Wall(GridObj):
            ...
    """

    def decorator(cls):
        # Lazy import to avoid circular dependency
        from cogrid.core.component_registry import (
            get_all_components,
            register_component_metadata,
            _validate_classmethod_signature,
        )

        properties = {
            "can_pickup": can_pickup,
            "can_overlap": can_overlap,
            "can_place_on": can_place_on,
            "can_pickup_from": can_pickup_from,
            "is_wall": is_wall,
        }

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

        # Store component metadata in the registry
        register_component_metadata(
            scope=scope,
            object_id=object_id,
            cls=cls,
            properties=properties,
            methods=discovered,
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

    return tables


def get_registered_object_ids(scope: str = "global") -> list[str]:
    """Return a list of the object_ids of available objects in a given scope."""
    return list(OBJECT_REGISTRY[scope].keys())


def get_object_char(object_id: str, scope: str = "global") -> str:
    return get_object_class(object_id, scope=scope).char


def get_object_id_from_char(object_char: str, scope: str = "global") -> str:
    # First check global scope, no matter what scope was passed (default to global).
    for object_id, object_class in OBJECT_REGISTRY["global"].items():
        if object_class.char == object_char:
            return object_id

    if scope != "global":
        for object_id, object_class in OBJECT_REGISTRY[scope].items():
            if object_class.char == object_char:
                return object_id

    raise ValueError(
        f"There is no registered object with character representation `{object_char}` in scope `{scope}`."
    )


def get_object_names(scope: str = "global") -> list[str]:
    """Return all registered object IDs in stable encoding order.

    Order: [None, "free_space", sorted globals, sorted scope, agent directions].
    """
    # Start with None and free_space which are special cases
    names = [None, "free_space"]

    # Add all registered global objects in sorted order (except free_space which we already added)
    global_objects = sorted(
        [
            obj_id
            for obj_id in OBJECT_REGISTRY.get("global", {}).keys()
            if obj_id != "free_space"
        ]
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
