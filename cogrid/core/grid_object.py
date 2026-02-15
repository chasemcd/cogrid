"""Backward-compatibility re-export shim.

This module previously contained all grid object code (base class, registry,
and concrete definitions) in a single 818-line file. The code has been split
into three focused modules:

- ``grid_object_base``: GridObj base class, GridAgent wrapper, helpers
- ``grid_object_registry``: OBJECT_REGISTRY, registration and lookup functions
- ``grid_objects``: Concrete global objects (Wall, Floor, Counter, Key, Door)

All public names are re-exported here so existing imports continue to work:

    from cogrid.core.grid_object import GridObj, Wall, make_object  # still works
    from cogrid.core import grid_object; grid_object.OBJECT_REGISTRY  # still works
"""

from cogrid.core.grid_object_base import (  # noqa: F401
    GridObj,
    GridAgent,
    _is_str,
    _is_int,
)

from cogrid.core.grid_object_registry import (  # noqa: F401
    OBJECT_REGISTRY,
    _OBJECT_TYPE_PROPERTIES,
    _COMPONENT_METHODS,
    make_object,
    get_object_class,
    register_object,
    register_object_type,
    build_lookup_tables,
    get_registered_object_ids,
    get_object_char,
    get_object_id_from_char,
    get_object_names,
    object_to_idx,
    idx_to_object,
)

from cogrid.core.grid_objects import (  # noqa: F401
    Wall,
    Floor,
    Counter,
    Key,
    Door,
)

__all__ = [
    # Base classes and helpers
    "GridObj",
    "GridAgent",
    "_is_str",
    "_is_int",
    # Registry data structures
    "OBJECT_REGISTRY",
    "_OBJECT_TYPE_PROPERTIES",
    "_COMPONENT_METHODS",
    # Registration and lookup functions
    "make_object",
    "get_object_class",
    "register_object",
    "register_object_type",
    "build_lookup_tables",
    "get_registered_object_ids",
    "get_object_char",
    "get_object_id_from_char",
    "get_object_names",
    "object_to_idx",
    "idx_to_object",
    # Concrete objects
    "Wall",
    "Floor",
    "Counter",
    "Key",
    "Door",
]
