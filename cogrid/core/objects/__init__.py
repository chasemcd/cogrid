"""Grid object system: base class, registry, concrete objects, containers.

Public names are re-exported here for convenient imports::

    from cogrid.core.objects import GridObj, Wall, when, Container
"""

from cogrid.core.objects.base import (  # noqa: F401
    GridAgent,
    GridObj,
    _is_int,
    _is_str,
)
from cogrid.core.objects.builtins import (  # noqa: F401
    Counter,
    Door,
    Floor,
    Key,
    Wall,
)
from cogrid.core.objects.containers import Container  # noqa: F401
from cogrid.core.objects.registry import (  # noqa: F401
    _COMPONENT_METHODS,
    _OBJECT_TYPE_PROPERTIES,
    OBJECT_REGISTRY,
    build_guard_tables,
    build_lookup_tables,
    get_object_char,
    get_object_class,
    get_object_id_from_char,
    get_object_names,
    get_registered_object_ids,
    idx_to_object,
    make_object,
    object_to_idx,
    register_object,
    register_object_type,
)
from cogrid.core.objects.when import When, when  # noqa: F401

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
    "build_guard_tables",
    "build_lookup_tables",
    "get_registered_object_ids",
    "get_object_char",
    "get_object_id_from_char",
    "get_object_names",
    "object_to_idx",
    "idx_to_object",
    # When descriptor
    "When",
    "when",
    # Container
    "Container",
    # Concrete objects
    "Wall",
    "Floor",
    "Counter",
    "Key",
    "Door",
]
