from cogrid.envs.overcooked import overcooked_features

from cogrid.core.scope_config import register_scope_config
from cogrid.envs.overcooked.array_config import build_overcooked_scope_config
from cogrid.core.layout_parser import register_symbols

register_scope_config("overcooked", build_overcooked_scope_config)

register_symbols("overcooked", {
    "#": {"object_id": "wall", "is_wall": True},
    "C": {"object_id": "counter"},
    "U": {"object_id": "pot"},
    "O": {"object_id": "onion_stack"},
    "=": {"object_id": "plate_stack"},
    "@": {"object_id": "delivery_zone"},
    "+": {"object_id": None, "is_spawn": True},
    " ": {"object_id": None},
})
