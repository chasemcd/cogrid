from cogrid.envs.overcooked import overcooked_features

from cogrid.core.scope_config import register_scope_config
from cogrid.envs.overcooked.array_config import build_overcooked_scope_config

register_scope_config("overcooked", build_overcooked_scope_config)
