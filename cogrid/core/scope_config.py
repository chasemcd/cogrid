"""Scope configuration registry for environment-specific array logic.

Each environment scope (e.g. "overcooked") registers a config builder
function that returns a dict describing its interaction tables, type IDs,
state extractors, and handler callbacks. Core modules consume this config
instead of hardcoding environment-specific logic.

Usage::

    from cogrid.core.scope_config import register_scope_config, get_scope_config

    # Registration (typically in envs/<scope>/__init__.py):
    register_scope_config("overcooked", build_overcooked_scope_config)

    # Consumption (in core modules):
    cfg = get_scope_config("overcooked")
    interaction_tables = cfg["interaction_tables"]
"""

from __future__ import annotations

from typing import Callable


# Global registry mapping scope name -> config builder callable.
SCOPE_CONFIG_REGISTRY: dict[str, Callable[[], dict]] = {}


def register_scope_config(scope: str, config_builder: Callable[[], dict]) -> None:
    """Register a scope configuration builder.

    Args:
        scope: Scope name (e.g. "overcooked").
        config_builder: A callable that takes no arguments and returns a
            scope config dict.
    """
    SCOPE_CONFIG_REGISTRY[scope] = config_builder


def get_scope_config(scope: str) -> dict:
    """Retrieve the scope configuration for the given scope.

    If the scope has a registered config builder, calls it and returns
    the result. Otherwise returns ``default_scope_config()``.

    Args:
        scope: Scope name to look up.

    Returns:
        A dict with keys: ``interaction_tables``, ``type_ids``,
        ``state_extractor``, ``interaction_handler``, ``tick_handler``,
        ``place_on_handlers``.
    """
    if scope in SCOPE_CONFIG_REGISTRY:
        return SCOPE_CONFIG_REGISTRY[scope]()
    return default_scope_config()


def default_scope_config() -> dict:
    """Return the default (empty) scope configuration.

    Used for scopes that have no custom interaction logic (e.g.
    search_rescue, goal_seeking). All fields are None or empty.

    Returns:
        Dict with default empty values for all scope config keys.
    """
    return {
        "interaction_tables": None,
        "type_ids": {},
        "state_extractor": None,
        "interaction_handler": None,
        "tick_handler": None,
        "place_on_handlers": {},
        # v1.1 additions
        "symbol_table": {},           # char -> {"object_id": str, "is_wall": bool, ...}
        "extra_state_schema": {},     # key -> {"shape": tuple, "dtype": str}
        "extra_state_builder": None,  # callable(parsed_arrays, scope) -> dict
    }
