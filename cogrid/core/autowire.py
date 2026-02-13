"""Auto-wire scope configuration from component registries.

Reads component metadata populated by ``@register_object_type`` and
produces a complete ``scope_config`` dict matching the shape consumed
by ``step_pipeline.step()``, ``layout_parser.parse_layout()``, and
``interactions.process_interactions()``.

Replaces manual dict assembly (e.g. ``build_overcooked_scope_config()``)
with automatic composition from registered components.
"""

from __future__ import annotations


def build_scope_config_from_components(
    scope: str,
    *,
    tick_handler=None,
    interaction_body=None,
    interaction_tables=None,
    state_extractor=None,
) -> dict:
    """Build a complete scope_config dict from registered component metadata.

    Queries the component registry for all GridObject subclasses in the
    given scope (plus global scope), and assembles type_ids, symbol_table,
    extra_state_schema, and static_tables automatically.

    Pass-through fields (tick_handler, interaction_body, interaction_tables,
    state_extractor) default to None and can be overridden via keyword
    arguments for backward compatibility with environments that have
    monolithic handler functions (e.g. Overcooked).

    Args:
        scope: Registry scope name (e.g. "overcooked").
        tick_handler: Optional tick handler callable.
        interaction_body: Optional per-agent interaction body callable.
        interaction_tables: Optional interaction tables dict.
        state_extractor: Optional state extractor callable.

    Returns:
        Dict with keys: scope, interaction_tables, type_ids, state_extractor,
        tick_handler, interaction_body, static_tables, symbol_table,
        extra_state_schema, extra_state_builder.
    """
    from cogrid.core.component_registry import (
        get_all_components,
        get_components_with_extra_state,
    )
    from cogrid.core.grid_object import (
        build_lookup_tables,
        get_object_names,
        object_to_idx,
    )

    # -- type_ids: map object names to integer indices --
    type_ids = {
        name: object_to_idx(name, scope)
        for name in get_object_names(scope=scope)
        if name is not None
    }

    # -- symbol_table: char -> {"object_id": str, ...} --
    symbol_table = _build_symbol_table(scope, get_all_components)

    # -- extra_state_schema: merged from all components, scope-prefixed, sorted --
    extra_state_schema = _build_extra_state_schema(
        scope, get_components_with_extra_state
    )

    # -- static_tables: CAN_PICKUP, CAN_OVERLAP, etc. from build_lookup_tables --
    static_tables = build_lookup_tables(scope=scope)

    return {
        "scope": scope,
        "interaction_tables": interaction_tables,
        "type_ids": type_ids,
        "state_extractor": state_extractor,
        "tick_handler": tick_handler,
        "interaction_body": interaction_body,
        "static_tables": static_tables,
        "symbol_table": symbol_table,
        "extra_state_schema": extra_state_schema,
        "extra_state_builder": None,
    }


def _build_symbol_table(scope: str, get_all_components) -> dict:
    """Build the symbol_table mapping char -> entry dict.

    Includes global components, scope-specific components, and the
    special "+" (spawn) and " " (empty) entries.
    """
    symbol_table = {}

    # Global components first (Wall, Counter, Key, Door, Floor, etc.)
    for meta in get_all_components("global"):
        entry = {"object_id": meta.object_id}
        # Include only True boolean properties
        for prop_name, prop_val in meta.properties.items():
            if prop_val:
                entry[prop_name] = True
        symbol_table[meta.char] = entry

    # Scope-specific components
    if scope != "global":
        for meta in get_all_components(scope):
            entry = {"object_id": meta.object_id}
            for prop_name, prop_val in meta.properties.items():
                if prop_val:
                    entry[prop_name] = True
            symbol_table[meta.char] = entry

    # Special entries (not GridObject subclasses)
    symbol_table["+"] = {"object_id": None, "is_spawn": True}
    symbol_table[" "] = {"object_id": None}

    return symbol_table


def _build_extra_state_schema(scope: str, get_components_with_extra_state) -> dict:
    """Build merged extra_state_schema from all components with extra_state.

    Each key is prefixed with ``{scope}.`` and the final dict is sorted
    by key for deterministic pytree structure.
    """
    merged = {}

    # Global components with extra state
    for meta in get_components_with_extra_state("global"):
        schema = meta.methods["extra_state_schema"]()
        for key, val in schema.items():
            merged[f"{scope}.{key}"] = val

    # Scope-specific components with extra state
    if scope != "global":
        for meta in get_components_with_extra_state(scope):
            schema = meta.methods["extra_state_schema"]()
            for key, val in schema.items():
                merged[f"{scope}.{key}"] = val

    return dict(sorted(merged.items()))


def build_reward_config_from_components(
    scope: str,
    n_agents: int,
    type_ids: dict,
    action_pickup_drop_idx: int = 4,
) -> dict:
    """Build a reward_config dict from registered ArrayReward components.

    Queries the component registry for all ArrayReward subclasses in the
    given scope (plus global scope), instantiates each with its default
    coefficient/common_reward, and composes a single ``compute_fn`` closure
    that calls each reward's ``compute()``, applies coefficient weighting
    and common_reward broadcasting, and sums results.

    Args:
        scope: Registry scope name (e.g. "overcooked").
        n_agents: Number of agents (determines reward array shape).
        type_ids: Mapping of object names to integer indices.
        action_pickup_drop_idx: Action index for pickup/drop. Defaults to 4.

    Returns:
        Dict with keys: compute_fn, type_ids, n_agents, action_pickup_drop_idx.
    """
    raise NotImplementedError("build_reward_config_from_components not yet implemented")
