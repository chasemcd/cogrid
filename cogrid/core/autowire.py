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
    raise NotImplementedError("build_scope_config_from_components not yet implemented")
