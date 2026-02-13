"""Auto-wire scope and reward configuration from component registries.

Reads component metadata populated by ``@register_object_type`` and
``@register_reward_type`` and produces complete ``scope_config`` and
``reward_config`` dicts matching the shapes consumed by
``step_pipeline.step()``, ``layout_parser.parse_layout()``, and
``interactions.process_interactions()``.

Composes scope_config and reward_config automatically from registered
components. This is the sole environment configuration path.
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
        get_tickable_components,
    )
    from cogrid.core.grid_object import (
        build_lookup_tables,
        get_object_names,
        object_to_idx,
    )

    # -- Collect all components for this scope (reused below) --
    all_components = get_all_components(scope)

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

    # -- Compose tick_handler from tickable components (if not overridden) --
    if tick_handler is None:
        tickable = get_tickable_components(scope)
        if len(tickable) == 1:
            tick_handler = tickable[0].methods["build_tick_fn"]()
        elif len(tickable) > 1:
            tick_fns = [m.methods["build_tick_fn"]() for m in tickable]

            def _composed_tick(state, scope_config, _fns=tick_fns):
                for fn in _fns:
                    state = fn(state, scope_config)
                return state

            tick_handler = _composed_tick

    # -- Compose interaction_body from components (if not overridden) --
    if interaction_body is None:
        interacting = [m for m in all_components if m.has_interaction]
        if len(interacting) == 1:
            interaction_body = interacting[0].methods["build_interaction_fn"]()
        elif len(interacting) > 1:
            raise ValueError(
                f"Multiple components in scope '{scope}' define build_interaction_fn. "
                f"Only one interaction_body per scope is supported. "
                f"Components: {[m.object_id for m in interacting]}"
            )

    # -- Compose extra_state_builder from components --
    extra_state_builder = None
    builders = [
        m for m in all_components
        if "extra_state_builder" in m.methods
    ]
    if builders:
        builder_fns = [m.methods["extra_state_builder"]() for m in builders]
        if len(builder_fns) == 1:
            extra_state_builder = builder_fns[0]
        else:
            def _composed_builder(parsed_arrays, scope=scope, _fns=builder_fns):
                merged = {}
                for fn in _fns:
                    merged.update(fn(parsed_arrays, scope))
                return merged

            extra_state_builder = _composed_builder

    # -- Merge component-specific static_tables --
    for meta in all_components:
        if meta.has_static_tables:
            extra_tables = meta.methods["build_static_tables"]()
            static_tables.update(extra_tables)

    # -- Compose render_sync from components (global + scope) --
    render_sync = None
    global_renderers = [m for m in get_all_components("global") if m.has_render_sync]
    scope_renderers = [m for m in all_components if m.has_render_sync] if scope != "global" else []
    all_renderers = global_renderers + scope_renderers
    if all_renderers:
        render_fns = [m.methods["build_render_sync_fn"]() for m in all_renderers]
        if len(render_fns) == 1:
            render_sync = render_fns[0]
        else:
            def _composed_render_sync(grid, env_state, scope, _fns=render_fns):
                for fn in _fns:
                    fn(grid, env_state, scope)
            render_sync = _composed_render_sync

    # -- Compose feature_fn_builder from components --
    feature_fn_builder = None
    feat_builders = [m for m in all_components if m.has_feature_fn]
    if len(feat_builders) == 1:
        feature_fn_builder = feat_builders[0].methods["build_feature_fn"]()
    elif len(feat_builders) > 1:
        raise ValueError(
            f"Multiple components in scope '{scope}' define build_feature_fn. "
            f"Only one feature_fn_builder per scope is supported. "
            f"Components: {[m.object_id for m in feat_builders]}"
        )

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
        "extra_state_builder": extra_state_builder,
        "render_sync": render_sync,
        "feature_fn_builder": feature_fn_builder,
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
    from cogrid.core.component_registry import get_reward_types

    # Collect reward metadata from scope-specific and global registries
    reward_metas = get_reward_types(scope)
    if scope != "global":
        reward_metas = reward_metas + get_reward_types("global")

    # Instantiate each reward with its registered defaults
    instances = [
        meta.cls(
            coefficient=meta.default_coefficient,
            common_reward=meta.default_common_reward,
        )
        for meta in reward_metas
    ]

    def compute_fn(prev_state, state, actions, reward_config):
        """Composed reward function that sums all registered rewards."""
        from cogrid.backend import xp

        total = xp.zeros(n_agents, dtype=xp.float32)
        for inst in instances:
            r = inst.compute(prev_state, state, actions, reward_config)
            r = r * inst.coefficient
            if inst.common_reward:
                r = xp.full(n_agents, xp.sum(r), dtype=xp.float32)
            total = total + r
        return total

    return {
        "compute_fn": compute_fn,
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": action_pickup_drop_idx,
    }
