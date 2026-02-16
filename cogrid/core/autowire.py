"""Auto-wire scope and reward configuration from component registries.

Reads component metadata populated by ``@register_object_type`` and
``@register_reward_type`` and produces complete ``scope_config`` and
``reward_config`` dicts matching the shapes consumed by
``step_pipeline.step()``, ``layout_parser.parse_layout()``, and
``interactions.process_interactions()``.

Composes scope_config, reward_config, and feature_config automatically
from registered components. This is the sole environment configuration path.
"""

from cogrid.backend import xp


def build_feature_config_from_components(
    scope: str,
    feature_names: list[str],
    n_agents: int,
    layout_idx: int = 0,
) -> dict:
    """Build feature_config from registered ArrayFeature subclasses.

    Composes features listed in *feature_names* into a single
    ``feature_fn(state, agent_idx) -> (obs_dim,) float32``.
    """
    from cogrid.core.array_features import compose_feature_fns, obs_dim_for_features
    from cogrid.core.component_registry import get_pre_compose_hook

    # Ensure global ArrayFeature subclasses are registered
    import cogrid.feature_space.array_features  # noqa: F401

    # Run scope-specific pre-compose hook (e.g. set layout index state
    # before feature closures capture it).
    pre_hook = get_pre_compose_hook(scope)
    if pre_hook is not None:
        pre_hook(layout_idx=layout_idx, scope=scope)

    lookup_scopes = [scope, "global"]

    composed_fn = compose_feature_fns(
        feature_names, scope, n_agents,
        scopes=lookup_scopes, preserve_order=True,
    )

    total_dim = obs_dim_for_features(
        feature_names, scope, n_agents, scopes=lookup_scopes,
    )

    return {
        "feature_fn": composed_fn,
        "obs_dim": total_dim,
        "feature_names": feature_names,
    }


def build_scope_config_from_components(
    scope: str,
    *,
    tick_handler=None,
    interaction_tables=None,
    state_extractor=None,
) -> dict:
    """Build a complete scope_config from registered component metadata.

    Assembles type_ids, symbol_table, extra_state_schema, static_tables,
    tick_handler, and render_sync automatically from GridObject classmethods
    registered in the given scope (plus global).

    Pass-through kwargs override auto-composed handlers.
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

    return {
        "scope": scope,
        "interaction_tables": interaction_tables,
        "type_ids": type_ids,
        "state_extractor": state_extractor,
        "tick_handler": tick_handler,
        "static_tables": static_tables,
        "symbol_table": symbol_table,
        "extra_state_schema": extra_state_schema,
        "extra_state_builder": extra_state_builder,
        "render_sync": render_sync,
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
    """Build reward_config from registered ArrayReward subclasses.

    Composes a single ``compute_fn`` that sums all registered rewards.
    Each reward's ``compute()`` returns final (n_agents,) float32 values --
    the composition layer just sums them.
    """
    from cogrid.core.component_registry import get_reward_types

    # Collect reward metadata from scope-specific and global registries
    reward_metas = get_reward_types(scope)
    if scope != "global":
        reward_metas = reward_metas + get_reward_types("global")

    instances = [meta.cls() for meta in reward_metas]

    def compute_fn(prev_state, state, actions, reward_config):
        """Composed reward function that sums all registered rewards."""
        total = xp.zeros(n_agents, dtype=xp.float32)
        for inst in instances:
            total = total + inst.compute(prev_state, state, actions, reward_config)
        return total

    return {
        "compute_fn": compute_fn,
        "type_ids": type_ids,
        "n_agents": n_agents,
        "action_pickup_drop_idx": action_pickup_drop_idx,
    }
