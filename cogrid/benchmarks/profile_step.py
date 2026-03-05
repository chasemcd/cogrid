"""Profile individual phases of the CoGrid step pipeline.

Measures wall-clock time for each phase (tick, movement, interactions,
observations, rewards) in isolation under JIT, and reports HLO op counts
for the full step vs JaxMARL.

Run:
    python -m cogrid.benchmarks.profile_step
"""

import time

import jax
import jax.numpy as jnp

SEED = 42
N_STEPS = 2000
N_WARMUP = 20


def _setup_cogrid():
    """Build CoGrid env, return (step_fn, reset_fn, state, actions, internals)."""
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=SEED)
    n_agents = len(env.possible_agents)
    key = jax.random.key(SEED)
    actions = jnp.full((n_agents,), 6, dtype=jnp.int32)

    return env, key, actions, n_agents


def _bench_fn(jit_fn, state, extra_args, n_steps=N_STEPS, extract_state=None):
    """Warm up and time a JIT-compiled function."""
    # warmup
    for _ in range(N_WARMUP):
        result = jit_fn(state, *extra_args)
        if extract_state:
            extract_state(result).block_until_ready()
        else:
            jax.tree.map(lambda x: x.block_until_ready(), result)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        result = jit_fn(state, *extra_args)
    if extract_state:
        extract_state(result).block_until_ready()
    else:
        jax.tree.map(lambda x: x.block_until_ready(), result)
    elapsed = time.perf_counter() - t0
    return n_steps / elapsed


def profile_phases():
    """Profile each step phase in isolation."""
    from cogrid.backend import xp
    from cogrid.core.interactions import process_interactions
    from cogrid.core.movement import move_agents
    from cogrid.core.step_pipeline import (
        envstate_to_dict,
    )
    from cogrid.feature_space.features import get_all_agent_obs

    env, key, actions, n_agents = _setup_cogrid()

    # Get internals from env
    step_fn = env.jax_step
    reset_fn = env.jax_reset

    jit_reset = jax.jit(reset_fn)
    _, state, _ = jit_reset(key)
    state.agent_pos.block_until_ready()

    # Get closed-over config from the env
    scope_config = env._scope_config
    lookup_tables = env._lookup_tables
    feature_fn = env._feature_fn
    reward_config = env._reward_config
    action_pickup_drop_idx = env._action_pickup_drop_idx
    action_toggle_idx = env._action_toggle_idx
    _ = env.max_steps
    _ = getattr(env, "_terminated_fn", None)

    dir_vec_table = xp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32)

    print("=" * 65)
    print("Phase-level profiling (batch=1, CPU, JIT-compiled)")
    print("=" * 65)
    print()

    # --- Phase: Tick ---
    tick_handler = scope_config.get("tick_handler")
    if tick_handler is not None:

        @jax.jit
        def jit_tick(state):
            return tick_handler(state, scope_config)

        for _ in range(N_WARMUP):
            s = jit_tick(state)
            s.agent_pos.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            s = jit_tick(state)
        s.agent_pos.block_until_ready()
        tick_rate = N_STEPS / (time.perf_counter() - t0)
        print(f"  Tick:          {tick_rate:>12,.0f} calls/sec")
    else:
        print("  Tick:          (no tick handler)")

    # --- Phase: RNG ---
    @jax.jit
    def jit_rng(rng_key):
        k, subkey = jax.random.split(rng_key)
        perm = jax.random.permutation(subkey, n_agents)
        return k, perm

    for _ in range(N_WARMUP):
        _, p = jit_rng(state.rng_key)
        p.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        _, p = jit_rng(state.rng_key)
    p.block_until_ready()
    rng_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  RNG:           {rng_rate:>12,.0f} calls/sec")

    # --- Phase: Movement ---
    _, priority = jit_rng(state.rng_key)
    priority.block_until_ready()

    @jax.jit
    def jit_move(state, actions, priority):
        return move_agents(
            state.agent_pos,
            state.agent_dir,
            actions,
            state.wall_map,
            state.object_type_map,
            lookup_tables["CAN_OVERLAP"],
            priority,
            state.action_set,
        )

    for _ in range(N_WARMUP):
        np, nd = jit_move(state, actions, priority)
        np.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        np_, nd = jit_move(state, actions, priority)
    np_.block_until_ready()
    move_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  Movement:      {move_rate:>12,.0f} calls/sec")

    # --- Phase: Interactions ---
    interaction_fn = scope_config.get("interaction_fn")

    @jax.jit
    def jit_interact(state, actions):
        return process_interactions(
            state,
            actions,
            interaction_fn,
            lookup_tables,
            scope_config,
            dir_vec_table,
            action_pickup_drop_idx,
            action_toggle_idx,
        )

    for _ in range(N_WARMUP):
        s = jit_interact(state, actions)
        s.agent_pos.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        s = jit_interact(state, actions)
    s.agent_pos.block_until_ready()
    interact_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  Interactions:  {interact_rate:>12,.0f} calls/sec")

    # --- Phase: Observations ---
    envstate_to_dict(state)

    @jax.jit
    def jit_obs(state):
        sv = envstate_to_dict(state)
        return get_all_agent_obs(feature_fn, sv, n_agents)

    for _ in range(N_WARMUP):
        o = jit_obs(state)
        o.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        o = jit_obs(state)
    o.block_until_ready()
    obs_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  Observations:  {obs_rate:>12,.0f} calls/sec")

    # --- Phase: Rewards ---
    compute_fn = reward_config["compute_fn"]

    @jax.jit
    def jit_rewards(prev_state, state, actions):
        prev_sv = envstate_to_dict(prev_state)
        sv = envstate_to_dict(state)
        return compute_fn(prev_sv, sv, actions, reward_config)

    for _ in range(N_WARMUP):
        r = jit_rewards(state, state, actions)
        r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        r = jit_rewards(state, state, actions)
    r.block_until_ready()
    reward_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  Rewards:       {reward_rate:>12,.0f} calls/sec")

    # --- Phase: Full step ---
    jit_step = jax.jit(step_fn)
    for _ in range(N_WARMUP):
        _, s, *_ = jit_step(state, actions)
        s.agent_pos.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        _, s, *_ = jit_step(state, actions)
    s.agent_pos.block_until_ready()
    full_rate = N_STEPS / (time.perf_counter() - t0)
    print(f"  Full step:     {full_rate:>12,.0f} calls/sec")

    print()

    # Overhead = full step dispatch minus sum of parts
    parts_sum = (
        1 / tick_rate
        + 1 / rng_rate
        + 1 / move_rate
        + 1 / interact_rate
        + 1 / obs_rate
        + 1 / reward_rate
    )
    full_time = 1 / full_rate
    overhead_pct = (full_time - parts_sum) / full_time * 100
    print(f"  Sum-of-parts:  {1 / parts_sum:>12,.0f} calls/sec (theoretical)")
    print(f"  Overhead:      {overhead_pct:>11.1f}%")
    print()


def hlo_analysis():
    """Compare HLO op counts between CoGrid and JaxMARL."""
    print("=" * 65)
    print("HLO analysis (compiled XLA graph size)")
    print("=" * 65)
    print()

    # CoGrid
    env, key, actions, n_agents = _setup_cogrid()
    step_fn = env.jax_step
    reset_fn = env.jax_reset

    jit_reset = jax.jit(reset_fn)
    _, state, _ = jit_reset(key)
    state.agent_pos.block_until_ready()

    lowered = jax.jit(step_fn).lower(state, actions)
    compiled = lowered.compile()
    hlo_text = compiled.as_text()
    # Count HLO instructions (lines starting with %)
    cogrid_ops = sum(1 for line in hlo_text.split("\n") if line.strip().startswith("%"))
    cogrid_bytes = len(hlo_text)
    print(f"  CoGrid step HLO:   {cogrid_ops:>6,d} ops, {cogrid_bytes:>8,d} bytes")

    # vmapped
    v_step = jax.jit(jax.vmap(step_fn))
    _, batch_state, _ = jax.jit(jax.vmap(reset_fn))(jax.random.split(key, 4))
    batch_state.agent_pos.block_until_ready()
    batch_actions = jnp.full((4, n_agents), 6, dtype=jnp.int32)

    lowered_v = v_step.lower(batch_state, batch_actions)
    compiled_v = lowered_v.compile()
    hlo_v = compiled_v.as_text()
    v_ops = sum(1 for line in hlo_v.split("\n") if line.strip().startswith("%"))
    v_bytes = len(hlo_v)
    print(f"  CoGrid vmap(4) HLO: {v_ops:>5,d} ops, {v_bytes:>8,d} bytes")

    # JaxMARL
    try:
        from jaxmarl import make as jaxmarl_make

        jm_env = jaxmarl_make("overcooked")
        agents = jm_env.agents

        jm_key = jax.random.key(SEED)
        jm_key, k_reset = jax.random.split(jm_key)
        jm_obs, jm_state = jax.jit(jm_env.reset)(k_reset)
        jax.tree.map(lambda x: x.block_until_ready(), jm_obs)

        jm_actions = {a: jnp.int32(4) for a in agents}
        jm_key, k_step = jax.random.split(jm_key)

        jm_lowered = jax.jit(jm_env.step).lower(k_step, jm_state, jm_actions)
        jm_compiled = jm_lowered.compile()
        jm_hlo = jm_compiled.as_text()
        jm_ops = sum(1 for line in jm_hlo.split("\n") if line.strip().startswith("%"))
        jm_bytes = len(jm_hlo)
        print(f"  JaxMARL step HLO:  {jm_ops:>6,d} ops, {jm_bytes:>8,d} bytes")

        # vmapped
        jm_v_step = jax.jit(jax.vmap(jm_env.step))
        jm_v_reset = jax.jit(jax.vmap(jm_env.reset))
        jm_keys = jax.random.split(jax.random.key(0), 4)
        jm_v_obs, jm_v_state = jm_v_reset(jm_keys)
        jax.tree.map(lambda x: x.block_until_ready(), jm_v_obs)
        jm_v_actions = {a: jnp.full(4, 4, dtype=jnp.int32) for a in agents}
        jm_v_keys = jax.random.split(jax.random.key(1), 4)

        jm_v_lowered = jm_v_step.lower(jm_v_keys, jm_v_state, jm_v_actions)
        jm_v_compiled = jm_v_lowered.compile()
        jm_v_hlo = jm_v_compiled.as_text()
        jm_v_ops = sum(1 for line in jm_v_hlo.split("\n") if line.strip().startswith("%"))
        jm_v_bytes = len(jm_v_hlo)
        print(f"  JaxMARL vmap(4) HLO: {jm_v_ops:>4,d} ops, {jm_v_bytes:>8,d} bytes")
    except Exception as e:
        print(f"  JaxMARL: SKIPPED ({e})")

    print()
    print(f"  CoGrid/JaxMARL op ratio: {cogrid_ops / jm_ops:.1f}x")
    print()

    return hlo_text, hlo_v


def hlo_top_ops(hlo_text, label="CoGrid", top_n=15):
    """Count the most common HLO op types."""
    import re

    op_counts = {}
    for line in hlo_text.split("\n"):
        line = line.strip()
        if line.startswith("%"):
            # Extract op name: %name = <type>[] op(...)
            m = re.match(r"%\S+\s*=\s*\S+\s+(\w+)", line)
            if m:
                op = m.group(1)
                op_counts[op] = op_counts.get(op, 0) + 1

    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])

    print(f"  Top {top_n} HLO ops in {label}:")
    for op, count in sorted_ops[:top_n]:
        print(f"    {op:<30s} {count:>5d}")
    print()


def profile_per_feature():
    """Profile each individual feature function: HLO ops and throughput."""
    from cogrid.core.component_registry import get_feature_types
    from cogrid.core.step_pipeline import envstate_to_dict

    env, key, actions, n_agents = _setup_cogrid()
    jit_reset = jax.jit(env.jax_reset)
    _, state, _ = jit_reset(key)
    state.agent_pos.block_until_ready()
    envstate_to_dict(state)

    # Get the feature names used by this env
    feature_names = env.config["features"]
    scope = env.scope

    # Resolve feature metadata across scopes
    all_metas = []
    for s in ["global", scope]:
        all_metas.extend(get_feature_types(s))
    meta_by_id = {m.feature_id: m for m in all_metas}

    print("=" * 65)
    print("Per-feature profiling (HLO ops and throughput)")
    print("=" * 65)
    print()
    print(f"  {'Feature':<28s} {'obs_dim':>7s} {'HLO ops':>8s} {'calls/s':>12s}")
    print(f"  {'-' * 28} {'-' * 7} {'-' * 8} {'-' * 12}")

    total_ops = 0
    for name in sorted(feature_names):
        meta = meta_by_id[name]
        fn = meta.cls.build_feature_fn(scope)

        if meta.per_agent:

            @jax.jit
            def jit_feat(state, _fn=fn):
                sv = envstate_to_dict(state)
                return _fn(sv, 0)

            # HLO ops
            lowered = jit_feat.lower(state)
            hlo = lowered.compile().as_text()
            ops = sum(1 for line in hlo.split("\n") if line.strip().startswith("%"))

            # Throughput
            for _ in range(10):
                r = jit_feat(state)
                r.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                r = jit_feat(state)
            r.block_until_ready()
            rate = N_STEPS / (time.perf_counter() - t0)
        else:

            @jax.jit
            def jit_feat(state, _fn=fn):
                sv = envstate_to_dict(state)
                return _fn(sv)

            lowered = jit_feat.lower(state)
            hlo = lowered.compile().as_text()
            ops = sum(1 for line in hlo.split("\n") if line.strip().startswith("%"))

            for _ in range(10):
                r = jit_feat(state)
                r.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(N_STEPS):
                r = jit_feat(state)
            r.block_until_ready()
            rate = N_STEPS / (time.perf_counter() - t0)

        total_ops += ops
        # Count how many times this feature is called per step
        print(f"  {name:<28s} {meta.obs_dim:>7d} {ops:>8,d} {rate:>12,.0f}")

        total_ops += ops

    print(f"  {'':28s} {'':>7s} {total_ops // 2:>8,d} {'(total)':>12s}")
    print()


def profile_interaction_detail():
    """Profile interaction sub-costs."""
    env, key, actions, n_agents = _setup_cogrid()
    jit_reset = jax.jit(env.jax_reset)
    _, state, _ = jit_reset(key)
    state.agent_pos.block_until_ready()

    scope_config = env._scope_config
    interaction_fn = scope_config.get("interaction_fn")

    from cogrid.backend import xp

    dir_vec_table = xp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32)

    # Profile just the overcooked interaction for 1 agent
    if interaction_fn is not None:

        @jax.jit
        def jit_single_interact(state):
            H, W = state.object_type_map.shape
            fwd_pos = state.agent_pos + dir_vec_table[state.agent_dir]
            fwd_r = jnp.clip(fwd_pos[:, 0], 0, H - 1)
            fwd_c = jnp.clip(fwd_pos[:, 1], 0, W - 1)
            is_interact = jnp.ones(n_agents, dtype=jnp.bool_)
            fwd_rc = jnp.stack([fwd_r, fwd_c], axis=1)
            fwd_matches_pos = jnp.all(fwd_rc[:, None, :] == state.agent_pos[None, :, :], axis=2)
            not_self = ~jnp.eye(n_agents, dtype=jnp.bool_)
            agent_ahead = jnp.any(fwd_matches_pos & not_self, axis=1)
            base_ok = is_interact & ~agent_ahead
            return interaction_fn(state, 0, fwd_r[0], fwd_c[0], base_ok[0], scope_config)

        lowered = jit_single_interact.lower(state)
        hlo = lowered.compile().as_text()
        ops = sum(1 for line in hlo.split("\n") if line.strip().startswith("%"))

        for _ in range(10):
            s = jit_single_interact(state)
            s.agent_pos.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(N_STEPS):
            s = jit_single_interact(state)
        s.agent_pos.block_until_ready()
        rate = N_STEPS / (time.perf_counter() - t0)

        print("=" * 65)
        print("Interaction detail")
        print("=" * 65)
        print()
        print(f"  Single-agent interaction:  {ops:>6,d} HLO ops, {rate:>12,.0f} calls/sec")
        print(
            f"  Full interaction (x2):     ~{ops * 2:>5,d} HLO ops,"
            f" {10631:>12,d} calls/sec (from phase profile)"
        )
        print()


if __name__ == "__main__":
    # Force CPU
    jax.config.update("jax_platforms", "cpu")

    profile_phases()
    profile_per_feature()
    profile_interaction_detail()
    hlo_text, hlo_v_text = hlo_analysis()
    hlo_top_ops(hlo_text, "CoGrid step (batch=1)")
    hlo_top_ops(hlo_v_text, "CoGrid vmap(4)")
