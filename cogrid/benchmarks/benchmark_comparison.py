"""Cross-library Overcooked Cramped Room benchmark.

Compares environment throughput across three implementations:
1. CoGrid (this library)          -- numpy single, JAX single, JAX vmap
2. overcooked_ai (original)       -- numpy single (step + featurize_state_mdp)
3. JaxMARL                        -- JAX single, JAX vmap

All implementations compute per-agent observations each step so the
comparison is apples-to-apples.  overcooked_ai's step() returns raw
game state, so we call featurize_state_mdp() after each step.

Install external dependencies before running:
    pip install overcooked-ai jaxmarl

Run:
    python -m cogrid.benchmarks.benchmark_comparison
"""

import statistics
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STEPS = 1000
N_WARMUP = 10
N_TRIALS = 5
BATCH_SIZE = 1024
SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(steps_per_sec):
    """Format steps/sec with commas."""
    return f"{steps_per_sec:>12,.0f}"


def _run_trials(run_one_trial, n_trials=N_TRIALS):
    """Run *run_one_trial()* n_trials times, return list of steps/sec."""
    return [run_one_trial() for _ in range(n_trials)]


# ===================================================================
# CoGrid -- NumPy single env
# ===================================================================


def bench_cogrid_numpy(n_steps=N_STEPS):
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="numpy")
    env.reset(seed=SEED)
    agent_ids = sorted(env.possible_agents)
    noop = {aid: 6 for aid in agent_ids}

    # warmup
    for _ in range(N_WARMUP):
        env.step(noop)

    def trial():
        env.reset(seed=SEED)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            env.step(noop)
        return n_steps / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# CoGrid -- JAX single env (JIT)
# ===================================================================


def bench_cogrid_jax_single(n_steps=N_STEPS):
    import jax
    import jax.numpy as jnp
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=SEED)
    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.possible_agents)
    actions = jnp.full((n_agents,), 6, dtype=jnp.int32)

    # warmup (includes JIT compilation)
    state, _ = reset_fn(jax.random.key(SEED))
    state.agent_pos.block_until_ready()
    for _ in range(N_WARMUP):
        state, *_ = step_fn(state, actions)
        state.agent_pos.block_until_ready()

    def trial():
        s, _ = reset_fn(jax.random.key(SEED))
        s.agent_pos.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            s, *_ = step_fn(s, actions)
        s.agent_pos.block_until_ready()
        return n_steps / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# CoGrid -- JAX vmap
# ===================================================================


def bench_cogrid_jax_vmap(n_steps=N_STEPS, batch_size=BATCH_SIZE):
    import jax
    import jax.numpy as jnp
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401
    from cogrid.envs import registry

    env = registry.make("Overcooked-CrampedRoom-V0", backend="jax")
    env.reset(seed=SEED)
    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.possible_agents)

    v_reset = jax.jit(jax.vmap(reset_fn))
    v_step = jax.jit(jax.vmap(step_fn))
    keys = jax.random.split(jax.random.key(0), batch_size)
    batch_actions = jnp.full((batch_size, n_agents), 6, dtype=jnp.int32)

    # warmup
    bs, _ = v_reset(keys)
    bs.agent_pos.block_until_ready()
    for _ in range(N_WARMUP):
        bs, *_ = v_step(bs, batch_actions)
        bs.agent_pos.block_until_ready()

    def trial():
        s, _ = v_reset(keys)
        s.agent_pos.block_until_ready()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            s, *_ = v_step(s, batch_actions)
        s.agent_pos.block_until_ready()
        return (n_steps * batch_size) / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# overcooked_ai -- NumPy single env
# ===================================================================


def bench_overcooked_ai(n_steps=N_STEPS):
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    noop = (Action.STAY, Action.STAY)

    # warmup -- trigger lazy mlam computation before timed section
    env.reset()
    _ = env.featurize_state_mdp(env.state)
    for _ in range(N_WARMUP):
        _, _, done, _ = env.step(noop)
        if done:
            env.reset()
        env.featurize_state_mdp(env.state)

    def trial():
        env.reset()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            _, _, done, _ = env.step(noop)
            if done:
                env.reset()
            env.featurize_state_mdp(env.state)
        return n_steps / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# JaxMARL -- JAX single env (JIT)
# ===================================================================


def bench_jaxmarl_single(n_steps=N_STEPS):
    import jax
    import jax.numpy as jnp
    from jaxmarl import make as jaxmarl_make

    env = jaxmarl_make("overcooked")
    agents = env.agents  # ["agent_0", "agent_1"]

    step_jit = jax.jit(env.step)
    reset_jit = jax.jit(env.reset)

    noop_actions = {agent: jnp.int32(4) for agent in agents}  # 4 = stay

    # warmup
    key = jax.random.key(SEED)
    key, k_reset = jax.random.split(key)
    obs, state = reset_jit(k_reset)
    jax.tree.map(lambda x: x.block_until_ready(), obs)
    for _ in range(N_WARMUP):
        key, k_step = jax.random.split(key)
        obs, state, rew, done, info = step_jit(k_step, state, noop_actions)
        jax.tree.map(lambda x: x.block_until_ready(), obs)

    def trial():
        nonlocal key
        key, k_reset = jax.random.split(key)
        obs, st = reset_jit(k_reset)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            key, k_step = jax.random.split(key)
            obs, st, rew, done, info = step_jit(k_step, st, noop_actions)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        return n_steps / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# JaxMARL -- JAX vmap
# ===================================================================


def bench_jaxmarl_vmap(n_steps=N_STEPS, batch_size=BATCH_SIZE):
    import jax
    import jax.numpy as jnp
    from jaxmarl import make as jaxmarl_make

    env = jaxmarl_make("overcooked")
    agents = env.agents

    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    noop_actions = {
        agent: jnp.full(batch_size, 4, dtype=jnp.int32) for agent in agents
    }

    # warmup
    keys = jax.random.split(jax.random.key(0), batch_size)
    obs, states = v_reset(keys)
    jax.tree.map(lambda x: x.block_until_ready(), obs)
    step_keys = jax.random.split(jax.random.key(1), batch_size)
    for _ in range(N_WARMUP):
        obs, states, rew, done, info = v_step(step_keys, states, noop_actions)
        jax.tree.map(lambda x: x.block_until_ready(), obs)

    def trial():
        rng = jax.random.key(SEED)
        rng, k_reset = jax.random.split(rng)
        reset_keys = jax.random.split(k_reset, batch_size)
        obs, st = v_reset(reset_keys)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            rng, k_step = jax.random.split(rng)
            s_keys = jax.random.split(k_step, batch_size)
            obs, st, rew, done, info = v_step(s_keys, st, noop_actions)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        return (n_steps * batch_size) / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# Runner
# ===================================================================

_BENCHMARKS = [
    ("CoGrid (NumPy)", "single", bench_cogrid_numpy),
    ("CoGrid (JAX JIT)", "single", bench_cogrid_jax_single),
    ("CoGrid (JAX vmap)", "vmap", bench_cogrid_jax_vmap),
    ("overcooked_ai (NumPy)", "single", bench_overcooked_ai),
    ("JaxMARL (JAX JIT)", "single", bench_jaxmarl_single),
    ("JaxMARL (JAX vmap)", "vmap", bench_jaxmarl_vmap),
]


def run_comparison():
    """Run all benchmarks and print a comparison table."""
    results = {}

    for label, mode, fn in _BENCHMARKS:
        print(f"  Running {label} ...")
        try:
            trials = fn()
            med = statistics.median(trials)
            results[label] = {"trials": trials, "median": med, "mode": mode}
        except Exception as e:
            print(f"    SKIPPED ({type(e).__name__}: {e})")
            results[label] = None

    # ---- print results ------------------------------------------------
    print()
    W = 78
    print("=" * W)
    print("Overcooked Cramped Room -- Environment Throughput Comparison")
    print("=" * W)
    print(
        f"  Steps per trial: {N_STEPS}   |   Trials: {N_TRIALS}   |   "
        f"vmap batch: {BATCH_SIZE}"
    )
    print("-" * W)
    print(
        f"  {'Library':<28} {'Mode':<10} "
        f"{'Total steps/s':>14} {'Per-env steps/s':>16}"
    )
    print("-" * W)

    for label, _, _ in _BENCHMARKS:
        r = results[label]
        if r is None:
            print(f"  {label:<28} {'--':<10} {'(not installed)':>14}")
        elif r["mode"] == "vmap":
            per_env = r["median"] / BATCH_SIZE
            print(
                f"  {label:<28} {r['mode']:<10} "
                f"{_fmt(r['median'])} {_fmt(per_env)}"
            )
        else:
            print(
                f"  {label:<28} {r['mode']:<10} "
                f"{_fmt(r['median'])} {_fmt(r['median'])}"
            )

    # ---- speedup summary ----------------------------------------------
    print()
    print("Speedup comparisons")
    print("-" * W)

    def _ratio_line(label_a, label_b):
        a, b = results.get(label_a), results.get(label_b)
        if a and b and b["median"] > 0:
            ratio = a["median"] / b["median"]
            print(f"  {label_a:<28} {ratio:>7.1f}x  vs {label_b}")

    # single-env: each library vs overcooked_ai
    _ratio_line("CoGrid (NumPy)", "overcooked_ai (NumPy)")
    _ratio_line("CoGrid (JAX JIT)", "overcooked_ai (NumPy)")
    _ratio_line("JaxMARL (JAX JIT)", "overcooked_ai (NumPy)")

    # single-env: JAX libs head-to-head
    _ratio_line("CoGrid (JAX JIT)", "JaxMARL (JAX JIT)")

    # vmap: total throughput comparison
    _ratio_line("CoGrid (JAX vmap)", "JaxMARL (JAX vmap)")

    # vmap vs single (within each library)
    _ratio_line("CoGrid (JAX vmap)", "CoGrid (JAX JIT)")
    _ratio_line("JaxMARL (JAX vmap)", "JaxMARL (JAX JIT)")
    print("=" * W)
    return results


if __name__ == "__main__":
    run_comparison()
