"""Overcooked Cramped Room -- Scaling Benchmark.

Sweeps over increasing numbers of vectorized environments and measures
throughput for CoGrid JAX and JaxMARL on both CPU and GPU,
with numpy baselines as reference lines.

Install external dependencies before running:
    pip install overcooked-ai jaxmarl matplotlib

Run:
    python -m cogrid.benchmarks.benchmark_comparison
"""

import statistics
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STEPS = 500
N_WARMUP = 10
N_TRIALS = 5
BATCH_SIZES = [1, 4, 16, 64, 256, 1024]
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
# overcooked_ai -- NumPy single env
# ===================================================================


def bench_overcooked_ai(n_steps=N_STEPS):
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld

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
# CoGrid -- JAX (parameterized by batch_size and device)
# ===================================================================


def bench_cogrid_jax(batch_size, device, n_steps=N_STEPS):
    """Returns list of steps/sec trials."""
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

    if batch_size == 1:
        jit_step = jax.jit(step_fn)
        jit_reset = jax.jit(reset_fn)
        actions = jax.device_put(jnp.full((n_agents,), 6, dtype=jnp.int32), device)
        key = jax.device_put(jax.random.key(SEED), device)

        # warmup (includes JIT compilation)
        state, _ = jit_reset(key)
        state.agent_pos.block_until_ready()
        for _ in range(N_WARMUP):
            state, *_ = jit_step(state, actions)
            state.agent_pos.block_until_ready()

        def trial():
            s, _ = jit_reset(key)
            s.agent_pos.block_until_ready()
            t0 = time.perf_counter()
            for _ in range(n_steps):
                s, *_ = jit_step(s, actions)
            s.agent_pos.block_until_ready()
            return n_steps / (time.perf_counter() - t0)
    else:
        v_step = jax.jit(jax.vmap(step_fn))
        v_reset = jax.jit(jax.vmap(reset_fn))
        keys = jax.device_put(jax.random.split(jax.random.key(0), batch_size), device)
        batch_actions = jax.device_put(
            jnp.full((batch_size, n_agents), 6, dtype=jnp.int32), device
        )

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
# JaxMARL -- JAX (parameterized by batch_size and device)
# ===================================================================


def bench_jaxmarl(batch_size, device, n_steps=N_STEPS):
    """Returns list of steps/sec trials."""
    import jax
    import jax.numpy as jnp
    from jaxmarl import make as jaxmarl_make

    env = jaxmarl_make("overcooked")
    agents = env.agents

    if batch_size == 1:
        step_jit = jax.jit(env.step)
        reset_jit = jax.jit(env.reset)
        noop_actions = {
            agent: jax.device_put(jnp.int32(4), device) for agent in agents
        }

        # warmup
        key = jax.device_put(jax.random.key(SEED), device)
        key, k_reset = jax.random.split(key)
        obs, state = reset_jit(k_reset)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        for _ in range(N_WARMUP):
            key, k_step = jax.random.split(key)
            obs, state, *_ = step_jit(k_step, state, noop_actions)
            jax.tree.map(lambda x: x.block_until_ready(), obs)

        def trial():
            nonlocal key
            key, k_reset = jax.random.split(key)
            obs, st = reset_jit(k_reset)
            jax.tree.map(lambda x: x.block_until_ready(), obs)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                key, k_step = jax.random.split(key)
                obs, st, *_ = step_jit(k_step, st, noop_actions)
            jax.tree.map(lambda x: x.block_until_ready(), obs)
            return n_steps / (time.perf_counter() - t0)
    else:
        v_reset = jax.jit(jax.vmap(env.reset))
        v_step = jax.jit(jax.vmap(env.step))
        noop_actions = {
            agent: jax.device_put(
                jnp.full(batch_size, 4, dtype=jnp.int32), device
            )
            for agent in agents
        }

        # warmup
        keys = jax.device_put(jax.random.split(jax.random.key(0), batch_size), device)
        obs, states = v_reset(keys)
        jax.tree.map(lambda x: x.block_until_ready(), obs)
        step_keys = jax.device_put(
            jax.random.split(jax.random.key(1), batch_size), device
        )
        for _ in range(N_WARMUP):
            obs, states, *_ = v_step(step_keys, states, noop_actions)
            jax.tree.map(lambda x: x.block_until_ready(), obs)

        def trial():
            rng = jax.device_put(jax.random.key(SEED), device)
            rng, k_reset = jax.random.split(rng)
            reset_keys = jax.random.split(k_reset, batch_size)
            obs, st = v_reset(reset_keys)
            jax.tree.map(lambda x: x.block_until_ready(), obs)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                rng, k_step = jax.random.split(rng)
                s_keys = jax.random.split(k_step, batch_size)
                obs, st, *_ = v_step(s_keys, st, noop_actions)
            jax.tree.map(lambda x: x.block_until_ready(), obs)
            return (n_steps * batch_size) / (time.perf_counter() - t0)

    return _run_trials(trial)


# ===================================================================
# Runner
# ===================================================================


def run_scaling_benchmark():
    """Sweep batch sizes for JAX backends on CPU and GPU."""
    import jax

    # Discover devices
    devices = {"cpu": jax.devices("cpu")[0]}
    try:
        devices["gpu"] = jax.devices("gpu")[0]
    except RuntimeError:
        pass
    has_gpu = "gpu" in devices

    device_names = ", ".join(
        f"{name} ({dev.device_kind})" for name, dev in devices.items()
    )
    print("Overcooked Cramped Room -- Scaling Benchmark")
    print(f"  Devices: {device_names}")
    print()

    results = {
        "batch_sizes": BATCH_SIZES,
        "has_gpu": has_gpu,
        "cogrid_numpy": None,
        "overcooked_ai": None,
        "cogrid_jax": {name: {} for name in devices},
        "jaxmarl": {name: {} for name in devices},
    }

    # --- numpy baselines (single env) ---------------------------------
    print("  Reference baselines:")
    for label, key, fn in [
        ("CoGrid NumPy", "cogrid_numpy", bench_cogrid_numpy),
        ("overcooked_ai", "overcooked_ai", bench_overcooked_ai),
    ]:
        try:
            trials = fn()
            med = statistics.median(trials)
            results[key] = med
            print(f"    {label + ':':<20s}{_fmt(med)} steps/sec")
        except Exception as e:
            print(f"    {label + ':':<20s} SKIPPED ({type(e).__name__}: {e})")
    print()

    # --- JAX scaling sweep per library --------------------------------
    dev_names = list(devices.keys())  # e.g. ["cpu", "gpu"]

    for lib_key, lib_label, bench_fn in [
        ("cogrid_jax", "CoGrid JAX", bench_cogrid_jax),
        ("jaxmarl", "JaxMARL", bench_jaxmarl),
    ]:
        print(f"  {lib_label}:")

        # header
        header = f"  {'Batch':>10s}"
        for name in dev_names:
            header += f"   {name.upper() + ' steps/s':>16s}"
        print(header)

        for bs in BATCH_SIZES:
            row = f"  {bs:>10,d}"

            for dev_name in dev_names:
                dev = devices[dev_name]
                try:
                    trials = bench_fn(bs, dev)
                    med = statistics.median(trials)
                    results[lib_key][dev_name][bs] = med
                    row += f"   {_fmt(med)}"
                except Exception:
                    results[lib_key][dev_name][bs] = None
                    row += f"   {'SKIP':>16s}"

            print(row)

        print()

    return results


# ===================================================================
# Plotting
# ===================================================================


def plot_scaling(results, output_path="cogrid/benchmarks/scaling_benchmark.png"):
    import matplotlib.pyplot as plt

    batch_sizes = results["batch_sizes"]
    has_gpu = results["has_gpu"]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Color families: CoGrid = blues, JaxMARL/overcooked_ai = oranges.
    # CPU vs GPU distinguished by marker shape and linestyle.
    COGRID_DARK = "#1a5fb4"   # dark blue  — JAX GPU
    COGRID_MED = "#4a90d9"    # mid blue   — JAX CPU
    COGRID_LIGHT = "#99c1f1"  # light blue — NumPy baseline

    JAXMARL_DARK = "#c64600"  # dark orange — JAX GPU
    JAXMARL_MED = "#e5841a"   # mid orange  — JAX CPU
    JAXMARL_LIGHT = "#f9b97a" # light orange — overcooked_ai baseline

    styles = [
        ("cogrid_jax", "cpu", "CoGrid JAX (CPU)", COGRID_MED, "o", "-"),
        ("jaxmarl", "cpu", "JaxMARL (CPU)", JAXMARL_MED, "s", "-"),
    ]
    if has_gpu:
        styles += [
            ("cogrid_jax", "gpu", "CoGrid JAX (GPU)", COGRID_DARK, "^", "--"),
            ("jaxmarl", "gpu", "JaxMARL (GPU)", JAXMARL_DARK, "D", "--"),
        ]

    for lib, dev, label, color, marker, ls in styles:
        data = results[lib].get(dev, {})
        xs = [bs for bs in batch_sizes if data.get(bs) is not None]
        ys = [data[bs] for bs in xs]
        if xs:
            ax.plot(
                xs, ys,
                linestyle=ls, color=color, marker=marker, markersize=6,
                label=label, linewidth=2,
            )

    # numpy reference lines — same color families, lighter shades
    baselines = [
        ("cogrid_numpy", "CoGrid NumPy", COGRID_LIGHT),
        ("overcooked_ai", "overcooked_ai", JAXMARL_LIGHT),
    ]
    for key, label, color in baselines:
        val = results.get(key)
        if val is not None:
            ax.axhline(val, color=color, linestyle="-", linewidth=2.5, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Number of parallel environments")
    ax.set_ylabel("Total throughput (steps/sec)")
    ax.set_title("Overcooked Cramped Room -- Throughput Scaling")
    ax.legend(fontsize=9)
    ax.grid(True, which="major", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Plot saved to {output_path}")
    plt.close(fig)


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    results = run_scaling_benchmark()

    try:
        plot_scaling(results)
    except Exception as e:
        print(f"  Plotting skipped ({type(e).__name__}: {e})")
