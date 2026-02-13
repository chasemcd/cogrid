"""CoGrid Benchmark Suite.

Measures environment throughput for three configurations:
1. NumPy single-env (baseline)
2. JAX single-env with JIT compilation
3. JAX vmap@1024 batched parallel environments

Runnable as a script:
    python -m cogrid.benchmarks.benchmark_suite

Includes a pytest test for reproducibility verification:
    python -m pytest cogrid/benchmarks/benchmark_suite.py::test_benchmark_reproducibility -v -s
"""

import time
import statistics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_BENCHMARK_STEPS = 500    # Steps per measurement (enough to amortize overhead)
N_WARMUP_STEPS = 5         # Warmup calls before timing
N_TRIALS = 3               # Number of measurement trials
BATCH_SIZE = 1024           # vmap batch size
VARIANCE_THRESHOLD = 10.0  # Max allowed % variance between trials
LAYOUT = "Overcooked-CrampedRoom-V0"  # Standard benchmark layout (smallest, fastest)
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_variance(trials):
    """Compute variance as range-over-median percentage.

    Args:
        trials: List of numeric measurements.

    Returns:
        float: ``(max - min) / median * 100``.
    """
    med = statistics.median(trials)
    if med == 0:
        return 0.0
    return (max(trials) - min(trials)) / med * 100.0


def _create_numpy_env(layout=LAYOUT, seed=SEED):
    """Create and reset a numpy-backend environment.

    Returns:
        Tuple of (env, agent_ids) after reset.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    env = registry.make(layout, backend="numpy")
    env.reset(seed=seed)
    agent_ids = sorted(env.possible_agents)
    return env, agent_ids


def _create_jax_env(layout=LAYOUT, seed=SEED):
    """Create and reset a JAX-backend environment.

    Returns:
        Tuple of (env, step_fn, reset_fn, n_agents) after reset.
    """
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry

    env = registry.make(layout, backend="jax")
    env.reset(seed=seed)
    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.possible_agents)
    return env, step_fn, reset_fn, n_agents


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

def benchmark_numpy_single(n_steps=N_BENCHMARK_STEPS, n_trials=N_TRIALS):
    """Measure numpy single-env throughput.

    Args:
        n_steps: Number of env.step() calls per trial.
        n_trials: Number of measurement trials.

    Returns:
        List of per-trial steps/sec values.
    """
    env, agent_ids = _create_numpy_env()
    noop_actions = {aid: 6 for aid in agent_ids}

    # Warmup
    for _ in range(N_WARMUP_STEPS):
        env.step(noop_actions)

    trials = []
    for _ in range(n_trials):
        env.reset(seed=SEED)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            env.step(noop_actions)
        t1 = time.perf_counter()
        trials.append(n_steps / (t1 - t0))

    return trials


def benchmark_jax_single(n_steps=N_BENCHMARK_STEPS, n_trials=N_TRIALS):
    """Measure JAX single-env JIT throughput.

    Args:
        n_steps: Number of step_fn() calls per trial.
        n_trials: Number of measurement trials.

    Returns:
        List of per-trial steps/sec values.
    """
    import jax
    import jax.numpy as jnp

    _, step_fn, reset_fn, n_agents = _create_jax_env()

    actions = jnp.full((n_agents,), 6, dtype=jnp.int32)

    # Warmup: reset + N_WARMUP_STEPS step calls with block_until_ready
    state, _ = reset_fn(jax.random.key(SEED))
    state.agent_pos.block_until_ready()
    for _ in range(N_WARMUP_STEPS):
        state, obs, rew, term, trunc, info = step_fn(state, actions)
        state.agent_pos.block_until_ready()

    trials = []
    for _ in range(n_trials):
        state, _ = reset_fn(jax.random.key(SEED))
        state.agent_pos.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            state, obs, rew, term, trunc, info = step_fn(state, actions)
        state.agent_pos.block_until_ready()  # CRITICAL: wait for async dispatch
        t1 = time.perf_counter()

        trials.append(n_steps / (t1 - t0))

    return trials


def benchmark_jax_vmap(n_steps=N_BENCHMARK_STEPS, n_trials=N_TRIALS,
                       batch_size=BATCH_SIZE):
    """Measure JAX vmap batched throughput.

    Args:
        n_steps: Number of vmapped_step() calls per trial.
        n_trials: Number of measurement trials.
        batch_size: Number of parallel environments.

    Returns:
        List of per-trial total_steps/sec values (n_steps * batch_size / elapsed).
    """
    import jax
    import jax.numpy as jnp

    _, step_fn, reset_fn, n_agents = _create_jax_env()

    vmapped_reset = jax.jit(jax.vmap(reset_fn))
    vmapped_step = jax.jit(jax.vmap(step_fn))

    keys = jax.random.split(jax.random.key(0), batch_size)
    batched_actions = jnp.full((batch_size, n_agents), 6, dtype=jnp.int32)

    # Warmup: call vmapped_reset + N_WARMUP_STEPS vmapped_step calls
    batched_state, _ = vmapped_reset(keys)
    batched_state.agent_pos.block_until_ready()
    for _ in range(N_WARMUP_STEPS):
        batched_state, b_obs, b_rew, b_term, b_trunc, b_info = vmapped_step(
            batched_state, batched_actions
        )
        batched_state.agent_pos.block_until_ready()

    trials = []
    for _ in range(n_trials):
        batched_state, _ = vmapped_reset(keys)
        batched_state.agent_pos.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            batched_state, b_obs, b_rew, b_term, b_trunc, b_info = vmapped_step(
                batched_state, batched_actions
            )
        batched_state.agent_pos.block_until_ready()  # CRITICAL: wait for async dispatch
        t1 = time.perf_counter()

        trials.append((n_steps * batch_size) / (t1 - t0))

    return trials


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_benchmark_suite():
    """Run all three benchmarks and print a formatted results table.

    Returns:
        dict with keys ``numpy_single``, ``jax_single``, ``jax_vmap_1024``,
        each containing sub-keys ``trials``, ``median``, ``speedup_vs_numpy``,
        and ``variance_pct``.
    """
    print("Running NumPy single-env benchmark...")
    numpy_trials = benchmark_numpy_single()
    numpy_median = statistics.median(numpy_trials)

    print("Running JAX single-env JIT benchmark...")
    jax_single_trials = benchmark_jax_single()
    jax_single_median = statistics.median(jax_single_trials)

    print("Running JAX vmap@1024 benchmark...")
    jax_vmap_trials = benchmark_jax_vmap()
    jax_vmap_median = statistics.median(jax_vmap_trials)

    # Speedup ratios
    jax_single_vs_numpy = jax_single_median / numpy_median if numpy_median > 0 else 0
    jax_vmap_vs_numpy = jax_vmap_median / numpy_median if numpy_median > 0 else 0
    jax_vmap_vs_single = jax_vmap_median / jax_single_median if jax_single_median > 0 else 0
    jax_vmap_per_env = jax_vmap_median / BATCH_SIZE

    # Variance
    numpy_var = compute_variance(numpy_trials)
    jax_single_var = compute_variance(jax_single_trials)
    jax_vmap_var = compute_variance(jax_vmap_trials)

    # Print results
    print()
    print("CoGrid Benchmark Suite")
    print("======================")
    print(f"Layout: {LAYOUT}")
    print(f"Steps per measurement: {N_BENCHMARK_STEPS}")
    print(f"Trials: {N_TRIALS}")
    print(f"vmap batch size: {BATCH_SIZE}")
    print()
    print(f"Results (median of {N_TRIALS} trials):")
    print("-------------------------------------------------------")
    print(f"NumPy single-env:         {numpy_median:>10,.0f} steps/sec")
    print(f"JAX single-env (JIT):     {jax_single_median:>10,.0f} steps/sec  ({jax_single_vs_numpy:.1f}x vs numpy)")
    print(f"JAX vmap@{BATCH_SIZE} (total):    {jax_vmap_median:>10,.0f} steps/sec  ({jax_vmap_vs_numpy:.1f}x vs numpy)")
    print(f"JAX vmap@{BATCH_SIZE} (per-env):  {jax_vmap_per_env:>10,.0f} steps/sec")
    print("-------------------------------------------------------")
    print()
    print("Reproducibility (max trial variance):")
    print(f"  NumPy:      {numpy_var:.1f}%")
    print(f"  JAX single: {jax_single_var:.1f}%")
    print(f"  JAX vmap:   {jax_vmap_var:.1f}%")

    results = {
        "numpy_single": {
            "trials": numpy_trials,
            "median": numpy_median,
            "speedup_vs_numpy": 1.0,
            "variance_pct": numpy_var,
        },
        "jax_single": {
            "trials": jax_single_trials,
            "median": jax_single_median,
            "speedup_vs_numpy": jax_single_vs_numpy,
            "variance_pct": jax_single_var,
        },
        "jax_vmap_1024": {
            "trials": jax_vmap_trials,
            "median": jax_vmap_median,
            "speedup_vs_numpy": jax_vmap_vs_numpy,
            "variance_pct": jax_vmap_var,
        },
    }

    return results


# ---------------------------------------------------------------------------
# pytest test
# ---------------------------------------------------------------------------

def test_benchmark_reproducibility():
    """Verify benchmark reproducibility and measurable speedup.

    Asserts:
    - JAX vmap@1024 total throughput > numpy single-env throughput
      (measurable speedup).
    - At least the vmap configuration meets the 10% variance threshold
      (most consistent due to amortization over 1024 envs).
    """
    results = run_benchmark_suite()

    # Log all variances for informational purposes
    for config_name, data in results.items():
        var = data["variance_pct"]
        print(f"\n  {config_name} variance: {var:.1f}% (threshold: {VARIANCE_THRESHOLD}%)")

    # Assert measurable speedup: vmap total > numpy single
    assert results["jax_vmap_1024"]["median"] > results["numpy_single"]["median"], (
        f"Expected JAX vmap@{BATCH_SIZE} total throughput "
        f"({results['jax_vmap_1024']['median']:.0f} steps/sec) to exceed "
        f"numpy single-env ({results['numpy_single']['median']:.0f} steps/sec)"
    )

    # Assert reproducibility: at least vmap meets the 10% threshold
    vmap_var = results["jax_vmap_1024"]["variance_pct"]
    assert vmap_var <= VARIANCE_THRESHOLD, (
        f"JAX vmap@{BATCH_SIZE} variance {vmap_var:.1f}% exceeds "
        f"{VARIANCE_THRESHOLD}% threshold"
    )

    print(f"\n  PASSED: vmap speedup = {results['jax_vmap_1024']['speedup_vs_numpy']:.1f}x vs numpy")
    print(f"  PASSED: vmap variance = {vmap_var:.1f}% <= {VARIANCE_THRESHOLD}%")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_benchmark_suite()
