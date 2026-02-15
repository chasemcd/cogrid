"""Goal-Finding Environment Example

Demonstrates how to create a custom CoGrid environment from scratch and
run it on both the numpy and JAX backends.

The environment is a simple grid where agents navigate to a goal cell.
It shows the component API -- all you need is:

    1. Register object types and a layout
    2. Register an ArrayReward subclass
    3. Use CoGridEnv directly (no env subclass needed)

Then run it:
    python examples/goal_finding.py
"""

import numpy as np

# -- 1. Register a "goal" object type ----------------------------------------
#
# CoGrid uses a global object registry. Each object type has a char
# (for ASCII layouts), a color (for rendering), and boolean properties
# that control how agents interact with it.

from cogrid.core.grid_object import GridObj, register_object_type
from cogrid.core.constants import Colors


@register_object_type("goal", can_overlap=True)
class Goal(GridObj):
    """A goal cell that agents can walk onto."""
    object_id = "goal"
    color = Colors.Green
    char = "g"

    def __init__(self, **kwargs):
        super().__init__(state=0)


# -- 2. Register a layout ----------------------------------------------------
#
# Layouts are ASCII strings. Special characters:
#   '#' = wall    '+' = spawn point    ' ' = empty    'g' = goal (ours)

from cogrid.core import layouts

layouts.register_layout(
    "goal_simple_v0",
    [
        "#######",
        "#  g  #",
        "#     #",
        "#     #",
        "#     #",
        "# +  +#",
        "#######",
    ],
)


# -- 3. Register an ArrayReward subclass --------------------------------------
#
# ArrayReward.compute() returns raw unweighted per-agent rewards. The
# autowire's composed compute_fn handles coefficient weighting and
# common_reward broadcasting automatically.

from cogrid.core.array_rewards import ArrayReward
from cogrid.core.component_registry import register_reward_type


@register_reward_type("goal", scope="global", coefficient=1.0, common_reward=True)
class GoalReward(ArrayReward):
    def compute(self, prev_state, state, actions, reward_config):
        from cogrid.backend import xp

        goal_id = reward_config["type_ids"].get("goal", -1)
        otm = state.object_type_map
        rows = state.agent_pos[:, 0]
        cols = state.agent_pos[:, 1]
        on_goal = (otm[rows, cols] == goal_id).astype(xp.float32)
        return on_goal


# -- 4. Termination function --------------------------------------------------
#
# terminated_fn is scope-specific termination logic that is not part of
# the ArrayReward interface. We patch it onto the env after creation.

def goal_terminated(prev_state, state, reward_config):
    """Terminate agents that are standing on the goal cell."""
    from cogrid.backend import xp

    goal_id = reward_config["type_ids"].get("goal", -1)
    otm = state.object_type_map
    rows = state.agent_pos[:, 0]
    cols = state.agent_pos[:, 1]
    return otm[rows, cols] == goal_id


# -- 5. Register the environment using CoGridEnv directly ---------------------
#
# No env subclass needed -- the component API (registered GoalReward +
# auto-wiring) handles everything. The registry entry points to CoGridEnv.

import functools
from cogrid.cogrid_env import CoGridEnv
from cogrid.envs import registry

goal_config = {
    "name": "goal_finding",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["agent_dir", "agent_position", "can_move_direction", "inventory"],
    "rewards": [],  # We use the ArrayReward component pipeline, not the legacy one
    "grid": {"layout": "goal_simple_v0"},
    "max_steps": 50,
    "scope": "global",
    "terminated_fn": goal_terminated,
}

registry.register(
    "GoalFinding-Simple-V0",
    functools.partial(CoGridEnv, config=goal_config),
)


# -- 6. Run on numpy ----------------------------------------------------------

def run_numpy():
    """Run the environment on the numpy backend."""
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    env = registry.make("GoalFinding-Simple-V0", backend="numpy")
    obs, info = env.reset(seed=42)

    print("=== NumPy Backend ===")
    print(f"Agents: {env.possible_agents}")
    print(f"Obs shapes: { {k: v.shape for k, v in obs.items()} }")

    rng = np.random.default_rng(0)
    total_reward = 0.0
    for step_i in range(50):
        # Random cardinal actions: 0=Up, 1=Down, 2=Left, 3=Right
        actions = {aid: rng.integers(0, 4) for aid in env.possible_agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        step_reward = sum(rewards.values())
        if step_reward > 0:
            print(f"  Step {step_i}: agent reached the goal! reward={step_reward:.1f}")
        total_reward += step_reward

    print(f"Total reward over 50 steps: {total_reward:.1f}")
    print()


# -- 7. Run on JAX ------------------------------------------------------------

def run_jax():
    """Run the environment on the JAX backend, then vmap over 1024 envs."""
    import jax
    import jax.numpy as jnp
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    # --- Single environment ---
    env = registry.make("GoalFinding-Simple-V0", backend="jax")
    obs, info = env.reset(seed=42)

    print("=== JAX Backend ===")
    print(f"Obs shapes: { {k: v.shape for k, v in obs.items()} }")

    # Step through the PettingZoo API (same as numpy)
    rng = np.random.default_rng(0)
    total_reward = 0.0
    for _ in range(50):
        actions = {aid: rng.integers(0, 4) for aid in env.possible_agents}
        obs, rewards, *_ = env.step(actions)
        total_reward += sum(rewards.values())

    print(f"Total reward over 50 steps: {total_reward:.1f}")
    print()

    # --- Direct functional API with JIT ---
    step_fn = env.jax_step    # JIT-compiled (state, actions) -> (state, obs, rew, terminateds, truncateds, info)
    reset_fn = env.jax_reset  # JIT-compiled (rng_key) -> (state, obs)

    key = jax.random.key(0)
    state, obs = reset_fn(key)
    actions = jnp.array([0, 3], dtype=jnp.int32)  # Agent 0: Up, Agent 1: Right
    state, obs, rew, terminateds_arr, truncateds_arr, _ = step_fn(state, actions)
    print(f"Functional API -- reward: {rew}, terminateds: {terminateds_arr}, truncateds: {truncateds_arr}")
    print()

    # --- Batched rollouts with vmap ---
    n_envs = 1024
    keys = jax.random.split(jax.random.key(1), n_envs)

    batched_reset = jax.jit(jax.vmap(reset_fn))
    batched_step = jax.jit(jax.vmap(step_fn))

    batched_state, batched_obs = batched_reset(keys)
    print(f"vmap reset -- {n_envs} envs, obs shape: {batched_obs.shape}")

    # Run 50 steps across all 1024 envs with random cardinal actions
    n_steps = 50
    action_key = jax.random.key(42)
    total_reward = jnp.float32(0.0)
    for _ in range(n_steps):
        action_key, subkey = jax.random.split(action_key)
        batched_actions = jax.random.randint(subkey, (n_envs, 2), 0, 4)
        batched_state, batched_obs, batched_rew, *_ = (
            batched_step(batched_state, batched_actions)
        )
        total_reward += batched_rew.sum()

    total_reward /= n_envs
    print(f"vmap step x{n_steps} -- total reward across batch: {float(total_reward):.1f}")
    print()


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    run_numpy()

    try:
        import jax
        run_jax()
    except ImportError:
        print("JAX not installed -- skipping JAX examples.")
        print("Install with: pip install jax jaxlib")
