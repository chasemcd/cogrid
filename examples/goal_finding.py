"""Goal-Finding Environment Example

Demonstrates how to create a custom CoGrid environment from scratch and
run it on both the numpy and JAX backends.

The environment is a simple grid where agents navigate to a goal cell.
It shows the three things you need to define a new environment:

    1. Register object types and a layout
    2. Write a reward function
    3. Create a thin env subclass

Then run it:
    python examples/goal_finding.py
"""

import numpy as np

# ── 1. Register a "goal" object type ────────────────────────────────
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


# ── 2. Register a layout ────────────────────────────────────────────
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


# ── 3. Write a reward function ──────────────────────────────────────
#
# Reward functions receive two state dicts (before and after the step),
# the actions taken, and a config dict.  They return a float32 array
# of shape (n_agents,).
#
# All array ops use `xp` so the same code runs on numpy and JAX.

def goal_reward(prev_state, state, actions, type_ids, n_agents,
                coefficient=1.0, common_reward=True, **kwargs):
    """Reward agents for standing on the goal cell."""
    from cogrid.backend import xp

    goal_id = type_ids.get("goal", -1)
    otm = state["object_type_map"]

    # Check which agents are on a goal cell
    rows = state["agent_pos"][:, 0]
    cols = state["agent_pos"][:, 1]
    on_goal = (otm[rows, cols] == goal_id).astype(xp.float32)

    rewards = on_goal * coefficient
    if common_reward:
        rewards = xp.full(n_agents, xp.sum(rewards), dtype=xp.float32)
    return rewards


def compute_rewards(prev_state, state, actions, reward_config):
    """Compose all reward functions (just goal_reward here)."""
    from cogrid.backend import xp

    n_agents = reward_config["n_agents"]
    total = xp.zeros(n_agents, dtype=xp.float32)
    for spec in reward_config["rewards"]:
        total = total + goal_reward(
            prev_state, state, actions,
            type_ids=reward_config["type_ids"],
            n_agents=n_agents,
            coefficient=spec.get("coefficient", 1.0),
            common_reward=spec.get("common_reward", True),
        )
    return total


def goal_terminated(prev_state, state, reward_config):
    """Terminate agents that are standing on the goal cell."""
    from cogrid.backend import xp

    goal_id = reward_config["type_ids"].get("goal", -1)
    otm = state["object_type_map"]
    rows = state["agent_pos"][:, 0]
    cols = state["agent_pos"][:, 1]
    return otm[rows, cols] == goal_id


# ── 4. Create the environment class ─────────────────────────────────
#
# A minimal subclass that sets the reward compute_fn.
# Everything else (step pipeline, observation building, JIT compilation)
# is handled by CoGridEnv.

from cogrid.cogrid_env import CoGridEnv


class GoalFindingEnv(CoGridEnv):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)

        # Build type_ids for the reward function
        from cogrid.core.grid_object import object_to_idx
        self._reward_config["type_ids"] = {
            "goal": object_to_idx("goal", scope=self.scope),
        }
        self._reward_config["compute_fn"] = compute_rewards
        self._reward_config["terminated_fn"] = goal_terminated
        self._reward_config["rewards"] = [
            {"fn": "goal", "coefficient": 1.0, "common_reward": True},
        ]


# ── 5. Register the environment ─────────────────────────────────────

import functools
from cogrid.envs import registry

goal_config = {
    "name": "goal_finding",
    "num_agents": 2,
    "action_set": "cardinal_actions",
    "features": ["agent_position", "full_map_encoding"],
    "rewards": [],  # We use the functional reward pipeline, not the legacy one
    "grid": {"layout": "goal_simple_v0"},
    "max_steps": 50,
    "scope": "global",
}

registry.register(
    "GoalFinding-Simple-V0",
    functools.partial(GoalFindingEnv, config=goal_config),
)


# ── 6. Run on numpy ─────────────────────────────────────────────────

def run_numpy():
    """Run the environment on the numpy backend."""
    from cogrid.backend._dispatch import _reset_backend_for_testing
    _reset_backend_for_testing()

    env = registry.make("GoalFinding-Simple-V0", backend="numpy")
    obs, info = env.reset(seed=42)

    print("=== NumPy Backend ===")
    print(f"Agents: {env.possible_agents}")
    print(f"Obs shapes: { {k: v.shape for k, v in obs.items()} }")

    total_reward = 0.0
    for step_i in range(50):
        # Random cardinal actions: 0=Up, 1=Down, 2=Left, 3=Right
        actions = {aid: np.random.randint(0, 4) for aid in env.possible_agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        step_reward = sum(rewards.values())
        if step_reward > 0:
            print(f"  Step {step_i}: agent reached the goal! reward={step_reward:.1f}")
        total_reward += step_reward

    print(f"Total reward over 50 steps: {total_reward:.1f}")
    print()


# ── 7. Run on JAX ───────────────────────────────────────────────────

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
    total_reward = 0.0
    for step_i in range(50):
        actions = {aid: np.random.randint(0, 4) for aid in env.possible_agents}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
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

    # Run 10 steps across all 1024 envs simultaneously
    batched_actions = jnp.zeros((n_envs, 2), dtype=jnp.int32)  # All noop
    for _ in range(10):
        batched_state, batched_obs, batched_rew, batched_term, batched_trunc, _ = (
            batched_step(batched_state, batched_actions)
        )

    print(f"vmap step x10 -- reward sum across batch: {float(batched_rew.sum()):.1f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_numpy()

    try:
        import jax
        run_jax()
    except ImportError:
        print("JAX not installed -- skipping JAX examples.")
        print("Install with: pip install jax jaxlib")
