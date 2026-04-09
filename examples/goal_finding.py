"""Goal-Finding Environment Example

Demonstrates how to create a custom CoGrid environment from scratch and
run it on both the numpy and JAX backends.

The environment is a simple grid where agents navigate to a goal cell.
It shows the component API -- all you need is:

    1. Register object types and a layout
    2. Register a Reward subclass
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

from cogrid.core.objects import GridObj, register_object_type, when
from cogrid.core.constants import Colors
from cogrid.core.pipeline.context import clear_facing_cell


@register_object_type("goal")
class Goal(GridObj):
    """A goal cell that agents can walk onto."""

    color = Colors.Green
    char = "g"
    can_overlap = when()

    def __init__(self, **kwargs):
        super().__init__(state=0)


# -- 2. Register a layout ----------------------------------------------------
#
# Layouts are ASCII strings. Special characters:
#   '#' = wall    '+' = spawn point    ' ' = empty    'g' = goal (ours)

from cogrid.core.grid import layouts

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


# -- 3. Define a Reward subclass ------------------------------------------------
#
# Reward.compute() returns final (n_agents,) rewards. The autowire
# layer just sums all reward instances -- coefficient weighting and
# broadcasting are the reward's responsibility.

from cogrid.core.pipeline.rewards import InteractionReward


class GoalReward(InteractionReward):
    """Reward for standing on a goal cell."""

    action = None
    overlaps = "goal"


# -- 4. Termination function --------------------------------------------------
#
# terminated_fn is scope-specific termination logic that is not part of
# the Reward interface. We patch it onto the env after creation.


def goal_terminated(prev_state, state, reward_config):
    """Terminate agents that are standing on the goal cell."""
    goal_id = reward_config["type_ids"].get("goal", -1)
    otm = state.object_type_map
    rows = state.agent_pos[:, 0]
    cols = state.agent_pos[:, 1]
    return otm[rows, cols] == goal_id


# -- 5. Interaction function ---------------------------------------------------
#
# Interaction functions have signature (ctx) -> (should_apply, changes).
# ctx is an InteractionContext with standard fields (facing_type, can_interact,
# agent_index, type_ids, etc.) plus any extra_state arrays declared by components.


def collect_goal(ctx):
    """Remove a goal when the agent interacts with it."""
    goal_id = ctx.type_ids["goal"]
    is_pickup = ctx.action == ctx.action_id.pickup_drop
    should_apply = ctx.can_interact & is_pickup & (ctx.facing_type == goal_id)
    changes = {
        "object_type_map": clear_facing_cell(ctx),
    }
    return should_apply, changes


# -- 6. Register the environment using CoGridEnv directly ---------------------
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
    "rewards": [GoalReward(coefficient=1.0, common_reward=True)],
    "grid": {"layout": "goal_simple_v0"},
    "max_steps": 50,
    "scope": "global",
    "terminated_fn": goal_terminated,
    "interactions": [collect_goal],
}

registry.register(
    "GoalFinding-Simple-V0",
    functools.partial(CoGridEnv, config=goal_config),
)


# -- 7. Run on numpy ----------------------------------------------------------


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


# -- 8. Run on JAX ------------------------------------------------------------


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
    step_fn = env.jax_step  # JIT-compiled step function
    reset_fn = env.jax_reset  # JIT-compiled reset function

    key = jax.random.key(0)
    obs, state, _ = reset_fn(key)
    actions = {0: jnp.int32(0), 1: jnp.int32(3)}  # Agent 0: Up, Agent 1: Right
    key, step_key = jax.random.split(key)
    obs, state, rew, terminateds_d, truncateds_d, _ = step_fn(step_key, state, actions)
    print(
        f"Functional API -- reward: {rew[0]}, terms: {terminateds_d[0]}, truncs: {truncateds_d[0]}"
    )
    print()

    # --- Batched rollouts with vmap ---
    n_envs = 1024
    keys = jax.random.split(jax.random.key(1), n_envs)

    batched_reset = jax.jit(jax.vmap(reset_fn))
    batched_step = jax.jit(jax.vmap(step_fn))

    batched_obs, batched_state, _ = batched_reset(keys)
    print(f"vmap reset -- {n_envs} envs, obs shape: {batched_obs[0].shape}")

    # Run 50 steps across all 1024 envs with random cardinal actions
    n_steps = 50
    action_key = jax.random.key(42)
    total_reward = jnp.float32(0.0)
    for _ in range(n_steps):
        action_key, subkey = jax.random.split(action_key)
        arr = jax.random.randint(subkey, (n_envs, 2), 0, 4)
        batched_actions = {0: arr[:, 0], 1: arr[:, 1]}
        subkey, step_subkey = jax.random.split(subkey)
        step_keys = jax.random.split(step_subkey, n_envs)
        batched_obs, batched_state, batched_rew, *_ = batched_step(
            step_keys, batched_state, batched_actions
        )
        total_reward += sum(v.sum() for v in batched_rew.values())

    total_reward /= n_envs
    print(f"vmap step x{n_steps} -- total reward across batch: {float(total_reward):.1f}")
    print()


# -- Main ---------------------------------------------------------------------

if __name__ == "__main__":
    run_numpy()

    try:
        import jax  # noqa: F401

        run_jax()
    except ImportError:
        print("JAX not installed -- skipping JAX examples.")
        print("Install with: pip install jax jaxlib")
