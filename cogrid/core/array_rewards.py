"""Generic array-based reward composition utility.

The ``compose_rewards`` function builds a composed reward callable from a list
of reward configs. It is environment-agnostic -- it simply calls whatever ``fn``
is provided in each config dict.

Environment-specific reward functions live in their respective envs/ modules:
- Overcooked: ``cogrid.envs.overcooked.array_rewards``
"""


def compose_rewards(reward_configs: list) -> callable:
    """Build a composed reward function from reward configs.

    Called at init time. Each config dict has:
        - 'fn': one of the reward functions above
        - 'coefficient': float scaling factor
        - 'common_reward': bool (optional, default False)

    Returns a closure ``(prev_state, state, actions, n_agents, type_ids, action_pickup_drop_idx) -> reward_array``
    that sums all configured rewards.
    """
    def composed_reward(prev_state, state, actions, n_agents, type_ids, action_pickup_drop_idx=4):
        from cogrid.backend import xp

        total = xp.zeros(n_agents, dtype=xp.float32)
        for config in reward_configs:
            r = config['fn'](
                prev_state, state, actions, type_ids, n_agents,
                coefficient=config['coefficient'],
                common_reward=config.get('common_reward', False),
                action_pickup_drop_idx=action_pickup_drop_idx,
            )
            total = total + r
        return total

    return composed_reward


# Backward-compatible re-exports (moved to cogrid.envs.overcooked.array_rewards)
# TODO: Remove after Phase 01.1 Plan 03 completes wiring
from cogrid.envs.overcooked.array_rewards import (
    delivery_reward_array,
    onion_in_pot_reward_array,
    soup_in_dish_reward_array,
)
