"""Overcooked-specific array-based reward functions.

These reward functions operate on state dictionaries (array state) instead of
Grid objects. They are specific to the Overcooked environment because they
reference Overcooked types: pot, onion_soup, delivery_zone, plate, onion.

Moved from cogrid/core/array_rewards.py (Phase 01.1, Plan 02) to enforce
separation of concerns: environment-specific logic lives under envs/.

Each reward function has the signature:
    reward_fn(prev_state, state, actions, type_ids, n_agents, ...) -> ndarray of shape (n_agents,)

State dicts contain:
    agent_pos:        (n_agents, 2) int32 -- [row, col]
    agent_dir:        (n_agents,)   int32 -- direction enum (Right=0, Down=1, Left=2, Up=3)
    agent_inv:        (n_agents, 1) int32 -- inventory type_id, -1 = empty
    object_type_map:  (H, W) int32        -- type_id at each cell
    object_state_map: (H, W) int32        -- state value at each cell
    pot_contents:     (n_pots, 3) int32   -- ingredient type_ids, -1 = empty slot
    pot_timer:        (n_pots,)   int32   -- cooking timer (0 = ready)
    pot_positions:    (n_pots, 2) int32   -- pot locations

All functions use ``from cogrid.backend import xp`` for backend-agnostic
array operations. Fully vectorized across agents -- no Python loops, no
int() casts.
"""

from cogrid.backend import xp
from cogrid.core.array_rewards import ArrayReward, register_reward_type


def _compute_fwd_positions(prev_state):
    """Compute forward positions, clipped indices, bounds mask, and forward type IDs."""
    # Direction vector table: Right=0, Down=1, Left=2, Up=3
    dir_vec_table = xp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32)

    fwd_pos = (
        prev_state.agent_pos + dir_vec_table[prev_state.agent_dir]
    )  # (n_agents, 2)
    H, W = prev_state.object_type_map.shape
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)
    in_bounds = (
        (fwd_pos[:, 0] >= 0)
        & (fwd_pos[:, 0] < H)
        & (fwd_pos[:, 1] >= 0)
        & (fwd_pos[:, 1] < W)
    )
    fwd_types = prev_state.object_type_map[fwd_r, fwd_c]  # (n_agents,)

    return fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types


def delivery_reward(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=1.0,
    common_reward=True,
    action_pickup_drop_idx=4,
):
    """Reward for delivering soup to a DeliveryZone. Fully vectorized."""
    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(
        prev_state
    )

    is_interact = actions == action_pickup_drop_idx  # (n_agents,)
    holds_soup = (
        prev_state.agent_inv[:, 0] == type_ids["onion_soup"]
    )  # (n_agents,)
    faces_delivery = fwd_types == type_ids["delivery_zone"]  # (n_agents,)

    earns_reward = (
        is_interact & holds_soup & faces_delivery & in_bounds
    )  # (n_agents,)

    # Apply reward: in common_reward mode, every earning agent adds coefficient
    # to ALL agents. This matches: `rewards = rewards + coefficient` per earner.
    if common_reward:
        n_earners = xp.sum(earns_reward.astype(xp.float32))
        rewards = xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
    else:
        rewards = earns_reward.astype(xp.float32) * coefficient

    return rewards


def onion_in_pot_reward(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=0.1,
    common_reward=False,
    action_pickup_drop_idx=4,
):
    """Reward for placing an onion into a pot with capacity. Fully vectorized."""
    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(
        prev_state
    )

    is_interact = actions == action_pickup_drop_idx
    holds_onion = prev_state.agent_inv[:, 0] == type_ids["onion"]
    faces_pot = fwd_types == type_ids["pot"]

    # Array-based pot position matching:
    # For each agent, check which pot (if any) their forward position matches.
    # fwd_pos[:, :] is (n_agents, 2), pot_positions is (n_pots, 2)
    agent_fwd = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2) clipped
    pot_positions = prev_state.pot_positions  # (n_pots, 2)

    # pos_match[i, j] = True iff agent i faces pot j
    pos_match = xp.all(
        pot_positions[None, :, :] == agent_fwd[:, None, :],
        axis=2,
    )  # (n_agents, n_pots)
    facing_any_pot = xp.any(pos_match, axis=1)  # (n_agents,)
    pot_idx = xp.argmax(
        pos_match, axis=1
    )  # (n_agents,) -- index of matched pot

    # Check pot capacity and type compatibility for each agent's matched pot.
    # pot_contents[pot_idx] gives (n_agents, 3) -- the contents of each agent's pot.
    pot_row = prev_state.pot_contents[pot_idx]  # (n_agents, 3)
    n_filled = xp.sum(pot_row != -1, axis=1)  # (n_agents,)
    has_capacity = n_filled < 3

    # Same-type check: all non-empty slots must be onion (or empty)
    is_onion_or_empty = (pot_row == -1) | (pot_row == type_ids["onion"])
    compatible = xp.all(is_onion_or_empty, axis=1)  # (n_agents,)

    earns_reward = (
        is_interact
        & holds_onion
        & faces_pot
        & in_bounds
        & facing_any_pot
        & has_capacity
        & compatible
    )

    if common_reward:
        n_earners = xp.sum(earns_reward.astype(xp.float32))
        rewards = xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
    else:
        rewards = earns_reward.astype(xp.float32) * coefficient

    return rewards


def soup_in_dish_reward(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=0.3,
    common_reward=False,
    action_pickup_drop_idx=4,
):
    """Reward for picking up a ready soup from a pot with a plate. Fully vectorized."""
    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(
        prev_state
    )

    is_interact = actions == action_pickup_drop_idx
    holds_plate = prev_state.agent_inv[:, 0] == type_ids["plate"]
    faces_pot = fwd_types == type_ids["pot"]

    # Array-based pot position matching
    agent_fwd = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2)
    pot_positions = prev_state.pot_positions  # (n_pots, 2)

    pos_match = xp.all(
        pot_positions[None, :, :] == agent_fwd[:, None, :],
        axis=2,
    )  # (n_agents, n_pots)
    facing_any_pot = xp.any(pos_match, axis=1)  # (n_agents,)
    pot_idx = xp.argmax(pos_match, axis=1)  # (n_agents,)

    # Check pot is ready: timer == 0
    pot_timer_vals = prev_state.pot_timer[pot_idx]  # (n_agents,)
    pot_ready = pot_timer_vals == 0

    earns_reward = (
        is_interact
        & holds_plate
        & faces_pot
        & in_bounds
        & facing_any_pot
        & pot_ready
    )

    if common_reward:
        n_earners = xp.sum(earns_reward.astype(xp.float32))
        rewards = xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
    else:
        rewards = earns_reward.astype(xp.float32) * coefficient

    return rewards


# ---------------------------------------------------------------------------
# ArrayReward subclasses (registered for autowire composition)
# ---------------------------------------------------------------------------
# Each compute() returns final (n_agents,) rewards with coefficient and
# broadcasting already applied. The autowire layer just sums them.


@register_reward_type("delivery", scope="overcooked")
class DeliveryReward(ArrayReward):
    def compute(self, prev_state, state, actions, reward_config):
        return delivery_reward(
            prev_state,
            state,
            actions,
            reward_config["type_ids"],
            reward_config["n_agents"],
            coefficient=1.0,
            common_reward=True,
            action_pickup_drop_idx=reward_config["action_pickup_drop_idx"],
        )


@register_reward_type("onion_in_pot", scope="overcooked")
class OnionInPotReward(ArrayReward):
    def compute(self, prev_state, state, actions, reward_config):
        return onion_in_pot_reward(
            prev_state,
            state,
            actions,
            reward_config["type_ids"],
            reward_config["n_agents"],
            coefficient=0.1,
            common_reward=False,
            action_pickup_drop_idx=reward_config["action_pickup_drop_idx"],
        )


@register_reward_type("soup_in_dish", scope="overcooked")
class SoupInDishReward(ArrayReward):
    def compute(self, prev_state, state, actions, reward_config):
        return soup_in_dish_reward(
            prev_state,
            state,
            actions,
            reward_config["type_ids"],
            reward_config["n_agents"],
            coefficient=0.3,
            common_reward=False,
            action_pickup_drop_idx=reward_config["action_pickup_drop_idx"],
        )
