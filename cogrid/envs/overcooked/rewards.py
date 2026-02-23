"""Overcooked-specific reward functions.

These reward classes operate on state dictionaries instead of Grid objects.
They are specific to the Overcooked environment because they reference
Overcooked types: pot, onion_soup, delivery_zone, plate, onion.

Each Reward subclass's compute() returns ndarray of shape (n_agents,).
Parameters (coefficient, common_reward, etc.) are passed via __init__
and stored in self.config.

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
from cogrid.core.rewards import Reward


def _compute_fwd_positions(prev_state):
    """Compute forward positions, clipped indices, bounds mask, and forward type IDs."""
    # Direction vector table: Right=0, Down=1, Left=2, Up=3
    dir_vec_table = xp.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=xp.int32)

    fwd_pos = prev_state.agent_pos + dir_vec_table[prev_state.agent_dir]  # (n_agents, 2)
    H, W = prev_state.object_type_map.shape
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)
    in_bounds = (
        (fwd_pos[:, 0] >= 0) & (fwd_pos[:, 0] < H) & (fwd_pos[:, 1] >= 0) & (fwd_pos[:, 1] < W)
    )
    fwd_types = prev_state.object_type_map[fwd_r, fwd_c]  # (n_agents,)

    return fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types


class DeliveryReward(Reward):
    """Reward for delivering soup to a delivery zone.

    Uses IS_DELIVERABLE lookup, per-recipe reward values, and order-aware
    gating when orders are enabled. Falls back to original behavior when
    static_tables or orders are not available.

    Parameters:
        coefficient: Reward scaling factor (default 1.0).
        common_reward: If True, all agents receive the reward (default True).
    """

    def compute(self, prev_state, state, actions, reward_config):
        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        action_pickup_drop_idx = reward_config["action_pickup_drop_idx"]
        coefficient = self.config.get("coefficient", 1.0)
        common_reward = self.config.get("common_reward", True)

        fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(prev_state)

        is_interact = actions == action_pickup_drop_idx  # (n_agents,)

        # --- Determine which agents hold a deliverable item ---
        static_tables = reward_config.get("static_tables", {})
        is_deliverable_table = static_tables.get("IS_DELIVERABLE", None)
        if is_deliverable_table is not None:
            holds_deliverable = is_deliverable_table[prev_state.agent_inv[:, 0]] == 1
        else:
            # Backward compat fallback
            holds_deliverable = prev_state.agent_inv[:, 0] == type_ids["onion_soup"]

        faces_delivery = fwd_types == type_ids["delivery_zone"]  # (n_agents,)
        earns_delivery = is_interact & holds_deliverable & faces_delivery & in_bounds

        # --- Per-recipe reward amounts ---
        recipe_result = static_tables.get("recipe_result", None)
        recipe_reward_arr = static_tables.get("recipe_reward", None)
        if recipe_result is not None and recipe_reward_arr is not None:
            delivered_type = prev_state.agent_inv[:, 0]  # (n_agents,)
            match = delivered_type[:, None] == recipe_result[None, :]  # (n_agents, n_recipes)
            recipe_idx = xp.argmax(match, axis=1)  # (n_agents,)
            per_agent_reward = recipe_reward_arr[recipe_idx]  # (n_agents,) float32
        else:
            per_agent_reward = xp.full(n_agents, coefficient, dtype=xp.float32)

        # --- Order matching and tip bonus ---
        tip_coefficient = reward_config.get("tip_coefficient", 0.0)
        prev_order = getattr(prev_state, "order_recipe", None)
        if prev_order is not None:
            # Orders are enabled -- check if a matching order was consumed this step
            was_active = prev_order >= 0  # (max_active,)
            now_inactive = state.order_recipe == -1  # (max_active,)
            consumed = was_active & now_inactive  # orders consumed this step
            any_consumed = xp.any(consumed)

            # Tip bonus: use prev timer of consumed order
            order_time_limit = static_tables.get("order_time_limit", None)
            if order_time_limit is not None and tip_coefficient > 0.0:
                consumed_idx = xp.argmax(consumed)
                remaining_time = prev_state.order_timer[consumed_idx]
                tip = xp.where(
                    any_consumed,
                    remaining_time.astype(xp.float32)
                    / xp.float32(order_time_limit)
                    * xp.float32(tip_coefficient),
                    xp.float32(0.0),
                )
            else:
                tip = xp.float32(0.0)

            # Only reward if order was consumed
            order_mask = xp.where(any_consumed, xp.float32(1.0), xp.float32(0.0))
        else:
            # No orders: always reward (backward compat)
            order_mask = xp.float32(1.0)
            tip = xp.float32(0.0)

        # --- Apply reward ---
        if common_reward:
            total = xp.sum(earns_delivery.astype(xp.float32) * per_agent_reward)
            total = total * order_mask + tip
            rewards = xp.full(n_agents, total, dtype=xp.float32)
        else:
            rewards = earns_delivery.astype(xp.float32) * per_agent_reward * order_mask + tip

        return rewards


class OnionInPotReward(Reward):
    """Reward for placing an onion into a pot.

    Parameters:
        coefficient: Reward scaling factor (default 0.1).
        common_reward: If True, all agents receive the reward (default False).
    """

    def compute(self, prev_state, state, actions, reward_config):
        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        action_pickup_drop_idx = reward_config["action_pickup_drop_idx"]
        coefficient = self.config.get("coefficient", 0.1)
        common_reward = self.config.get("common_reward", False)

        fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(prev_state)

        is_interact = actions == action_pickup_drop_idx
        holds_onion = prev_state.agent_inv[:, 0] == type_ids["onion"]
        faces_pot = fwd_types == type_ids["pot"]

        # Array-based pot position matching:
        agent_fwd = xp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2) clipped
        pot_positions = prev_state.pot_positions  # (n_pots, 2)

        # pos_match[i, j] = True iff agent i faces pot j
        pos_match = xp.all(
            pot_positions[None, :, :] == agent_fwd[:, None, :],
            axis=2,
        )  # (n_agents, n_pots)
        facing_any_pot = xp.any(pos_match, axis=1)  # (n_agents,)
        pot_idx = xp.argmax(pos_match, axis=1)  # (n_agents,) -- index of matched pot

        # Check pot capacity and type compatibility for each agent's matched pot.
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


class SoupInDishReward(Reward):
    """Reward for picking up completed soup from a pot.

    Parameters:
        coefficient: Reward scaling factor (default 0.3).
        common_reward: If True, all agents receive the reward (default False).
    """

    def compute(self, prev_state, state, actions, reward_config):
        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        action_pickup_drop_idx = reward_config["action_pickup_drop_idx"]
        coefficient = self.config.get("coefficient", 0.3)
        common_reward = self.config.get("common_reward", False)

        fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(prev_state)

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

        earns_reward = is_interact & holds_plate & faces_pot & in_bounds & facing_any_pot & pot_ready

        if common_reward:
            n_earners = xp.sum(earns_reward.astype(xp.float32))
            rewards = xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
        else:
            rewards = earns_reward.astype(xp.float32) * coefficient

        return rewards


class ExpiredOrderPenalty(Reward):
    """Penalty for expired orders (common reward -- all agents penalized).

    Uses prev/curr diff on order_n_expired to detect newly expired orders
    this step. Returns zero when orders are disabled (no order_n_expired
    in state).

    Parameters:
        penalty: Penalty value per expired order (default -5.0).
    """

    def compute(self, prev_state, state, actions, reward_config):
        prev_expired = getattr(prev_state, "order_n_expired", None)
        if prev_expired is None:
            return xp.zeros(reward_config["n_agents"], dtype=xp.float32)

        curr_expired = state.order_n_expired
        newly_expired = curr_expired - prev_expired  # scalar int32

        n_agents = reward_config["n_agents"]
        penalty = self.config.get("penalty", -5.0)

        # Broadcast to all agents (common penalty)
        return xp.full(n_agents, newly_expired.astype(xp.float32) * penalty, dtype=xp.float32)
