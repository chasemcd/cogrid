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
from cogrid.core.rewards import (
    InteractionReward,
    Reward,
    _compute_fwd_positions,
)

# ---------------------------------------------------------------------------
# Simple declarative reward: onion soup delivery
# ---------------------------------------------------------------------------


class OnionSoupDeliveryReward(InteractionReward):
    """Reward when an agent delivers onion soup to a delivery zone.

    Triggers when: agent performs pickup_drop + holds onion_soup + faces delivery_zone.
    This is the simplest delivery reward -- fixed coefficient, no recipe lookup,
    no order gating. Use DeliveryReward for multi-recipe support or
    OrderDeliveryReward for order-gated delivery with tip bonus.
    """

    action = "pickup_drop"
    holds = "onion_soup"
    faces = "delivery_zone"


# ---------------------------------------------------------------------------
# Multi-recipe delivery reward (no order gating)
# ---------------------------------------------------------------------------


class DeliveryReward(Reward):
    """Reward for delivering any deliverable item to a delivery zone.

    Supports multiple recipe types via IS_DELIVERABLE and per-recipe reward
    amounts from static_tables. Does NOT gate on active orders -- fires
    whenever a valid delivery occurs.

    Use OrderDeliveryReward if you need order matching and tip bonuses.

    Config keys (passed via ``__init__(**kwargs)``, stored in ``self.config``):
        - ``coefficient``: Reward scaling factor (default 1.0). Used as uniform
          reward when per-recipe tables are absent.
        - ``common_reward``: If True, all agents receive the reward (default True).
    """

    def compute(self, prev_state, state, actions, reward_config):
        """Compute delivery rewards using IS_DELIVERABLE and per-recipe amounts."""
        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        action_pickup_drop_idx = reward_config["action_pickup_drop_idx"]
        coefficient = self.config.get("coefficient", 1.0)
        common_reward = self.config.get("common_reward", True)

        # Step 1: compute each agent's forward cell position and type
        fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(prev_state)

        # Step 2: check which agents performed the pickup_drop action
        is_interact = actions == action_pickup_drop_idx  # (n_agents,)

        # Step 3: check which agents hold a deliverable item
        # IS_DELIVERABLE is a boolean lookup table indexed by type_id
        static_tables = reward_config.get("static_tables", {})
        is_deliverable_table = static_tables.get("IS_DELIVERABLE", None)
        if is_deliverable_table is not None:
            holds_deliverable = is_deliverable_table[prev_state.agent_inv[:, 0]] == 1
        else:
            # Backward compat: only onion_soup is deliverable
            holds_deliverable = prev_state.agent_inv[:, 0] == type_ids["onion_soup"]

        # Step 4: check which agents face a delivery zone
        faces_delivery = fwd_types == type_ids["delivery_zone"]  # (n_agents,)

        # Step 5: combine all conditions into a single delivery mask
        earns_delivery = is_interact & holds_deliverable & faces_delivery & in_bounds

        # Step 6: determine per-agent reward amounts from recipe tables
        # recipe_result maps recipe index -> result type_id
        # recipe_reward maps recipe index -> reward value
        recipe_result = static_tables.get("recipe_result", None)
        recipe_reward_arr = static_tables.get("recipe_reward", None)
        if recipe_result is not None and recipe_reward_arr is not None:
            delivered_type = prev_state.agent_inv[:, 0]  # (n_agents,)
            # Match each agent's held type to a recipe result
            match = delivered_type[:, None] == recipe_result[None, :]  # (n_agents, n_recipes)
            recipe_idx = xp.argmax(match, axis=1)  # (n_agents,)
            per_agent_reward = recipe_reward_arr[recipe_idx]  # (n_agents,) float32
        else:
            # No recipe tables: use uniform coefficient
            per_agent_reward = xp.full(n_agents, coefficient, dtype=xp.float32)

        # Step 7: apply reward with optional broadcast to all agents
        if common_reward:
            total = xp.sum(earns_delivery.astype(xp.float32) * per_agent_reward)
            rewards = xp.full(n_agents, total, dtype=xp.float32)
        else:
            rewards = earns_delivery.astype(xp.float32) * per_agent_reward

        return rewards


# ---------------------------------------------------------------------------
# Order-gated delivery reward with tip bonus
# ---------------------------------------------------------------------------


class OrderDeliveryReward(DeliveryReward):
    """Delivery reward gated on active order consumption, with optional tip bonus.

    Extends DeliveryReward by:
    1. Only firing when a matching active order was consumed this step
       (detected via prev_state.order_recipe vs state.order_recipe diff).
    2. Adding a time-based tip bonus proportional to remaining order time.

    Falls back to base DeliveryReward behavior (no gating, no tip) when
    order_recipe is not present in state (orders disabled).

    Config keys (passed via ``__init__(**kwargs)``, stored in ``self.config``):
        - ``coefficient``: Reward scaling factor (default 1.0).
        - ``common_reward``: If True, all agents receive the reward (default True).
    """

    def compute(self, prev_state, state, actions, reward_config):
        """Compute delivery rewards gated on order consumption with tip bonus."""
        # Get base delivery reward from parent (fires for any valid delivery)
        base_rewards = super().compute(prev_state, state, actions, reward_config)

        # Check if order system is active
        prev_order = getattr(prev_state, "order_recipe", None)
        if prev_order is None:
            # Orders disabled: fall back to base behavior (no gating)
            return base_rewards

        # Detect which orders were consumed this step:
        # an order slot that was active (recipe >= 0) and is now inactive (recipe == -1)
        was_active = prev_order >= 0  # (max_active,)
        now_inactive = state.order_recipe == -1  # (max_active,)
        consumed = was_active & now_inactive
        any_consumed = xp.any(consumed)

        # Gate: only reward if at least one order was consumed
        order_mask = xp.where(any_consumed, xp.float32(1.0), xp.float32(0.0))

        # Tip bonus: proportional to remaining time on the consumed order
        # tip = (remaining_time / time_limit) * tip_coefficient
        static_tables = reward_config.get("static_tables", {})
        tip_coefficient = reward_config.get("tip_coefficient", 0.0)
        order_time_limit = static_tables.get("order_time_limit", None)
        if order_time_limit is not None and tip_coefficient > 0.0:
            # Use the timer of the first consumed order slot
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

        # Apply order gate and add tip to every agent's reward
        n_agents = reward_config["n_agents"]
        return base_rewards * order_mask + xp.full(n_agents, tip, dtype=xp.float32)


# ---------------------------------------------------------------------------
# Onion-in-pot reward
# ---------------------------------------------------------------------------


class OnionInPotReward(InteractionReward):
    """Reward for placing an onion into a pot via pickup_drop action.

    Triggers when: agent performs pickup_drop + holds onion + faces pot,
    AND the pot has capacity (< 3 ingredients) and contains only onions
    or empty slots (type compatibility).

    Config keys (passed via ``__init__(**kwargs)``, stored in ``self.config``):
        - ``coefficient``: Reward scaling factor (default 0.1).
        - ``common_reward``: If True, all agents receive the reward (default False).
    """

    action = "pickup_drop"
    holds = "onion"
    faces = "pot"

    def extra_condition(self, mask, prev_state, fwd_r, fwd_c, reward_config):
        """Narrow mask to pots with capacity and compatible contents."""
        type_ids = reward_config["type_ids"]

        # Match each agent's forward position to a pot index
        agent_fwd = xp.stack([fwd_r, fwd_c], axis=1)
        pos_match = xp.all(prev_state.pot_positions[None, :, :] == agent_fwd[:, None, :], axis=2)
        pot_idx = xp.argmax(pos_match, axis=1)

        # Check pot has room (fewer than 3 non-empty slots)
        pot_row = prev_state.pot_contents[pot_idx]
        has_capacity = xp.sum(pot_row != -1, axis=1) < 3

        # Check pot only contains onions or empty slots (no mixed recipes)
        compatible = xp.all((pot_row == -1) | (pot_row == type_ids["onion"]), axis=1)

        return mask & xp.any(pos_match, axis=1) & has_capacity & compatible


# ---------------------------------------------------------------------------
# Soup-in-dish reward
# ---------------------------------------------------------------------------


class SoupInDishReward(InteractionReward):
    """Reward for picking up completed soup from a pot via pickup_drop action.

    Triggers when: agent performs pickup_drop + holds plate + faces pot,
    AND the pot is ready (timer == 0, meaning cooking is complete).

    Config keys (passed via ``__init__(**kwargs)``, stored in ``self.config``):
        - ``coefficient``: Reward scaling factor (default 0.3).
        - ``common_reward``: If True, all agents receive the reward (default False).
    """

    action = "pickup_drop"
    holds = "plate"
    faces = "pot"

    def extra_condition(self, mask, prev_state, fwd_r, fwd_c, reward_config):
        """Narrow mask to pots that are done cooking."""
        # Match each agent's forward position to a pot index
        agent_fwd = xp.stack([fwd_r, fwd_c], axis=1)
        pos_match = xp.all(prev_state.pot_positions[None, :, :] == agent_fwd[:, None, :], axis=2)
        pot_idx = xp.argmax(pos_match, axis=1)

        # Check pot is done cooking (timer == 0 means ready to serve)
        pot_ready = prev_state.pot_timer[pot_idx] == 0

        return mask & xp.any(pos_match, axis=1) & pot_ready


# ---------------------------------------------------------------------------
# Expired order penalty
# ---------------------------------------------------------------------------


class ExpiredOrderPenalty(Reward):
    """Penalty for expired orders (common reward -- all agents penalized).

    Detects newly expired orders by diffing prev_state.order_n_expired
    vs state.order_n_expired. Returns zero when orders are disabled
    (no order_n_expired in state).

    Config keys (passed via ``__init__(**kwargs)``, stored in ``self.config``):
        - ``penalty``: Penalty value per expired order (default -5.0).
    """

    def compute(self, prev_state, state, actions, reward_config):
        """Compute penalty for newly expired orders this step."""
        # Check if order system is active
        prev_expired = getattr(prev_state, "order_n_expired", None)
        if prev_expired is None:
            return xp.zeros(reward_config["n_agents"], dtype=xp.float32)

        # Count how many orders expired this step
        curr_expired = state.order_n_expired
        newly_expired = curr_expired - prev_expired  # scalar int32

        n_agents = reward_config["n_agents"]
        penalty = self.config.get("penalty", -5.0)

        # Broadcast penalty to all agents (shared responsibility)
        return xp.full(n_agents, newly_expired.astype(xp.float32) * penalty, dtype=xp.float32)
