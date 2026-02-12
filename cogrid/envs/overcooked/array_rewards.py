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
    pot_positions:    list of (row, col) OR (n_pots, 2) int32 -- pot locations

Numpy-path functions use Python loops over agents with int() casts.
JAX-path functions (_jax suffix) use fully vectorized ops across all agents
simultaneously, with array-based pot position matching instead of Python loops.
"""

from cogrid.core.agent import get_dir_vec_table


def delivery_reward_array(
    prev_state: dict,
    state: dict,
    actions,
    type_ids: dict,
    n_agents: int,
    coefficient: float = 1.0,
    common_reward: bool = True,
    action_pickup_drop_idx: int = 4,
):
    """Reward for delivering OnionSoup to a DeliveryZone.

    Mirrors SoupDeliveryReward.calculate_reward():
    - Agent must be performing PickupDrop action
    - Agent must hold OnionSoup (prev_state inventory)
    - Agent must face a DeliveryZone (prev_state grid)
    - If common_reward: all agents receive the reward

    Uses prev_state for both inventory and grid checks, matching the
    existing implementation where ``state`` is ``self.prev_grid``.
    """
    from cogrid.backend import xp

    rewards = xp.zeros(n_agents, dtype=xp.float32)
    dir_vec_table = get_dir_vec_table()

    # PHASE2: vectorize across agents
    for i in range(n_agents):
        if int(actions[i]) != action_pickup_drop_idx:
            continue

        # Check if agent holds OnionSoup (prev_state inventory)
        agent_holds_soup = int(prev_state['agent_inv'][i, 0]) == type_ids['onion_soup']

        # Check if agent faces DeliveryZone
        fwd_pos = prev_state['agent_pos'][i] + dir_vec_table[int(prev_state['agent_dir'][i])]
        fwd_r, fwd_c = int(fwd_pos[0]), int(fwd_pos[1])

        # Bounds check
        h, w = prev_state['object_type_map'].shape
        if fwd_r < 0 or fwd_r >= h or fwd_c < 0 or fwd_c >= w:
            continue

        fwd_type = int(prev_state['object_type_map'][fwd_r, fwd_c])
        facing_delivery = fwd_type == type_ids['delivery_zone']

        if agent_holds_soup and facing_delivery:
            if common_reward:
                rewards = rewards + coefficient  # all agents get it
            else:
                rewards[i] += coefficient

    return rewards


def onion_in_pot_reward_array(
    prev_state: dict,
    state: dict,
    actions,
    type_ids: dict,
    n_agents: int,
    coefficient: float = 0.1,
    common_reward: bool = False,
    action_pickup_drop_idx: int = 4,
):
    """Reward for placing an onion into a pot with capacity.

    Mirrors OnionInPotReward.calculate_reward():
    - Agent must be performing PickupDrop action
    - Agent must hold Onion (prev_state inventory)
    - Agent must face a Pot (prev_state grid)
    - Pot must have capacity (< 3 ingredients)
    - Pot contents must be compatible (all same type or empty)

    The can_place_on logic for Pot checks:
    1. len(objects_in_pot) < capacity (capacity = 3)
    2. ingredient is legal (Onion or Tomato)
    3. ingredient matches type already in pot (or pot is empty)
    """
    from cogrid.backend import xp

    rewards = xp.zeros(n_agents, dtype=xp.float32)
    dir_vec_table = get_dir_vec_table()

    # PHASE2: vectorize across agents
    for i in range(n_agents):
        if int(actions[i]) != action_pickup_drop_idx:
            continue

        # Check if agent holds Onion
        agent_inv_type = int(prev_state['agent_inv'][i, 0])
        agent_holds_onion = agent_inv_type == type_ids['onion']
        if not agent_holds_onion:
            continue

        # Check if agent faces a Pot
        fwd_pos = prev_state['agent_pos'][i] + dir_vec_table[int(prev_state['agent_dir'][i])]
        fwd_r, fwd_c = int(fwd_pos[0]), int(fwd_pos[1])

        # Bounds check
        h, w = prev_state['object_type_map'].shape
        if fwd_r < 0 or fwd_r >= h or fwd_c < 0 or fwd_c >= w:
            continue

        fwd_type = int(prev_state['object_type_map'][fwd_r, fwd_c])
        if fwd_type != type_ids['pot']:
            continue

        # Find pot index by position
        pot_idx = _find_pot_index(prev_state['pot_positions'], fwd_r, fwd_c)
        if pot_idx < 0:
            continue

        # Check pot has capacity (< 3 filled slots)
        pot_row = prev_state['pot_contents'][pot_idx]
        n_filled = int(xp.sum(pot_row != -1))
        if n_filled >= 3:
            continue

        # Check ingredient type compatibility:
        # All existing contents must be the same type as what we're adding,
        # or the pot must be empty. Onion type_id is what the agent holds.
        compatible = True
        for slot in range(3):
            slot_val = int(pot_row[slot])
            if slot_val != -1 and slot_val != type_ids['onion']:
                compatible = False
                break

        if not compatible:
            continue

        if common_reward:
            rewards = rewards + coefficient
        else:
            rewards[i] += coefficient

    return rewards


def soup_in_dish_reward_array(
    prev_state: dict,
    state: dict,
    actions,
    type_ids: dict,
    n_agents: int,
    coefficient: float = 0.3,
    common_reward: bool = False,
    action_pickup_drop_idx: int = 4,
):
    """Reward for picking up a ready soup from a pot with a plate.

    Mirrors SoupInDishReward.calculate_reward():
    - Agent must be performing PickupDrop action
    - Agent must hold Plate (prev_state inventory)
    - Agent must face a Pot (prev_state grid)
    - Pot must be ready (cooking_timer == 0, i.e. dish_ready)
    """
    from cogrid.backend import xp

    rewards = xp.zeros(n_agents, dtype=xp.float32)
    dir_vec_table = get_dir_vec_table()

    # PHASE2: vectorize across agents
    for i in range(n_agents):
        if int(actions[i]) != action_pickup_drop_idx:
            continue

        # Check if agent holds Plate
        agent_holds_plate = int(prev_state['agent_inv'][i, 0]) == type_ids['plate']
        if not agent_holds_plate:
            continue

        # Check if agent faces a Pot
        fwd_pos = prev_state['agent_pos'][i] + dir_vec_table[int(prev_state['agent_dir'][i])]
        fwd_r, fwd_c = int(fwd_pos[0]), int(fwd_pos[1])

        # Bounds check
        h, w = prev_state['object_type_map'].shape
        if fwd_r < 0 or fwd_r >= h or fwd_c < 0 or fwd_c >= w:
            continue

        fwd_type = int(prev_state['object_type_map'][fwd_r, fwd_c])
        if fwd_type != type_ids['pot']:
            continue

        # Find pot index by position
        pot_idx = _find_pot_index(prev_state['pot_positions'], fwd_r, fwd_c)
        if pot_idx < 0:
            continue

        # Check pot is ready (timer == 0 means dish_ready)
        pot_timer_val = int(prev_state['pot_timer'][pot_idx])
        if pot_timer_val != 0:
            continue

        if common_reward:
            rewards = rewards + coefficient
        else:
            rewards[i] += coefficient

    return rewards


def _find_pot_index(pot_positions, row: int, col: int) -> int:
    """Find the index of a pot at the given (row, col) position.

    Returns -1 if not found.
    """
    for idx, pos in enumerate(pot_positions):
        if int(pos[0]) == row and int(pos[1]) == col:
            return idx
    return -1


# ---------------------------------------------------------------------------
# JAX-path reward functions (JIT-compatible, fully vectorized)
# ---------------------------------------------------------------------------


def _compute_fwd_positions_jax(prev_state):
    """Compute forward positions for all agents (JAX path).

    Shared helper used by all JAX reward functions. Returns the forward position,
    clipped coordinates for safe indexing, in-bounds mask, and forward type IDs.

    Args:
        prev_state: dict of jnp state arrays.

    Returns:
        Tuple of (fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types) where:
            fwd_pos: (n_agents, 2) int32 -- raw forward positions
            fwd_r: (n_agents,) int32 -- clipped row indices
            fwd_c: (n_agents,) int32 -- clipped col indices
            in_bounds: (n_agents,) bool -- whether forward position is in grid
            fwd_types: (n_agents,) int32 -- object type IDs at forward positions
    """
    import jax.numpy as jnp

    # Direction vector table: Right=0, Down=1, Left=2, Up=3
    dir_vec_table = jnp.array(
        [[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=jnp.int32
    )

    fwd_pos = prev_state['agent_pos'] + dir_vec_table[prev_state['agent_dir']]  # (n_agents, 2)
    H, W = prev_state['object_type_map'].shape
    fwd_r = jnp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = jnp.clip(fwd_pos[:, 1], 0, W - 1)
    in_bounds = (
        (fwd_pos[:, 0] >= 0)
        & (fwd_pos[:, 0] < H)
        & (fwd_pos[:, 1] >= 0)
        & (fwd_pos[:, 1] < W)
    )
    fwd_types = prev_state['object_type_map'][fwd_r, fwd_c]  # (n_agents,)

    return fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types


def delivery_reward_jax(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=1.0,
    common_reward=True,
    action_pickup_drop_idx=4,
):
    """Reward for delivering OnionSoup to a DeliveryZone (JAX path).

    Fully vectorized across all agents. No Python loops or int() casts.
    type_ids is a Python dict used at trace time (not traced).
    common_reward and coefficient are Python values used at trace time.

    Args:
        prev_state: dict of jnp state arrays (agent_pos, agent_dir, agent_inv,
            object_type_map).
        state: dict of jnp state arrays after step (unused, matches numpy signature).
        actions: (n_agents,) int32 action indices.
        type_ids: Python dict mapping type names to int IDs.
        n_agents: int (Python, not traced).
        coefficient: float reward value.
        common_reward: bool, if True all agents share the reward.
        action_pickup_drop_idx: int action index for pickup/drop.

    Returns:
        (n_agents,) float32 array of rewards.
    """
    import jax.numpy as jnp

    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions_jax(prev_state)

    is_interact = (actions == action_pickup_drop_idx)  # (n_agents,)
    holds_soup = (prev_state['agent_inv'][:, 0] == type_ids['onion_soup'])  # (n_agents,)
    faces_delivery = (fwd_types == type_ids['delivery_zone'])  # (n_agents,)

    earns_reward = is_interact & holds_soup & faces_delivery & in_bounds  # (n_agents,)

    # Apply reward: in common_reward mode, every earning agent adds coefficient
    # to ALL agents. This matches numpy: `rewards = rewards + coefficient` per earner.
    if common_reward:
        n_earners = jnp.sum(earns_reward.astype(jnp.float32))
        rewards = jnp.full(n_agents, n_earners * coefficient, dtype=jnp.float32)
    else:
        rewards = earns_reward.astype(jnp.float32) * coefficient

    return rewards


def onion_in_pot_reward_jax(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=0.1,
    common_reward=False,
    action_pickup_drop_idx=4,
):
    """Reward for placing an onion into a pot with capacity (JAX path).

    Fully vectorized across all agents. Uses array-based pot position matching
    instead of _find_pot_index Python loop.

    pot_positions must be a (n_pots, 2) jnp.array (not a Python list).

    Args:
        prev_state: dict of jnp state arrays including pot_positions (n_pots, 2),
            pot_contents (n_pots, 3).
        state: dict (unused, matches numpy signature).
        actions: (n_agents,) int32 action indices.
        type_ids: Python dict mapping type names to int IDs.
        n_agents: int.
        coefficient: float reward value.
        common_reward: bool.
        action_pickup_drop_idx: int.

    Returns:
        (n_agents,) float32 array of rewards.
    """
    import jax.numpy as jnp

    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions_jax(prev_state)

    is_interact = (actions == action_pickup_drop_idx)
    holds_onion = (prev_state['agent_inv'][:, 0] == type_ids['onion'])
    faces_pot = (fwd_types == type_ids['pot'])

    # Array-based pot position matching:
    # For each agent, check which pot (if any) their forward position matches.
    # fwd_pos[:, :] is (n_agents, 2), pot_positions is (n_pots, 2)
    agent_fwd = jnp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2) clipped
    pot_positions = prev_state['pot_positions']  # (n_pots, 2)

    # pos_match[i, j] = True iff agent i faces pot j
    pos_match = jnp.all(
        pot_positions[None, :, :] == agent_fwd[:, None, :],
        axis=2,
    )  # (n_agents, n_pots)
    facing_any_pot = jnp.any(pos_match, axis=1)  # (n_agents,)
    pot_idx = jnp.argmax(pos_match, axis=1)  # (n_agents,) -- index of matched pot

    # Check pot capacity and type compatibility for each agent's matched pot.
    # pot_contents[pot_idx] gives (n_agents, 3) -- the contents of each agent's pot.
    pot_row = prev_state['pot_contents'][pot_idx]  # (n_agents, 3)
    n_filled = jnp.sum(pot_row != -1, axis=1)  # (n_agents,)
    has_capacity = n_filled < 3

    # Same-type check: all non-empty slots must be onion (or empty)
    is_onion_or_empty = (pot_row == -1) | (pot_row == type_ids['onion'])
    compatible = jnp.all(is_onion_or_empty, axis=1)  # (n_agents,)

    earns_reward = (
        is_interact & holds_onion & faces_pot & in_bounds
        & facing_any_pot & has_capacity & compatible
    )

    if common_reward:
        n_earners = jnp.sum(earns_reward.astype(jnp.float32))
        rewards = jnp.full(n_agents, n_earners * coefficient, dtype=jnp.float32)
    else:
        rewards = earns_reward.astype(jnp.float32) * coefficient

    return rewards


def soup_in_dish_reward_jax(
    prev_state,
    state,
    actions,
    type_ids,
    n_agents,
    coefficient=0.3,
    common_reward=False,
    action_pickup_drop_idx=4,
):
    """Reward for picking up a ready soup from a pot with a plate (JAX path).

    Fully vectorized across all agents. Uses array-based pot position matching.

    Args:
        prev_state: dict of jnp state arrays including pot_positions (n_pots, 2),
            pot_timer (n_pots,).
        state: dict (unused, matches numpy signature).
        actions: (n_agents,) int32 action indices.
        type_ids: Python dict mapping type names to int IDs.
        n_agents: int.
        coefficient: float reward value.
        common_reward: bool.
        action_pickup_drop_idx: int.

    Returns:
        (n_agents,) float32 array of rewards.
    """
    import jax.numpy as jnp

    fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions_jax(prev_state)

    is_interact = (actions == action_pickup_drop_idx)
    holds_plate = (prev_state['agent_inv'][:, 0] == type_ids['plate'])
    faces_pot = (fwd_types == type_ids['pot'])

    # Array-based pot position matching
    agent_fwd = jnp.stack([fwd_r, fwd_c], axis=1)  # (n_agents, 2)
    pot_positions = prev_state['pot_positions']  # (n_pots, 2)

    pos_match = jnp.all(
        pot_positions[None, :, :] == agent_fwd[:, None, :],
        axis=2,
    )  # (n_agents, n_pots)
    facing_any_pot = jnp.any(pos_match, axis=1)  # (n_agents,)
    pot_idx = jnp.argmax(pos_match, axis=1)  # (n_agents,)

    # Check pot is ready: timer == 0
    pot_timer_vals = prev_state['pot_timer'][pot_idx]  # (n_agents,)
    pot_ready = (pot_timer_vals == 0)

    earns_reward = (
        is_interact & holds_plate & faces_pot & in_bounds
        & facing_any_pot & pot_ready
    )

    if common_reward:
        n_earners = jnp.sum(earns_reward.astype(jnp.float32))
        rewards = jnp.full(n_agents, n_earners * coefficient, dtype=jnp.float32)
    else:
        rewards = earns_reward.astype(jnp.float32) * coefficient

    return rewards


def compute_rewards_jax(prev_state, state, actions, reward_config):
    """Compute combined rewards from all configured reward functions (JAX path).

    This is the `compute_rewards` function referenced in Phase 2 success criteria #5.
    Composes individual JAX reward functions and sums their outputs.

    The Python for loop over reward_config["rewards"] is fine because reward_config
    is a static Python dict (not traced). Each iteration adds a traced computation
    to the graph. JAX traces through the loop at compile time and fuses the
    individual reward computations.

    Args:
        prev_state: dict of jnp state arrays before step.
        state: dict of jnp state arrays after step.
        actions: (n_agents,) int32 action indices.
        reward_config: Python dict with keys:
            - "type_ids": dict mapping type names to int IDs
            - "n_agents": int (static)
            - "rewards": list of dicts, each with:
                - "fn": str name ("delivery", "onion_in_pot", "soup_in_dish")
                - "coefficient": float
                - "common_reward": bool
            - "action_pickup_drop_idx": int (static)

    Returns:
        (n_agents,) float32 array of combined rewards.
    """
    import jax.numpy as jnp

    n_agents = reward_config["n_agents"]
    type_ids = reward_config["type_ids"]
    action_idx = reward_config["action_pickup_drop_idx"]

    # Map function name strings to JAX reward functions
    fn_map = {
        "delivery": delivery_reward_jax,
        "onion_in_pot": onion_in_pot_reward_jax,
        "soup_in_dish": soup_in_dish_reward_jax,
    }

    total_rewards = jnp.zeros(n_agents, dtype=jnp.float32)
    for reward_spec in reward_config["rewards"]:
        fn = fn_map[reward_spec["fn"]]
        r = fn(
            prev_state, state, actions, type_ids, n_agents,
            coefficient=reward_spec["coefficient"],
            common_reward=reward_spec["common_reward"],
            action_pickup_drop_idx=action_idx,
        )
        total_rewards = total_rewards + r

    return total_rewards
