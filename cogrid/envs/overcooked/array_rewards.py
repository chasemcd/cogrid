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
    pot_positions:    list of (row, col)   -- pot locations
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
