"""Generic reward composition utility.

Provides the ``Reward`` base class for reward components, the declarative
``InteractionReward`` base for condition-triggered rewards, and the
generic ``_compute_fwd_positions`` helper.

Reward instances are passed explicitly in the env config ``"rewards"`` list.
Composition is handled by ``cogrid.core.autowire.build_reward_config()``.

Environment-specific reward functions live in their respective envs/ modules:
- Overcooked: ``cogrid.envs.overcooked.rewards``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cogrid.backend import xp

if TYPE_CHECKING:
    from cogrid.core.typing import ArrayLike


def _compute_fwd_positions(
    prev_state: Any,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Compute forward positions, clipped indices, bounds mask, and forward type IDs."""
    # Direction vector table: Right=0, Down=1, Left=2, Up=3
    from cogrid.core.agent import get_dir_vec_table

    dir_vec_table = get_dir_vec_table()

    fwd_pos = prev_state.agent_pos + dir_vec_table[prev_state.agent_dir]  # (n_agents, 2)
    H, W = prev_state.object_type_map.shape
    fwd_r = xp.clip(fwd_pos[:, 0], 0, H - 1)
    fwd_c = xp.clip(fwd_pos[:, 1], 0, W - 1)
    in_bounds = (
        (fwd_pos[:, 0] >= 0) & (fwd_pos[:, 0] < H) & (fwd_pos[:, 1] >= 0) & (fwd_pos[:, 1] < W)
    )
    fwd_types = prev_state.object_type_map[fwd_r, fwd_c]  # (n_agents,)

    return fwd_pos, fwd_r, fwd_c, in_bounds, fwd_types


class Reward:
    """Base class for reward functions.

    Subclasses define compute() which receives StateView objects and returns
    (n_agents,) float32 reward arrays. The returned values are the final
    rewards -- apply any scaling or broadcasting inside compute().

    Every reward has a ``coefficient`` that controls its magnitude.
    At runtime, coefficients are stored as a dynamic array in
    ``EnvState.extra_state["reward_coefficients"]`` (accessible as
    ``state.reward_coefficients`` in compute).  This allows coefficient
    updates without re-JIT on the JAX backend.

    ``_reward_index`` is assigned by ``build_reward_config()`` and maps
    this instance to its position in the coefficients array.

    Usage::

        class DeliveryReward(Reward):
            def compute(self, prev_state, state, actions, reward_config):
                coefficient = state.reward_coefficients[self._reward_index]
                ...
                return rewards  # (n_agents,) float32


        config = {
            "rewards": [DeliveryReward(coefficient=1.0, common_reward=True)],
        }
    """

    def __init__(self, coefficient: float = 1.0, **kwargs: Any) -> None:
        """Store config kwargs for use in compute()."""
        self.coefficient = coefficient
        self._reward_index: int | None = None
        self.config = kwargs

    def get_coefficient(self, state: Any) -> float:
        """Read the dynamic coefficient from state, falling back to ``self.coefficient``.

        The coefficients array lives in ``state.reward_coefficients``
        (populated from ``EnvState.extra_state``).  If absent (e.g. in
        unit tests that build states manually), falls back to the value
        set at init time.
        """
        rc = getattr(state, "reward_coefficients", None)
        if rc is not None and self._reward_index is not None:
            return rc[self._reward_index]
        return self.coefficient

    def compute(
        self,
        prev_state: Any,
        state: Any,
        actions: ArrayLike,
        reward_config: dict[str, Any],
    ) -> ArrayLike:
        """Compute and return (n_agents,) float32 reward array.

        Subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.compute() is not implemented. "
            f"Subclasses must override compute()."
        )


_UNSET = object()


class InteractionReward(Reward):
    """Declarative base for condition-triggered rewards.

    Class attributes (declare what triggers the reward):
        action:    "pickup_drop", "toggle", or None (any/no action check).
                   Subclasses MUST set this explicitly -- there is no default.
        holds:     str type name agent must hold, or None
        faces:     str type name agent must face (forward cell), or None
        overlaps:  str type name agent must stand on, or None
        direction: int direction agent must face (0=R,1=D,2=L,3=U), or None

    Instance config (runtime tuning via __init__ kwargs):
        coefficient:   float scaling factor (default 1.0)
        common_reward: bool broadcast to all agents (default False)

    Override extra_condition() for domain-specific checks beyond
    the standard conditions (pot capacity, timers, etc.).

    Examples::

        class OnionInPotReward(InteractionReward):
            action = "pickup_drop"
            holds = "onion"
            faces = "pot"


        class GoalReward(InteractionReward):
            action = None
            overlaps = "goal"
    """

    action = _UNSET  # subclasses MUST set to "pickup_drop", "toggle", or None
    holds = None  # type name agent must hold
    faces = None  # type name in forward cell
    overlaps = None  # type name agent stands on
    direction = None  # direction agent must face (0-3)

    def extra_condition(
        self,
        mask: ArrayLike,
        prev_state: Any,
        fwd_r: ArrayLike | None,
        fwd_c: ArrayLike | None,
        reward_config: dict[str, Any],
    ) -> ArrayLike:
        """Override to add conditions beyond the declarative attributes.

        Return narrowed boolean mask.
        """
        return mask

    def compute(
        self,
        prev_state: Any,
        state: Any,
        actions: ArrayLike,
        reward_config: dict[str, Any],
    ) -> ArrayLike:
        """Compute (n_agents,) float32 rewards from declarative conditions."""
        if self.action is _UNSET:
            raise TypeError(
                f"{type(self).__name__} must set action to 'pickup_drop', 'toggle', or None"
            )

        type_ids = reward_config["type_ids"]
        n_agents = reward_config["n_agents"]
        coefficient = self.get_coefficient(state)
        common_reward = self.config.get("common_reward", False)

        mask = xp.ones(n_agents, dtype=xp.bool_)

        # Action check
        if self.action == "pickup_drop":
            mask = mask & (actions == reward_config["action_pickup_drop_idx"])
        elif self.action == "toggle":
            mask = mask & (actions == reward_config["action_toggle_idx"])
        # action=None -> no action filter

        # Inventory check
        if self.holds is not None:
            mask = mask & (prev_state.agent_inv[:, 0] == type_ids[self.holds])

        # Direction check
        if self.direction is not None:
            mask = mask & (prev_state.agent_dir == self.direction)

        # Forward-cell check (only compute fwd positions if needed)
        fwd_r = fwd_c = None
        if self.faces is not None:
            _, fwd_r, fwd_c, in_bounds, fwd_types = _compute_fwd_positions(prev_state)
            mask = mask & in_bounds & (fwd_types == type_ids[self.faces])

        # Overlap check (agent's current cell)
        if self.overlaps is not None:
            rows = prev_state.agent_pos[:, 0]
            cols = prev_state.agent_pos[:, 1]
            mask = mask & (prev_state.object_type_map[rows, cols] == type_ids[self.overlaps])

        # Extra condition hook
        mask = self.extra_condition(mask, prev_state, fwd_r, fwd_c, reward_config)

        # Apply coefficient + broadcast
        if common_reward:
            n_earners = xp.sum(mask.astype(xp.float32))
            return xp.full(n_agents, n_earners * coefficient, dtype=xp.float32)
        else:
            return mask.astype(xp.float32) * coefficient
