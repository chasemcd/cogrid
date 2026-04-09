"""Validate V2 extra observation channels against known states.

Tests the 27 extra channels added by OvercookedLocalView:
  - 4 pot state (is_cooking, is_ready, fill_norm, timer_norm)
  - 4 pot ingredient breakdown (onion, tomato, broccoli, mushroom)
  - 12 decomposed inventory (6 per agent: plate, cooked, 4 ingredients)
  - 6 recipe indicator decomposition (plate, cooked, 4 ingredients)
  - 1 correct delivery indicator
"""

import numpy as np
import pytest

import cogrid.envs

cogrid.envs._ensure_v2_types()

from cogrid.core.objects.registry import object_to_idx
from cogrid.envs.overcooked.features import _INGREDIENT_NAMES, OvercookedLocalView

SCOPE = "overcooked"
H, W = 4, 5
INV_FIELDS = ["plate", "cooked"] + list(_INGREDIENT_NAMES)

# Channel offsets within the 27 extra layers
POT_STATE = 0  # 4 channels
POT_INGS = 4  # 4 channels
SELF_INV = 8  # 6 channels
OTHER_INV = 14  # 6 channels
RECIPE = 20  # 6 channels
DELIVERY = 26  # 1 channel


@pytest.fixture(scope="module")
def extra_fn():
    env_config = {"n_agents": 2, "target_recipes": ["onion_soup", "tomato_soup"]}
    return OvercookedLocalView.build_extra_channel_fn(SCOPE, env_config)


@pytest.fixture(scope="module")
def type_ids():
    return {
        name: object_to_idx(name, scope=SCOPE)
        for name in [
            "onion",
            "tomato",
            "plate",
            "onion_soup",
            "tomato_soup",
            "open_pot",
            "recipe_indicator",
            "open_delivery_zone",
        ]
    }


class _State:
    """Minimal state object for testing extra channel functions."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _mk(type_ids, **overrides):
    defaults = dict(
        agent_pos=np.array([[1, 1], [1, 3]], dtype=np.int32),
        agent_dir=np.array([0, 2], dtype=np.int32),
        agent_inv=np.full((2, 1), -1, dtype=np.int32),
        object_type_map=np.zeros((H, W), dtype=np.int32),
        object_state_map=np.zeros((H, W), dtype=np.int32),
        open_pot_positions=np.array([[0, 2]], dtype=np.int32),
        open_pot_contents=np.array([[-1, -1, -1]], dtype=np.int32),
        open_pot_timer=np.array([0], dtype=np.int32),
    )
    # Default: place open_pot in object_type_map
    otm = defaults["object_type_map"].copy()
    otm[0, 2] = type_ids["open_pot"]
    defaults["object_type_map"] = otm
    defaults.update(overrides)
    return _State(**defaults)


def _at(layers, ch_idx, r, c):
    return float(layers[ch_idx][r, c])


# -----------------------------------------------------------------------
# Pot state channels
# -----------------------------------------------------------------------


class TestPotStateChannels:
    def test_empty_pot(self, extra_fn, type_ids):
        layers = extra_fn(_mk(type_ids), H, W, 0)
        assert len(layers) == 27
        for i in range(4):
            assert _at(layers, POT_STATE + i, 0, 2) == 0.0

    def test_cooking_pot(self, extra_fn, type_ids):
        oid, tid = type_ids["onion"], type_ids["tomato"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, oid, tid]], dtype=np.int32),
            open_pot_timer=np.array([15], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_STATE + 0, 0, 2) == pytest.approx(1.0)  # is_cooking
        assert _at(layers, POT_STATE + 1, 0, 2) == pytest.approx(0.0)  # is_ready
        assert _at(layers, POT_STATE + 2, 0, 2) == pytest.approx(1.0)  # fill
        assert _at(layers, POT_STATE + 3, 0, 2) == pytest.approx(0.75)  # timer

    def test_ready_pot(self, extra_fn, type_ids):
        oid = type_ids["onion"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, oid, oid]], dtype=np.int32),
            open_pot_timer=np.array([0], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_STATE + 0, 0, 2) == pytest.approx(0.0)  # not cooking
        assert _at(layers, POT_STATE + 1, 0, 2) == pytest.approx(1.0)  # ready

    def test_partial_fill(self, extra_fn, type_ids):
        oid = type_ids["onion"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, -1, -1]], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_STATE + 0, 0, 2) == pytest.approx(0.0)  # not cooking
        assert _at(layers, POT_STATE + 1, 0, 2) == pytest.approx(0.0)  # not ready
        assert _at(layers, POT_STATE + 2, 0, 2) == pytest.approx(1 / 3)  # fill


# -----------------------------------------------------------------------
# Pot ingredient breakdown channels
# -----------------------------------------------------------------------


class TestPotIngredientChannels:
    def test_empty_pot(self, extra_fn, type_ids):
        layers = extra_fn(_mk(type_ids), H, W, 0)
        for i in range(4):
            assert _at(layers, POT_INGS + i, 0, 2) == pytest.approx(0.0)

    def test_mixed_contents(self, extra_fn, type_ids):
        oid, tid = type_ids["onion"], type_ids["tomato"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, oid, tid]], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_INGS + 0, 0, 2) == pytest.approx(2 / 3)  # onion
        assert _at(layers, POT_INGS + 1, 0, 2) == pytest.approx(1 / 3)  # tomato
        assert _at(layers, POT_INGS + 2, 0, 2) == pytest.approx(0.0)  # broccoli
        assert _at(layers, POT_INGS + 3, 0, 2) == pytest.approx(0.0)  # mushroom

    def test_single_ingredient(self, extra_fn, type_ids):
        oid = type_ids["onion"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, -1, -1]], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_INGS + 0, 0, 2) == pytest.approx(1 / 3)
        assert _at(layers, POT_INGS + 1, 0, 2) == pytest.approx(0.0)

    def test_non_pot_cell_is_zero(self, extra_fn, type_ids):
        oid = type_ids["onion"]
        state = _mk(
            type_ids,
            open_pot_contents=np.array([[oid, oid, oid]], dtype=np.int32),
        )
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, POT_INGS + 0, 1, 1) == pytest.approx(0.0)


# -----------------------------------------------------------------------
# Decomposed inventory channels
# -----------------------------------------------------------------------


class TestDecomposedInventory:
    def test_holding_raw_onion(self, extra_fn, type_ids):
        inv = np.array([[type_ids["onion"]], [-1]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, 0)
        expected = [0, 0, 1, 0, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, SELF_INV + i, 1, 1) == pytest.approx(exp), INV_FIELDS[i]

    def test_holding_plate(self, extra_fn, type_ids):
        inv = np.array([[type_ids["plate"]], [-1]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, 0)
        expected = [1, 0, 0, 0, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, SELF_INV + i, 1, 1) == pytest.approx(exp), INV_FIELDS[i]

    def test_holding_onion_soup(self, extra_fn, type_ids):
        inv = np.array([[type_ids["onion_soup"]], [-1]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, 0)
        expected = [1, 1, 3, 0, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, SELF_INV + i, 1, 1) == pytest.approx(exp), INV_FIELDS[i]

    def test_holding_tomato_soup(self, extra_fn, type_ids):
        inv = np.array([[type_ids["tomato_soup"]], [-1]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, 0)
        expected = [1, 1, 0, 3, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, SELF_INV + i, 1, 1) == pytest.approx(exp), INV_FIELDS[i]

    def test_empty_inventory(self, extra_fn, type_ids):
        layers = extra_fn(_mk(type_ids), H, W, 0)
        for i in range(6):
            assert _at(layers, SELF_INV + i, 1, 1) == pytest.approx(0.0), INV_FIELDS[i]
            assert _at(layers, OTHER_INV + i, 1, 3) == pytest.approx(0.0), INV_FIELDS[i]

    def test_other_agent_inventory(self, extra_fn, type_ids):
        inv = np.array([[-1], [type_ids["plate"]]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, 0)
        expected = [1, 0, 0, 0, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, OTHER_INV + i, 1, 3) == pytest.approx(exp), INV_FIELDS[i]

    def test_agent_ordering_flipped(self, extra_fn, type_ids):
        """agent_idx=1 should put agent 1's inventory in self channels."""
        inv = np.array([[type_ids["onion"]], [type_ids["plate"]]], dtype=np.int32)
        layers = extra_fn(_mk(type_ids, agent_inv=inv), H, W, agent_idx=1)
        # Self = agent 1 at (1,3) holding plate
        assert _at(layers, SELF_INV + 0, 1, 3) == pytest.approx(1.0)  # plate
        assert _at(layers, SELF_INV + 2, 1, 3) == pytest.approx(0.0)  # no onion
        # Other = agent 0 at (1,1) holding onion
        assert _at(layers, OTHER_INV + 0, 1, 1) == pytest.approx(0.0)  # no plate
        assert _at(layers, OTHER_INV + 2, 1, 1) == pytest.approx(1.0)  # onion


# -----------------------------------------------------------------------
# Recipe indicator decomposition channels
# -----------------------------------------------------------------------


class TestRecipeIndicatorChannels:
    def _state_with_indicator(self, type_ids, target_recipe_idx):
        otm = np.zeros((H, W), dtype=np.int32)
        otm[0, 2] = type_ids["open_pot"]
        otm[2, 4] = type_ids["recipe_indicator"]
        osm = np.zeros((H, W), dtype=np.int32)
        osm[2, 4] = 1  # active indicator
        return _mk(
            type_ids,
            object_type_map=otm,
            object_state_map=osm,
            target_recipe=np.int32(target_recipe_idx),
        )

    def test_onion_soup_recipe(self, extra_fn, type_ids):
        state = self._state_with_indicator(type_ids, 0)
        layers = extra_fn(state, H, W, 0)
        expected = [1, 1, 3, 0, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, RECIPE + i, 2, 4) == pytest.approx(exp), INV_FIELDS[i]

    def test_tomato_soup_recipe(self, extra_fn, type_ids):
        state = self._state_with_indicator(type_ids, 1)
        layers = extra_fn(state, H, W, 0)
        expected = [1, 1, 0, 3, 0, 0]
        for i, exp in enumerate(expected):
            assert _at(layers, RECIPE + i, 2, 4) == pytest.approx(exp), INV_FIELDS[i]

    def test_non_indicator_cell_is_zero(self, extra_fn, type_ids):
        state = self._state_with_indicator(type_ids, 0)
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, RECIPE, 0, 0) == pytest.approx(0.0)

    def test_no_target_recipe_system(self, extra_fn, type_ids):
        """Without target_recipe in state, recipe channels are all zero."""
        layers = extra_fn(_mk(type_ids), H, W, 0)
        for i in range(6):
            assert _at(layers, RECIPE + i, 2, 4) == pytest.approx(0.0)


# -----------------------------------------------------------------------
# Correct delivery indicator channel
# -----------------------------------------------------------------------


class TestDeliveryIndicator:
    def _state_with_delivery_zone(self, type_ids, flag):
        otm = np.zeros((H, W), dtype=np.int32)
        otm[0, 2] = type_ids["open_pot"]
        otm[3, 4] = type_ids["open_delivery_zone"]
        return _mk(
            type_ids,
            object_type_map=otm,
            delivery_occurred=np.int32(flag),
        )

    def test_delivery_occurred(self, extra_fn, type_ids):
        state = self._state_with_delivery_zone(type_ids, 1)
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, DELIVERY, 3, 4) == pytest.approx(1.0)

    def test_no_delivery(self, extra_fn, type_ids):
        state = self._state_with_delivery_zone(type_ids, 0)
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, DELIVERY, 3, 4) == pytest.approx(0.0)

    def test_non_delivery_cell_is_zero(self, extra_fn, type_ids):
        state = self._state_with_delivery_zone(type_ids, 1)
        layers = extra_fn(state, H, W, 0)
        assert _at(layers, DELIVERY, 0, 0) == pytest.approx(0.0)


# -----------------------------------------------------------------------
# End-to-end pipeline test with a real V2 environment
# -----------------------------------------------------------------------

# Actions
UP, DOWN, LEFT, RIGHT, INTERACT, TOGGLE, NOOP = 0, 1, 2, 3, 4, 5, 6

# Base channel offsets in the full 35-channel observation
CH_TYPE_MAP = 0
CH_SELF_POS = 1
CH_OTHER_POS = 2
CH_SELF_DIR = 3
CH_OTHER_DIR = 4
CH_SELF_INV_BASE = 5  # 1 base channel (normalized type ID)
CH_OTHER_INV_BASE = 6
CH_OSM = 7
# Extra channels start at 8
CH_POT_STATE = 8  # 4: cook, ready, fill, timer
CH_POT_INGS = 12  # 4: onion, tomato, broccoli, mushroom
CH_SELF_INV_DECOMP = 16  # 6: plate, cooked, onion, tomato, broccoli, mushroom
CH_OTHER_INV_DECOMP = 22  # 6
CH_RECIPE_DECOMP = 28  # 6
CH_DELIVERY_IND = 34  # 1


def _reshape_obs(obs_flat, grid_h, grid_w, n_channels=35):
    """Reshape flat obs to (H, W, C)."""
    return obs_flat.reshape(grid_h, grid_w, n_channels)


def _register_test_layout():
    """Register the test layout once (idempotent)."""
    from cogrid.core.grid import layouts

    _id = "_test_v2_obs_pipeline"
    if _id not in layouts.LAYOUT_REGISTRY:
        layouts.register_layout(
            _id,
            [
                "CuC=C",
                "O+ +X",
                "CCRCC",
            ],
        )


class TestEndToEndPipeline:
    """Full pipeline test with a real V2 env.

    Steps through a cooking sequence, verifying observations at each stage.

    Layout (3 rows x 5 cols)::

        C u C = C
        O +   + X
        C C R C C

    Agent 0 spawns at (1,1), agent 1 at (1,3).
    - Onion stack O at (1,0): agent 0 faces left to pickup
    - Open pot u at (0,1): agent 0 faces up to interact with pot
    - Plate stack = at (0,3): agent 1 faces up to pickup
    - Open delivery zone X at (1,4): agent 1 faces right to deliver
    - Recipe indicator R at (2,2): visible in observations

    Each test begins with a directional move to establish a known facing
    direction (initial direction is randomized on reset).
    """

    @pytest.fixture(autouse=True)
    def setup_env(self):
        import copy

        from cogrid.envs import _ensure_v2_types, _make_v2_env, _v2_base_config
        from cogrid.envs.overcooked.config import (
            build_target_recipe_extra_state,
            build_target_recipe_tick,
        )
        from cogrid.envs.overcooked.rewards import (
            TargetRecipeDeliveryReward,
            TargetRecipeIngredientInPotReward,
            TargetRecipeSoupInDishReward,
        )

        _ensure_v2_types()
        _register_test_layout()

        config = copy.deepcopy(_v2_base_config)
        config["grid"] = {"layout": "_test_v2_obs_pipeline"}
        config["grid_height"] = 3
        config["grid_width"] = 5
        config.pop("local_view_radius", None)  # full grid obs
        config["rewards"] = [
            TargetRecipeDeliveryReward(
                coefficient=20.0,
                common_reward=True,
                penalize_incorrect=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
            TargetRecipeIngredientInPotReward(
                coefficient=3.0,
                common_reward=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
            TargetRecipeSoupInDishReward(
                coefficient=5.0,
                common_reward=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
        ]
        config["target_recipes"] = ["onion_soup", "tomato_soup"]
        config["resample_on_delivery"] = False  # keep recipe fixed for testing
        config["tick_fn"] = build_target_recipe_tick(config)
        config["extra_state_init_fn"] = build_target_recipe_extra_state(config)
        config["pickupable_types"] = [
            "onion",
            "tomato",
            "plate",
            "onion_soup",
            "tomato_soup",
        ]

        self.env = _make_v2_env(config)
        np.random.seed(0)
        self.obs, _ = self.env.reset(seed=0)
        self.grid_h = 3
        self.grid_w = 5
        self.n_ch = 35

        from cogrid.core.objects.registry import object_to_idx

        self.type_ids = {
            name: object_to_idx(name, scope="overcooked")
            for name in [
                "onion",
                "tomato",
                "plate",
                "onion_soup",
                "tomato_soup",
                "open_pot",
                "onion_stack",
                "plate_stack",
                "recipe_indicator",
                "open_delivery_zone",
                "counter",
            ]
        }

    def _obs(self, agent=0):
        return _reshape_obs(self.obs[agent], self.grid_h, self.grid_w, self.n_ch)

    def _step(self, a0, a1=NOOP):
        self.obs, rew, terms, truncs, info = self.env.step({0: a0, 1: a1})
        return rew

    def _a0_pos(self):
        return int(self.env._env_state.agent_pos[0, 0]), int(self.env._env_state.agent_pos[0, 1])

    def _a1_pos(self):
        return int(self.env._env_state.agent_pos[1, 0]), int(self.env._env_state.agent_pos[1, 1])

    def _pickup_onion_a0(self):
        """Move agent 0 left (sets facing), then interact to pickup onion."""
        self._step(LEFT)  # move/face left toward onion_stack
        self._step(INTERACT)  # pickup

    def _drop_in_pot_a0(self):
        """Move agent 0 up (sets facing toward pot), then interact to drop."""
        self._step(UP)  # face up toward pot at (0,1)
        self._step(INTERACT)  # drop

    def test_initial_state(self):
        """On reset: pot empty, agents empty inv, recipe indicator active."""
        # Step once to establish known direction (doesn't matter which)
        self._step(NOOP)
        obs = self._obs(0)

        # Pot at (0,1) empty
        assert obs[0, 1, CH_POT_STATE + 0] == pytest.approx(0.0)  # not cooking
        assert obs[0, 1, CH_POT_STATE + 1] == pytest.approx(0.0)  # not ready
        assert obs[0, 1, CH_POT_STATE + 2] == pytest.approx(0.0)  # fill = 0

        # Pot ingredients all zero
        for i in range(4):
            assert obs[0, 1, CH_POT_INGS + i] == pytest.approx(0.0)

        # Agent 0 inventory empty
        r, c = self._a0_pos()
        for i in range(6):
            assert obs[r, c, CH_SELF_INV_DECOMP + i] == pytest.approx(0.0)

        # Recipe indicator at (2,2): plate and cooked bits set
        assert obs[2, 2, CH_RECIPE_DECOMP + 0] == pytest.approx(1.0)  # plate
        assert obs[2, 2, CH_RECIPE_DECOMP + 1] == pytest.approx(1.0)  # cooked

    def test_pickup_onion_updates_inventory_channels(self):
        """After picking up an onion, self inventory channels show onion=1."""
        self._pickup_onion_a0()
        obs = self._obs(0)

        held = int(self.env._env_state.agent_inv[0, 0])
        assert held == self.type_ids["onion"], f"Expected onion, got {held}"

        r, c = self._a0_pos()
        assert obs[r, c, CH_SELF_INV_DECOMP + 0] == pytest.approx(0.0)  # plate
        assert obs[r, c, CH_SELF_INV_DECOMP + 1] == pytest.approx(0.0)  # cooked
        assert obs[r, c, CH_SELF_INV_DECOMP + 2] == pytest.approx(1.0)  # onion
        assert obs[r, c, CH_SELF_INV_DECOMP + 3] == pytest.approx(0.0)  # tomato

    def test_place_onion_in_pot_updates_pot_channels(self):
        """After placing an onion in pot, pot ingredient channels update."""
        self._pickup_onion_a0()
        self._drop_in_pot_a0()
        obs = self._obs(0)

        held = int(self.env._env_state.agent_inv[0, 0])
        assert held == -1, f"Expected empty inv, got {held}"

        assert obs[0, 1, CH_POT_STATE + 2] == pytest.approx(1 / 3)  # fill
        assert obs[0, 1, CH_POT_INGS + 0] == pytest.approx(1 / 3)  # onion
        assert obs[0, 1, CH_POT_INGS + 1] == pytest.approx(0.0)  # tomato

    def test_full_pot_starts_cooking(self):
        """After 3 onions, pot starts cooking."""
        for _ in range(3):
            self._pickup_onion_a0()
            self._drop_in_pot_a0()

        obs = self._obs(0)
        assert obs[0, 1, CH_POT_STATE + 2] == pytest.approx(1.0)  # fill
        assert obs[0, 1, CH_POT_STATE + 0] == pytest.approx(1.0)  # is_cooking
        assert obs[0, 1, CH_POT_STATE + 1] == pytest.approx(0.0)  # not ready
        assert obs[0, 1, CH_POT_INGS + 0] == pytest.approx(1.0)  # onion = 3/3

    def test_cooked_pot_shows_ready(self):
        """After cooking completes, is_ready=1 and timer=0."""
        for _ in range(3):
            self._pickup_onion_a0()
            self._drop_in_pot_a0()

        # Wait for cooking (cook_time=20)
        for _ in range(25):
            self._step(NOOP)

        obs = self._obs(0)
        assert obs[0, 1, CH_POT_STATE + 0] == pytest.approx(0.0)  # not cooking
        assert obs[0, 1, CH_POT_STATE + 1] == pytest.approx(1.0)  # ready
        assert obs[0, 1, CH_POT_STATE + 3] == pytest.approx(0.0)  # timer = 0

    def test_other_agent_inventory_visible(self):
        """Agent 1's inventory appears in agent 0's other-inv channels."""
        # Agent 1 faces up (toward plate_stack at (0,3)) and picks up
        self._step(NOOP, UP)
        self._step(NOOP, INTERACT)
        obs = self._obs(0)

        held = int(self.env._env_state.agent_inv[1, 0])
        assert held == self.type_ids["plate"], f"Expected plate, got {held}"

        r, c = self._a1_pos()
        assert obs[r, c, CH_OTHER_INV_DECOMP + 0] == pytest.approx(1.0)  # plate
        assert obs[r, c, CH_OTHER_INV_DECOMP + 1] == pytest.approx(0.0)  # not cooked
        assert obs[r, c, CH_OTHER_INV_DECOMP + 2] == pytest.approx(0.0)  # no onion


# -----------------------------------------------------------------------
# Partial observability + recipe indicators + buttons
# -----------------------------------------------------------------------


def _register_partial_obs_layout():
    """Register layout for partial-obs indicator/button tests (idempotent)."""
    from cogrid.core.grid import layouts

    _id = "_test_v2_partial_obs"
    if _id not in layouts.LAYOUT_REGISTRY:
        # 5 rows x 7 cols. Radius=2 → 5x5 window.
        #
        #   R C u C = L C
        #   O +       + X
        #   C C C C C C C
        #   C C C C C C C
        #   C C C C C C C
        #
        # Agent 0 at (1,1): can see R at (0,0).
        #   Cannot see L at (0,5) — 4 cols away, outside radius=2.
        # Agent 1 at (1,5): can see L at (0,5), faces up to toggle.
        #   Cannot see R at (0,0).
        layouts.register_layout(
            _id,
            [
                "RCuC=LC",
                "O+   +X",
                "CCCCCCC",
                "CCCCCCC",
                "CCCCCCC",
            ],
        )


class TestPartialObsIndicators:
    """Test recipe indicator and button visibility under partial observability.

    Layout (5x7) with local_view_radius=2 (5x5 window)::

        R C u C = L C
        O +         + X
        C C C C C C C
        C C C C C C C
        C C C C C C C

    Agent 0 at (1,1): R at (0,0) visible, L at (0,5) NOT visible.
    Agent 1 at (1,5): L at (0,5) visible (faces up to toggle), R at (0,0) NOT visible.
    """

    @pytest.fixture(autouse=True)
    def setup_env(self):
        import copy

        from cogrid.envs import _ensure_v2_types, _make_v2_env, _v2_base_config
        from cogrid.envs.overcooked.config import (
            build_branch_activate_button,
            build_branch_flag_delivery,
            build_target_recipe_extra_state,
            build_target_recipe_tick,
        )
        from cogrid.envs.overcooked.rewards import (
            ButtonActivationCost,
            TargetRecipeDeliveryReward,
            TargetRecipeIngredientInPotReward,
            TargetRecipeSoupInDishReward,
        )

        _ensure_v2_types()
        _register_partial_obs_layout()

        config = copy.deepcopy(_v2_base_config)
        config["grid"] = {"layout": "_test_v2_partial_obs"}
        config["local_view_radius"] = 2
        config["interactions"] = [
            build_branch_activate_button(activation_time=10),
            build_branch_flag_delivery(),
        ]
        config["rewards"] = [
            TargetRecipeDeliveryReward(
                coefficient=20.0,
                common_reward=True,
                penalize_incorrect=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
            ButtonActivationCost(coefficient=-5.0),
            TargetRecipeIngredientInPotReward(
                coefficient=3.0,
                common_reward=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
            TargetRecipeSoupInDishReward(
                coefficient=5.0,
                common_reward=True,
                target_recipes=["onion_soup", "tomato_soup"],
            ),
        ]
        config["target_recipes"] = ["onion_soup", "tomato_soup"]
        config["resample_on_delivery"] = False
        config["tick_fn"] = build_target_recipe_tick(config)
        config["extra_state_init_fn"] = build_target_recipe_extra_state(config)
        config["pickupable_types"] = [
            "onion",
            "tomato",
            "plate",
            "onion_soup",
            "tomato_soup",
        ]

        self.env = _make_v2_env(config)
        np.random.seed(0)
        self.obs, _ = self.env.reset(seed=0)
        self.radius = 2
        self.window = 2 * self.radius + 1  # 5
        self.n_ch = 35

        from cogrid.core.objects.registry import object_to_idx

        self.ri_id = object_to_idx("recipe_indicator", scope="overcooked")
        self.bi_id = object_to_idx("button_indicator", scope="overcooked")

    def _obs_spatial(self, agent=0):
        return self.obs[agent].reshape(self.window, self.window, self.n_ch)

    def _step(self, a0, a1=NOOP):
        self.obs, rew, terms, truncs, info = self.env.step({0: a0, 1: a1})
        return rew

    def _agent_pos(self, idx):
        return int(self.env._env_state.agent_pos[idx, 0]), int(
            self.env._env_state.agent_pos[idx, 1]
        )

    def _grid_to_local(self, grid_r, grid_c, agent_idx):
        """Convert grid coords to local-view coords for an agent."""
        ar, ac = self._agent_pos(agent_idx)
        lr = grid_r - ar + self.radius
        lc = grid_c - ac + self.radius
        return lr, lc

    def _in_view(self, grid_r, grid_c, agent_idx):
        lr, lc = self._grid_to_local(grid_r, grid_c, agent_idx)
        return 0 <= lr < self.window and 0 <= lc < self.window

    def test_recipe_indicator_visible_to_nearby_agent(self):
        """Agent 0 at (1,1) can see recipe indicator R at (0,0) in local view."""
        # Step to establish facing (direction doesn't matter for obs)
        self._step(NOOP)
        obs = self._obs_spatial(0)

        assert self._in_view(0, 0, 0), "R should be in agent 0's view"
        lr, lc = self._grid_to_local(0, 0, 0)

        # Recipe indicator should show decomposed recipe
        assert obs[lr, lc, CH_RECIPE_DECOMP + 0] == pytest.approx(1.0)  # plate
        assert obs[lr, lc, CH_RECIPE_DECOMP + 1] == pytest.approx(1.0)  # cooked
        # At least one ingredient channel should be non-zero
        ing_sum = sum(obs[lr, lc, CH_RECIPE_DECOMP + 2 + i] for i in range(4))
        assert ing_sum == pytest.approx(3.0), f"Recipe should have 3 ingredients, got {ing_sum}"

    def test_recipe_indicator_not_visible_to_far_agent(self):
        """Agent 1 at (1,5) cannot see recipe indicator R at (0,0)."""
        self._step(NOOP)
        assert not self._in_view(0, 0, 1), "R should NOT be in agent 1's view"

    def test_button_indicator_visible_to_nearby_agent(self):
        """Agent 1 at (1,5) can see button indicator L at (0,5) in local view."""
        self._step(NOOP)
        assert self._in_view(0, 5, 1), "L should be in agent 1's view"

    def test_button_inactive_shows_no_recipe(self):
        """Button indicator at (0,5) shows zero recipe channels when inactive."""
        self._step(NOOP)
        obs = self._obs_spatial(1)
        lr, lc = self._grid_to_local(0, 5, 1)

        for i in range(6):
            assert obs[lr, lc, CH_RECIPE_DECOMP + i] == pytest.approx(0.0), (
                f"Inactive button recipe channel {i} should be 0"
            )

    def test_button_activated_shows_recipe(self):
        """After agent 1 toggles button, it shows the target recipe."""
        # Agent 1 at (1,5): face up toward button L at (0,5), then toggle.
        self._step(NOOP, UP)  # agent 1 faces up
        self._step(NOOP, TOGGLE)  # agent 1 activates button (timer set)
        self._step(NOOP, NOOP)  # tick syncs OSM on next step

        obs = self._obs_spatial(1)
        lr, lc = self._grid_to_local(0, 5, 1)

        # Button now active → should show recipe decomposition
        assert obs[lr, lc, CH_RECIPE_DECOMP + 0] == pytest.approx(1.0)  # plate
        assert obs[lr, lc, CH_RECIPE_DECOMP + 1] == pytest.approx(1.0)  # cooked
        # Should have 3 ingredients total
        ing_sum = sum(obs[lr, lc, CH_RECIPE_DECOMP + 2 + i] for i in range(4))
        assert ing_sum == pytest.approx(3.0), f"Expected 3 ingredients, got {ing_sum}"

    def test_button_recipe_not_visible_to_far_agent(self):
        """After button activation, agent 0 can't see it (out of view)."""
        self._step(NOOP, UP)
        self._step(NOOP, TOGGLE)
        self._step(NOOP, NOOP)  # tick syncs

        assert not self._in_view(0, 5, 0), "L should NOT be in agent 0's view"

    def test_button_deactivates_after_timer(self):
        """Button recipe channels return to zero after timer expires."""
        self._step(NOOP, UP)
        self._step(NOOP, TOGGLE)  # activate (timer=10)
        self._step(NOOP, NOOP)  # tick syncs

        # Wait for timer to expire (9 remaining after sync step + extra margin)
        for _ in range(12):
            self._step(NOOP, NOOP)

        obs = self._obs_spatial(1)
        lr, lc = self._grid_to_local(0, 5, 1)

        for i in range(6):
            assert obs[lr, lc, CH_RECIPE_DECOMP + i] == pytest.approx(0.0), (
                f"Deactivated button recipe channel {i} should be 0"
            )
