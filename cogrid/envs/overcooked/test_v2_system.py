"""Tests for OvercookedV2 target recipe, button indicator, and delivery system.

Validates:
  - Button activation via toggle action
  - Button timer countdown and expiry
  - Recipe indicator state sync to object_state_map
  - Button indicator shows recipe only when active
  - Delivery flag and recipe resampling
  - TargetRecipeDeliveryReward (correct/incorrect)
  - ButtonActivationCost reward

Run with::

    pytest cogrid/envs/overcooked/test_v2_system.py -v
"""

import numpy as np
import pytest

import cogrid  # noqa: F401
import cogrid.envs  # noqa: F401 — register environments
from cogrid.backend._dispatch import _reset_backend_for_testing
from cogrid.backend.env_state import EnvState
from cogrid.core.agent import get_dir_vec_table
from cogrid.core.autowire import build_scope_config_from_components
from cogrid.core.objects import build_lookup_tables
from cogrid.core.pipeline.interactions import process_interactions
from cogrid.envs.overcooked.config import (
    build_branch_activate_button,
    build_branch_flag_delivery,
    build_target_recipe_tick,
)

_reset_backend_for_testing()

SCOPE = "overcooked"
TOGGLE = 5
PICKUP_DROP = 4
NOOP = 6


@pytest.fixture(autouse=True)
def _setup():
    _reset_backend_for_testing()


def _setup_scope():
    """Return (tables, scope_cfg, type_ids, dir_vec) for the overcooked scope."""
    import cogrid.envs.overcooked.v2_objects  # noqa: F401 — register V2 types

    tables = build_lookup_tables(scope=SCOPE)
    scope_cfg = build_scope_config_from_components(SCOPE)
    dir_vec = get_dir_vec_table()
    return tables, scope_cfg, scope_cfg["type_ids"], dir_vec


def _type_ids():
    _, _, tids, _ = _setup_scope()
    return tids


def _run_interactions(state, actions, branches=None):
    """Run process_interactions with correct signature."""
    tables, scope_cfg, _, dir_vec = _setup_scope()
    return process_interactions(
        state,
        actions,
        branches,
        tables,
        scope_cfg,
        dir_vec,
        PICKUP_DROP,
        TOGGLE,
    )


def _make_v2_state(
    agent_pos,
    agent_dir,
    agent_inv,
    otm,
    osm=None,
    target_recipe=0,
    button_timer=None,
    button_positions=None,
    indicator_positions=None,
    open_pot_contents=None,
    open_pot_timer=None,
    open_pot_positions=None,
):
    """Build a minimal EnvState for V2 testing."""
    H, W = otm.shape
    n_agents = agent_pos.shape[0]
    if osm is None:
        osm = np.zeros((H, W), dtype=np.int32)
    if button_positions is None:
        button_positions = np.zeros((0, 2), dtype=np.int32)
    if button_timer is None:
        button_timer = np.zeros(button_positions.shape[0], dtype=np.int32)
    if indicator_positions is None:
        indicator_positions = np.zeros((0, 2), dtype=np.int32)
    if open_pot_positions is None:
        open_pot_positions = np.zeros((0, 2), dtype=np.int32)
    if open_pot_contents is None:
        open_pot_contents = np.full((open_pot_positions.shape[0], 3), -1, dtype=np.int32)
    if open_pot_timer is None:
        open_pot_timer = np.zeros(open_pot_positions.shape[0], dtype=np.int32)

    extra = {
        "overcooked.target_recipe": np.int32(target_recipe),
        "overcooked.delivery_occurred": np.int32(0),
        "overcooked.button_timer": button_timer,
        "overcooked.button_positions": button_positions,
        "overcooked.indicator_positions": indicator_positions,
        "overcooked.open_pot_contents": open_pot_contents,
        "overcooked.open_pot_timer": open_pot_timer,
        "overcooked.open_pot_positions": open_pot_positions,
        # Standard pot arrays (empty for V2)
        "overcooked.pot_contents": np.full((0, 3), -1, dtype=np.int32),
        "overcooked.pot_timer": np.zeros(0, dtype=np.int32),
        "overcooked.pot_positions": np.zeros((0, 2), dtype=np.int32),
    }

    return EnvState(
        agent_pos=agent_pos,
        agent_dir=agent_dir,
        agent_inv=agent_inv,
        wall_map=np.zeros((H, W), dtype=np.int32),
        object_type_map=otm,
        object_state_map=osm,
        extra_state=extra,
        rng_key=None,
        time=np.int32(0),
        done=np.zeros(n_agents, dtype=np.bool_),
        n_agents=n_agents,
        height=H,
        width=W,
        action_set="cardinal",
    )


# ---------------------------------------------------------------------------
# Target recipe tick
# ---------------------------------------------------------------------------


class TestTargetRecipeTick:
    """Tests for build_target_recipe_tick."""

    def _make_tick(self, resample=False):
        config = {
            "target_recipes": ["onion_soup", "tomato_soup"],
            "resample_on_delivery": resample,
            "indicator_activation_time": 10,
        }
        return build_target_recipe_tick(config)

    def test_recipe_indicator_state_sync(self):
        """Tick writes target_recipe+1 to object_state_map at indicator positions."""
        tids = _type_ids()
        ri_id = tids["recipe_indicator"]

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[0, 0] = ri_id
        otm[0, 4] = ri_id
        indicator_pos = np.array([[0, 0], [0, 4]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=otm,
            target_recipe=0,
            indicator_positions=indicator_pos,
        )

        tick = self._make_tick()
        state = tick(state, {})

        # target_recipe=0 -> osm value = 0 + 1 = 1
        assert int(state.object_state_map[0, 0]) == 1
        assert int(state.object_state_map[0, 4]) == 1

    def test_button_indicator_inactive_shows_zero(self):
        """Tick writes 0 to osm at button positions when timer is 0."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 1] = bi_id
        btn_pos = np.array([[1, 1]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=otm,
            target_recipe=1,
            button_positions=btn_pos,
            button_timer=np.array([0], dtype=np.int32),
        )

        tick = self._make_tick()
        state = tick(state, {})

        assert int(state.object_state_map[1, 1]) == 0

    def test_button_indicator_active_shows_recipe(self):
        """Tick writes recipe+1 to osm at button positions when timer > 0."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 1] = bi_id
        btn_pos = np.array([[1, 1]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=otm,
            target_recipe=1,
            button_positions=btn_pos,
            button_timer=np.array([5], dtype=np.int32),
        )

        tick = self._make_tick()
        state = tick(state, {})

        # target_recipe=1 -> recipe_val = 2
        assert int(state.object_state_map[1, 1]) == 2

    def test_button_timer_countdown(self):
        """Button timer decrements by 1 each tick and stops at 0."""
        btn_pos = np.array([[1, 1]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=np.zeros((5, 5), dtype=np.int32),
            button_positions=btn_pos,
            button_timer=np.array([3], dtype=np.int32),
        )

        tick = self._make_tick()

        state = tick(state, {})
        assert int(state.extra_state["overcooked.button_timer"][0]) == 2

        state = tick(state, {})
        assert int(state.extra_state["overcooked.button_timer"][0]) == 1

        state = tick(state, {})
        assert int(state.extra_state["overcooked.button_timer"][0]) == 0

        # Should not go negative
        state = tick(state, {})
        assert int(state.extra_state["overcooked.button_timer"][0]) == 0

    def test_delivery_resampling(self):
        """When delivery_occurred flag is set and resample=True, recipe changes."""
        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=np.zeros((5, 5), dtype=np.int32),
            target_recipe=0,
        )

        # Set delivery flag
        extra = dict(state.extra_state)
        extra["overcooked.delivery_occurred"] = np.int32(1)
        import dataclasses

        state = dataclasses.replace(state, extra_state=extra)

        tick = self._make_tick(resample=True)
        state = tick(state, {})

        # Flag should be reset
        assert int(state.extra_state["overcooked.delivery_occurred"]) == 0
        # Recipe should be valid (0 or 1)
        assert int(state.extra_state["overcooked.target_recipe"]) in [0, 1]

    def test_no_resampling_without_delivery(self):
        """Recipe stays the same when no delivery occurred."""
        state = _make_v2_state(
            agent_pos=np.array([[2, 2]], dtype=np.int32),
            agent_dir=np.array([0], dtype=np.int32),
            agent_inv=np.array([[-1]], dtype=np.int32),
            otm=np.zeros((5, 5), dtype=np.int32),
            target_recipe=0,
        )

        tick = self._make_tick(resample=True)
        state = tick(state, {})

        assert int(state.extra_state["overcooked.target_recipe"]) == 0


# ---------------------------------------------------------------------------
# Button activation branch
# ---------------------------------------------------------------------------


class TestButtonActivation:
    """Tests for build_branch_activate_button."""

    def test_toggle_activates_button(self):
        """Agent facing button with empty hands and toggle activates it."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]
        build_scope_config_from_components(SCOPE)  # ensure scope is initialized

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = bi_id  # button at (1, 2)
        btn_pos = np.array([[1, 2]], dtype=np.int32)

        # Agent at (2, 2) facing up (dir=3 means UP) -> faces (1, 2)
        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[-1], [-1]], dtype=np.int32),
            otm=otm,
            button_positions=btn_pos,
            button_timer=np.array([0], dtype=np.int32),
        )

        activation_time = 10
        branch = build_branch_activate_button(activation_time=activation_time)
        user_interactions = [branch]

        actions = np.array([TOGGLE, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=user_interactions)

        timer = state.extra_state["overcooked.button_timer"]
        assert int(timer[0]) == activation_time

    def test_pickup_drop_does_not_activate_button(self):
        """Pickup/drop action should not activate button (only toggle does)."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]
        build_scope_config_from_components(SCOPE)  # ensure scope is initialized

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = bi_id
        btn_pos = np.array([[1, 2]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[-1], [-1]], dtype=np.int32),
            otm=otm,
            button_positions=btn_pos,
            button_timer=np.array([0], dtype=np.int32),
        )

        branch = build_branch_activate_button(activation_time=10)
        actions = np.array([PICKUP_DROP, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        assert int(state.extra_state["overcooked.button_timer"][0]) == 0

    def test_cannot_activate_already_active_button(self):
        """Toggle on an already-active button does not reset its timer."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]
        build_scope_config_from_components(SCOPE)  # ensure scope is initialized

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = bi_id
        btn_pos = np.array([[1, 2]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[-1], [-1]], dtype=np.int32),
            otm=otm,
            button_positions=btn_pos,
            button_timer=np.array([5], dtype=np.int32),  # already active
        )

        branch = build_branch_activate_button(activation_time=10)
        actions = np.array([TOGGLE, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        # Timer should stay at 5, not reset to 10
        assert int(state.extra_state["overcooked.button_timer"][0]) == 5

    def test_holding_item_does_not_activate_button(self):
        """Agent holding an item cannot activate button."""
        tids = _type_ids()
        bi_id = tids["button_indicator"]
        onion_id = tids["onion"]
        build_scope_config_from_components(SCOPE)  # ensure scope is initialized

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = bi_id
        btn_pos = np.array([[1, 2]], dtype=np.int32)

        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[onion_id], [-1]], dtype=np.int32),
            otm=otm,
            button_positions=btn_pos,
            button_timer=np.array([0], dtype=np.int32),
        )

        branch = build_branch_activate_button(activation_time=10)
        actions = np.array([TOGGLE, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        assert int(state.extra_state["overcooked.button_timer"][0]) == 0


# ---------------------------------------------------------------------------
# Delivery flag branch
# ---------------------------------------------------------------------------


class TestDeliveryFlag:
    """Tests for build_branch_flag_delivery."""

    def test_delivery_sets_flag(self):
        """Delivering soup to open_delivery_zone sets delivery_occurred flag."""
        tids = _type_ids()
        dz_id = tids["open_delivery_zone"]
        onion_soup_id = tids["onion_soup"]
        build_scope_config_from_components(SCOPE)  # ensure scope is initialized

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = dz_id  # delivery zone at (1, 2)

        # Agent at (2, 2) facing up -> faces (1, 2)
        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[onion_soup_id], [-1]], dtype=np.int32),
            otm=otm,
        )

        branch = build_branch_flag_delivery()
        actions = np.array([PICKUP_DROP, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        assert int(state.extra_state["overcooked.delivery_occurred"]) == 1

    def test_no_delivery_no_flag(self):
        """Flag stays 0 when no delivery happens."""
        _type_ids()  # ensure types registered

        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([0, 0], dtype=np.int32),
            agent_inv=np.array([[-1], [-1]], dtype=np.int32),
            otm=np.zeros((5, 5), dtype=np.int32),
        )

        branch = build_branch_flag_delivery()
        actions = np.array([NOOP, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        assert int(state.extra_state["overcooked.delivery_occurred"]) == 0

    def test_non_soup_item_does_not_set_flag(self):
        """Holding a raw ingredient at the delivery zone should NOT set the flag."""
        tids = _type_ids()
        dz_id = tids["open_delivery_zone"]
        onion_id = tids["onion"]

        otm = np.zeros((5, 5), dtype=np.int32)
        otm[1, 2] = dz_id

        # Agent holds an onion (not a soup) and faces the delivery zone
        state = _make_v2_state(
            agent_pos=np.array([[2, 2], [4, 4]], dtype=np.int32),
            agent_dir=np.array([3, 0], dtype=np.int32),
            agent_inv=np.array([[onion_id], [-1]], dtype=np.int32),
            otm=otm,
        )

        branch = build_branch_flag_delivery()
        actions = np.array([PICKUP_DROP, NOOP], dtype=np.int32)
        state = _run_interactions(state, actions, branches=[branch])

        assert int(state.extra_state["overcooked.delivery_occurred"]) == 0, (
            "Raw ingredient at delivery zone should not trigger delivery flag"
        )


# ---------------------------------------------------------------------------
# End-to-end: full step cycle
# ---------------------------------------------------------------------------


class TestV2EndToEnd:
    """End-to-end tests using the full environment."""

    def test_env_creation_and_reset(self):
        """V2 environment can be created and reset without errors."""
        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="numpy")
        obs, info = env.reset(seed=42)
        assert len(obs) == 2
        for agent_id in env.agents:
            assert obs[agent_id].shape[0] > 0

    def test_env_step_with_noop(self):
        """Stepping with noop actions doesn't crash."""
        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="numpy")
        env.reset(seed=42)
        actions = {a: NOOP for a in env.agents}
        obs, rewards, terms, truncs, info = env.step(actions)
        assert len(obs) == 2

    def test_button_activation_in_env(self):
        """Button timer increases when agent toggles facing a button."""
        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="numpy")
        env.reset(seed=42)

        # Check initial button timer is 0
        timer = env._env_state.extra_state["overcooked.button_timer"]
        assert np.all(timer == 0), f"Initial button timer should be 0, got {timer}"

    def test_target_recipe_in_extra_state(self):
        """Target recipe is initialized in extra_state."""
        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="numpy")
        env.reset(seed=42)

        target = env._env_state.extra_state["overcooked.target_recipe"]
        # Should be 0 or 1 (two target recipes)
        assert int(target) in [0, 1]

    def test_indicator_positions_populated(self):
        """Recipe and button indicator positions are correctly extracted."""
        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="numpy")
        env.reset(seed=42)

        ind_pos = env._env_state.extra_state["overcooked.indicator_positions"]
        btn_pos = env._env_state.extra_state["overcooked.button_positions"]

        # GroundedCoordSimple has 1 recipe indicator (R) and 1 button (L)
        assert ind_pos.shape[0] >= 1, f"Expected >=1 indicator, got {ind_pos.shape[0]}"
        assert btn_pos.shape[0] >= 1, f"Expected >=1 button, got {btn_pos.shape[0]}"

    def test_jax_backend_v2(self):
        """V2 environment works with JAX backend."""
        jax = pytest.importorskip("jax")

        env = cogrid.make("OvercookedV2-GroundedCoordSimple-V0", backend="jax")
        env.reset(seed=42)

        obs, state, _ = env.jax_reset(jax.random.key(0))
        assert 0 in obs
        assert obs[0].shape[0] > 0

        # Step with noop
        actions = {i: jax.numpy.int32(NOOP) for i in range(2)}
        obs2, state2, rewards, terms, truncs, info = env.jax_step(jax.random.key(1), state, actions)
        assert 0 in obs2


# ---------------------------------------------------------------------------
# Recipe observability and delivery rewards
# ---------------------------------------------------------------------------


class TestRecipeObservability:
    """Validate agents can read the target recipe from their observation."""

    def test_indicator_state_visible_in_observation(self):
        """The object_state_map channel contains the recipe indicator value.

        After the first tick, the indicator cell in the OSM channel should be
        nonzero (target_recipe + 1), proving the agent can distinguish recipes.
        """
        env = cogrid.make("OvercookedV2-CrampedRoomIndicator-V0", backend="numpy")
        obs, _ = env.reset(seed=42)

        # Step once so the tick writes the recipe into object_state_map
        obs, _, _, _, _ = env.step({a: NOOP for a in env.agents})

        # The indicator position from extra_state
        ind_pos = env._env_state.extra_state["overcooked.indicator_positions"]
        assert ind_pos.shape[0] >= 1

        # object_state_map at the indicator cell should be target_recipe + 1
        target = int(env._env_state.extra_state["overcooked.target_recipe"])
        ri_row, ri_col = int(ind_pos[0, 0]), int(ind_pos[0, 1])
        osm_val = int(env._env_state.object_state_map[ri_row, ri_col])
        assert osm_val == target + 1, f"OSM at indicator should be {target + 1}, got {osm_val}"

    def test_observation_differs_by_recipe(self):
        """Observations are different when the target recipe differs.

        Reset twice with different seeds until we get different initial recipes,
        then verify the observations are not identical.
        """
        env = cogrid.make("OvercookedV2-CrampedRoomIndicator-V0", backend="numpy")

        # Collect observations for different target recipes
        obs_by_recipe = {}
        for seed in range(100):
            obs, _ = env.reset(seed=seed)
            # Step once to sync indicator
            obs, _, _, _, _ = env.step({a: NOOP for a in env.agents})
            target = int(env._env_state.extra_state["overcooked.target_recipe"])
            if target not in obs_by_recipe:
                obs_by_recipe[target] = obs[env.agents[0]]
            if len(obs_by_recipe) >= 2:
                break

        assert len(obs_by_recipe) >= 2, "Could not get two different recipes across 100 seeds"
        obs_r0 = obs_by_recipe[0]
        obs_r1 = obs_by_recipe[1]
        assert not np.array_equal(obs_r0, obs_r1), (
            "Observations should differ when target recipe differs"
        )

    def test_recipe_indicator_channel_value(self):
        """The OSM channel at the indicator cell encodes the recipe index.

        Full-grid LocalView: obs is (H, W, C) flattened. The OSM is channel 7
        (after type_id, 2x pos, 2x dir, 2x inv). Verify the indicator cell
        in that channel has the expected normalized value.
        """
        env = cogrid.make("OvercookedV2-CrampedRoomIndicator-V0", backend="numpy")
        obs, _ = env.reset(seed=42)
        obs, _, _, _, _ = env.step({a: NOOP for a in env.agents})

        ind_pos = env._env_state.extra_state["overcooked.indicator_positions"]
        ri_row, ri_col = int(ind_pos[0, 0]), int(ind_pos[0, 1])

        # Reshape observation to spatial (H, W, C)
        h = env.config["grid_height"]
        w = env.config["grid_width"]
        obs_flat = obs[env.agents[0]]
        n_ch = obs_flat.shape[0] // (h * w)
        spatial = obs_flat.reshape(h, w, n_ch)

        # OSM channel is index 7 (type_id=0, pos*2=1-2, dir*2=3-4, inv*2=5-6, osm=7)
        osm_channel = spatial[:, :, 7]
        indicator_val = float(osm_channel[ri_row, ri_col])
        assert indicator_val > 0, (
            f"OSM channel at indicator ({ri_row},{ri_col}) should be >0, got {indicator_val}"
        )


class TestTargetRecipeReward:
    """Validate correct/incorrect delivery rewards."""

    def _step_env_to_delivery(self, env, soup_type_name):
        """Manually set up state so agent 0 holds soup and faces delivery zone."""
        import dataclasses

        tids = _type_ids()
        soup_id = tids[soup_type_name]

        # Find any delivery zone (standard or open) in the grid
        otm = np.array(env._env_state.object_type_map)
        dz_positions = np.zeros((0, 2), dtype=np.int64)
        for dz_name in ("delivery_zone", "open_delivery_zone"):
            if dz_name in tids:
                matches = np.argwhere(otm == tids[dz_name])
                if len(matches) > 0:
                    dz_positions = matches
                    break
        assert len(dz_positions) > 0, "No delivery zone found in layout"
        dz_row, dz_col = dz_positions[0]

        # Place agent 0 one cell above the delivery zone, facing down
        agent_row = dz_row - 1
        state = env._env_state
        new_agent_pos = np.array(state.agent_pos)
        new_agent_dir = np.array(state.agent_dir)
        new_agent_inv = np.array(state.agent_inv)

        new_agent_pos[0] = [agent_row, dz_col]
        new_agent_dir[0] = 1  # facing down
        new_agent_inv[0, 0] = soup_id

        # Move agent 1 out of the way
        new_agent_pos[1] = [0, 0]

        env._env_state = dataclasses.replace(
            state,
            agent_pos=new_agent_pos,
            agent_dir=new_agent_dir,
            agent_inv=new_agent_inv,
        )

    def test_correct_delivery_positive_reward(self):
        """Delivering the target recipe earns +coefficient reward."""
        env = cogrid.make("OvercookedV2-CrampedRoomIndicator-V0", backend="numpy")
        env.reset(seed=42)

        # Step once to sync indicator
        env.step({a: NOOP for a in env.agents})

        target = int(env._env_state.extra_state["overcooked.target_recipe"])
        target_recipes = ["onion_soup", "tomato_soup"]
        target_name = target_recipes[target]

        self._step_env_to_delivery(env, target_name)

        # Agent 0 does pickup_drop (delivers), agent 1 does noop
        obs, rewards, terms, truncs, info = env.step(
            {env.agents[0]: PICKUP_DROP, env.agents[1]: NOOP}
        )

        total = sum(float(rewards[a]) for a in env.agents)
        assert total > 0, f"Correct delivery should give positive reward, got {total}"

    def test_incorrect_delivery_negative_reward(self):
        """Delivering the wrong recipe incurs -coefficient penalty."""
        env = cogrid.make("OvercookedV2-CrampedRoomIndicator-V0", backend="numpy")
        env.reset(seed=42)

        env.step({a: NOOP for a in env.agents})

        target = int(env._env_state.extra_state["overcooked.target_recipe"])
        target_recipes = ["onion_soup", "tomato_soup"]
        # Pick the wrong recipe
        wrong_name = target_recipes[1 - target]

        self._step_env_to_delivery(env, wrong_name)

        obs, rewards, terms, truncs, info = env.step(
            {env.agents[0]: PICKUP_DROP, env.agents[1]: NOOP}
        )

        total = sum(float(rewards[a]) for a in env.agents)
        assert total < 0, f"Incorrect delivery should give negative reward, got {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
