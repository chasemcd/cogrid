"""Validate order observation encoding for CrampedMixedKitchen.

Tests that order_recipe, order_timer, and the order_observation feature
correctly encode active orders as recipe one-hot + normalized timer,
and that orders spawn, count down, and expire as expected.

Both unit tests (direct feature function calls with known states) and
end-to-end tests (real CrampedMixedKitchen env) are included.
"""

import numpy as np
import pytest

import cogrid
import cogrid.envs
from cogrid.envs.overcooked.features import OrderObservation

NOOP = 6


# -----------------------------------------------------------------------
# Unit tests: feature function with synthetic states
# -----------------------------------------------------------------------


class FakeState:
    """Minimal state for testing the order observation function directly."""

    def __init__(self, order_recipe, order_timer):
        self.order_recipe = np.array(order_recipe, dtype=np.int32)
        self.order_timer = np.array(order_timer, dtype=np.int32)


@pytest.fixture(scope="module")
def order_fn():
    """Build order_observation fn with n_recipes=2, max_active=3, time_limit=200."""
    env_config = {
        "orders": {"max_active": 3, "time_limit": 200},
        "recipes": ["onion_soup", "tomato_soup"],
    }
    return OrderObservation.build_feature_fn("overcooked", env_config)


class TestOrderObservationEncoding:
    """Test the order_observation feature function against known states."""

    def test_no_active_orders(self, order_fn):
        """All orders inactive (-1): observation should be all zeros."""
        state = FakeState([-1, -1, -1], [0, 0, 0])
        obs = order_fn(state)
        assert obs.shape == (9,)
        np.testing.assert_array_equal(obs, np.zeros(9))

    def test_single_onion_soup_order(self, order_fn):
        """One active onion_soup order (recipe=0) with timer=100."""
        state = FakeState([0, -1, -1], [100, 0, 0])
        obs = order_fn(state)

        # Slot 0: [1, 0, 0.5]  (onion_soup one-hot, 100/200 timer)
        assert obs[0] == pytest.approx(1.0)  # onion_soup
        assert obs[1] == pytest.approx(0.0)  # not tomato_soup
        assert obs[2] == pytest.approx(0.5)  # 100/200

        # Slots 1-2: inactive
        np.testing.assert_array_almost_equal(obs[3:], np.zeros(6))

    def test_single_tomato_soup_order(self, order_fn):
        """One active tomato_soup order (recipe=1) with timer=200."""
        state = FakeState([1, -1, -1], [200, 0, 0])
        obs = order_fn(state)

        assert obs[0] == pytest.approx(0.0)  # not onion_soup
        assert obs[1] == pytest.approx(1.0)  # tomato_soup
        assert obs[2] == pytest.approx(1.0)  # 200/200 = full time

    def test_multiple_orders(self, order_fn):
        """Three active orders with different recipes and timers."""
        state = FakeState([0, 1, 0], [50, 150, 200])
        obs = order_fn(state)

        # Slot 0: onion_soup, 50/200
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(0.0)
        assert obs[2] == pytest.approx(0.25)

        # Slot 1: tomato_soup, 150/200
        assert obs[3] == pytest.approx(0.0)
        assert obs[4] == pytest.approx(1.0)
        assert obs[5] == pytest.approx(0.75)

        # Slot 2: onion_soup, 200/200
        assert obs[6] == pytest.approx(1.0)
        assert obs[7] == pytest.approx(0.0)
        assert obs[8] == pytest.approx(1.0)

    def test_timer_at_zero(self, order_fn):
        """Active order with timer=0 should show recipe but 0.0 timer."""
        state = FakeState([1, -1, -1], [0, 0, 0])
        obs = order_fn(state)
        assert obs[0] == pytest.approx(0.0)
        assert obs[1] == pytest.approx(1.0)  # tomato_soup
        assert obs[2] == pytest.approx(0.0)  # timer at zero

    def test_observation_is_global(self):
        """order_observation is per_agent=False (same for all agents)."""
        assert OrderObservation.per_agent is False


# -----------------------------------------------------------------------
# End-to-end tests: real CrampedMixedKitchen environment
# -----------------------------------------------------------------------


class TestOrdersInCrampedMixedKitchen:
    """End-to-end validation using the real CrampedMixedKitchen environment.

    Tests verify that:
    - Order observation slice is in the correct position
    - Orders spawn, tick down, and appear in observations
    - Both agents see identical order state
    - Observation values match underlying extra_state arrays
    - Feature dimensions match the actual order config
    """

    @pytest.fixture(autouse=True)
    def setup_env(self):
        self.env = cogrid.make("Overcooked-CrampedMixedKitchen-V0")
        self.obs, _ = self.env.reset(seed=42)

        # Read actual config from the env
        order_cfg = self.env.config["orders"]
        self.max_active = order_cfg["max_active"]
        self.time_limit = order_cfg["time_limit"]
        self.n_recipes = 2  # onion_soup, tomato_soup
        self.order_dim = self.max_active * (self.n_recipes + 1)

        # order_observation is the last feature
        total_dim = self.obs[0].shape[0]
        self.order_slice = slice(total_dim - self.order_dim, total_dim)

    def _step(self):
        actions = {a: NOOP for a in self.env.agents}
        self.obs, rew, terms, truncs, info = self.env.step(actions)
        return rew

    def _order_obs(self, agent=0):
        return self.obs[agent][self.order_slice]

    def _extra(self, key):
        return np.array(self.env._env_state.extra_state[f"overcooked.{key}"])

    def test_order_dim_matches_config(self):
        """Order obs dimension matches max_active * (n_recipes + 1)."""
        obs = self._order_obs()
        assert obs.shape == (self.order_dim,), f"Expected ({self.order_dim},), got {obs.shape}"

    def test_initial_orders_empty(self):
        """At reset, no orders are active — obs and state both zero."""
        order_recipe = self._extra("order_recipe")
        assert np.all(order_recipe == -1), f"Expected all -1, got {order_recipe}"

        obs = self._order_obs()
        np.testing.assert_array_almost_equal(obs, np.zeros(self.order_dim))

    def test_orders_same_for_both_agents(self):
        """Both agents see identical order observation after orders spawn."""
        for _ in range(500):
            self._step()

        obs0 = self._order_obs(0)
        obs1 = self._order_obs(1)
        np.testing.assert_array_equal(obs0, obs1)

    def test_order_spawns_eventually(self):
        """Orders appear in extra_state within a reasonable number of steps."""
        spawned = False
        for _ in range(2000):
            self._step()
            if np.any(self._extra("order_recipe") >= 0):
                spawned = True
                break
        assert spawned, "No orders spawned in 2000 steps"

    def test_spawned_order_visible_in_obs(self):
        """When an order spawns, its recipe one-hot and timer appear in obs."""
        # Step until an order spawns
        for _ in range(2000):
            self._step()
            recipes = self._extra("order_recipe")
            if np.any(recipes >= 0):
                break

        if not np.any(recipes >= 0):
            pytest.skip("No order spawned")

        obs = self._order_obs()
        active_idx = int(np.argmax(recipes >= 0))
        recipe_id = int(recipes[active_idx])
        timer = int(self._extra("order_timer")[active_idx])

        offset = active_idx * (self.n_recipes + 1)

        # Recipe one-hot
        expected_onehot = np.zeros(self.n_recipes, dtype=np.float32)
        expected_onehot[recipe_id] = 1.0
        np.testing.assert_array_almost_equal(
            obs[offset : offset + self.n_recipes],
            expected_onehot,
            err_msg=f"Order {active_idx} recipe one-hot wrong",
        )

        # Timer normalized
        expected_timer = timer / self.time_limit
        actual_timer = obs[offset + self.n_recipes]
        assert actual_timer == pytest.approx(expected_timer, abs=1e-5), (
            f"Order {active_idx} timer: expected {expected_timer}, got {actual_timer}"
        )

    def test_order_timer_decrements(self):
        """Active order timer decreases by 1 each step."""
        for _ in range(2000):
            self._step()
            if np.any(self._extra("order_recipe") >= 0):
                break

        recipes = self._extra("order_recipe")
        if not np.any(recipes >= 0):
            pytest.skip("No order spawned")

        active_idx = int(np.argmax(recipes >= 0))
        timer_before = int(self._extra("order_timer")[active_idx])
        assert timer_before > 1, "Need timer > 1 to test decrement"

        self._step()

        # If order is still active, timer should have decremented
        if self._extra("order_recipe")[active_idx] >= 0:
            timer_after = int(self._extra("order_timer")[active_idx])
            assert timer_after == timer_before - 1

    def test_timer_obs_decreases_over_steps(self):
        """The normalized timer in the observation decreases as steps pass."""
        for _ in range(2000):
            self._step()
            if np.any(self._extra("order_recipe") >= 0):
                break

        recipes = self._extra("order_recipe")
        if not np.any(recipes >= 0):
            pytest.skip("No order spawned")

        active_idx = int(np.argmax(recipes >= 0))
        offset = active_idx * (self.n_recipes + 1) + self.n_recipes

        timer_obs_before = float(self._order_obs()[offset])

        # Step a few times
        for _ in range(5):
            self._step()
            if self._extra("order_recipe")[active_idx] < 0:
                break  # order expired or was consumed

        if self._extra("order_recipe")[active_idx] >= 0:
            timer_obs_after = float(self._order_obs()[offset])
            assert timer_obs_after < timer_obs_before, (
                f"Timer obs should decrease: {timer_obs_before} -> {timer_obs_after}"
            )

    def test_full_observation_matches_state_for_all_slots(self):
        """Cross-check every order slot in obs against extra_state arrays."""
        # Step enough for orders to appear
        for _ in range(500):
            self._step()

        recipes = self._extra("order_recipe")
        timers = self._extra("order_timer")
        obs = self._order_obs()

        for i in range(self.max_active):
            offset = i * (self.n_recipes + 1)
            if recipes[i] >= 0:
                # Active: check one-hot
                for r in range(self.n_recipes):
                    expected = 1.0 if r == recipes[i] else 0.0
                    assert obs[offset + r] == pytest.approx(expected), (
                        f"Slot {i} recipe[{r}]: expected {expected}, got {obs[offset + r]}"
                    )
                # Check timer
                expected_t = timers[i] / self.time_limit
                assert obs[offset + self.n_recipes] == pytest.approx(expected_t, abs=1e-5), (
                    f"Slot {i} timer: expected {expected_t}, got {obs[offset + self.n_recipes]}"
                )
            else:
                # Inactive: all zeros
                for j in range(self.n_recipes + 1):
                    assert obs[offset + j] == pytest.approx(0.0), (
                        f"Inactive slot {i}[{j}] should be 0, got {obs[offset + j]}"
                    )

    def test_expired_order_clears_from_obs(self):
        """After an order expires, its observation slot returns to zeros."""
        # Step until an order spawns
        for _ in range(2000):
            self._step()
            if np.any(self._extra("order_recipe") >= 0):
                break

        recipes = self._extra("order_recipe")
        if not np.any(recipes >= 0):
            pytest.skip("No order spawned")

        active_idx = int(np.argmax(recipes >= 0))
        timer = int(self._extra("order_timer")[active_idx])

        # Step until this order expires
        for _ in range(timer + 5):
            self._step()
            if self._extra("order_recipe")[active_idx] < 0:
                break

        if self._extra("order_recipe")[active_idx] >= 0:
            pytest.skip("Order didn't expire (may have been replaced)")

        # Verify obs slot is zeroed (unless a new order filled it)
        obs = self._order_obs()
        offset = active_idx * (self.n_recipes + 1)
        if self._extra("order_recipe")[active_idx] < 0:
            for j in range(self.n_recipes + 1):
                assert obs[offset + j] == pytest.approx(0.0), (
                    f"Expired slot {active_idx}[{j}] should be 0"
                )

    def test_n_expired_increments(self):
        """order_n_expired increases when orders time out."""
        expired_before = int(self._extra("order_n_expired"))

        # Step a long time to guarantee at least one expiry
        for _ in range(self.time_limit + 2000):
            self._step()

        expired_after = int(self._extra("order_n_expired"))
        assert expired_after > expired_before, (
            f"order_n_expired should increase: {expired_before} -> {expired_after}"
        )

    def test_inactive_slots_always_zero(self):
        """Any slot with order_recipe=-1 must have all-zero observation."""
        # Step a moderate amount (some slots active, some not)
        for _ in range(300):
            self._step()

        recipes = self._extra("order_recipe")
        obs = self._order_obs()

        for i in range(self.max_active):
            if recipes[i] < 0:
                offset = i * (self.n_recipes + 1)
                slot_obs = obs[offset : offset + self.n_recipes + 1]
                np.testing.assert_array_almost_equal(
                    slot_obs,
                    np.zeros(self.n_recipes + 1),
                    err_msg=f"Inactive slot {i} should be all zeros",
                )
