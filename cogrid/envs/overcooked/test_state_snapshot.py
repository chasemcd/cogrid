"""Regression tests for CoGridEnv.get_state / set_state round-trip.

Uses an OvercookedV2 layout because it exposes both bugs simultaneously:
  - Bug 1 (shape collapse): the layout registers V1 ``pot`` and
    ``button_indicator`` containers but instantiates none, so their
    ``_contents`` / ``_positions`` entries are 2D arrays with a zero-sized
    leading dim. ``np.array(v).tolist()`` collapses these to ``[]``, which
    then deserializes to shape ``(0,)`` instead of ``(0, capacity)``.
  - Bug 2 (dtype coercion): ``reward_coefficients`` is float32 and was
    previously coerced to int32 on restore, truncating fractional values.

Run with::

    pytest cogrid/envs/overcooked/test_state_snapshot.py -v
"""

import dataclasses
import json

import numpy as np
import pytest

import cogrid  # noqa: F401
import cogrid.envs  # noqa: F401 — register environments
from cogrid.backend._dispatch import _reset_backend_for_testing
from cogrid.envs import registry
from cogrid.envs.overcooked import v2_objects  # noqa: F401 — register V2 types


@pytest.fixture(autouse=True)
def _setup():
    _reset_backend_for_testing()


def _shapes_and_dtypes(extra_state):
    return {k: (np.asarray(v).shape, str(np.asarray(v).dtype)) for k, v in extra_state.items()}


def test_round_trip_preserves_all_extra_state_shapes_and_dtypes():
    env = registry.make("OvercookedV2-TestTimeSimple-V0")
    env.reset()

    before = _shapes_and_dtypes(env._env_state.extra_state)

    snapshot = json.loads(json.dumps(env.get_state()))
    env.set_state(snapshot)

    after = _shapes_and_dtypes(env._env_state.extra_state)

    assert before == after, (
        f"extra_state shape/dtype changed across round-trip:\n  before={before}\n  after={after}"
    )


def test_round_trip_preserves_empty_2d_container_shape():
    """Bug 1 regression: pot_contents shape (0, 3) must not collapse to (0,)."""
    env = registry.make("OvercookedV2-TestTimeSimple-V0")
    env.reset()

    before = np.asarray(env._env_state.extra_state["overcooked.pot_contents"])
    assert before.shape == (0, 3), (
        f"fixture assumption violated: expected (0, 3), got {before.shape}"
    )

    env.set_state(json.loads(json.dumps(env.get_state())))

    after = np.asarray(env._env_state.extra_state["overcooked.pot_contents"])
    assert after.shape == (0, 3)


def test_round_trip_preserves_float_dtype():
    """Bug 2 regression: float32 reward_coefficients must not become int32."""
    env = registry.make("OvercookedV2-TestTimeSimple-V0")
    env.reset()

    key = "overcooked.reward_coefficients"
    before = np.asarray(env._env_state.extra_state[key])
    assert before.dtype == np.float32, (
        f"fixture assumption violated: expected float32, got {before.dtype}"
    )

    env.set_state(json.loads(json.dumps(env.get_state())))

    after = np.asarray(env._env_state.extra_state[key])
    assert after.dtype == np.float32
    np.testing.assert_array_equal(after, before)


def test_step_succeeds_after_round_trip():
    """Bug 1 consequence: container tick_fn raised AxisError on collapsed arrays."""
    env = registry.make("OvercookedV2-TestTimeSimple-V0")
    env.reset()

    env.set_state(json.loads(json.dumps(env.get_state())))

    env.step({0: 6, 1: 6})


def test_round_trip_preserves_fractional_reward_coefficient_value():
    """Bug 2 consequence: 0.5 was truncated to 0 by int32 coercion."""
    env = registry.make("OvercookedV2-TestTimeSimple-V0")
    env.reset()

    key = "overcooked.reward_coefficients"
    coeffs = np.asarray(env._env_state.extra_state[key]).copy()
    coeffs[0] = 0.5
    env._env_state = dataclasses.replace(
        env._env_state,
        extra_state={**env._env_state.extra_state, key: coeffs.astype(np.float32)},
    )

    env.set_state(json.loads(json.dumps(env.get_state())))

    restored = np.asarray(env._env_state.extra_state[key])
    assert restored.dtype == np.float32
    assert restored[0] == pytest.approx(0.5)
