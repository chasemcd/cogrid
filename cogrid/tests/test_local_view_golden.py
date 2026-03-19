"""Golden output tests for LocalView and OvercookedLocalView.

Captures exact tensor baselines on a seeded CrampedRoom state before
any code changes begin.  Any refactoring that alters observation output
will trigger an immediate assertion failure with a channel-level diff
message identifying which specific channels diverged.

Golden values were captured with seed=42 on the CrampedRoom layout.
"""

import numpy as np
from numpy.testing import assert_array_equal

from cogrid.feature_space.local_view import LocalView
from cogrid.envs.overcooked.features import OvercookedLocalView


# ---------------------------------------------------------------------------
# Golden arrays (sparse representation: size, nonzero indices, nonzero values)
#
# Captured from Overcooked-CrampedRoom-V0 with env.reset(seed=42).
# Base LocalView: 7x7 window, 31 channels = 1519 elements
# OvercookedLocalView: 7x7 window, 39 channels = 1911 elements
# ---------------------------------------------------------------------------

def _build_golden(size, indices, values):
    """Reconstruct a golden array from sparse (index, value) pairs."""
    arr = np.zeros(size, dtype=np.float32)
    arr[indices] = values
    return arr


GOLDEN_BASE_AGENT0 = _build_golden(
    1519,
    [496, 527, 589, 620, 717, 754, 759, 841, 930, 1003, 1009, 1054, 1147, 1184, 1209, 1241, 1271],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

GOLDEN_BASE_AGENT1 = _build_golden(
    1519,
    [248, 279, 341, 372, 469, 507, 515, 593, 682, 754, 757, 806, 899, 936, 961, 993, 1023],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

GOLDEN_OC_AGENT0 = _build_golden(
    1911,
    [624, 663, 733, 738, 741, 780, 901, 946, 951, 1057, 1170, 1259, 1265, 1326, 1443, 1488, 1521, 1561, 1599],
    [1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

GOLDEN_OC_AGENT1 = _build_golden(
    1911,
    [312, 351, 421, 426, 429, 468, 589, 635, 643, 745, 858, 946, 949, 1014, 1131, 1176, 1209, 1249, 1287],
    [1.0, 1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

# ---------------------------------------------------------------------------
# Env config matching CrampedRoom defaults
# ---------------------------------------------------------------------------

_ENV_CONFIG = {
    "observable_radius": 3,
    "n_agents": 2,
    "pickupable_types": ["onion", "onion_soup", "plate", "tomato", "tomato_soup"],
    "local_view_type_names": [
        "counter",
        "delivery_zone",
        "onion",
        "onion_soup",
        "onion_stack",
        "plate",
        "plate_stack",
        "tomato",
        "tomato_soup",
        "tomato_stack",
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _channel_diff_msg(actual, expected, window, n_channels):
    """Return a message identifying which channels differ between two arrays."""
    actual_3d = actual.reshape(window, window, n_channels)
    expected_3d = expected.reshape(window, window, n_channels)
    diff_channels = np.where(np.any(actual_3d != expected_3d, axis=(0, 1)))[0]
    return f"Channels {diff_channels.tolist()} differ out of {n_channels}"


def _make_state():
    """Create a deterministic CrampedRoom state with seed=42."""
    from cogrid.backend._dispatch import _reset_backend_for_testing
    from cogrid.backend import set_backend

    _reset_backend_for_testing()
    set_backend("numpy")

    import cogrid.envs  # noqa: F401 -- trigger registration
    from cogrid.envs import registry
    from cogrid.backend.state_view import StateView

    env = registry.make("Overcooked-CrampedRoom-V0")
    env.reset(seed=42)
    sd = env._state

    state = StateView(
        agent_pos=np.array(sd["agent_pos"], dtype=np.int32),
        agent_dir=np.array(sd["agent_dir"], dtype=np.int32),
        agent_inv=np.array(sd["agent_inv"], dtype=np.int32),
        wall_map=np.array(sd["wall_map"], dtype=np.int32),
        object_type_map=np.array(sd["object_type_map"], dtype=np.int32),
        object_state_map=np.array(sd["object_state_map"], dtype=np.int32),
        extra={
            "pot_positions": np.array(sd["pot_positions"], dtype=np.int32),
            "pot_contents": np.array(sd["pot_contents"], dtype=np.int32),
            "pot_timer": np.array(sd["pot_timer"], dtype=np.int32),
        },
    )
    return state, _ENV_CONFIG


# ---------------------------------------------------------------------------
# Golden tests
# ---------------------------------------------------------------------------


def test_golden_base_local_view_agent0():
    """Base LocalView output for agent 0 matches golden baseline exactly."""
    state, env_config = _make_state()
    fn = LocalView.build_feature_fn("overcooked", env_config)
    actual = fn(state, 0)
    assert actual.shape == GOLDEN_BASE_AGENT0.shape
    assert actual.dtype == np.float32
    window = 2 * env_config["observable_radius"] + 1
    n_ch = len(actual) // (window * window)
    msg = _channel_diff_msg(actual, GOLDEN_BASE_AGENT0, window, n_ch)
    assert_array_equal(actual, GOLDEN_BASE_AGENT0, err_msg=msg)


def test_golden_base_local_view_agent1():
    """Base LocalView output for agent 1 matches golden baseline exactly."""
    state, env_config = _make_state()
    fn = LocalView.build_feature_fn("overcooked", env_config)
    actual = fn(state, 1)
    assert actual.shape == GOLDEN_BASE_AGENT1.shape
    assert actual.dtype == np.float32
    window = 2 * env_config["observable_radius"] + 1
    n_ch = len(actual) // (window * window)
    msg = _channel_diff_msg(actual, GOLDEN_BASE_AGENT1, window, n_ch)
    assert_array_equal(actual, GOLDEN_BASE_AGENT1, err_msg=msg)


def test_golden_overcooked_local_view_agent0():
    """OvercookedLocalView output for agent 0 matches golden baseline exactly."""
    state, env_config = _make_state()
    fn = OvercookedLocalView.build_feature_fn("overcooked", env_config)
    actual = fn(state, 0)
    assert actual.shape == GOLDEN_OC_AGENT0.shape
    assert actual.dtype == np.float32
    window = 2 * env_config["observable_radius"] + 1
    n_ch = len(actual) // (window * window)
    msg = _channel_diff_msg(actual, GOLDEN_OC_AGENT0, window, n_ch)
    assert_array_equal(actual, GOLDEN_OC_AGENT0, err_msg=msg)


def test_golden_overcooked_local_view_agent1():
    """OvercookedLocalView output for agent 1 matches golden baseline exactly."""
    state, env_config = _make_state()
    fn = OvercookedLocalView.build_feature_fn("overcooked", env_config)
    actual = fn(state, 1)
    assert actual.shape == GOLDEN_OC_AGENT1.shape
    assert actual.dtype == np.float32
    window = 2 * env_config["observable_radius"] + 1
    n_ch = len(actual) // (window * window)
    msg = _channel_diff_msg(actual, GOLDEN_OC_AGENT1, window, n_ch)
    assert_array_equal(actual, GOLDEN_OC_AGENT1, err_msg=msg)
