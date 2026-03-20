"""Tests for the new LocalView API contract (Phase 2).

Covers:
- Import re-exports from cogrid.feature_space
- _scatter_to_grid helper (NumPy and JAX)
- __init_subclass__ validation for missing n_extra_channels
- Channel count mismatch runtime assertion
- JAX JIT tracing with instance-in-closure pattern
- End-to-end new subclass API with extra channels
"""

import pytest


# ===================================================================
# Test 1: Import re-exports (should PASS immediately)
# ===================================================================


def test_import_from_feature_space():
    """Verify `from cogrid.feature_space import LocalView, register_feature_type` works."""
    from cogrid.feature_space import LocalView, register_feature_type

    assert LocalView is not None
    assert register_feature_type is not None


# ===================================================================
# Test 2: _scatter_to_grid with NumPy backend
# ===================================================================


def test_scatter_to_grid_numpy():
    """Verify LocalView._scatter_to_grid produces correct output with NumPy."""
    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    set_backend("numpy")

    from cogrid.feature_space.local_view import LocalView

    import numpy as np

    rows = np.array([0, 2], dtype=np.int32)
    cols = np.array([1, 3], dtype=np.int32)
    values = np.array([1.0, 0.5], dtype=np.float32)
    result = LocalView._scatter_to_grid(4, 5, rows, cols, values)
    assert result.shape == (4, 5)
    assert result.dtype == np.float32
    assert result[0, 1] == 1.0
    assert result[2, 3] == 0.5
    assert result[0, 0] == 0.0  # untouched cell


# ===================================================================
# Test 3: _scatter_to_grid with JAX backend
# ===================================================================


jax = pytest.importorskip("jax")


def test_scatter_to_grid_jax():
    """Verify LocalView._scatter_to_grid produces correct output with JAX."""
    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    set_backend("jax")

    from cogrid.feature_space.local_view import LocalView

    import jax.numpy as jnp

    rows = jnp.array([0, 2], dtype=jnp.int32)
    cols = jnp.array([1, 3], dtype=jnp.int32)
    values = jnp.array([1.0, 0.5], dtype=jnp.float32)
    result = LocalView._scatter_to_grid(4, 5, rows, cols, values)
    assert result.shape == (4, 5)
    assert float(result[0, 1]) == 1.0
    assert float(result[2, 3]) == 0.5


# ===================================================================
# Test 4: __init_subclass__ validates missing n_extra_channels
# ===================================================================


def test_missing_n_extra_channels_error():
    """Verify __init_subclass__ raises TypeError when n_extra_channels is missing."""
    from cogrid.feature_space.local_view import LocalView

    with pytest.raises(TypeError, match="must define n_extra_channels"):

        class BadSubclass(LocalView):
            pass


# ===================================================================
# Test 5: Channel count mismatch runtime assertion
# ===================================================================


def test_channel_count_mismatch_error():
    """Verify runtime error when n_extra_channels mismatches extra_channels() output."""
    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    set_backend("numpy")

    import numpy as np

    from cogrid.feature_space.local_view import LocalView
    from cogrid.core.component_registry import register_feature_type

    @register_feature_type("test_mismatch_lv", scope="test_mismatch_scope")
    class MismatchView(LocalView):
        n_extra_channels = 3

        def extra_channels(self, state, H, W):
            return np.zeros((H, W, 5), dtype=np.float32)  # says 3, returns 5

    fn = MismatchView.build_feature_fn(
        "test_mismatch_scope", env_config={"observable_radius": 1, "n_agents": 1}
    )
    # Create a minimal state to trigger the call
    from cogrid.backend.state_view import StateView

    state = StateView(
        agent_pos=np.array([[1, 1]], dtype=np.int32),
        agent_dir=np.array([0], dtype=np.int32),
        agent_inv=np.full((1, 1), -1, dtype=np.int32),
        wall_map=np.zeros((3, 3), dtype=np.int32),
        object_type_map=np.zeros((3, 3), dtype=np.int32),
        object_state_map=np.zeros((3, 3), dtype=np.int32),
    )
    with pytest.raises(
        ValueError,
        match=r"extra_channels\(\) returned shape \(3, 3, 5\), expected \(3, 3, 3\)",
    ):
        fn(state, 0)


# ===================================================================
# Test 6: JAX JIT tracing with instance-in-closure
# ===================================================================


def test_jax_jit_tracing():
    """Verify the refactored LocalView works under jax.jit."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    set_backend("jax")

    from cogrid.feature_space.local_view import LocalView
    from cogrid.core.component_registry import register_feature_type
    from cogrid.backend.state_view import StateView, register_stateview_pytree

    # StateView must be registered as a JAX pytree before JIT tracing
    register_stateview_pytree()

    @register_feature_type("test_jit_lv", scope="test_jit_scope")
    class JitTestView(LocalView):
        n_extra_channels = 1

        def extra_channels(self, state, H, W):
            ch = jnp.zeros((H, W, 1), dtype=jnp.float32)
            return ch

    env_config = {"observable_radius": 1, "n_agents": 1}
    fn = JitTestView.build_feature_fn("test_jit_scope", env_config=env_config)
    state = StateView(
        agent_pos=jnp.array([[1, 1]], dtype=jnp.int32),
        agent_dir=jnp.array([0], dtype=jnp.int32),
        agent_inv=jnp.full((1, 1), -1, dtype=jnp.int32),
        wall_map=jnp.zeros((3, 3), dtype=jnp.int32),
        object_type_map=jnp.zeros((3, 3), dtype=jnp.int32),
        object_state_map=jnp.zeros((3, 3), dtype=jnp.int32),
    )
    jitted = jax.jit(fn, static_argnums=(1,))
    result = jitted(state, 0)
    assert result.shape[0] > 0  # produces output without error


# ===================================================================
# Test 7: End-to-end new subclass API with extra channels
# ===================================================================


def test_new_subclass_with_extra_channels():
    """End-to-end test: define subclass with n_extra_channels=2, verify output."""
    from cogrid.backend import set_backend
    from cogrid.backend._dispatch import _reset_backend_for_testing

    _reset_backend_for_testing()
    set_backend("numpy")

    import numpy as np

    from cogrid.feature_space.local_view import LocalView
    from cogrid.core.component_registry import register_feature_type
    from cogrid.backend.state_view import StateView

    @register_feature_type("test_new_api_lv", scope="test_new_api_scope")
    class NewApiView(LocalView):
        n_extra_channels = 2

        def extra_channels(self, state, H, W):
            ch0 = np.ones((H, W), dtype=np.float32)
            ch1 = np.full((H, W), 0.5, dtype=np.float32)
            return np.stack([ch0, ch1], axis=-1)  # (H, W, 2)

    env_config = {"observable_radius": 1, "n_agents": 1}
    dim = NewApiView.compute_obs_dim("test_new_api_scope", env_config)
    fn = NewApiView.build_feature_fn("test_new_api_scope", env_config=env_config)
    state = StateView(
        agent_pos=np.array([[1, 1]], dtype=np.int32),
        agent_dir=np.array([0], dtype=np.int32),
        agent_inv=np.full((1, 1), -1, dtype=np.int32),
        wall_map=np.zeros((3, 3), dtype=np.int32),
        object_type_map=np.zeros((3, 3), dtype=np.int32),
        object_state_map=np.zeros((3, 3), dtype=np.int32),
    )
    result = fn(state, 0)
    assert result.shape == (dim,)
    assert result.dtype == np.float32
