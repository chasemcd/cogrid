"""Backend dispatch module for cogrid.

Manages the active array library (numpy or jax.numpy) used by all simulation code.
The backend is set once per process and cannot be changed after first use.

Usage:
    from cogrid.backend import xp
    arr = xp.zeros((3, 3))

To switch to JAX:
    from cogrid.backend import set_backend
    set_backend('jax')  # Must be called before any other cogrid imports use xp
"""

import numpy

# Module-level globals
_backend_name: str = "numpy"
_backend_set: bool = False
xp_module = numpy


class BackendProxy:
    """Proxy that delegates attribute access to the current backend module.

    This allows ``from cogrid.backend import xp`` at module level.
    When code calls ``xp.zeros(...)``, the proxy resolves ``zeros`` on the
    *current* backend (numpy or jax.numpy), not the one active at import time.
    """

    def __getattr__(self, name):
        return getattr(xp_module, name)

    def __repr__(self):
        return f"BackendProxy(current={_backend_name})"


xp = BackendProxy()


def set_backend(name: str) -> None:
    """Set the active array backend ('numpy' or 'jax').

    Must be called before any simulation code accesses ``xp``. Once set,
    the backend cannot be changed within the same process.
    """
    global _backend_name, _backend_set, xp_module

    if _backend_set and name != _backend_name:
        raise RuntimeError(
            f"Backend already set to '{_backend_name}'. "
            "Cannot change backend after first environment creation."
        )

    if name == "numpy":
        xp_module = numpy
    elif name == "jax":
        try:
            import jax.numpy as jnp

            xp_module = jnp
        except ImportError:
            raise ImportError(
                "Backend 'jax' requested but JAX is not installed.\n"
                "Install JAX with: pip install jax jaxlib\n"
                "Or use the numpy backend (default): CoGridEnv(config)"
            )
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'numpy' or 'jax'.")

    _backend_set = True
    _backend_name = name


def get_backend() -> str:
    """Return the currently active backend name ('numpy' or 'jax')."""
    return _backend_name


def _reset_backend_for_testing() -> None:
    """Reset global backend state for test isolation.

    WARNING: This is a private test-only function. Do NOT use in production
    code. It clears the backend lock so tests can switch between numpy and
    JAX backends within a single process.

    Also resets the EnvState pytree registration flag since a fresh backend
    requires fresh registration.
    """
    global _backend_name, _backend_set, xp_module
    _backend_name = "numpy"
    _backend_set = False
    xp_module = numpy

    # NOTE: Do NOT reset _pytree_registered here. JAX's internal pytree
    # registry is process-global and cannot be un-registered. Setting
    # _pytree_registered = False would cause a duplicate registration
    # error on the next call to register_envstate_pytree(). The flag
    # stays True after the first successful registration.
