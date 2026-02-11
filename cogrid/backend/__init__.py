"""cogrid.backend -- Array backend dispatch.

Provides a single array API that resolves to numpy or jax.numpy:

    from cogrid.backend import xp
    arr = xp.zeros((3, 3))

The backend defaults to numpy. Call set_backend('jax') before any other
cogrid imports to switch to JAX.

The ``xp`` name is resolved lazily via ``__getattr__`` so that
``from cogrid.backend import xp`` always returns the *current* backend
module, even if ``set_backend()`` was called after the initial import
of this package.
"""

from cogrid.backend._dispatch import set_backend, get_backend

__all__ = ["xp", "set_backend", "get_backend"]


def __getattr__(name: str):
    if name == "xp":
        from cogrid.backend import _dispatch
        return _dispatch.xp
    raise AttributeError(f"module 'cogrid.backend' has no attribute {name!r}")
