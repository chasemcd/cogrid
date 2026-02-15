"""Backend-aware array mutation helpers.

This is the ONLY module that branches on numpy vs JAX for mutation.
All other code uses ``xp`` for operations that work identically.
"""
from cogrid.backend._dispatch import get_backend


def set_at(arr, idx, value):
    """Return new array with arr[idx] = value.

    idx can be a single index, a tuple, or any valid numpy/jax indexing.
    numpy: copies then assigns. JAX: uses .at[idx].set(value).
    """
    if get_backend() == "jax":
        return arr.at[idx].set(value)
    out = arr.copy()
    out[idx] = value
    return out


def set_at_2d(arr, row, col, value):
    """Return new array with arr[row, col] = value. Convenience wrapper."""
    return set_at(arr, (row, col), value)
