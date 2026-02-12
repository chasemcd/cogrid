"""Backward-compatibility aliases for the unified step pipeline.

The actual implementation lives in :mod:`cogrid.core.step_pipeline`.
These aliases maintain import compatibility during the Phase 8->9
transition. All callers (``cogrid_env.py``, tests) that import from
this module continue to work unchanged.

.. deprecated::
    Import from ``cogrid.core.step_pipeline`` instead. This module
    will be removed in Phase 9.
"""
from cogrid.core.step_pipeline import (
    step as jax_step,
    reset as jax_reset,
    build_step_fn as make_jitted_step,
    build_reset_fn as make_jitted_reset,
    envstate_to_dict,
)
