"""Type aliases shared across the codebase."""

import typing

import numpy as np
from pettingzoo.utils import env

ActionType = env.ActionType
AgentID = env.AgentID
ObsType = env.ObsType
Any = typing.Any
EnvType = env.ParallelEnv

try:
    import jax

    ArrayLike = np.ndarray | jax.Array
except ImportError:
    ArrayLike = np.ndarray
