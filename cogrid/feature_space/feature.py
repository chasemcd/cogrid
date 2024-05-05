import numpy as np
from gymnasium import spaces

from cogrid.core import typing


class Feature:
    """Base class for creating observation features.

    :param low: Minimum value for this feature.
    :type low: float
    :param high: Maximum value for this feature.
    :type high: float
    :param name: Descriptive name for this feature.
    :type name: str
    :param shape: Shape of the feature, defaults to None
    :type shape: tuple | np.ndarray, optional
    :param space: Gymnasium Space for this features, defaults to None
    :type space: spaces.Space | None, optional
    """

    shape = None

    def __init__(
        self,
        low: float,
        high: float,
        name: str,
        shape: tuple | np.ndarray = None,
        space: spaces.Space | None = None,
        **kwargs
    ):
        """Generator Method"""
        self.low = low
        self.high = high

        if shape is not None:
            self.shape = shape
        assert self.shape is not None, "Must specify shape via class or init!"

        self.space = spaces.Box(low, high, shape) if space is None else space
        self.name = name

    def generate(
        self, env: typing.EnvType, player_id: typing.AgentID, **kwargs
    ):
        """Generate the feature for the specified player in environment.

        :param env: The CoGridEnv instance to calculate the feature from (e.g., state S).
        :type env: cogrid_env.CoGridEnv
        :param player_id: The AgentID to calculate the feature for.
        :type player_id: typing.AgentID
        :raises NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError
