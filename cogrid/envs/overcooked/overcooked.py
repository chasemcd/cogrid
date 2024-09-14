import copy

import numpy as np

from cogrid.core import directions
from cogrid import cogrid_env
from cogrid.envs.overcooked import agent
from cogrid.envs import registry
from cogrid.envs.overcooked import rewards


class Overcooked(cogrid_env.CoGridEnv):
    """The Overcooked task is a reproduction of the Overcooked-AI
    environment from Carroll et al. (2019). The original paper
    can be found here: https://arxiv.org/abs/1910.05789 and
    source code is provided through
    https://github.com/HumanCompatibleAI/overcooked_ai
    """

    def __init__(self, config, render_mode=None, **kwargs):
        """Constructor method"""
        super().__init__(
            config=config,
            render_mode=render_mode,
            agent_class=agent.OvercookedAgent,
            **kwargs,
        )
