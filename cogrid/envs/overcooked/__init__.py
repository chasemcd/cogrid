"""Overcooked cooperative cooking environment."""

from cogrid.envs.overcooked import (
    overcooked_grid_objects,  # noqa: F401 -- triggers @register_object_type (must load before features)
    features,  # noqa: F401 -- triggers @register_feature_type
    rewards,  # noqa: F401 -- triggers @register_reward_type
)
