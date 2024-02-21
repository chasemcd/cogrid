from cogrid.feature_space import feature_space
from cogrid.feature_space import features

feature_space.register_feature("full_map_image", features.FullMapImage)
feature_space.register_feature(
    "stacked_full_map_resized_grayscale_image", features.StackedFullMapResizedGrayscale
)
feature_space.register_feature(
    "full_map_resized_grayscale_image", features.FullMapResizedGrayscale
)
feature_space.register_feature("fov_image", features.FoVImage)
feature_space.register_feature("full_map_encoding", features.FullMapEncoding)
feature_space.register_feature("fov_encoding", features.FoVEncoding)
feature_space.register_feature("full_map_ascii", features.FullMapASCII)
feature_space.register_feature("fov_ascii", features.FoVASCII)
feature_space.register_feature("agent_position", features.AgentPosition)
feature_space.register_feature("agent_positions", features.AgentPositions)
feature_space.register_feature("agent_dir", features.AgentDir)
feature_space.register_feature("other_agent_actions", features.OtherAgentActions)
feature_space.register_feature("other_agent_visibility", features.OtherAgentVisibility)
feature_space.register_feature("role", features.Role)
feature_space.register_feature("inventory", features.Inventory)
feature_space.register_feature("action_mask", features.ActionMask)
feature_space.register_feature("agent_id", features.AgentID)
