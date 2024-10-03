from gymnasium.spaces import Dict

from cogrid.feature_space import feature

FEATURE_SPACE_REGISTRY: dict[str, feature.Feature] = {}


def register_feature(feature_id: str, feature_class: feature.Feature) -> None:
    if feature_id in FEATURE_SPACE_REGISTRY:
        print(
            f"A feature is already registered with the ID `{feature_id}`. Overwriting it."
        )

    FEATURE_SPACE_REGISTRY[feature_id] = feature_class


def make_feature_generator(feature_id: str, **kwargs) -> feature.Feature:
    if feature_id not in FEATURE_SPACE_REGISTRY:
        raise ValueError(f"Feature {feature_id} not registered.")
    return FEATURE_SPACE_REGISTRY[feature_id](**kwargs)


class FeatureSpace:
    def __init__(self, feature_names: list[str], env, agent_id: str, **kwargs):
        self.env = env
        self.feature_generators = self._setup_feature_generators(feature_names)
        self.observation_space = Dict(
            {feature.name: feature.space for feature in self.feature_generators}
        )
        self.agent_id = agent_id

    def generate_features(self):
        generated_features = {
            feature.name: feature.generate(self.env, self.agent_id)
            for feature in self.feature_generators
        }
        return generated_features

    def _setup_feature_generators(self, feature_names):
        return [
            self._fetch_feature_generator(feature_name, env=self.env)
            for feature_name in feature_names
        ]

    def _fetch_feature_generator(self, feat_name, **kwargs):
        return make_feature_generator(feature_id=feat_name, **kwargs)

        # TODO(chase): all of these features should have args replaced with env
        # if feat_name == "full_map_image":
        #     return features.FullMapImage(self.env.shape, self.env.tile_size)
        # if feat_name == "fov_image":
        #     return features.FoVImage(self.env.agent_view_size, self.env.tile_size)
        # if feat_name == "full_map_encoding":
        #     return features.FullMapEncoding(self.env.shape)
        # if feat_name == "fov_encoding":
        #     return features.FoVEncoding(self.env.agent_view_size)
        # if feat_name == "full_map_ascii":
        #     return features.FullMapASCII(self.env.shape)
        # if feat_name == "fov_ascii":
        #     return features.FoVASCII(self.env.agent_view_size)
        # if feat_name == "agent_positions":
        #     return features.AgentPositions(self.env.shape)
        # if feat_name == "other_agent_actions":
        #     return features.OtherAgentActions(
        #         len(self.env.agent_ids), self.env.action_space.n
        #     )
        # if feat_name == "inventory":
        #     # TODO(chase): If functionality added for >1 inventory item, must change this.
        #     return features.Inventory(inventory_capacity=1)
        # if feat_name == "other_agent_visibility":
        #     return features.OtherAgentVisibility(
        #         len(self.env.agent_ids), self.env.agent_view_size
        #     )
        # if feat_name == "role":
        #     return features.Role(self.env.num_roles)
        # if feat_name == "action_mask":
        #     return features.ActionMask(self.env.action_space.n)
        # if feat_name == "agent_id":
        #     return features.AgentID(len(self.env.agent_ids))
        # if feat_name == "stacked_full_map_resized_grayscale_image":
        #     return features.StackedFullMapResizedGrayscale()
        # if feat_name == "full_map_resized_grayscale_image":
        #     return features.FullMapResizedGrayscale()
        # raise NotImplementedError(f"Feature {feat_name} not implemented.")
