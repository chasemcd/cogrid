"""Interactive human-play interface using pygame."""

import numpy as np

from cogrid.cogrid_env import CoGridEnv
from cogrid.core import typing
from cogrid.core.actions import Actions
from cogrid.envs import registry

try:
    import pygame
except ImportError:
    pygame = None
    raise ImportError("Must `pip install pygame` to use interactive visualizer!")

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from scipy import special
except ImportError:
    special = None

ORT_SESSIONS: dict[str, "ort.InferenceSession"] = {}

REWARDS = []


def sample_action_via_softmax(logits: np.ndarray) -> int:
    """Given logits sample an action via softmax."""
    action_distribution = special.softmax(logits)
    action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)
    return action


def inference_onnx_model(
    input_dict: dict[str, np.ndarray],
    model_path: str,
) -> np.ndarray:
    """Given an input dict and path to an ONNX model, return the model outputs."""
    outputs = ORT_SESSIONS[model_path].run(["output"], input_dict)
    return outputs


def onnx_model_inference_fn(observation: dict[str, np.ndarray] | np.ndarray, onnx_model_path: str):
    """Run inference on an ONNX model and sample an action via softmax."""
    # if it's a dictionary observation, the onnx model expects a flattened input array
    if isinstance(observation, dict):
        observation = np.hstack(list(observation.values())).reshape((1, -1))

    model_outputs = inference_onnx_model(
        {
            "obs": observation.astype(np.float32),
            "state_ins": np.array([0.0], dtype=np.float32),  # rllib artifact
        },
        model_path=onnx_model_path,
    )[0].reshape(-1)  # outputs list of a batch. batch size always 1 so index list and reshape

    action = sample_action_via_softmax(model_outputs)

    return action


def load_onnx_policy_fn(onnx_model_path: str) -> str:
    """Initialize the ORT session and return the string to access it."""
    if ORT_SESSIONS.get(onnx_model_path) is None:
        ORT_SESSIONS[onnx_model_path] = ort.InferenceSession(onnx_model_path, None)

    return onnx_model_path


model = None  # load_onnx_policy_fn("path/to/model.onnx")


ACTION_MESSAGE = ""
HUMAN_AGENT_ID = 1

ACTION_SET = "cardinal_actions"

if ACTION_SET == "cardinal_actions":
    KEY_TO_ACTION = {
        "left": Actions.MoveLeft,
        "right": Actions.MoveRight,
        "up": Actions.MoveUp,
        "down": Actions.MoveDown,
        "q": Actions.Toggle,
        "w": Actions.PickupDrop,
        "space": Actions.Noop,
    }
elif ACTION_SET == "rotation_actions":
    KEY_TO_ACTION = {
        "left": Actions.RotateLeft,
        "right": Actions.RotateRight,
        "up": Actions.Forward,
        "q": Actions.Toggle,
        "w": Actions.PickupDrop,
        "space": Actions.Noop,
    }


class HumanPlay:
    """Interactive human-controlled environment player."""

    def __init__(
        self,
        env: CoGridEnv,
        human_agent_id: typing.AgentID = None,
        seed: int = None,
    ) -> None:
        """Initialize with environment, human agent ID, and seed."""
        self.env = env
        self.seed = seed
        self.closed = False
        self.human_agent_id = human_agent_id
        self.obs = None
        self.cumulative_reward = 0

    def run(self):
        """Run the main game loop until window is closed."""
        self.reset(self.seed)
        while not self.closed:
            actions = {agent_id: Actions.Noop for agent_id in self.env.agent_ids}

            for a_id, obs in self.obs.items():
                if a_id == self.human_agent_id:
                    continue
                actions[a_id] = self.env.action_spaces[a_id].sample()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))

                    if event.key == "escape":
                        self.env.close()
                        return

                    if event.key == "backspace":
                        self.reset(self.seed)
                        return

                    if self.human_agent_id is not None and event.key in KEY_TO_ACTION.keys():
                        actions[self.human_agent_id] = KEY_TO_ACTION[event.key]
                    else:
                        print(f"Invalid action: {event.key}")

            self.step(actions)

    def step(self, actions: dict[str, Actions]):
        """Execute one step with the given actions."""
        # Convert action enum strings to integer indices
        action_set = self.env.action_set
        int_actions = {
            aid: action_set.index(a) if isinstance(a, str) else int(a) for aid, a in actions.items()
        }
        self.obs, rewards, terminateds, truncateds, _ = self.env.step(int_actions)
        self.cumulative_reward += [*rewards.values()][0]

        if not self.env.agents:
            print("All agents done!")
            REWARDS.append(self.cumulative_reward)
            if len(REWARDS) == 50:
                print(REWARDS)
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed):
        """Reset the environment with the given seed."""
        self.obs, _ = self.env.reset(seed=seed)
        self.cumulative_reward = 0
        self.env.render()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        help="environment to load",
        default="search_rescue",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=42,
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        help="size at which to render tiles",
        default=32,
    )
    parser.add_argument(
        "--agent-pov",
        action="store_true",
        help="draw the agent sees (partially observable view)",
    )
    parser.add_argument(
        "--agent-view-size",
        type=int,
        default=7,
        help="set the number of grid spaces visible in agent-view ",
    )
    parser.add_argument(
        "--screen-size",
        type=int,
        default=512,
        help="set the resolution for pygame rendering (width and height)",
    )

    args = parser.parse_args()

    def env_creator(render_mode: str | None = None, render_message="") -> CoGridEnv:
        """Create an Overcooked environment instance."""
        return registry.make(
            "Overcooked-CrampedRoom-V0",
            highlight=False,
            render_mode=render_mode,
            screen_size=args.screen_size,
            render_message=render_message,
        )

    policy_mapping = {0: "random", 1: "random"}

    # NOTE: If you need to pass a config to your policy, specify it here and the
    # policy class will be initialized with it.
    configs = {agent_id: {} for agent_id in policy_mapping.keys()}

    env: CoGridEnv = env_creator(render_mode="human")
    manual_control = HumanPlay(
        env,
        human_agent_id=HUMAN_AGENT_ID,
        seed=args.seed,
    )
    manual_control.run()
