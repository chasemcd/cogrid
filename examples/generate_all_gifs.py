"""Train IPPO on all Overcooked layouts and generate showcase GIFs.

Produces two combined APNGs (animated PNGs with transparency):
  examples/v1_layouts.png — standard Overcooked environments
  examples/v2_layouts.png — OvercookedV2 benchmark environments

Trained parameters are saved under examples/checkpoints/ so that GIFs
can be re-generated without retraining (``--skip-training``).

Usage:
    python examples/generate_all_gifs.py                       # 5M steps (quick)
    python examples/generate_all_gifs.py --timesteps 50000000  # 50M steps
    python examples/generate_all_gifs.py --skip-training       # re-gen from checkpoints
    python examples/generate_all_gifs.py --v1-only
    python examples/generate_all_gifs.py --v2-only
"""

import argparse
import gc
import math
import os
import re
import sys

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes, to_bytes
from PIL import Image, ImageDraw, ImageFont

# Allow imports from the examples/ directory and the project root
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_overcooked_jax import ActorCritic
from train_overcooked_jax import make_train as make_train_mlp
from train_overcooked_jax import visualize_policy as visualize_policy_mlp
from train_overcooked_v2_jax import ActorCriticRNN, gru_initialize_carry
from train_overcooked_v2_jax import make_train as make_train_cnn
from train_overcooked_v2_jax import visualize_policy as visualize_policy_cnn

import cogrid

# ---------------------------------------------------------------------------
# Environment lists
# ---------------------------------------------------------------------------

CMK_ENVS = [
    "Overcooked-CrampedMixedKitchen-V0",
]

V1_ENVS = [
    # Row 1
    "Overcooked-CrampedRoom-V0",
    "Overcooked-ForcedCoordination-V0",
    "Overcooked-CoordinationRing-V0",
    # Row 2
    "Overcooked-AsymmetricAdvantages-V0",
    "Overcooked-CounterCircuit-V0",
]

V2_ENVS = [
    "OvercookedV2-GroundedCoordSimple-V0",
    "OvercookedV2-GroundedCoordRing-V0",
    "OvercookedV2-TestTimeSimple-V0",
    "OvercookedV2-TestTimeWide-V0",
    "OvercookedV2-DemoCookSimple-V0",
    "OvercookedV2-DemoCookWide-V0",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXAMPLES_DIR = os.path.dirname(__file__)
GIF_DIR = os.path.join(EXAMPLES_DIR, "gifs")
CKPT_DIR = os.path.join(EXAMPLES_DIR, "checkpoints")

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _ckpt_path(env_id: str) -> str:
    return os.path.join(CKPT_DIR, f"{env_id}.params")


def save_params(params, env_id: str):
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = _ckpt_path(env_id)
    with open(path, "wb") as f:
        f.write(to_bytes(params))
    print(f"  Saved checkpoint to {path}")


def load_params(env_id: str, template_params):
    path = _ckpt_path(env_id)
    with open(path, "rb") as f:
        return from_bytes(template_params, f.read())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def env_id_to_label(env_id: str) -> str:
    """Convert env ID to a human-readable label."""
    parts = env_id.split("-")
    middle = "-".join(parts[1:-1])
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", middle)


def make_title_frame(text: str, width: int, height: int, n_frames: int = 15):
    """Create title card frames (dark background, white text)."""
    img = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(img)

    font_size = max(14, width // 20)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((width - tw) // 2, (height - th) // 2),
        text,
        fill=(255, 255, 255),
        font=font,
    )

    frame = np.array(img)
    return [frame] * n_frames


def pad_frame(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center-pad a frame to target dimensions with transparency."""
    h, w = frame.shape[:2]
    padded = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    y_off = target_h - h
    x_off = (target_w - w) // 2
    padded[y_off : y_off + h, x_off : x_off + w, :3] = frame[..., :3]
    padded[y_off : y_off + h, x_off : x_off + w, 3] = 255
    return padded


def resize_frame(frame: np.ndarray, target_w: int) -> np.ndarray:
    """Resize frame to target width, preserving aspect ratio (nearest neighbor)."""
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / w
    target_h = int(h * scale)
    img = Image.fromarray(frame)
    img = img.resize((target_w, target_h), Image.NEAREST)
    return np.array(img)


# ---------------------------------------------------------------------------
# Train + visualize (V1 MLP)
# ---------------------------------------------------------------------------


def _shaped_reward_args_v1(env, config):
    """Build shaped-reward annealing kwargs for make_train_mlp."""
    from cogrid.envs.overcooked.rewards import (
        ButtonActivationCost,
        DeliveryReward,
        ExpiredOrderPenalty,
    )

    _sparse_types = (DeliveryReward, ExpiredOrderPenalty, ButtonActivationCost)
    reward_instances = env.config.get("rewards", [])
    shaped_indices = [
        inst._reward_index for inst in reward_instances if not isinstance(inst, _sparse_types)
    ]
    return {
        "set_reward_coefficients_fn": env.set_reward_coefficients,
        "initial_reward_coefficients": env._reward_config["initial_reward_coefficients"],
        "shaped_reward_indices": shaped_indices,
    }


def train_and_visualize_v1(env_id: str, config: dict, gif_dir: str) -> str:
    """Train MLP IPPO on one V1/CMK environment and generate an episode GIF."""
    gif_path = os.path.join(gif_dir, f"{env_id}.gif")
    print(f"\n{'=' * 60}")
    print(f"Training (MLP): {env_id}")
    print(f"{'=' * 60}")

    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=config["SEED"])

    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.possible_agents)
    n_actions = len(env.action_set)

    test_obs, _, _ = reset_fn(jax.random.key(0))
    obs_dim = test_obs[0].shape[-1]
    print(f"  {n_agents} agents, {n_actions} actions, obs_dim={obs_dim}")

    train_fn = jax.jit(
        make_train_mlp(
            config,
            step_fn,
            reset_fn,
            n_agents,
            n_actions,
            obs_dim,
            **_shaped_reward_args_v1(env, config),
        )
    )
    print("  Compiling...")
    out = train_fn(jax.random.key(config["SEED"]))

    _print_results(out)

    params = out["runner_state"][0].params
    save_params(params, env_id)

    network = ActorCritic(n_actions, activation=config["ACTIVATION"])
    max_steps = env.max_steps
    visualize_policy_mlp(network, params, env_id, max_steps=max_steps, gif_path=gif_path, fps=15)

    del out, train_fn, env
    gc.collect()
    return gif_path


# ---------------------------------------------------------------------------
# Train + visualize (V2 CNN+RNN)
# ---------------------------------------------------------------------------


def _get_obs_shape(env):
    """Infer spatial obs shape from a local_view environment."""
    test_obs, _, _ = env.jax_reset(jax.random.key(0))
    obs_dim_flat = test_obs[0].shape[-1]
    radius = env.config.get("local_view_radius")
    if radius is not None:
        h = w = 2 * radius + 1
    else:
        h = env.config["grid_height"]
        w = env.config["grid_width"]
    n_channels = obs_dim_flat // (h * w)
    return (h, w, n_channels)


def _shaped_reward_args_v2(env, config):
    """Build shaped-reward annealing kwargs for make_train_cnn."""
    from cogrid.envs.overcooked.rewards import (
        ButtonActivationCost,
        DeliveryReward,
        ExpiredOrderPenalty,
    )

    _sparse_types = (DeliveryReward, ExpiredOrderPenalty, ButtonActivationCost)
    reward_instances = env.config.get("rewards", [])
    shaped_indices = [
        inst._reward_index for inst in reward_instances if not isinstance(inst, _sparse_types)
    ]
    return {
        "set_reward_coefficients_fn": env.set_reward_coefficients,
        "initial_reward_coefficients": env._reward_config["initial_reward_coefficients"],
        "shaped_reward_indices": shaped_indices,
    }


def train_and_visualize_v2(env_id: str, config: dict, gif_dir: str) -> str:
    """Train CNN+RNN IPPO on one V2 environment and generate an episode GIF."""
    gif_path = os.path.join(gif_dir, f"{env_id}.gif")
    print(f"\n{'=' * 60}")
    print(f"Training (CNN+RNN): {env_id}")
    print(f"{'=' * 60}")

    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=config["SEED"])

    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.possible_agents)
    n_actions = len(env.action_set)
    obs_shape = _get_obs_shape(env)
    print(f"  {n_agents} agents, {n_actions} actions, obs_shape={obs_shape}")

    train_fn = jax.jit(
        make_train_cnn(
            config,
            step_fn,
            reset_fn,
            n_agents,
            n_actions,
            obs_shape,
            **_shaped_reward_args_v2(env, config),
        )
    )
    print("  Compiling...")
    out = train_fn(jax.random.key(config["SEED"]))

    _print_results(out)

    params = out["runner_state"][0].params
    save_params(params, env_id)

    network = ActorCriticRNN(
        action_dim=n_actions,
        gru_hidden_dim=config["GRU_HIDDEN_DIM"],
        fc_dim=config["FC_DIM"],
        activation=config["ACTIVATION"],
    )
    max_steps = env.max_steps
    visualize_policy_cnn(
        network,
        params,
        env_id,
        obs_shape,
        gru_hidden_dim=config["GRU_HIDDEN_DIM"],
        max_steps=max_steps,
        gif_path=gif_path,
        fps=15,
    )

    del out, train_fn, env
    gc.collect()
    return gif_path


# ---------------------------------------------------------------------------
# Regenerate GIF from saved checkpoint (no training)
# ---------------------------------------------------------------------------


def regenerate_gif_v1(env_id: str, config: dict, gif_dir: str) -> str:
    """Load saved params and generate a GIF for a V1/CMK environment."""
    gif_path = os.path.join(gif_dir, f"{env_id}.gif")
    ckpt = _ckpt_path(env_id)
    if not os.path.exists(ckpt):
        print(f"  Skipping {env_id} (no checkpoint at {ckpt})")
        return gif_path

    print(f"  Regenerating GIF: {env_id}")
    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=config["SEED"])
    n_actions = len(env.action_set)

    network = ActorCritic(n_actions, activation=config["ACTIVATION"])

    # Build template params for deserialization
    test_obs, _, _ = env.jax_reset(jax.random.key(0))
    obs_dim = test_obs[0].shape[-1]
    template = network.init(jax.random.key(0), jnp.zeros(obs_dim, dtype=jnp.float32))
    params = load_params(env_id, template)

    visualize_policy_mlp(
        network,
        params,
        env_id,
        max_steps=env.max_steps,
        gif_path=gif_path,
        fps=15,
    )
    return gif_path


def regenerate_gif_v2(env_id: str, config: dict, gif_dir: str) -> str:
    """Load saved params and generate a GIF for a V2 environment."""
    gif_path = os.path.join(gif_dir, f"{env_id}.gif")
    ckpt = _ckpt_path(env_id)
    if not os.path.exists(ckpt):
        print(f"  Skipping {env_id} (no checkpoint at {ckpt})")
        return gif_path

    print(f"  Regenerating GIF: {env_id}")
    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=config["SEED"])
    n_agents = len(env.possible_agents)
    n_actions = len(env.action_set)
    obs_shape = _get_obs_shape(env)

    network = ActorCriticRNN(
        action_dim=n_actions,
        gru_hidden_dim=config["GRU_HIDDEN_DIM"],
        fc_dim=config["FC_DIM"],
        activation=config["ACTIVATION"],
    )

    # Build template params for deserialization
    init_hstate = gru_initialize_carry(n_agents, config["GRU_HIDDEN_DIM"])
    init_obs = jnp.zeros((1, n_agents, *obs_shape), dtype=jnp.float32)
    init_dones = jnp.zeros((1, n_agents), dtype=jnp.float32)
    template = network.init(jax.random.key(0), init_hstate, (init_obs, init_dones))
    params = load_params(env_id, template)

    visualize_policy_cnn(
        network,
        params,
        env_id,
        obs_shape,
        gru_hidden_dim=config["GRU_HIDDEN_DIM"],
        max_steps=env.max_steps,
        gif_path=gif_path,
        fps=15,
    )
    return gif_path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _print_results(out):
    ep_returns = np.array(out["metrics"]["returned_episode_returns"])
    ep_dones = np.array(out["metrics"]["returned_episode"])
    completed = ep_returns[ep_dones > 0]
    if len(completed) > 0:
        tail = max(1, len(completed) // 10)
        mean_ret = completed[-tail:].mean()
        print(f"  {len(completed)} episodes, mean return (last {tail}): {mean_ret:.2f}")
    else:
        print("  No episodes completed")


def stitch_gifs(
    gif_paths: list[str],
    env_ids: list[str],
    output_path: str,
    cell_w: int = 640,
    gap: int = 12,
    max_frames_per_layout: int = 300,
):
    """Arrange per-layout GIFs in a grid so all layouts play simultaneously."""
    max_h = 0

    raw_layout_frames = []
    max_raw_w = 0
    for gif_path, env_id in zip(gif_paths, env_ids):
        if not os.path.exists(gif_path):
            print(f"  Skipping {env_id} (no GIF found)")
            continue
        reader = imageio.get_reader(gif_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames_per_layout:
                break
            frames.append(frame)
        reader.close()
        max_raw_w = max(max_raw_w, frames[0].shape[1])
        raw_layout_frames.append((env_id, frames))

    scale = cell_w / max_raw_w if max_raw_w > 0 else 1.0
    layout_frames = []
    for env_id, frames in raw_layout_frames:
        target_w = int(frames[0].shape[1] * scale)
        resized = [resize_frame(f, target_w) for f in frames]
        max_h = max(max_h, resized[0].shape[0])
        layout_frames.append((env_id, resized))

    if not layout_frames:
        print("  No GIFs to stitch")
        return

    n = len(layout_frames)
    n_cols = math.ceil(math.sqrt(n))
    n_rows = math.ceil(n / n_cols)

    label_h = max(20, max_h // 8)
    cell_h = max_h + label_h
    grid_w = n_cols * cell_w + (n_cols - 1) * gap
    grid_h = n_rows * cell_h

    font_size = max(10, label_h // 2)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size,
        )
    except OSError:
        font = ImageFont.load_default()

    row_counts = []
    for r in range(n_rows):
        start = r * n_cols
        count = min(n_cols, n - start)
        row_counts.append(count)

    def _row_x_offset(row):
        items = row_counts[row]
        row_w = items * cell_w + (items - 1) * gap
        return (grid_w - row_w) // 2

    content_x_offsets = [(cell_w - frames[0].shape[1]) // 2 for _, frames in layout_frames]

    n_frames = max(len(frames) for _, frames in layout_frames)

    all_frames = []
    for t in range(n_frames):
        canvas = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

        for idx, (env_id, frames) in enumerate(layout_frames):
            row, col = divmod(idx, n_cols)
            frame = frames[min(t, len(frames) - 1)]
            padded = pad_frame(frame, max_h, cell_w)

            y0 = row * cell_h
            x0 = _row_x_offset(row) + col * (cell_w + gap)
            canvas[y0 : y0 + max_h, x0 : x0 + cell_w] = padded

        img = Image.fromarray(canvas, "RGBA")
        draw = ImageDraw.Draw(img)
        for idx, (env_id, _) in enumerate(layout_frames):
            row, col = divmod(idx, n_cols)
            label = env_id_to_label(env_id)
            x0 = _row_x_offset(row) + col * (cell_w + gap) + content_x_offsets[idx]
            y0 = row * cell_h + max_h + 2
            draw.text((x0, y0), label, fill=(200, 200, 200, 255), font=font)

        all_frames.append(np.array(img))

    apng_path = re.sub(r"\.gif$", ".png", output_path)
    pil_frames = [Image.fromarray(f, "RGBA") for f in all_frames]
    pil_frames[0].save(
        apng_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // 15,
        loop=0,
    )
    print(f"Saved grid APNG ({n_frames} frames, {n_rows}x{n_cols}) to {apng_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train all Overcooked layouts and generate GIFs")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5_000_000,
        help="Total timesteps per env",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Regenerate GIFs from saved checkpoints",
    )
    parser.add_argument("--cmk-only", action="store_true", help="Only Cramped Mixed Kitchen")
    parser.add_argument("--v1-only", action="store_true", help="Only V1 envs")
    parser.add_argument("--v2-only", action="store_true", help="Only V2 envs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(GIF_DIR, exist_ok=True)

    v1_config = {
        "LR": 4e-4,
        "NUM_ENVS": 64,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": args.timesteps,
        "SHAPED_REWARD_ANNEAL_TIMESTEPS": args.timesteps // 2,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 16,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.2,
        "ENT_COEF_FINAL": 3e-3,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "SEED": args.seed,
    }

    v2_config = {
        "LR": 0.00025,
        "NUM_ENVS": 256,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": args.timesteps,
        "SHAPED_REWARD_ANNEAL_TIMESTEPS": args.timesteps // 2,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 64,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "ENT_COEF_FINAL": 1e-3,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "ANNEAL_LR": True,
        "GRU_HIDDEN_DIM": 64,
        "FC_DIM": 64,
        "SEED": args.seed,
    }

    # Determine which groups to process
    explicit = args.cmk_only or args.v1_only or args.v2_only
    do_cmk = args.cmk_only or not explicit
    do_v1 = args.v1_only or not explicit
    do_v2 = args.v2_only or not explicit

    if not args.skip_training:
        if do_cmk:
            for env_id in CMK_ENVS:
                train_and_visualize_v1(env_id, v1_config, GIF_DIR)
        if do_v1:
            for env_id in V1_ENVS:
                train_and_visualize_v1(env_id, v1_config, GIF_DIR)
        if do_v2:
            for env_id in V2_ENVS:
                train_and_visualize_v2(env_id, v2_config, GIF_DIR)
    else:
        if do_cmk:
            for env_id in CMK_ENVS:
                regenerate_gif_v1(env_id, v1_config, GIF_DIR)
        if do_v1:
            for env_id in V1_ENVS:
                regenerate_gif_v1(env_id, v1_config, GIF_DIR)
        if do_v2:
            for env_id in V2_ENVS:
                regenerate_gif_v2(env_id, v2_config, GIF_DIR)

    # Stitch into combined APNGs
    if do_cmk:
        cmk_paths = [os.path.join(GIF_DIR, f"{eid}.gif") for eid in CMK_ENVS]
        stitch_gifs(cmk_paths, CMK_ENVS, os.path.join(EXAMPLES_DIR, "cmk_layout.png"))

    if do_v1:
        v1_paths = [os.path.join(GIF_DIR, f"{eid}.gif") for eid in V1_ENVS]
        stitch_gifs(v1_paths, V1_ENVS, os.path.join(EXAMPLES_DIR, "v1_layouts.png"))

    if do_v2:
        v2_paths = [os.path.join(GIF_DIR, f"{eid}.gif") for eid in V2_ENVS]
        stitch_gifs(v2_paths, V2_ENVS, os.path.join(EXAMPLES_DIR, "v2_layouts.png"))


if __name__ == "__main__":
    main()
