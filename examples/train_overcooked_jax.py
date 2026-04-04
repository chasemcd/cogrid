"""Train shared-parameter IPPO on CoGrid Overcooked (JAX backend).

Demonstrates CoGrid's JAX-native step/reset pipeline with:
- jax.vmap for parallel environments
- jax.lax.scan for the training loop
- Auto-reset on episode completion

Usage:
    python examples/train_overcooked_jax.py



IMPORTANT! All credit for this script goes to JAXMarl. We've simply copied their
implementation and dropped in the CoGrid Overcooked environment.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from flax.training.train_state import TrainState

import cogrid


ENVIRONMENT_NAME = "Overcooked-CrampedMixedKitchen-V0"


# ---- Categorical distribution helpers ----


def categorical_sample(rng, logits):
    return jax.random.categorical(rng, logits)


def categorical_log_prob(logits, actions):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.take_along_axis(log_probs, actions[..., None].astype(jnp.int32), axis=-1).squeeze(-1)


def categorical_entropy(logits):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)


# ---- Network ----


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor = activation(actor)
        actor = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return logits, jnp.squeeze(critic, axis=-1)


# ---- Transition storage ----


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# ---- Training ----


def make_train(config, step_fn, reset_fn, n_agents, n_actions, obs_dim):
    """Build a fully JIT-compilable IPPO train function.

    Args:
        config: Hyperparameter dict.
        step_fn: CoGrid step function with signature
            ``(key, state, actions_dict) -> (obs, state, rew, term, trunc, info)``.
        reset_fn: CoGrid reset function with signature
            ``(rng) -> (obs_dict, state, infos_dict)``.
        n_agents: Number of agents per env.
        n_actions: Number of discrete actions.
        obs_dim: Flat observation size per agent.
    """
    num_envs = config["NUM_ENVS"]
    num_actors = n_agents * num_envs
    num_steps = config["NUM_STEPS"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // num_steps // num_envs)
    num_minibatches = config["NUM_MINIBATCHES"]

    network = ActorCritic(n_actions, activation=config["ACTIVATION"])

    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    def entropy_schedule(update_step):
        """Linearly decay entropy coefficient from ENT_COEF to ENT_COEF_FINAL."""
        frac = 1.0 - update_step / num_updates
        return config["ENT_COEF_FINAL"] + (config["ENT_COEF"] - config["ENT_COEF_FINAL"]) * frac

    def train(rng):
        # ---- Init network ----
        rng, init_rng = jax.random.split(rng)
        params = network.init(init_rng, jnp.zeros(obs_dim, dtype=jnp.float32))

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        # ---- Init envs (vmapped) ----
        rng, reset_rng = jax.random.split(rng)
        obs_dict, env_state, _ = jax.vmap(reset_fn)(jax.random.split(reset_rng, num_envs))
        obs = jnp.stack([obs_dict[i] for i in range(n_agents)], axis=1)
        # obs: (NUM_ENVS, n_agents, obs_dim)

        # Episode return tracking per env (summed across agents)
        ep_return = jnp.zeros(num_envs, dtype=jnp.float32)

        # ---- Outer loop: one iteration = collect + update ----
        def _update_step(runner_state, update_step):
            def _env_step(carry, unused):
                train_state, env_state, last_obs, ep_return, rng = carry

                # Batchify: (NUM_ENVS, n_agents, obs_dim) -> (num_actors, obs_dim)
                obs_batch = last_obs.reshape(num_actors, -1).astype(jnp.float32)

                # Forward pass (shared params for all agents)
                rng, action_rng = jax.random.split(rng)
                logits, value = network.apply(train_state.params, obs_batch)
                action = categorical_sample(action_rng, logits)
                log_prob = categorical_log_prob(logits, action)

                # Unbatchify actions: (num_actors,) -> (NUM_ENVS, n_agents)
                env_actions = action.reshape(num_envs, n_agents)
                env_actions_dict = {i: env_actions[:, i] for i in range(n_agents)}

                # Step all envs in parallel
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                new_obs_dict, new_state, rewards_dict, terms_dict, truncs_dict, _ = jax.vmap(
                    step_fn
                )(step_keys, env_state, env_actions_dict)
                new_obs = jnp.stack([new_obs_dict[i] for i in range(n_agents)], axis=1)
                rewards = jnp.stack([rewards_dict[i] for i in range(n_agents)], axis=1)
                terms = jnp.stack([terms_dict[i] for i in range(n_agents)], axis=1)
                truncs = jnp.stack([truncs_dict[i] for i in range(n_agents)], axis=1)

                done = terms | truncs  # (NUM_ENVS, n_agents)
                any_done = jnp.any(done, axis=-1)  # (NUM_ENVS,)

                # Episode return tracking
                new_ep_return = ep_return + rewards.sum(axis=-1)
                returned_ep_return = jnp.where(any_done, new_ep_return, 0.0)
                returned_episode = any_done.astype(jnp.float32)
                ep_return_next = jnp.where(any_done, 0.0, new_ep_return)

                # Auto-reset done envs
                rng, reset_rng = jax.random.split(rng)
                reset_obs_dict, reset_state, _ = jax.vmap(reset_fn)(
                    jax.random.split(reset_rng, num_envs)
                )
                reset_obs = jnp.stack([reset_obs_dict[i] for i in range(n_agents)], axis=1)

                def _select(reset_val, step_val):
                    shape = (num_envs,) + (1,) * (reset_val.ndim - 1)
                    return jnp.where(any_done.reshape(shape), reset_val, step_val)

                final_state = jax.tree.map(_select, reset_state, new_state)
                final_obs = _select(reset_obs, new_obs)

                transition = Transition(
                    done=done.reshape(num_actors).astype(jnp.float32),
                    action=action,
                    value=value,
                    reward=rewards.reshape(num_actors),
                    log_prob=log_prob,
                    obs=obs_batch,
                )
                carry = (
                    train_state,
                    final_state,
                    final_obs,
                    ep_return_next,
                    rng,
                )
                return carry, (transition, returned_ep_return, returned_episode)

            # Collect trajectories
            carry, (traj_batch, ep_returns, ep_dones) = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )
            train_state, env_state, last_obs, ep_return, rng = carry

            # ---- GAE ----
            last_obs_batch = last_obs.reshape(num_actors, -1).astype(jnp.float32)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # ---- PPO update epochs ----
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        logits, value = network.apply(params, traj_batch.obs)
                        log_prob = categorical_log_prob(logits, traj_batch.action)
                        entropy = categorical_entropy(logits)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - targets),
                                jnp.square(value_pred_clipped - targets),
                            ).mean()
                        )

                        # Actor loss (clipped surrogate)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae,
                        ).mean()

                        ent_coef = entropy_schedule(update_step)
                        total_loss = (
                            loss_actor + config["VF_COEF"] * value_loss - ent_coef * entropy.mean()
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy.mean(),
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch_size = num_actors * num_steps
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(
                    lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]),
                    shuffled,
                )
                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (
                    train_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ), total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metrics = {
                "returned_episode_returns": ep_returns,
                "returned_episode": ep_dones,
            }
            runner_state = (train_state, env_state, last_obs, ep_return, rng)
            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, ep_return, train_rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(num_updates))
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def visualize_policy(
    network,
    params,
    env_id,
    max_steps=1000,
    gif_path="examples/episode.gif",
    fps=30,
    seed=0,
):
    """Roll out trained policy stochastically and save as GIF."""
    import imageio

    rng = jax.random.key(seed)
    env = cogrid.make(env_id, render_mode="rgb_array", backend="jax")
    obs, info = env.reset(seed=seed)
    frames = [env.render()]

    for _ in range(max_steps):
        obs_array = jnp.stack([obs[a] for a in env.agents])  # (n_agents, obs_dim)
        logits, _ = network.apply(params, obs_array)
        rng, action_rng = jax.random.split(rng)
        actions_arr = jax.random.categorical(action_rng, logits)
        actions = {a: int(actions_arr[i]) for i, a in enumerate(env.agents)}

        obs, rewards, terms, truncs, info = env.step(actions)
        frames.append(env.render())

        if any(terms.values()) or any(truncs.values()):
            break

    env.close()

    # Stamp a 2px step-progress bar at the bottom of each frame so the GIF
    # encoder keeps every frame (GIF deduplicates identical images).
    for i, frame in enumerate(frames):
        fill = max(1, int((i / len(frames)) * frame.shape[1]))
        frame[-2:, :fill] = [60, 60, 60]

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF to {gif_path}")


def export_to_onnx(
    params,
    obs_dim,
    n_actions,
    activation="tanh",
    onnx_path="examples/policy.onnx",
):
    """Export the trained ActorCritic Flax model to ONNX format.

    Builds the ONNX graph directly from Flax parameters. Exports both the
    actor (logits) and critic (value) heads.

    Args:
        params: Flax parameter dict (the 'params' key from TrainState).
        obs_dim: Observation vector size.
        n_actions: Number of discrete actions (actor output size).
        activation: "tanh" or "relu".
        onnx_path: Output file path for the .onnx model.
    """
    import onnx
    from onnx import TensorProto, helper

    p = params["params"]

    # Build initializers (weights & biases) from Flax params
    initializers = []
    for name in [
        "Dense_0",
        "Dense_1",
        "Dense_2",
        "Dense_3",
        "Dense_4",
        "Dense_5",
    ]:
        kernel = np.array(p[name]["kernel"])
        bias = np.array(p[name]["bias"])
        initializers.append(
            helper.make_tensor(
                f"{name}_weight",
                TensorProto.FLOAT,
                kernel.shape,
                kernel.flatten().tolist(),
            )
        )
        initializers.append(
            helper.make_tensor(
                f"{name}_bias",
                TensorProto.FLOAT,
                bias.shape,
                bias.flatten().tolist(),
            )
        )

    act_type = "Tanh" if activation == "tanh" else "Relu"

    # Actor head: Dense_0 -> act -> Dense_1 -> act -> Dense_2
    actor_nodes = [
        helper.make_node(
            "Gemm",
            ["input", "Dense_0_weight", "Dense_0_bias"],
            ["actor_h0"],
            transB=0,
        ),
        helper.make_node(act_type, ["actor_h0"], ["actor_a0"]),
        helper.make_node(
            "Gemm",
            ["actor_a0", "Dense_1_weight", "Dense_1_bias"],
            ["actor_h1"],
            transB=0,
        ),
        helper.make_node(act_type, ["actor_h1"], ["actor_a1"]),
        helper.make_node(
            "Gemm",
            ["actor_a1", "Dense_2_weight", "Dense_2_bias"],
            ["logits"],
            transB=0,
        ),
    ]

    # Critic head: Dense_3 -> act -> Dense_4 -> act -> Dense_5 -> squeeze
    squeeze_axes = helper.make_tensor("squeeze_axes", TensorProto.INT64, [1], [1])
    initializers.append(squeeze_axes)

    critic_nodes = [
        helper.make_node(
            "Gemm",
            ["input", "Dense_3_weight", "Dense_3_bias"],
            ["critic_h0"],
            transB=0,
        ),
        helper.make_node(act_type, ["critic_h0"], ["critic_a0"]),
        helper.make_node(
            "Gemm",
            ["critic_a0", "Dense_4_weight", "Dense_4_bias"],
            ["critic_h1"],
            transB=0,
        ),
        helper.make_node(act_type, ["critic_h1"], ["critic_a1"]),
        helper.make_node(
            "Gemm",
            ["critic_a1", "Dense_5_weight", "Dense_5_bias"],
            ["value_2d"],
            transB=0,
        ),
        helper.make_node("Squeeze", ["value_2d", "squeeze_axes"], ["value"]),
    ]

    graph = helper.make_graph(
        actor_nodes + critic_nodes,
        "ActorCritic",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, obs_dim])],
        outputs=[
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, n_actions]),
            helper.make_tensor_value_info("value", TensorProto.FLOAT, [None]),
        ],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"Exported ONNX model to {onnx_path}")

    # Verify with onnxruntime
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    test_input = np.random.randn(1, obs_dim).astype(np.float32)
    ort_logits, ort_value = sess.run(None, {"input": test_input})
    print(f"  ONNX verification passed — logits: {ort_logits.shape}, value: {ort_value.shape}")


if __name__ == "__main__":
    config = {
        "LR": 4e-4,
        "NUM_ENVS": 32,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 500_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.25,
        "ENT_COEF_FINAL": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "SEED": 42,
    }

    # Build CoGrid env (JAX backend) to get pure step/reset functions
    env = cogrid.make(ENVIRONMENT_NAME, backend="jax")
    env.reset(seed=config["SEED"])

    # Extract pure JAX functions (already JIT-compiled)
    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = 2
    n_actions = len(env.action_set)

    # Infer obs dim from a sample observation.
    # Observation features are auto-discovered from Feature subclasses
    # registered to the scope via autowire.
    test_obs, _, _ = reset_fn(jax.random.key(0))
    obs_dim = test_obs[0].shape[-1]
    print(f"Training IPPO: {n_agents} agents, {n_actions} actions, obs_dim={obs_dim}")
    print(f"  {config['NUM_ENVS']} parallel envs, {config['TOTAL_TIMESTEPS']:.0f} total timesteps")

    train_fn = jax.jit(make_train(config, step_fn, reset_fn, n_agents, n_actions, obs_dim))

    print("Compiling...")
    out = train_fn(jax.random.key(config["SEED"]))

    # Summarize results
    ep_returns = np.array(out["metrics"]["returned_episode_returns"])
    ep_dones = np.array(out["metrics"]["returned_episode"])

    completed_returns = ep_returns[ep_dones > 0]
    total_episodes = len(completed_returns)

    print(f"\nDone! {total_episodes} episodes completed")
    tail = max(1, total_episodes // 10)
    print(f"Mean return (last {tail} episodes): {completed_returns[-tail:].mean():.2f}")

    try:
        import matplotlib.pyplot as plt

        num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        num_steps = config["NUM_STEPS"]
        num_envs = config["NUM_ENVS"]

        # ep_returns: (num_updates, num_steps, num_envs)
        # ep_dones:   (num_updates, num_steps, num_envs)
        ep_ret = np.array(ep_returns)
        ep_done = np.array(ep_dones)

        # For each episode completion, compute the env step it occurred at
        # and collect its return
        all_steps = []
        all_returns = []
        for u in range(ep_ret.shape[0]):
            for s in range(ep_ret.shape[1]):
                env_step = (u * num_steps + s + 1) * num_envs
                for e in range(ep_ret.shape[2]):
                    if ep_done[u, s, e] > 0:
                        all_steps.append(env_step)
                        all_returns.append(ep_ret[u, s, e])

        all_steps = np.array(all_steps)
        all_returns = np.array(all_returns)

        # Rolling mean over completed episodes
        window = max(1, len(all_returns) // 50)
        smoothed = np.convolve(all_returns, np.ones(window) / window, mode="valid")
        smoothed_steps = all_steps[window - 1 :]

        plt.figure(figsize=(8, 5))
        plt.plot(smoothed_steps, smoothed, linewidth=1.5)
        plt.xlabel("Environment Steps")
        plt.ylabel("Mean Episode Return")
        plt.title(f"IPPO on {ENVIRONMENT_NAME}")
        plt.tight_layout()
        plt.savefig("examples/overcooked_training.png", dpi=150)
        print("Saved learning curve to examples/overcooked_training.png")
    except ImportError:
        pass

    # Export trained policy to ONNX
    params = out["runner_state"][0].params
    export_to_onnx(params, obs_dim, n_actions, activation=config["ACTIVATION"])

    # Visualize trained policy as a GIF
    network = ActorCritic(n_actions, activation=config["ACTIVATION"])
    visualize_policy(network, params, ENVIRONMENT_NAME)
