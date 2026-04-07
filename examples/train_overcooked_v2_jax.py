"""Train shared-parameter IPPO with CNN+GRU on CoGrid Overcooked (JAX backend).

Designed for environments that use the ``local_view`` feature (channel-based
partial observability).  The network encodes the (2r+1, 2r+1, C) spatial
observation with a small CNN, feeds the embedding through a GRU for temporal
memory, then produces actor (logits) and critic (value) heads.

Validation: run with ``--env Overcooked-CrampedRoom-LocalView-V0`` to confirm
reward parity (~50) with the MLP baseline on the fully-observable CrampedRoom.

Usage:
    python examples/train_overcooked_v2_jax.py
    python examples/train_overcooked_v2_jax.py --env OvercookedV2-TestTimeSimple-V0

IMPORTANT! The RNN training loop is adapted from JaxMARL (De Witt et al.).
We've dropped in the CoGrid Overcooked environment and removed the
hydra/wandb dependencies to keep the script self-contained.
"""

import argparse
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from flax.training.train_state import TrainState

import cogrid


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


def gru_initialize_carry(batch_size, hidden_size):
    """Return zeros initial carry for a GRUCell."""
    return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class CNN(nn.Module):
    """Small CNN for (H, W, C) local-view observations.

    Architecture: three 1x1 convs for channel mixing, then up to two 3x3
    spatial convs (skipped when the spatial dimension is too small), followed
    by a flatten and dense projection.
    """

    output_size: int = 64
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh

        # 1x1 channel mixing
        x = act(
            nn.Conv(
                32,
                (1, 1),
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        x = act(
            nn.Conv(
                32,
                (1, 1),
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        x = act(
            nn.Conv(
                8,
                (1, 1),
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )

        # 3x3 spatial convs (only if spatial dims allow)
        x = act(
            nn.Conv(
                16,
                (3, 3),
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        if x.shape[-3] >= 3 and x.shape[-2] >= 3:
            x = act(
                nn.Conv(
                    32,
                    (3, 3),
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
            )

        x = x.reshape((x.shape[0], -1))
        x = act(
            nn.Dense(
                self.output_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
        )
        return x


class ActorCriticRNN(nn.Module):
    """CNN encoder -> GRU -> actor/critic heads.

    Processes a (T, B, ...) sequence: the CNN encodes each timestep, then
    the GRU runs over the time axis with episode-boundary resets.
    """

    action_dim: int
    gru_hidden_dim: int = 128
    fc_dim: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x  # obs: (T, B, H, W, C), dones: (T, B)

        act = nn.relu if self.activation == "relu" else nn.tanh
        cnn = CNN(output_size=self.gru_hidden_dim, activation=self.activation)

        # Encode each timestep's observations with the CNN (vmapped over T)
        embedding = jax.vmap(cnn)(obs)  # (T, B, gru_hidden_dim)
        embedding = nn.LayerNorm()(embedding)

        # GRU with episode-boundary resets.
        # Initialize GRUCell to capture its params, then run the scan using
        # the cell's apply method with the captured variables.
        gru_cell = nn.GRUCell(features=self.gru_hidden_dim)
        # Materialize params by calling the cell once
        dummy_carry = jnp.zeros_like(hidden)
        dummy_in = jnp.zeros_like(embedding[0])
        gru_vars = gru_cell.bind(
            {
                "params": self.param(
                    "gru",
                    lambda rng: gru_cell.init(rng, dummy_carry, dummy_in)["params"],
                )
            }
        )

        def _gru_step(carry, inputs):
            emb_t, done_t = inputs  # (B, D), (B,)
            fresh = jnp.zeros_like(carry)
            carry = jnp.where(done_t[:, jnp.newaxis], fresh, carry)
            new_carry, y = gru_vars(carry, emb_t)
            return new_carry, y

        hidden, gru_out = jax.lax.scan(
            _gru_step, hidden, (embedding, dones)
        )  # gru_out: (T, B, gru_hidden_dim)

        # Actor head
        actor = act(
            nn.Dense(self.fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(gru_out)
        )
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        # Critic head
        critic = act(
            nn.Dense(self.fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0))(gru_out)
        )
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, logits, jnp.squeeze(value, axis=-1)


# ---- Transition storage ----


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


# ---- Training ----


def make_train(
    config,
    step_fn,
    reset_fn,
    n_agents,
    n_actions,
    obs_shape,
    set_reward_coefficients_fn=None,
    initial_reward_coefficients=None,
    shaped_reward_indices=None,
):
    """Build a fully JIT-compilable CNN+RNN IPPO train function.

    Args:
        config: Hyperparameter dict.
        step_fn: CoGrid step function —
            ``(key, state, actions_dict) -> (obs, state, rew, term, trunc, info)``.
        reset_fn: CoGrid reset function —
            ``(rng) -> (obs_dict, state, infos_dict)``.
        n_agents: Number of agents per env.
        n_actions: Number of discrete actions.
        obs_shape: Spatial observation shape per agent, e.g. ``(11, 11, 12)``.
        set_reward_coefficients_fn: Callable ``(env_state, coefficients) ->
            env_state`` from ``CoGridEnv.set_reward_coefficients``.
            Required when ``SHAPED_REWARD_ANNEAL_TIMESTEPS`` is set.
        initial_reward_coefficients: The initial coefficients array from
            ``build_reward_config``.  Used as the starting point for
            the annealing schedule.
        shaped_reward_indices: List of integer indices into the coefficients
            array identifying which rewards are shaped (will be annealed
            to zero).
    """
    num_envs = config["NUM_ENVS"]
    num_actors = n_agents * num_envs
    num_steps = config["NUM_STEPS"]
    num_updates = int(config["TOTAL_TIMESTEPS"] // num_steps // num_envs)
    num_minibatches = config["NUM_MINIBATCHES"]
    gru_hidden_dim = config["GRU_HIDDEN_DIM"]

    network = ActorCriticRNN(
        action_dim=n_actions,
        gru_hidden_dim=gru_hidden_dim,
        fc_dim=config["FC_DIM"],
        activation=config["ACTIVATION"],
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (num_minibatches * config["UPDATE_EPOCHS"])) / num_updates
        return config["LR"] * frac

    def entropy_schedule(update_step):
        frac = 1.0 - update_step / num_updates
        return config["ENT_COEF_FINAL"] + (config["ENT_COEF"] - config["ENT_COEF_FINAL"]) * frac

    # --- Shaped reward annealing ---
    anneal_timesteps = config.get("SHAPED_REWARD_ANNEAL_TIMESTEPS", 0)
    if anneal_timesteps > 0 and set_reward_coefficients_fn is not None:
        anneal_updates = int(anneal_timesteps // num_steps // num_envs)
        _init_coeffs = jnp.array(initial_reward_coefficients, dtype=jnp.float32)
        _shaped_mask = jnp.zeros(len(_init_coeffs), dtype=jnp.float32)
        for idx in shaped_reward_indices:
            _shaped_mask = _shaped_mask.at[idx].set(1.0)

        def _anneal_reward_coefficients(env_state, update_step):
            """Linearly anneal shaped reward coefficients to zero."""
            frac = jnp.clip(1.0 - update_step / anneal_updates, 0.0, 1.0)
            new_coeffs = _init_coeffs * (1.0 - _shaped_mask + _shaped_mask * frac)
            return set_reward_coefficients_fn(env_state, new_coeffs)

    else:
        _anneal_reward_coefficients = None

    def train(rng):
        # ---- Init network ----
        rng, init_rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, num_envs, *obs_shape), dtype=jnp.float32)
        init_dones = jnp.zeros((1, num_envs), dtype=jnp.float32)
        init_hstate = gru_initialize_carry(num_actors, gru_hidden_dim)

        # For init we use num_envs batch dim; at runtime we'll use num_actors
        init_hstate_net = gru_initialize_carry(num_envs, gru_hidden_dim)
        params = network.init(init_rng, init_hstate_net, (init_obs, init_dones))

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
        # obs_dict[i]: (NUM_ENVS, obs_dim_flat) — reshape to spatial
        obs = jnp.stack([obs_dict[i] for i in range(n_agents)], axis=1)
        # obs: (NUM_ENVS, n_agents, obs_dim_flat)

        ep_return = jnp.zeros(num_envs, dtype=jnp.float32)

        # ---- Init RNN hidden state for all actors ----
        init_hstate = gru_initialize_carry(num_actors, gru_hidden_dim)
        init_done_batch = jnp.zeros(num_actors, dtype=jnp.float32)

        # ---- Outer loop: one iteration = collect + update ----
        def _update_step(runner_state, update_step):
            # Anneal shaped reward coefficients on the env state
            if _anneal_reward_coefficients is not None:
                runner_state = (
                    runner_state[0],
                    _anneal_reward_coefficients(runner_state[1], update_step),
                    *runner_state[2:],
                )

            def _env_step(carry, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    ep_return,
                    hstate,
                    rng,
                ) = carry

                # Batchify: (NUM_ENVS, n_agents, flat) -> (num_actors, *obs_shape)
                obs_batch = last_obs.reshape(num_actors, *obs_shape).astype(jnp.float32)

                # Forward: add time dim (T=1) for scanned RNN
                rng, action_rng = jax.random.split(rng)
                ac_in = (
                    obs_batch[jnp.newaxis, :],  # (1, num_actors, H, W, C)
                    last_done[jnp.newaxis, :],  # (1, num_actors)
                )
                hstate, logits, value = network.apply(train_state.params, hstate, ac_in)
                # Remove time dim
                logits = logits.squeeze(0)  # (num_actors, n_actions)
                value = value.squeeze(0)  # (num_actors,)

                action = categorical_sample(action_rng, logits)
                log_prob = categorical_log_prob(logits, action)

                # Unbatchify actions: (num_actors,) -> (NUM_ENVS, n_agents)
                env_actions = action.reshape(num_envs, n_agents)
                env_actions_dict = {i: env_actions[:, i] for i in range(n_agents)}

                # Step all envs in parallel
                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, num_envs)
                (
                    new_obs_dict,
                    new_state,
                    rewards_dict,
                    terms_dict,
                    truncs_dict,
                    _,
                ) = jax.vmap(step_fn)(step_keys, env_state, env_actions_dict)
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
                    if reset_val.size == 0:
                        return reset_val
                    shape = (num_envs,) + (1,) * (reset_val.ndim - 1)
                    return jnp.where(any_done.reshape(shape), reset_val, step_val)

                final_state = jax.tree.map(_select, reset_state, new_state)
                final_obs = _select(reset_obs, new_obs)

                done_batch = done.reshape(num_actors).astype(jnp.float32)

                transition = Transition(
                    done=done_batch,
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
                    done_batch,
                    ep_return_next,
                    hstate,
                    rng,
                )
                return carry, (transition, returned_ep_return, returned_episode)

            # Collect trajectories
            initial_hstate = runner_state[5]
            carry, (traj_batch, ep_returns, ep_dones) = jax.lax.scan(
                _env_step, runner_state, None, num_steps
            )
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                ep_return,
                hstate,
                rng,
            ) = carry

            # ---- GAE ----
            last_obs_batch = last_obs.reshape(num_actors, *obs_shape).astype(jnp.float32)
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

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
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        _, logits, value = network.apply(
                            params,
                            init_hstate.squeeze(0),
                            (traj_batch.obs, traj_batch.done),
                        )
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
                    total_loss, grads = grad_fn(
                        train_state.params,
                        init_hstate,
                        traj_batch,
                        advantages,
                        targets,
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, perm_rng = jax.random.split(rng)

                # init_hstate: (num_actors, gru_hidden_dim) -> (1, num_actors, gru_hidden_dim)
                init_hstate_t = jnp.reshape(init_hstate, (1, num_actors, -1))
                batch = (init_hstate_t, traj_batch, advantages, targets)

                # Shuffle along actor dimension and split into minibatches
                permutation = jax.random.permutation(perm_rng, num_actors)
                shuffled = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], num_minibatches, -1] + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled,
                )

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ), total_loss

            update_state = (
                train_state,
                initial_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metrics = {
                "returned_episode_returns": ep_returns,
                "returned_episode": ep_dones,
            }
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                ep_return,
                hstate,
                rng,
            )
            return runner_state, metrics

        rng, train_rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obs,
            init_done_batch,
            ep_return,
            init_hstate,
            train_rng,
        )
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(num_updates))
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def visualize_policy(
    network,
    params,
    env_id,
    obs_shape,
    gru_hidden_dim=64,
    max_steps=1000,
    gif_path="examples/episode_v2.gif",
    fps=30,
    seed=0,
):
    """Roll out trained policy stochastically and save as GIF."""
    import imageio

    rng = jax.random.key(seed)
    env = cogrid.make(env_id, render_mode="rgb_array", backend="jax")
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    n_agents = len(env.agents)

    hstate = gru_initialize_carry(n_agents, gru_hidden_dim)

    done_arr = jnp.zeros(n_agents, dtype=jnp.float32)

    for _ in range(max_steps):
        obs_array = jnp.stack([obs[a] for a in env.agents])  # (n_agents, flat)
        obs_spatial = obs_array.reshape(n_agents, *obs_shape).astype(jnp.float32)

        ac_in = (
            obs_spatial[jnp.newaxis, :],  # (1, n_agents, H, W, C)
            done_arr[jnp.newaxis, :],  # (1, n_agents)
        )

        hstate, logits, _ = network.apply(params, hstate, ac_in)
        logits = logits.squeeze(0)

        rng, action_rng = jax.random.split(rng)
        actions_arr = jax.random.categorical(action_rng, logits)
        actions = {a: int(actions_arr[i]) for i, a in enumerate(env.agents)}

        obs, rewards, terms, truncs, info = env.step(actions)
        frames.append(env.render())

        if any(terms.values()) or any(truncs.values()):
            break

        done_arr = jnp.zeros(n_agents, dtype=jnp.float32)

    env.close()

    for i, frame in enumerate(frames):
        fill = max(1, int((i / len(frames)) * frame.shape[1]))
        frame[-2:, :fill] = [60, 60, 60]

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF to {gif_path}")


def save_checkpoint(params, config, env_id, obs_shape, path=None):
    """Save trained parameters and metadata to a checkpoint file."""
    from flax.serialization import to_bytes

    if path is None:
        os.makedirs("examples/checkpoints", exist_ok=True)
        path = f"examples/checkpoints/{env_id}.params"
    with open(path, "wb") as f:
        f.write(to_bytes(params))

    # Save metadata alongside for loading
    import json

    meta = {
        "env_id": env_id,
        "obs_shape": list(obs_shape),
        "gru_hidden_dim": config["GRU_HIDDEN_DIM"],
        "fc_dim": config["FC_DIM"],
        "activation": config["ACTIVATION"],
    }
    meta_path = path.replace(".params", ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(path, env_id):
    """Load parameters from a checkpoint, returning (params, metadata)."""
    import json

    from flax.serialization import from_bytes

    meta_path = path.replace(".params", ".json")
    with open(meta_path) as f:
        meta = json.load(f)

    obs_shape = tuple(meta["obs_shape"])
    gru_hidden_dim = meta["gru_hidden_dim"]
    fc_dim = meta["fc_dim"]
    activation = meta["activation"]

    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=0)
    n_agents = len(env.agents)
    n_actions = len(env.action_set)

    network = ActorCriticRNN(
        action_dim=n_actions,
        gru_hidden_dim=gru_hidden_dim,
        fc_dim=fc_dim,
        activation=activation,
    )
    init_hstate = gru_initialize_carry(n_agents, gru_hidden_dim)
    init_obs = jnp.zeros((1, n_agents, *obs_shape), dtype=jnp.float32)
    init_dones = jnp.zeros((1, n_agents), dtype=jnp.float32)
    template = network.init(jax.random.key(0), init_hstate, (init_obs, init_dones))

    with open(path, "rb") as f:
        params = from_bytes(template, f.read())

    return params, meta, network, obs_shape


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN+RNN IPPO on CoGrid Overcooked")
    p.add_argument(
        "--env",
        default="Overcooked-CrampedRoom-LocalView-V0",
        help="Environment ID (must use local_view features)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total-timesteps", type=int, default=30_000_000)
    p.add_argument("--no-gif", action="store_true", help="Skip GIF generation")
    p.add_argument(
        "--shaped-reward-anneal-steps",
        type=int,
        default=0,
        help="Linearly anneal shaped reward coefficients to 0 over this many "
        "timesteps (0 = no annealing)",
    )
    p.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to .params file — skip training and generate GIF only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env_id = args.env

    # ------------------------------------------------------------------
    # Load from checkpoint — skip training, just generate GIF
    # ------------------------------------------------------------------
    if args.load_checkpoint:
        params, meta, vis_network, obs_shape = load_checkpoint(args.load_checkpoint, env_id)
        gif_path = f"examples/{env_id.lower().replace('-', '_')}_episode.gif"
        visualize_policy(
            vis_network,
            params,
            env_id,
            obs_shape,
            gru_hidden_dim=meta["gru_hidden_dim"],
            gif_path=gif_path,
        )
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    config = {
        "LR": 0.00025,
        "NUM_ENVS": 256,
        "NUM_STEPS": 256,
        "TOTAL_TIMESTEPS": args.total_timesteps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.25,
        "ENT_COEF_FINAL": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.25,
        "ACTIVATION": "relu",
        "ANNEAL_LR": True,
        "GRU_HIDDEN_DIM": 128,
        "FC_DIM": 128,
        "SEED": args.seed,
        "SHAPED_REWARD_ANNEAL_TIMESTEPS": args.total_timesteps // 2,
    }

    # Build CoGrid env (JAX backend) to get pure step/reset functions
    env = cogrid.make(env_id, backend="jax")
    env.reset(seed=config["SEED"])

    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = len(env.agents)
    n_actions = len(env.action_set)

    # Infer observation shape from a sample.
    # LocalView returns flat obs — recover spatial shape from env config.
    test_obs, _, _ = reset_fn(jax.random.key(0))
    obs_dim_flat = test_obs[0].shape[-1]

    radius = env.config.get("local_view_radius")
    if radius is not None:
        h = w = 2 * radius + 1
    else:
        h = env.config["grid_height"]
        w = env.config["grid_width"]
    n_channels = obs_dim_flat // (h * w)
    obs_shape = (h, w, n_channels)

    assert h * w * n_channels == obs_dim_flat, (
        f"obs_dim={obs_dim_flat} doesn't factor into ({h}, {w}, C). "
        f"This script requires local_view observations."
    )

    print(f"Training CNN+RNN IPPO on {env_id}")
    print(
        f"  {n_agents} agents, {n_actions} actions, "
        f"obs_shape={obs_shape} (local_view_radius={radius})"
    )
    print(f"  {config['NUM_ENVS']} parallel envs, {config['TOTAL_TIMESTEPS']:.0f} total timesteps")

    # Identify shaped reward indices for annealing
    from cogrid.envs.overcooked.rewards import (
        TargetRecipeIngredientInPotReward,
        TargetRecipeSoupInDishReward,
    )

    _shaped_types = (
        TargetRecipeIngredientInPotReward,
        TargetRecipeSoupInDishReward,
    )
    reward_instances = env.config.get("rewards", [])
    shaped_reward_indices = [
        inst._reward_index for inst in reward_instances if isinstance(inst, _shaped_types)
    ]
    initial_coefficients = env._reward_config["initial_reward_coefficients"]

    if config["SHAPED_REWARD_ANNEAL_TIMESTEPS"] > 0:
        print(
            f"  Annealing {len(shaped_reward_indices)} shaped rewards "
            f"over {config['SHAPED_REWARD_ANNEAL_TIMESTEPS']:,} timesteps"
        )

    train_fn = jax.jit(
        make_train(
            config,
            step_fn,
            reset_fn,
            n_agents,
            n_actions,
            obs_shape,
            set_reward_coefficients_fn=env.set_reward_coefficients,
            initial_reward_coefficients=initial_coefficients,
            shaped_reward_indices=shaped_reward_indices,
        )
    )

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

    # Save checkpoint
    params = out["runner_state"][0].params
    save_checkpoint(params, config, env_id, obs_shape)

    try:
        import matplotlib.pyplot as plt

        num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        num_steps = config["NUM_STEPS"]
        num_envs = config["NUM_ENVS"]

        ep_ret = np.array(ep_returns)
        ep_done = np.array(ep_dones)

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

        window_size = max(1, len(all_returns) // 50)
        smoothed = np.convolve(all_returns, np.ones(window_size) / window_size, mode="valid")
        smoothed_steps = all_steps[window_size - 1 :]

        plt.figure(figsize=(8, 5))
        plt.plot(smoothed_steps, smoothed, linewidth=1.5)
        plt.xlabel("Environment Steps")
        plt.ylabel("Mean Episode Return")
        plt.title(f"CNN+RNN IPPO on {env_id}")
        plt.tight_layout()
        plot_path = f"examples/{env_id.lower().replace('-', '_')}_training.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved learning curve to {plot_path}")
    except ImportError:
        pass

    # Visualize trained policy as a GIF
    if not args.no_gif:
        vis_network = ActorCriticRNN(
            action_dim=n_actions,
            gru_hidden_dim=config["GRU_HIDDEN_DIM"],
            fc_dim=config["FC_DIM"],
            activation=config["ACTIVATION"],
        )
        gif_path = f"examples/{env_id.lower().replace('-', '_')}_episode.gif"
        visualize_policy(
            vis_network,
            params,
            env_id,
            obs_shape,
            gru_hidden_dim=config["GRU_HIDDEN_DIM"],
            gif_path=gif_path,
        )
