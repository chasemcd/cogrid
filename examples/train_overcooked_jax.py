"""Train shared-parameter IPPO on CoGrid Overcooked (JAX backend).

Demonstrates CoGrid's JAX-native step/reset pipeline with:
- jax.vmap for parallel environments
- jax.lax.scan for the training loop
- Auto-reset on episode completion

Usage:
    python examples/train_overcooked_jax.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple
from flax.training.train_state import TrainState

import cogrid.envs  # noqa: F401 -- register layouts & grid objects
from cogrid.cogrid_env import CoGridEnv


# ---- Categorical distribution helpers ----


def categorical_sample(rng, logits):
    return jax.random.categorical(rng, logits)


def categorical_log_prob(logits, actions):
    log_probs = logits - jax.nn.logsumexp(logits, axis=-1, keepdims=True)
    return jnp.take_along_axis(
        log_probs, actions[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)


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
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor)

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
        step_fn: CoGrid step -- (state, actions) -> (state, obs, rewards, terms, truncs, infos).
        reset_fn: CoGrid reset -- (rng) -> (state, obs).
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
        frac = (
            1.0
            - (count // (num_minibatches * config["UPDATE_EPOCHS"])) / num_updates
        )
        return config["LR"] * frac

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
        train_state = TrainState.create(
            apply_fn=network.apply, params=params, tx=tx
        )

        # ---- Init envs (vmapped) ----
        rng, reset_rng = jax.random.split(rng)
        env_state, obs = jax.vmap(reset_fn)(
            jax.random.split(reset_rng, num_envs)
        )
        # obs: (NUM_ENVS, n_agents, obs_dim)

        # Episode return tracking per env (summed across agents)
        ep_return = jnp.zeros(num_envs, dtype=jnp.float32)

        # ---- Outer loop: one iteration = collect + update ----
        def _update_step(runner_state, unused):
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

                # Step all envs in parallel
                new_state, new_obs, rewards, terms, truncs, _ = jax.vmap(
                    step_fn
                )(env_state, env_actions)
                done = terms | truncs  # (NUM_ENVS, n_agents)
                any_done = jnp.any(done, axis=-1)  # (NUM_ENVS,)

                # Episode return tracking
                new_ep_return = ep_return + rewards.sum(axis=-1)
                returned_ep_return = jnp.where(any_done, new_ep_return, 0.0)
                returned_episode = any_done.astype(jnp.float32)
                ep_return_next = jnp.where(any_done, 0.0, new_ep_return)

                # Auto-reset done envs
                rng, reset_rng = jax.random.split(rng)
                reset_state, reset_obs = jax.vmap(reset_fn)(
                    jax.random.split(reset_rng, num_envs)
                )

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
                    delta = (
                        reward
                        + config["GAMMA"] * next_value * (1 - done)
                        - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1 - done)
                        * gae
                    )
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
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
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

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy.mean()
                        )
                        return total_loss, (value_loss, loss_actor, entropy.mean())

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                batch_size = num_actors * num_steps
                permutation = jax.random.permutation(perm_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]),
                    shuffled,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
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
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 32,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 50_000_000,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "SEED": 42,
    }

    # Build CoGrid env (JAX backend) to get pure step/reset functions
    env = CoGridEnv(
        config={
            "name": "overcooked",
            "scope": "overcooked",
            "num_agents": 2,
            "max_steps": 400,
            "action_set": "cardinal_actions",
            "features": ["agent_position"],
            "grid": {"layout": "overcooked_cramped_room_v0"},
        },
        backend="jax",
    )
    env.reset(seed=config["SEED"])

    # Extract pure JAX functions (already JIT-compiled)
    step_fn = env.jax_step
    reset_fn = env.jax_reset
    n_agents = 2
    n_actions = len(env.action_set)

    # Infer obs dim
    _, test_obs = reset_fn(jax.random.key(0))
    obs_dim = test_obs.shape[-1]
    print(f"Training IPPO: {n_agents} agents, {n_actions} actions, obs_dim={obs_dim}")
    print(f"  {config['NUM_ENVS']} parallel envs, {config['TOTAL_TIMESTEPS']:.0f} total timesteps")

    train_fn = jax.jit(
        make_train(config, step_fn, reset_fn, n_agents, n_actions, obs_dim)
    )

    print("Compiling...")
    out = train_fn(jax.random.key(config["SEED"]))

    # Summarize results
    ep_returns = out["metrics"]["returned_episode_returns"]
    ep_dones = out["metrics"]["returned_episode"]

    total_returns = (ep_returns * ep_dones).sum(axis=(1, 2))
    total_episodes = ep_dones.sum(axis=(1, 2))
    mean_return = jnp.where(total_episodes > 0, total_returns / total_episodes, 0.0)

    print(f"\nDone! {int(total_episodes.sum())} episodes completed")
    print(f"Mean return (last 10 updates): {float(mean_return[-10:].mean()):.2f}")

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
        smoothed_steps = all_steps[window - 1:]

        plt.figure(figsize=(8, 5))
        plt.plot(smoothed_steps, smoothed, linewidth=1.5)
        plt.xlabel("Environment Steps")
        plt.ylabel("Mean Episode Return")
        plt.title("IPPO on CoGrid Overcooked (cramped room)")
        plt.tight_layout()
        plt.savefig("examples/overcooked_training.png", dpi=150)
        print("Saved learning curve to examples/overcooked_training.png")
    except ImportError:
        pass
