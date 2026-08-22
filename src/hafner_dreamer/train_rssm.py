"""Train the Dreamer RSSM jointly on CarRacing-v3.

Workflow
--------
1. Collect *num_collect_episodes* episodes with a **random policy**,
   repeating each sampled action for *action_repeat* environment steps
   (rewards are summed over the repeat).
2. Push all episodes into a ``SequenceReplayBuffer``.
3. Overfit the world model on this fixed buffer for *train_steps* gradient
   steps, sampling a fresh mini-batch each step.

Loss terms
----------
* reconstruction  – pixel MSE between decoder output and input frame
* kl              – one-sample MC KL:  log q(z|h,o) − log p(z|h)
* reward          – MSE on the reward head
* continue        – binary cross-entropy on the continue head (logit form)
"""

from __future__ import annotations

import os

# The trainer and the inference-server/self-play processes are separate JAX
# processes sharing one GPU, so none may greedily preallocate the whole
# device. Must be set before jax is imported anywhere in this process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import multiprocessing as mp
import sys
from pathlib import Path

# Expose the models package (local imports: encoder_decoder, rssm, …) and the
# shared inference-server module (sibling package under src/).
sys.path.insert(0, str(Path(__file__).parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "multithreading"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
import orbax.checkpoint as ocp
from jaxtyping import Float
from flax import nnx
from tqdm import tqdm

from replay_buffer import SequenceReplayBuffer
from rssm import RSSM
from controller import ActionValue, Critic, EncoderPolicy, Policy, Temperature, build_encoder_policy
from inference_server import extract_weights, run_inference_server
from self_play_worker import run_worker


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Numerically stable sigmoid binary cross-entropy."""
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def rssm_loss(
    rssm: RSSM,
    images: jax.Array,          # (B, T, 64, 64, C) float32 [0, 1]
    actions: jax.Array,         # (B, T, action_dim)
    rewards: jax.Array,         # (B, T)
    continues: jax.Array,       # (B, T)
    kl_weight: jax.Array,       # scalar
    reward_weight: jax.Array,
    continue_weight: jax.Array,
    free_bits: jax.Array,       # scalar nats; KL below this per-step is ignored
) -> tuple[jax.Array, dict[str, jax.Array]]:
    _, out = rssm.teacher_forcing_forward(images, actions, initial_carry=None, return_carry=True)

    # images are frame-stacked (image_channels * frame_stack channels) and
    # the decoder reconstructs the whole stack, so the target is just images
    # itself — no slicing needed.
    # Reconstruction: compare decoder output (all T frames) against all images.
    recon_loss = jnp.mean((out["reconstruction"] - images) ** 2)

    z_sg = jax.lax.stop_gradient(out["post_stoch"])  # (B, T-1, D)
    log_post_dist  = RSSM.log_prob(z_sg, out["post_pi_logits"],  out["post_mu"],  out["post_log_var"])
    log_prior_dist = RSSM.log_prob(z_sg, out["prior_pi_logits"], out["prior_mu"], out["prior_log_var"])
    kl_per_step = jax.lax.stop_gradient(log_post_dist) - log_prior_dist  
    # The 20% term regularises the encoder to stay in a space the prior can
    # reach, providing a gradient signal that pure reconstruction alone lacks.
    # z      = out["post_stoch"]                     # (B, T-1, D) reparameterised
    # z_sg   = jax.lax.stop_gradient(z)

    # log_q      = RSSM.log_prob(z,    out["post_pi_logits"],  out["post_mu"],  out["post_log_var"])
    # log_q_sg   = jax.lax.stop_gradient(log_q)
    # log_p      = RSSM.log_prob(z_sg, out["prior_pi_logits"], out["prior_mu"], out["prior_log_var"])
    # log_p_z    = RSSM.log_prob(z,    out["prior_pi_logits"], out["prior_mu"], out["prior_log_var"])
    # log_p_z_sg = jax.lax.stop_gradient(log_p_z)

    # kl_prior   = log_q_sg - log_p              # gradient → prior only
    # kl_post    = log_q    - log_p_z_sg         # gradient → encoder only (via z)
    # kl_per_step = 0.8 * kl_prior + 0.2 * kl_post
    kl_loss = jnp.mean(jnp.maximum(kl_per_step, free_bits))

    # Reward/continue: actions[:, :-1] drove the T-1 imagination steps,
    # so the predicted reward at step t corresponds to rewards[:, t+1].
    reward_loss   = jnp.mean((out["reward_logit"][..., 0]   - rewards[:, 1:])   ** 2)
    continue_loss = jnp.mean(_bce_with_logits(out["continue_logit"][..., 0], continues[:, 1:]))

    loss = recon_loss + kl_weight * kl_loss  + reward_weight * reward_loss + continue_weight * continue_loss
    return loss, {
        "loss":          loss,
        "recon_loss":    recon_loss,
        "kl_loss":       kl_loss,
        "reward_loss":   reward_loss,
        "continue_loss": continue_loss,
        # debug: track individual log-probs to diagnose divergence
        "log_post_dist":    jnp.mean(log_post_dist),
        "log_prior_dist":    jnp.mean(log_prior_dist),
    }

def policy_loss(
    rssm: RSSM,
    policy: Policy,
    # Starting latent states sampled from a teacher-forcing rollout.
    # Shape (B, features_dim) = concat([deter, post_stoch]) at some real step.
    start_features: jax.Array,
    horizon: jax.Array,    # imagination horizon (number of steps)
    gamma: jax.Array,      # discount factor
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """REINFORCE loss computed entirely in imagination (latent space).

    Algorithm
    ---------
    For each starting (h, z) in the batch:
      1. For t = 0 … H-1:
           a_t   ~ π(· | h_t, z_t)              # sample policy
           h_{t+1}, z_{t+1} = RSSM.prior_step(h_t, z_t, a_t)
           r_t, c_t = reward_head(h_t, z_t)     # imagined reward & continue
      2. Compute discounted returns:
           R_t = Σ_{k=t}^{H-1}  γ^{k-t} · r_k · Π_{j=t}^{k-1} c_j
      3. Baseline: subtract mean return to reduce variance.
      4. Loss: -mean_t [ R_t · log π(a_t | h_t, z_t) ]
    """
    B = start_features.shape[0]
    mem_d = rssm.memory_dim
    # Split starting features back into (h, z).
    h0 = start_features[:, :mem_d]          # (B, memory_dim)
    z0 = start_features[:, mem_d:]          # (B, stoch_dim)

    # Initialise LSTM carry from h0: both cell and hidden start at h0.
    initial_carry = (h0, h0)

    def scan_step(carry, rng_t):
        lstm_carry, prev_z = carry

        # Stop gradient through the world-model state so RSSM parameters are
        # not updated during the policy step — the world model is a frozen
        # simulator here; only policy weights should receive gradient.
        features_t = jax.lax.stop_gradient(
            jnp.concatenate([lstm_carry[0], prev_z], axis=-1)
        )

        # Split key: half for policy sample, half for prior sample.
        rng_pol, rng_prior = jax.random.split(rng_t)
        action_t, log_pi_t = policy.sample(features_t, rng_key=rng_pol)
        _, std_t = policy._mu_and_std(features_t)  # for diagnostics only

        # Step the deterministic RNN.
        rnn_input = jnp.concatenate([prev_z, action_t], axis=-1)
        lstm_carry, deter_t = rssm.cell(lstm_carry, rnn_input)

        # Sample next z from prior.
        z_t, _, _, _ = rssm.prior(deter_t, rng_key=rng_prior)

        # Imagined reward and continue (from current features_t).
        reward_t, continue_logit_t = rssm.reward_continue(features_t)
        r_t = reward_t[..., 0]                          # (B,)
        c_t = jax.nn.sigmoid(continue_logit_t[..., 0])  # (B,) ∈ (0, 1)

        new_carry = (lstm_carry, z_t)
        return new_carry, (r_t, c_t, log_pi_t, std_t)

    step_rngs = jax.random.split(rssm.rngs.noise(), horizon)
    _, (rewards_h, continues_h, log_pis_h, stds_h) = jax.lax.scan(
        scan_step, (initial_carry, z0), step_rngs
    )
    # rewards_h, continues_h, log_pis_h: (H, B)

    # --- discounted returns ---
    # R_t = r_t + γ·c_t·r_{t+1} + γ²·c_t·c_{t+1}·r_{t+2} + …
    # Computed backwards in a scan.
    def return_step(future_return, rc_t):
        r_t, c_t = rc_t
        G_t = r_t + gamma * c_t * future_return
        return G_t, G_t

    _, returns_h = jax.lax.scan(
        return_step,
        jnp.zeros(B),
        (rewards_h, continues_h),
        reverse=True,
    )
    # returns_h: (H, B)

    # Treat returns as a constant weight — REINFORCE does not differentiate
    # through R_t (it is the "environment signal", not a network output).
    returns_h = jax.lax.stop_gradient(returns_h.T)

    # Normalise returns (baseline = mean over time × batch).
    returns_h = (returns_h - returns_h.mean()) #/ (returns_h.std() + 1e-8)

    # REINFORCE: maximise E[R · log π]  →  minimise -E[R · log π].
    loss = -jnp.mean(returns_h * log_pis_h)

    return loss, {
        "policy_loss":      loss,
        "imagined_return":  jnp.mean(returns_h * (returns_h.std() + 1e-8) + returns_h.mean()),
        "mean_reward":      jnp.mean(rewards_h),
        "mean_log_pi":      jnp.mean(log_pis_h),
        "policy_std":       jnp.mean(stds_h),
    }


def _l2_penalty(module: nnx.Module) -> jax.Array:
    """Sum of squared parameters — plain (coupled) L2 weight decay term."""
    leaves = jax.tree.leaves(nnx.state(module, nnx.Param))
    return sum(jnp.sum(jnp.square(p)) for p in leaves)


def q_loss_real(
    critic: Critic,
    critic_target: Critic,
    policy: Policy,
    temperature: Temperature,
    features: Float[jax.Array, "batch seq stoch_dim"],   # RSSM encoder output (frozen)
    actions: Float[jax.Array, "batch seq action_dim"],    # raw actions from replay buffer
    rewards: Float[jax.Array, "batch seq"],
    continues: Float[jax.Array, "batch seq"],
    gamma: Float[jax.Array, ()],
    weight_decay: Float[jax.Array, ()],
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """SAC critic (soft Bellman residual) loss on one-step real transitions.

    ``collect_episode`` stores frame ``t``, the raw action sampled at ``t``,
    the reward earned and whether the *next* frame is terminal — so
    ``(features[t], actions[t], rewards[t], continues[t], features[t+1])``
    is exactly a one-step transition; no imagined rollout is involved.

    Target: y = r + γ·c·(min(Q1', Q2')(s', a') − α·log π(a'|s'))
    with a' ~ π(·|s') (current policy, target *critics*).
    """
    s      = features[:, :-1]
    s_next = features[:, 1:]
    a      = actions[:, :-1]          # already squashed/bounded (from the buffer)
    r      = rewards[:, :-1]
    c      = continues[:, :-1]

    # policy.sample already returns a bounded (tanh-squashed) action.
    next_action, next_log_pi = policy.sample(s_next, rng_key=rng_key)
    q1_next, q2_next = critic_target(s_next, next_action)

    # For a tanh-squashed Gaussian, log π is UNBOUNDED ABOVE: as the policy
    # learns decisive, near-boundary actions (e.g. flooring the gas once it
    # can drive well), the tanh change-of-variables correction pushes log π
    # large-positive, so the soft-target term −α·log π flips from an entropy
    # *bonus* into a large entropy *penalty* that drags the bootstrapped
    # target down. With γ≈0.99 this compounds and collapses Q — which is why
    # reward peaks then craters and the actor loss (−q_min) starts growing.
    # Clip only the entropy term *in the target* so a transient low-entropy
    # spike can't sink the critic. The actor and temperature still see the
    # true (unclipped) log π, so their entropy-restoring signal is intact.
    q_next = jnp.minimum(q1_next, q2_next) - temperature.value * jnp.clip(next_log_pi, -10.0, 10.0)
    target = jax.lax.stop_gradient(r + gamma * c * q_next)

    q1_pred, q2_pred = critic(s, a)
    bellman_loss = jnp.mean((q1_pred - target) ** 2) + jnp.mean((q2_pred - target) ** 2)

    # L2 weight decay on the critic's own params only — keeps Q from
    # inflating its weights to fit noisy/out-of-distribution bootstrap
    # targets, a cheap extra guard against the Q-divergence seen previously.
    l2 = _l2_penalty(critic)
    loss = bellman_loss + weight_decay * l2

    return loss, {
        "q_loss":        loss,
        "q_bellman_loss": bellman_loss,
        "q_l2":          l2,
        "q1_mean":     jnp.mean(q1_pred),
        "q2_mean":     jnp.mean(q2_pred),
        "target_mean": jnp.mean(target),
        "reward_mean": jnp.mean(r),
        "next_log_pi": jnp.mean(next_log_pi),
    }


def policy_loss_real(
    policy: Policy,
    critic: Critic,
    temperature: Temperature,
    features: Float[jax.Array, "batch stoch_dim"],
    rng_key: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """SAC actor loss trained directly on real (buffer) states — off-policy.

    Actions are sampled with the reparameterisation trick so gradients flow
    a ~ π(s) → Q(s, a) → policy params. ``critic`` is only ever passed as a
    non-differentiated argument (see ``argnums`` in ``train_step_sac_real``),
    so its own parameters never receive gradient here — only the policy does.
    """
    # policy.sample already returns a bounded (tanh-squashed) action.
    # Reparameterised sample: gradients must flow a ~ π(s) → Q(s, a) → policy
    # params. Do NOT stop_gradient q_min — that would erase the value signal
    # and leave the actor maximising entropy only (policy_std saturates and
    # mu parks at the centre → slow, indecisive driving). The critic's own
    # params are already protected: train_step_sac_real differentiates this
    # loss w.r.t. the policy only (argnums=0), so critic weights never update
    # here regardless.
    action, log_pi = policy.sample(features, rng_key=rng_key)
    q1, q2 = critic(features, action)
    q_min = jnp.minimum(q1, q2)

    alpha = temperature.value
    loss = jnp.mean(alpha * log_pi - q_min)

    _, std = policy._mu_and_std(features)  # for diagnostics only
    return loss, {
        "policy_loss": loss,
        "mean_log_pi": jnp.mean(log_pi),
        "mean_q":      jnp.mean(q_min),
        "policy_std":  jnp.mean(std),
        "alpha":       alpha,
        "log_pi":      log_pi,  # consumed by temperature_loss_real, stripped before logging
    }


def temperature_loss_real(
    temperature: Temperature,
    log_pi: jax.Array,
    target_entropy: Float[jax.Array, ()],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Automatic entropy-coefficient tuning (Haarnoja et al., 2018).

    ``log_pi`` comes from the actor step and is treated as a constant here —
    log_alpha is pushed up when entropy is below target, down when above.
    """
    loss = -jnp.mean(temperature.log_alpha[...] * jax.lax.stop_gradient(log_pi + target_entropy))
    return loss, {"temperature_loss": loss, "alpha": temperature.value}


def soft_update(target: nnx.Module, source: nnx.Module, tau: jax.Array) -> None:
    """Polyak-average ``target``'s params towards ``source``: target ← (1-τ)·target + τ·source."""
    target_state = nnx.state(target, nnx.Param)
    source_state = nnx.state(source, nnx.Param)
    new_state = jax.tree.map(lambda t, s: (1.0 - tau) * t + tau * s, target_state, source_state)
    nnx.update(target, new_state)


def _global_grad_norm(grads) -> jax.Array:
    """Global L2 norm of a gradient pytree (matches optax.clip_by_global_norm).

    Useful as an early-warning signal: a gradient norm that trends up without
    bound is the fingerprint of the divergence (KL / mean-Q blow-up) we're
    chasing here. Computed inside the jitted train step and returned as a
    metric so it lands in the same MLflow log as the losses.
    """
    leaves = jax.tree.leaves(grads)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


@nnx.jit
def train_step_sac_real(
    rssm: RSSM,
    policy: Policy,
    critic: Critic,
    critic_target: Critic,
    temperature: Temperature,
    policy_optimizer: nnx.Optimizer,
    critic_optimizer: nnx.Optimizer,
    temp_optimizer: nnx.Optimizer,
    images: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
    gamma: jax.Array,
    tau: jax.Array,
    target_entropy: jax.Array,
    q_weight_decay: jax.Array,
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    """One off-policy SAC update (critic → actor → temperature → target sync)."""
    z, _, _, _ = rssm.encoder(images)  # (B, T, stoch_dim)
    z = jax.lax.stop_gradient(z)       # world model is a frozen feature extractor here

    rng_critic, rng_actor = jax.random.split(rng_key)

    # --- critic (twin Q) update ------------------------------------------
    q_grad_fn = nnx.value_and_grad(q_loss_real, argnums=0, has_aux=True)
    (_, q_metrics), q_grads = q_grad_fn(
        critic, critic_target, policy, temperature,
        z, actions, rewards, continues, gamma, q_weight_decay, rng_critic,
    )
    q_metrics["q_grad_norm"] = _global_grad_norm(q_grads)
    critic_optimizer.update(critic, q_grads)

    # --- actor update ------------------------------------------------------
    s = z[:, :-1]  # states with a valid "next" transition, same as critic loss
    pi_grad_fn = nnx.value_and_grad(policy_loss_real, argnums=0, has_aux=True)
    (_, pi_metrics), pi_grads = pi_grad_fn(policy, critic, temperature, s, rng_actor)
    pi_metrics["policy_grad_norm"] = _global_grad_norm(pi_grads)
    policy_optimizer.update(policy, pi_grads)

    # --- temperature (entropy coefficient) update --------------------------
    temp_grad_fn = nnx.value_and_grad(temperature_loss_real, argnums=0, has_aux=True)
    (_, temp_metrics), temp_grads = temp_grad_fn(temperature, pi_metrics["log_pi"], target_entropy)
    temp_metrics["temp_grad_norm"] = _global_grad_norm(temp_grads)
    temp_optimizer.update(temperature, temp_grads)

    # --- target critic Polyak update ---------------------------------------
    soft_update(critic_target, critic, tau)

    metrics = {**q_metrics, **temp_metrics}
    metrics.update({k: v for k, v in pi_metrics.items() if k != "log_pi"})
    return metrics

@nnx.jit
def train_step_world_model(
    rssm: RSSM,
    optimizer: nnx.Optimizer,
    images: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
    kl_weight: jax.Array,
    free_bits: jax.Array,
) -> dict[str, jax.Array]:
    _ones = jnp.ones(())
    grad_fn = nnx.value_and_grad(rssm_loss, has_aux=True)
    (_, metrics), grads = grad_fn(rssm, images, actions, rewards, continues,
                                  kl_weight, _ones, _ones, free_bits)
    metrics["grad_norm"] = _global_grad_norm(grads)
    optimizer.update(rssm, grads)
    return metrics


@nnx.jit
def eval_step_world_model(
    rssm: RSSM,
    images: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    continues: jax.Array,
    kl_weight: jax.Array,
    free_bits: jax.Array,
) -> dict[str, jax.Array]:
    _ones = jnp.ones(())
    _, metrics = rssm_loss(rssm, images, actions, rewards, continues,
                           kl_weight, _ones, _ones, free_bits)
    return metrics


@nnx.jit
def train_step_policy(
    rssm: RSSM,
    policy: Policy,
    optimizer: nnx.Optimizer,
    start_features: jax.Array,  # (B, features_dim) concat([h, z])
    horizon: jax.Array,
    gamma: jax.Array,
) -> dict[str, jax.Array]:
    grad_fn = nnx.value_and_grad(policy_loss, has_aux=True)
    (_, metrics), grads = grad_fn(rssm, policy, start_features, horizon, gamma)
    optimizer.update(policy, grads)
    return metrics


# ---------------------------------------------------------------------------
# Reconstruction visualisation
# ---------------------------------------------------------------------------

def save_reconstruction_grid(
    inputs: np.ndarray,          # (B, T,   H, W, C) float32
    reconstructions: np.ndarray, # (B, T,   H, W, C)
    prior_recons: np.ndarray,    # (B, T-1, H, W, C)
    path: Path,
    *,
    num_items: int = 4,
    title: str | None = None,
    image_channels: int | None = None,
) -> None:
    """Save a 3-row grid per batch item: real | posterior recon | prior imagination."""
    B = min(inputs.shape[0], num_items)
    T = inputs.shape[1]
    # Pick frames from the T-1 range so all three rows have valid data.
    t_indices = [0, (T - 1) // 2, T - 2]

    # All three tensors may be frame-stacked (image_channels * frame_stack
    # channels) — imshow only understands 1/3/4 channels, so display just
    # the current-frame slice (the last image_channels channels) of each.
    display_channels = image_channels or inputs.shape[-1]
    real_inputs      = inputs[..., -display_channels:]
    reconstructions  = reconstructions[..., -display_channels:]
    prior_recons     = prior_recons[..., -display_channels:]

    num_cols = len(t_indices)
    row_labels = ["real", "posterior", "prior"]
    fig, axes = plt.subplots(
        B * 3, num_cols, figsize=(num_cols * 2, B * 6), squeeze=False
    )
    for b in range(B):
        for col, t in enumerate(t_indices):
            rows = [
                real_inputs[b, t],
                reconstructions[b, t],
                prior_recons[b, t],      # prior_recons is T-1, t < T-1 always
            ]
            for row_offset, (img, label) in enumerate(zip(rows, row_labels)):
                ax = axes[b * 3 + row_offset, col]
                ax.imshow(np.clip(img, 0, 1))
                ax.axis("off")
                if b == 0:
                    ax.set_title(f"{label} t={t}", fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def log_reconstruction_images(rssm: RSSM, batch: dict, step: int | str) -> None:
    images = batch["images"]
    actions = batch["actions"]
    _, out = rssm.teacher_forcing_forward(images, actions, initial_carry=None, return_carry=True)
    recon       = np.asarray(out["reconstruction"])
    prior_recon = np.asarray(out["prior_reconstruction"])
    step_label = f"{step:06d}" if isinstance(step, int) else step
    grid_path = Path(f"data/hafner_dreamer/recon_grid_step_{step_label}.png")
    save_reconstruction_grid(
        np.asarray(images), recon, prior_recon, grid_path,
        title=f"step {step_label}",
        image_channels=rssm.image_channels,
    )
    mlflow.log_artifact(str(grid_path), artifact_path="recon_images")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def make_checkpoint_manager(
    directory: str | Path, *, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(directory, options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager, models: dict[str, nnx.Module], step: int
) -> None:
    """Save every model in *models* (e.g. rssm/policy/critic/...) as one
    checkpoint step, each under its own named item via ``ocp.args.Composite``
    so a single ``manager.save``/``restore`` call covers the whole agent."""
    manager.save(
        step,
        args=ocp.args.Composite(
            **{name: ocp.args.StandardSave(nnx.state(m)) for name, m in models.items()}
        ),
    )


def load_checkpoint(models: dict[str, nnx.Module], directory: str | Path) -> int:
    """Restore the latest checkpoint from *directory* into every model in
    *models* (matched by name) in-place.

    Returns the restored step number, or 0 if no checkpoint was found.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        print(f"[checkpoint] directory {directory} not found — starting from scratch")
        return 0
    manager = ocp.CheckpointManager(
        directory, options=ocp.CheckpointManagerOptions()
    )
    step = manager.latest_step()
    if step is None:
        print(f"[checkpoint] no checkpoint in {directory} — starting from scratch")
        manager.close()
        return 0
    restore_args = ocp.args.Composite(
        **{name: ocp.args.StandardRestore(nnx.state(m)) for name, m in models.items()}
    )
    restored = manager.restore(step, args=restore_args)
    for name, m in models.items():
        nnx.update(m, restored[name])
    manager.close()
    print(f"[checkpoint] loaded step {step} from {directory} ({', '.join(models)})")
    return step


def metrics_to_floats(metrics: dict) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        f = float(v)
        if not (f == f):  # NaN check
            print(f"[warn] metric '{k}' is NaN — skipping MLflow log")
        elif abs(f) == float("inf"):
            print(f"[warn] metric '{k}' is Inf — skipping MLflow log")
        else:
            out[k] = f
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    config = dict(
        # env
        action_repeat=3,
        max_episode_steps=500,
        buffer_capacity=1000,       # max episodes stored
        # self-play workers (each owns its own env + calls the inference server)
        num_workers=16,
        episodes_per_worker=3,     # collect this many episodes *per worker*, then train
        max_inference_batch=16,    # inference-server batch size (>= num_workers)
        # collect / train interleave
        train_per_collect=200,     # world-model gradient steps per collect phase
        policy_steps_per_collect=50,  # policy gradient steps per collect phase
        num_cycles=200,             # total collect→train cycles
        # model
        image_channels=3,
        frame_stack=3,             # encoder sees the last N frames stacked as channels
        action_dim=3,              # CarRacing-v3 continuous: [steering, gas, brake]
        memory_dim=256,
        stoch_dim=256,
        num_gaussian_components=8,
        predictor_hidden_dim=256,
        # policy (SAC)
        policy_hidden_dim=256,
        policy_lr=1e-4,
        q_hidden_dim=256,
        q_lr=3e-4,
        q_weight_decay=1e-4,   # L2 penalty on critic params, added directly to q_loss_real
        alpha_lr=1e-3,
        init_alpha=1./5.,             # initial entropy coefficient (auto-tuned afterwards)
        # SAC entropy target (desired -E[log π]). The textbook heuristic
        # -action_dim (=-3) asks for a *near-deterministic* policy; but with a
        # tanh-squashed action "low entropy" is reached by pushing mass into
        # the flat tanh tails, i.e. actions pinned at the box boundaries
        # (full-lock steering ⇒ the car spins in a circle). Empirically log π
        # then climbs to ~+3 and reward collapses into that circling attractor.
        # Target ~0 keeps the policy stochastic (log π ≈ 0), preserving
        # exploration instead of annealing α → 0 and saturating.
        target_entropy=0.0,
        target_update_tau=0.005,    # Polyak averaging rate for target critics
        gamma=0.99,
        # training
        seq_len=50,
        batch_size=16,
        learning_rate=8e-4,
        grad_clip=100.0,
        kl_weight=1.0,
        free_bits=1.0,
        log_every=50,
        image_every=400,
        checkpoint_every=800,
        # Full-agent resume from data/checkpoints/hafner_dreamer (below) is
        # automatic and always tried first. This is only a fallback for
        # warm-starting *just* the RSSM from a separate world-model-only
        # checkpoint dir when there's nothing to resume yet; None = skip it.
        pretrained_rssm=None,
    )

    # ------------------------------------------------------------------
    # Init buffer, model, optimiser
    # ------------------------------------------------------------------
    rngs = nnx.Rngs(0, noise=1)
    buffer = SequenceReplayBuffer(capacity=config["buffer_capacity"], rngs=rngs)

    rssm = RSSM(
        image_channels=config["image_channels"],
        action_dim=config["action_dim"],
        memory_dim=config["memory_dim"],
        stoch_dim=config["stoch_dim"],
        num_gaussian_components=config["num_gaussian_components"],
        predictor_hidden_dim=config["predictor_hidden_dim"],
        frame_stack=config["frame_stack"],
        rngs=rngs,
    )
    policy = Policy(
        features_dim=config["stoch_dim"],
        hidden_dim=config["policy_hidden_dim"],
        action_dim=config["action_dim"],
        rngs=nnx.Rngs(42, noise=43),
    )
    critic = Critic(
        features_dim=config["stoch_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["q_hidden_dim"],
        rngs=nnx.Rngs(44, noise=45),
    )
    critic_target = Critic(
        features_dim=config["stoch_dim"],
        action_dim=config["action_dim"],
        hidden_dim=config["q_hidden_dim"],
        rngs=nnx.Rngs(46, noise=47),
    )
    nnx.update(critic_target, nnx.state(critic))  # start target == online critic
    temperature = Temperature(initial_alpha=config["init_alpha"])

    # Everything a checkpoint needs to fully resume the agent (world model +
    # actor-critic + entropy coefficient) — saved/restored together as one step.
    all_models = {
        "rssm": rssm,
        "policy": policy,
        "critic": critic,
        "critic_target": critic_target,
        "temperature": temperature,
    }

    checkpoint_dir = Path("data/checkpoints/hafner_dreamer")

    # Resume the *whole* agent (rssm + policy + critic(s) + temperature) from
    # the latest checkpoint in checkpoint_dir, if one exists — this is what
    # "continue training the saved model" restores. World-model and policy
    # steps are tracked separately (they advance at different rates:
    # train_per_collect vs policy_steps_per_collect per cycle), so each metric
    # family lands on its own clean x-axis in MLflow. Checkpoints are keyed by
    # the world-model step, so wm_step resumes from the checkpoint step and
    # policy_step is derived from it via the per-cycle ratio so the sac/* curve
    # continues at a sensible position too. Falls back to warm-starting *only*
    # the RSSM from `pretrained_rssm` when there's nothing to resume yet.
    #
    # Caveat: only model *parameters* are checkpointed, not optimizer state
    # (Adam's m/v moments, step count) — so on resume, the world-model LR
    # schedule restarts its warmup/decay from a fresh optimizer, and Adam's
    # momentum rebuilds from zero for a few steps. Usually a minor blip
    # given the short 200-step warmup; ask if you want optimizer state
    # checkpointed too for a fully seamless resume.
    wm_step = load_checkpoint(all_models, checkpoint_dir)
    if wm_step == 0 and config["pretrained_rssm"] is not None:
        load_checkpoint({"rssm": rssm}, config["pretrained_rssm"])
    policy_step = (
        wm_step * config["policy_steps_per_collect"] // config["train_per_collect"]
    )

    ckpt_manager = make_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    total_train_steps = config["num_cycles"] * config["train_per_collect"]
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=200,
        decay_steps=total_train_steps,
    )
    tx = optax.chain(
        # optax.clip_by_global_norm(config["grad_clip"]),
        optax.adam(schedule),
    )
    optimizer = nnx.Optimizer(rssm, tx, wrt=nnx.Param)
    policy_optimizer = nnx.Optimizer(policy, optax.adam(config["policy_lr"]), wrt=nnx.Param)
    critic_optimizer = nnx.Optimizer(critic, optax.adam(config["q_lr"]), wrt=nnx.Param)
    temp_optimizer = nnx.Optimizer(temperature, optax.adam(config["alpha_lr"]), wrt=nnx.Param)
    target_entropy = jnp.asarray(float(config["target_entropy"]), dtype=jnp.float32)

    gamma = jnp.asarray(config["gamma"], dtype=jnp.float32)
    tau   = jnp.asarray(config["target_update_tau"], dtype=jnp.float32)
    q_weight_decay = jnp.asarray(config["q_weight_decay"], dtype=jnp.float32)
    sac_rng = jax.random.PRNGKey(7)

    kl_w      = jnp.asarray(config["kl_weight"], dtype=jnp.float32)
    free_bits = jnp.asarray(config["free_bits"],  dtype=jnp.float32)

    # ------------------------------------------------------------------
    # Spawn the inference server + self-play workers
    # ------------------------------------------------------------------
    # The server owns a copy of encoder+policy (only what's needed to act);
    # workers only ever exchange frames/actions with it, never touching the
    # accelerator or the RSSM/Policy classes themselves.
    ctx = mp.get_context("spawn")
    request_queue     = ctx.Queue()
    weight_queue       = ctx.Queue()
    result_queue       = ctx.Queue()
    response_queues    = [ctx.Queue() for _ in range(config["num_workers"])]
    command_queues     = [ctx.Queue() for _ in range(config["num_workers"])]
    stop_event  = ctx.Event()
    ready_event = ctx.Event()

    def sync_target_weights() -> None:
        """Push the live encoder+policy weights to the inference server."""
        weight_queue.put(extract_weights(EncoderPolicy(rssm.encoder, policy)))

    remote_model_kwargs = dict(
        image_channels=config["image_channels"],
        memory_dim=config["memory_dim"],
        stoch_dim=config["stoch_dim"],
        num_gaussian_components=config["num_gaussian_components"],
        policy_hidden_dim=config["policy_hidden_dim"],
        action_dim=config["action_dim"],
        frame_stack=config["frame_stack"],
    )
    server = ctx.Process(
        target=run_inference_server,
        args=(
            build_encoder_policy,
            remote_model_kwargs,
            request_queue,
            response_queues,
            weight_queue,
            stop_event,
            ready_event,
            config["max_inference_batch"],
        ),
        daemon=True,
    )
    server.start()

    # Seed the server with the live encoder/policy weights and wait until it
    # is serving before launching workers.
    sync_target_weights()
    ready_event.wait()

    sp_config = {
        "action_repeat": config["action_repeat"],
        "max_episode_steps": config["max_episode_steps"],
        "frame_stack": config["frame_stack"],
    }
    workers = [
        ctx.Process(
            target=run_worker,
            args=(
                i,
                command_queues[i],
                request_queue,
                response_queues[i],
                result_queue,
                stop_event,
                sp_config,
            ),
            daemon=True,
        )
        for i in range(config["num_workers"])
    ]
    for worker in workers:
        worker.start()

    # ------------------------------------------------------------------
    # Collect → train loop
    # ------------------------------------------------------------------
    mlflow.set_experiment("hafner_dreamer")
    try:
        with mlflow.start_run(run_name="policy_collect_train", log_system_metrics=True):
            mlflow.log_params(config)
            if wm_step > 0:
                print(f"[resume] continuing from wm_step={wm_step}, policy_step={policy_step}")
            mlflow.log_param("resumed_from_wm_step", wm_step)
            mlflow.log_param("resumed_from_policy_step", policy_step)

            val_batch = None
            # Image/checkpoint cadence is keyed off the world-model step and
            # uses a threshold (fire when >= N steps elapsed since the last
            # event) rather than exact ``wm_step % N == 0`` so nothing is
            # skipped after resuming at an unaligned offset.
            last_image_step = wm_step
            last_ckpt_step  = wm_step

            def _save_and_upload_checkpoint(step: int) -> None:
                save_checkpoint(ckpt_manager, all_models, step)
                ckpt_manager.wait_until_finished()
                # Upload incrementally so a later divergence/crash/kill doesn't
                # lose everything — the end-of-run upload alone is too fragile.
                mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")

            for cycle in range(1, config["num_cycles"] + 1):
                # --- collect phase -----------------------------------------
                # Dispatch a "collect n episodes" command to every worker,
                # then block until all num_workers * episodes_per_worker
                # episodes have arrived before starting the train phase.
                print(f"\n[cycle {cycle}/{config['num_cycles']}] collecting "
                    f"{config['episodes_per_worker']} episodes x {config['num_workers']} workers …")
                for command_queue in command_queues:
                    command_queue.put(("collect", config["episodes_per_worker"]))

                expected = config["num_workers"] * config["episodes_per_worker"]
                ep_rewards, ep_lengths = [], []
                with tqdm(total=expected, desc=f"collect cycle {cycle}", leave=False) as pbar:
                    while len(ep_rewards) < expected:
                        _worker_id, images, actions, rewards, continues = result_queue.get()
                        buffer.add_episode(images, actions, rewards, continues)
                        ep_rewards.append(float(rewards.sum()))
                        ep_lengths.append(int(images.shape[0]))
                        pbar.update(1)

                collect_stats = {
                    "collect/ep_reward_mean": float(np.mean(ep_rewards)),
                    "collect/ep_reward_max":  float(np.max(ep_rewards)),
                    "collect/ep_reward_min":  float(np.min(ep_rewards)),
                    "collect/ep_length_mean": float(np.mean(ep_lengths)),
                    "collect/buffer_size":    float(len(buffer)),
                }
                mlflow.log_metrics(collect_stats, step=policy_step)
                print(f"  buffer: {len(buffer)} episodes  |  "
                      f"reward mean/max: {collect_stats['collect/ep_reward_mean']:.1f} / "
                      f"{collect_stats['collect/ep_reward_max']:.1f}")

                # Refresh validation batch each cycle so it reflects new data.
                val_batch = buffer.sample(
                    batch_size=config["batch_size"], seq_len=config["seq_len"]
                )

                # --- train phase  world model---------------------------------------------
                for _ in tqdm(range(config["train_per_collect"]),
                            desc=f"world model train cycle {cycle}", leave=False):
                    wm_step += 1
                    batch = buffer.sample(
                        batch_size=config["batch_size"], seq_len=config["seq_len"]
                    )
                    metrics = train_step_world_model(
                        rssm, optimizer,
                        batch["images"], batch["actions"],
                        batch["rewards"], batch["continues"],
                        kl_w, free_bits,
                    )

                    if wm_step % config["log_every"] == 0:
                        mlflow.log_metrics(
                            {f"train/{k}": v
                            for k, v in metrics_to_floats(metrics).items()},
                            step=wm_step,
                        )

                    if wm_step - last_image_step >= config["image_every"]:
                        last_image_step = wm_step
                        val_metrics = eval_step_world_model(
                            rssm,
                            val_batch["images"], val_batch["actions"],
                            val_batch["rewards"], val_batch["continues"],
                            kl_w, free_bits,
                        )
                        mlflow.log_metrics(
                            {f"val/{k}": v
                            for k, v in metrics_to_floats(val_metrics).items()},
                            step=wm_step,
                        )
                        log_reconstruction_images(rssm, val_batch, wm_step)

                    if wm_step - last_ckpt_step >= config["checkpoint_every"]:
                        last_ckpt_step = wm_step
                        _save_and_upload_checkpoint(wm_step)

                # --- train phase  controller (off-policy SAC, real buffer) -----------
                for _ in tqdm(range(config["policy_steps_per_collect"]),
                              desc=f"policy train cycle {cycle}", leave=False):
                    policy_step += 1
                    batch = buffer.sample(
                        batch_size=config["batch_size"], seq_len=config["seq_len"]
                    )
                    sac_rng, step_rng = jax.random.split(sac_rng)
                    pol_metrics = train_step_sac_real(
                        rssm, policy, critic, critic_target, temperature,
                        policy_optimizer, critic_optimizer, temp_optimizer,
                        batch["images"], batch["actions"],
                        batch["rewards"], batch["continues"],
                        gamma, tau, target_entropy, q_weight_decay, step_rng,
                    )

                    if policy_step % config["log_every"] == 0:
                        mlflow.log_metrics(
                            {f"sac/{k}": v
                             for k, v in metrics_to_floats(pol_metrics).items()},
                            step=policy_step,
                        )

                # Push the freshly-trained encoder+policy to the inference
                # server so the *next* collect phase uses updated weights.
                sync_target_weights()

            # Final artefacts
            if val_batch is not None:
                log_reconstruction_images(rssm, val_batch, "final")
            if wm_step != last_ckpt_step:
                _save_and_upload_checkpoint(wm_step)
            ckpt_manager.close()
    finally:
        stop_event.set()
        for worker in workers:
            worker.terminate()
        server.terminate()
        for worker in workers:
            worker.join(timeout=5)
        server.join(timeout=5)
