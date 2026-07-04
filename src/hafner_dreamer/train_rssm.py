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

import sys
from pathlib import Path

# Expose the models package (local imports: encoder_decoder, rssm, …)
sys.path.insert(0, str(Path(__file__).parent / "models"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image
from tqdm import tqdm

import gymnasium as gym

from replay_buffer import SequenceReplayBuffer
from rssm import RSSM


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

IMAGE_SIZE = (64, 64)  # MDNEncoder architecture is fixed to 64×64 input


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize a (H, W, 3) uint8 frame to IMAGE_SIZE uint8."""
    img = Image.fromarray(frame).resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def collect_episode(
    env: gym.Env,
    *,
    action_repeat: int = 2,
    max_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect one episode with a random policy.

    Each timestep: sample a random action, repeat it *action_repeat* times,
    sum the rewards, record the final observation.

    Returns
    -------
    images   : uint8  (T, 64, 64, 3)
    actions  : float32 (T, action_dim)
    rewards  : float32 (T,)
    continues: float32 (T,)   1 = episode not done, 0 = done
    """
    obs, _ = env.reset()
    images, actions, rewards, continues = [], [], [], []

    done = False
    step = 0
    while not done and step < max_steps:
        action = env.action_space.sample()
        total_reward = 0.0
        last_obs = obs
        for _ in range(action_repeat):
            last_obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            done = terminated or truncated
            if done:
                break

        images.append(preprocess_frame(last_obs))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(total_reward))
        continues.append(0.0 if done else 1.0)
        obs = last_obs
        step += 1

    return (
        np.stack(images),
        np.stack(actions),
        np.array(rewards, dtype=np.float32),
        np.array(continues, dtype=np.float32),
    )


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
    _, out = rssm(images, actions, initial_carry=None, return_carry=True)

    # Reconstruction: compare decoder output (all T frames) against all images.
    recon_loss = jnp.mean((out["reconstruction"] - images) ** 2)

    z_sg = jax.lax.stop_gradient(out["post_stoch"])  # (B, T-1, D)
    log_post_dist  = RSSM.log_prob(z_sg, out["post_pi_logits"],  out["post_mu"],  out["post_log_var"])
    log_prior_dist = RSSM.log_prob(z_sg, out["prior_pi_logits"], out["prior_mu"], out["prior_log_var"])
    kl_per_step = jax.lax.stop_gradient(log_post_dist) - log_prior_dist  # (B, T-1)
    # KL: post_stoch / post_* are already aligned to z_1..z_{T-1};
    # prior_* covers the same range — no off-by-one here.
    #
    # DreamerV2-style 80/20 KL balancing:
    #   80% — trains only the prior  (stop_gradient on z and on log_q)
    #   20% — trains only the encoder (stop_gradient on log_p, gradient flows
    #          through z via reparameterisation back to the encoder)
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


@nnx.jit
def train_step(
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
    optimizer.update(rssm, grads)
    return metrics


@nnx.jit
def eval_step(
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
) -> None:
    """Save a 3-row grid per batch item: real | posterior recon | prior imagination."""
    B = min(inputs.shape[0], num_items)
    T = inputs.shape[1]
    # Pick frames from the T-1 range so all three rows have valid data.
    t_indices = [0, (T - 1) // 2, T - 2]

    num_cols = len(t_indices)
    row_labels = ["real", "posterior", "prior"]
    fig, axes = plt.subplots(
        B * 3, num_cols, figsize=(num_cols * 2, B * 6), squeeze=False
    )
    for b in range(B):
        for col, t in enumerate(t_indices):
            rows = [
                inputs[b, t],
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
    _, out = rssm(images, actions, initial_carry=None, return_carry=True)
    recon       = np.asarray(out["reconstruction"])
    prior_recon = np.asarray(out["prior_reconstruction"])
    step_label = f"{step:06d}" if isinstance(step, int) else step
    grid_path = Path(f"data/hafner_dreamer/recon_grid_step_{step_label}.png")
    save_reconstruction_grid(
        np.asarray(images), recon, prior_recon, grid_path,
        title=f"step {step_label}",
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


def save_checkpoint(manager: ocp.CheckpointManager, model: RSSM, step: int) -> None:
    manager.save(step, args=ocp.args.StandardSave(nnx.state(model)))


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
        num_collect_episodes=5,
        max_episode_steps=500,
        # model
        image_channels=3,
        action_dim=3,          # CarRacing-v3 continuous: [steering, gas, brake]
        memory_dim=256,
        stoch_dim=32,
        num_gaussian_components=2,
        predictor_hidden_dim=256,
        # training
        seq_len=50,
        batch_size=8,
        train_steps=5000,
        learning_rate=3e-4,
        grad_clip=100.0,
        kl_weight=1.0,
        free_bits=1.0,         # nats; per-step KL below this is not penalised
        log_every=50,
        image_every=500,
        checkpoint_every=1000,
    )

    # ------------------------------------------------------------------
    # Collect data
    # ------------------------------------------------------------------
    print(f"Collecting {config['num_collect_episodes']} episodes …")
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    rngs = nnx.Rngs(0, noise=1)
    buffer = SequenceReplayBuffer(capacity=config["num_collect_episodes"] * 2, rngs=rngs)

    total_steps = 0
    for ep_idx in range(config["num_collect_episodes"]):
        images, actions, rewards, continues = collect_episode(
            env,
            action_repeat=config["action_repeat"],
            max_steps=config["max_episode_steps"],
        )
        buffer.add_episode(images, actions, rewards, continues)
        total_steps += images.shape[0]
        print(
            f"  episode {ep_idx + 1}: {images.shape[0]} steps, "
            f"total reward {rewards.sum():.1f}"
        )
    env.close()
    print(f"Buffer: {len(buffer)} episodes, ~{total_steps} steps total.\n")

    # One fixed validation batch (sampled once, reused for visual logging)
    val_batch = buffer.sample(batch_size=config["batch_size"], seq_len=config["seq_len"])

    # ------------------------------------------------------------------
    # Model + optimiser
    # ------------------------------------------------------------------
    rssm = RSSM(
        image_channels=config["image_channels"],
        action_dim=config["action_dim"],
        memory_dim=config["memory_dim"],
        stoch_dim=config["stoch_dim"],
        num_gaussian_components=config["num_gaussian_components"],
        predictor_hidden_dim=config["predictor_hidden_dim"],
        rngs=rngs,
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["learning_rate"],
        warmup_steps=200,
        decay_steps=config["train_steps"],
    )
    tx = optax.chain(
        # optax.clip_by_global_norm(config["grad_clip"]),
        optax.adam(schedule),
    )
    optimizer = nnx.Optimizer(rssm, tx, wrt=nnx.Param)

    checkpoint_dir = Path("data/checkpoints/hafner_dreamer/rssm")
    ckpt_manager = make_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    mlflow.set_experiment("hafner_dreamer")
    with mlflow.start_run(run_name="rssm_overfit_one_batch", log_system_metrics=True):
        mlflow.log_params(config)

        kl_w = jnp.asarray(config["kl_weight"], dtype=jnp.float32)
        free_bits = jnp.asarray(config["free_bits"], dtype=jnp.float32)

        for step in tqdm(range(1, config["train_steps"] + 1), desc="train"):
            batch = buffer.sample(
                batch_size=config["batch_size"], seq_len=config["seq_len"]
            )
            metrics = train_step(
                rssm, optimizer,
                batch["images"], batch["actions"],
                batch["rewards"], batch["continues"],
                kl_w, free_bits,
            )

            if step % config["log_every"] == 0:
                mlflow.log_metrics(
                    {f"train/{k}": v for k, v in metrics_to_floats(metrics).items()},
                    step=step,
                )

            if step % config["image_every"] == 0:
                val_metrics = eval_step(
                    rssm,
                    val_batch["images"], val_batch["actions"],
                    val_batch["rewards"], val_batch["continues"],
                    kl_w, free_bits,
                )
                mlflow.log_metrics(
                    {f"val/{k}": v for k, v in metrics_to_floats(val_metrics).items()},
                    step=step,
                )
                log_reconstruction_images(rssm, val_batch, step)

            if step % config["checkpoint_every"] == 0:
                save_checkpoint(ckpt_manager, rssm, step)

        log_reconstruction_images(rssm, val_batch, "final")
        save_checkpoint(ckpt_manager, rssm, config["train_steps"])
        ckpt_manager.wait_until_finished()
        mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")
        ckpt_manager.close()
