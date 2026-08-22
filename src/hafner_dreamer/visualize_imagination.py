"""Visualise real vs imagined rollout from a trained RSSM+Policy checkpoint.

The script:
  1. Loads the RSSM and Policy weights from a checkpoint saved by
     ``train_rssm.py`` (a single ``Composite`` step with named items —
     see ``load_checkpoint`` there).
  2. Runs one episode in CarRacing-v3, acting with the trained policy's
     deterministic (squashed-mean) action at each step.
  3. Encodes the first frame to z_0 with the RSSM encoder.
  4. Imagines T steps using the actions actually taken and the prior.
  5. Saves a side-by-side GIF:  real | reconstruction | imagined
     and individual PNG frames to data/hafner_dreamer/imagination/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "models"))

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image
import gymnasium as gym

from rssm import RSSM
from controller import Policy
from self_play_worker import preprocess_frame, FrameStacker, IMAGE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_episode(
    env, rssm: RSSM, policy: Policy, *, action_repeat: int = 3, horizon: int = 100, frame_stack: int = 3,
):
    """Collect up to *horizon* steps acting greedily with the trained policy.

    At each step: stack the current frame with the ``frame_stack - 1``
    previous frames along the channel axis (zeros for missing history at
    episode start — matching self-play collection via :class:`FrameStacker`),
    encode that stack to z, take the policy's deterministic squashed-mean
    action (already inside the env's valid box — no clipping needed), and
    repeat it *action_repeat* times.

    Returns
    -------
    real_frames   : uint8  (T, H, W, 3)               — single frames, for display only
    stacked_frames: uint8  (T, H, W, 3*frame_stack)    — what the encoder actually saw
    actions       : float32 (T, action_dim)
    """
    obs, _ = env.reset()
    real_frames, stacked_frames, actions = [], [], []
    stacker = FrameStacker(frame_stack, (*IMAGE_SIZE, 3))
    done = False
    while not done and len(real_frames) < horizon:
        frame = preprocess_frame(obs)
        stacked_frame = stacker.push(frame)

        stacked_jax = jnp.asarray(stacked_frame[None] / 255.0, dtype=jnp.float32)  # (1, H, W, C)
        z, _, _, _ = rssm.encoder(stacked_jax)
        action = np.asarray(policy(z), dtype=np.float32)[0]

        for _ in range(action_repeat):
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break
        real_frames.append(frame)
        stacked_frames.append(stacked_frame)
        actions.append(action)
    return np.stack(real_frames), np.stack(stacked_frames), np.stack(actions)


def load_models(checkpoint_path: str, rssm_config: dict, policy_config: dict) -> tuple[RSSM, Policy]:
    """Build a fresh RSSM + Policy and restore both from one checkpoint step.

    Expects the ``Composite`` layout written by ``train_rssm.py``'s
    ``save_checkpoint``: named items ``"rssm"`` and ``"policy"`` (among
    others) under the same step.
    """
    rssm = RSSM(
        image_channels=rssm_config["image_channels"],
        action_dim=rssm_config["action_dim"],
        memory_dim=rssm_config["memory_dim"],
        stoch_dim=rssm_config["stoch_dim"],
        num_gaussian_components=rssm_config["num_gaussian_components"],
        predictor_hidden_dim=rssm_config["predictor_hidden_dim"],
        frame_stack=rssm_config["frame_stack"],
        rngs=nnx.Rngs(0, noise=1),
    )
    policy = Policy(
        features_dim=rssm_config["stoch_dim"],
        hidden_dim=policy_config["policy_hidden_dim"],
        action_dim=rssm_config["action_dim"],
        rngs=nnx.Rngs(42, noise=43),
    )

    manager = ocp.CheckpointManager(
        Path(checkpoint_path).expanduser().resolve(),
        options=ocp.CheckpointManagerOptions(),
    )
    step = manager.latest_step()
    if step is None:
        print(f"[warn] no checkpoint found at {checkpoint_path}; using random weights")
    else:
        restore_args = ocp.args.Composite(
            rssm=ocp.args.StandardRestore(nnx.state(rssm)),
            policy=ocp.args.StandardRestore(nnx.state(policy)),
        )
        restored = manager.restore(step, args=restore_args)
        nnx.update(rssm, restored["rssm"])
        nnx.update(policy, restored["policy"])
        print(f"Loaded checkpoint step {step} from {checkpoint_path} (rssm, policy)")
    manager.close()
    return rssm, policy


def frames_to_gif(frames: list[np.ndarray], path: Path, fps: int = 10) -> None:
    """Save a list of uint8 (H, W, 3) arrays as an animated GIF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    print(f"Saved GIF → {path}  ({len(imgs)} frames @ {fps} fps)")


def make_strip(*panels: np.ndarray) -> np.ndarray:
    """Stack uint8 (H, W, C) panels side-by-side with a 2-px grey divider."""
    H = panels[0].shape[0]
    divider = np.full((H, 2, 3), 180, dtype=np.uint8)
    parts = []
    for i, p in enumerate(panels):
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(divider)
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint  = "data/checkpoints/hafner_dreamer"
    horizon     = 300
    out_path    = Path("data/hafner_dreamer/imagination.gif")
    fps         = 10
    action_repeat = 3

    model_config = dict(
        image_channels=3,
        frame_stack=3,
        action_dim=3,
        memory_dim=256,
        stoch_dim=256, # 32
        num_gaussian_components=8,
        predictor_hidden_dim=256,
    )
    policy_config = dict(policy_hidden_dim=256)

    # --- load model ---
    rssm, policy = load_models(checkpoint, model_config, policy_config)

    # --- collect real episode, acting with the trained policy ---
    print("Collecting real episode …")
    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    real_frames_u8, stacked_frames_u8, actions = collect_episode(
        env, rssm, policy, action_repeat=action_repeat, horizon=horizon,
        frame_stack=model_config["frame_stack"],
    )
    env.close()
    T = len(actions)
    print(f"  collected {T} steps")

    # Shared JAX inputs — encoder expects frame-stacked (H, W, C*frame_stack)
    # observations, matching what it saw during self-play collection.
    all_frames_b = jnp.asarray(stacked_frames_u8 / 255.0, dtype=jnp.float32)[np.newaxis]  # (1, T, H, W, C)
    first_frame  = all_frames_b[:, 0]   # (1, H, W, C) — already includes zero-padded history
    actions_jax  = jnp.asarray(actions[np.newaxis], dtype=jnp.float32)

    # --- direct reconstruction: encode every real (stacked) frame, decode straight back ---
    post_stoch, _, _, _ = rssm.encoder(all_frames_b[0])   # (T, D)
    recon_frames = np.asarray(rssm.decoder(post_stoch))    # (T, H, W, C*frame_stack) — full stack reconstruction
    # Decoder reconstructs the whole stack; only display the current-frame
    # slice (last image_channels channels) for a plain RGB comparison.
    recon_frames = recon_frames[..., -rssm.image_channels:]
    recon_u8 = np.clip(recon_frames * 255, 0, 255).astype(np.uint8)

    # --- imagination: seed from z_0, roll forward with prior ---
    _, out = rssm.autoregressive_forward(first_frame, actions_jax, return_carry=True)
    imagined = np.asarray(out["imagined_frames"][0])       # (T, H, W, C*frame_stack)
    imagined = imagined[..., -rssm.image_channels:]
    imagined_u8 = np.clip(imagined * 255, 0, 255).astype(np.uint8)

    real_64 = real_frames_u8   # already 64×64 from preprocess_frame

    # --- build  real | reconstruction | imagination  GIF ---
    frames_sb = []
    for t in range(min(T, imagined_u8.shape[0])):
        frames_sb.append(make_strip(real_64[t], recon_u8[t], imagined_u8[t]))

    frames_to_gif(frames_sb, out_path, fps=fps)

    # --- also save individual PNG strips ---
    strips_dir = out_path.parent / "imagination_frames"
    strips_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in enumerate(frames_sb):
        Image.fromarray(frame).save(strips_dir / f"frame_{t:04d}.png")

    # --- print predicted rewards ---
    rewards = np.asarray(out["reward_logit"][0, :, 0])
    conts   = np.asarray(jax.nn.sigmoid(out["continue_logit"][0, :, 0]))
    print(f"\nImagined rewards  — mean {rewards.mean():.3f}  "
          f"min {rewards.min():.3f}  max {rewards.max():.3f}")
    print(f"Continue probs    — mean {conts.mean():.3f}  "
          f"min {conts.min():.3f}  max {conts.max():.3f}")
    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
