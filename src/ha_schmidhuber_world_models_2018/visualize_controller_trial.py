from pathlib import Path

import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from ha_schmidhuber_world_models_2018.controller import CMAESController
from ha_schmidhuber_world_models_2018.evolve_controller import (
    encode_observations,
    make_checkpoint_manager,
    restore_checkpoint,
    squash_car_racing_actions,
    unpack_controller_parameters,
)
from ha_schmidhuber_world_models_2018.vae import VAE
from ha_schmidhuber_world_models_2018.world_model import WorldModel


def save_gif(frames: list[np.ndarray], path: str | Path, *, fps: int) -> None:
    if not frames:
        raise ValueError("Cannot save a GIF with no frames.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(frames[0])
    ax.axis("off")

    def update(frame: np.ndarray) -> tuple[object]:
        image.set_data(frame)
        return (image,)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=True,
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def save_reward_plot(rewards: list[float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cumulative_rewards = np.cumsum(np.asarray(rewards, dtype=np.float32))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative_rewards)
    ax.set_title("Controller Trial Cumulative Reward")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_reconstruction_comparison_frame(
    rendered_frame: np.ndarray,
    vae: VAE,
    z: jax.Array,
    *,
    image_size: tuple[int, int],
) -> np.ndarray:
    real_frame = jax.image.resize(
        jnp.asarray(rendered_frame, dtype=jnp.float32),
        (*image_size, rendered_frame.shape[-1]),
        method="bilinear",
    )
    real_frame = np.asarray(real_frame / 255.0)
    reconstruction = np.asarray(vae.decoder(z)[0])

    real_frame = np.clip(real_frame, 0.0, 1.0)
    reconstruction = np.clip(reconstruction, 0.0, 1.0)
    combined = np.concatenate([real_frame, reconstruction], axis=1)
    return np.asarray(combined * 255.0, dtype=np.uint8)


def main() -> None:
    env_id = "CarRacing-v3"
    vae_checkpoint_dir = Path("data/checkpoints/vae")
    world_model_checkpoint_dir = Path("data/checkpoints/world_model")
    controller_path = Path("data/checkpoints/controller/best_controller.npz")
    gif_path = Path("data/controller_trials/controller_trial.gif")
    reward_plot_path = Path("data/controller_trials/controller_trial_rewards.png")

    seed = 0
    max_episode_steps = 1000
    fps = 30
    save_every = 2
    image_size = (64, 64)
    image_channels = 3
    latent_dim = 32
    hidden_dim = 256
    action_dim = 3
    num_gaussian_components = 5
    temperature = 1.0
    sample_latent = False

    vae = VAE(
        image_channels=image_channels,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(seed, noise=seed + 1),
    )
    vae_manager = make_checkpoint_manager(vae_checkpoint_dir)
    restore_checkpoint(vae_manager, vae)
    vae_manager.close()

    world_model = WorldModel(
        in_features=latent_dim + action_dim,
        hidden_features=hidden_dim,
        out_features=latent_dim,
        num_gaussian_components=num_gaussian_components,
        temperature=temperature,
        rngs=nnx.Rngs(seed + 2, noise=seed + 3),
    )
    world_model_manager = make_checkpoint_manager(world_model_checkpoint_dir)
    restore_checkpoint(world_model_manager, world_model)
    world_model_manager.close()

    assert vae is not None
    assert world_model is not None

    if controller_path.exists():
        with np.load(controller_path) as controller_archive:
            parameters = np.asarray(controller_archive["parameters"], dtype=np.float32)
    else:
        print(f"{controller_path} does not exist; visualizing a zero controller.")
        parameters = np.zeros(
            CMAESController.num_parameters(
                z_dim=latent_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            ),
            dtype=np.float32,
        )
    weights, bias = unpack_controller_parameters(
        parameters[None, ...],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )

    env = gym.make(env_id, continuous=True, render_mode="rgb_array")
    frames: list[np.ndarray] = []
    rewards: list[float] = []

    try:
        observation, _ = env.reset(seed=seed)
        z = encode_observations(
            vae,
            observation[None, ...],
            image_size=image_size,
            sample_latent=sample_latent,
        )
        frames.append(
            make_reconstruction_comparison_frame(
                np.asarray(env.render()),
                vae,
                z,
                image_size=image_size,
            )
        )
        carry = (
            jnp.zeros((1, hidden_dim), dtype=jnp.float32),
            jnp.zeros((1, hidden_dim), dtype=jnp.float32),
        )

        for step in range(max_episode_steps):
            controller_input = jnp.concatenate([z, carry[1]], axis=-1)
            raw_action = jnp.einsum("bi,bai->ba", controller_input, weights) + bias
            action = np.asarray(squash_car_racing_actions(raw_action)[0], dtype=np.float32)

            observation, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            if step % save_every == 0:
                frame_z = encode_observations(
                    vae,
                    observation[None, ...],
                    image_size=image_size,
                    sample_latent=sample_latent,
                )
                frames.append(
                    make_reconstruction_comparison_frame(
                        np.asarray(env.render()),
                        vae,
                        frame_z,
                        image_size=image_size,
                    )
                )

            action_batch = jnp.asarray(action, dtype=jnp.float32)[None, None, :]
            carry, _ = world_model(
                z[:, None, :],
                action_batch,
                initial_carry=carry,
                return_carry=True,
            )
            if terminated or truncated:
                break

            z = encode_observations(
                vae,
                observation[None, ...],
                image_size=image_size,
                sample_latent=sample_latent,
            )
    finally:
        env.close()

    save_gif(frames, gif_path, fps=fps)
    save_reward_plot(rewards, reward_plot_path)
    print(
        f"saved trial GIF to {gif_path} with total_reward={sum(rewards):.2f}; "
        f"saved reward plot to {reward_plot_path}"
    )


if __name__ == "__main__":
    main()
