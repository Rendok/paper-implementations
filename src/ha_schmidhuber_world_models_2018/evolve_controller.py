from pathlib import Path

import cma
import gymnasium as gym
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from ha_schmidhuber_world_models_2018.controller import CMAESController
from ha_schmidhuber_world_models_2018.vae import VAE
from ha_schmidhuber_world_models_2018.world_model import WorldModel


def make_checkpoint_manager(
    directory: str | Path, *, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    directory = Path(directory).expanduser().resolve()
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(directory, options=options)


def restore_checkpoint(
    manager: ocp.CheckpointManager, model: nnx.Module, *, step: int | None = None
) -> None:
    target_step = manager.latest_step() if step is None else step
    if target_step is None:
        raise FileNotFoundError("No checkpoints found to restore.")
    restored = manager.restore(
        target_step,
        args=ocp.args.StandardRestore(nnx.state(model)),
    )
    nnx.update(model, restored)


def preprocess_observations(
    observations: np.ndarray, *, image_size: tuple[int, int]
) -> jax.Array:
    frames = jnp.asarray(observations, dtype=jnp.float32)
    target_shape = (*frames.shape[:-3], *image_size, frames.shape[-1])
    frames = jax.image.resize(frames, target_shape, method="bilinear")
    return frames / 255.0


@nnx.jit
def jit_vae_encoder(vae: VAE, frames: jax.Array) -> tuple[jax.Array, jax.Array]:
    z, mu, _ = vae.encoder(frames)
    return z, mu


def encode_observations(
    vae: VAE,
    observations: np.ndarray,
    *,
    image_size: tuple[int, int],
    sample_latent: bool,
) -> jax.Array:
    frames = preprocess_observations(observations, image_size=image_size)
    z, mu = jit_vae_encoder(vae, frames)
    return z if sample_latent else mu


def make_vector_env(
    *,
    env_id: str,
    num_envs: int,
    render_mode: str | None,
) -> gym.vector.VectorEnv:
    def make_env() -> gym.Env:
        return gym.make(env_id, continuous=True, render_mode=render_mode)

    return gym.vector.SyncVectorEnv([make_env for _ in range(num_envs)])


def unpack_controller_parameters(
    parameters: np.ndarray,
    *,
    latent_dim: int,
    hidden_dim: int,
    action_dim: int,
) -> tuple[jax.Array, jax.Array]:
    parameters = jnp.asarray(parameters, dtype=jnp.float32)
    input_dim = latent_dim + hidden_dim
    expected_parameters = CMAESController.num_parameters(
        z_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )
    if parameters.ndim != 2 or parameters.shape[-1] != expected_parameters:
        raise ValueError(
            "Expected controller parameters with shape "
            f"(batch, {expected_parameters}), got {parameters.shape}."
        )

    weights_size = action_dim * input_dim
    weights = parameters[:, :weights_size].reshape(-1, action_dim, input_dim)
    bias = parameters[:, weights_size:]
    return weights, bias


def squash_car_racing_actions(raw_actions: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.tanh(raw_actions[..., :1]),
            jax.nn.sigmoid(raw_actions[..., 1:]),
        ],
        axis=-1,
    )


@nnx.jit
def jit_world_model(
    world_model: WorldModel,
    z: jax.Array,
    action_batch: jax.Array,
    carry: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]:
    next_carry, _ = world_model(
        z[:, None, :],
        action_batch,
        initial_carry=carry,
        return_carry=True,
    )
    return next_carry


def evaluate_population(
    population: list[np.ndarray],
    *,
    env: gym.vector.VectorEnv,
    vae: VAE,
    world_model: WorldModel,
    base_seed: int,
    episodes_per_candidate: int,
    max_episode_steps: int,
    image_size: tuple[int, int],
    latent_dim: int,
    hidden_dim: int,
    action_dim: int,
    sample_latent: bool,
) -> np.ndarray:
    population_size = len(population)
    candidate_indices = np.repeat(np.arange(population_size), episodes_per_candidate)
    parameters = np.asarray(population, dtype=np.float32)[candidate_indices]
    num_envs = parameters.shape[0]

    weights, bias = unpack_controller_parameters(
        parameters,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )

    observations, _ = env.reset(seed=[base_seed + i for i in range(num_envs)])

    z = encode_observations(
        vae,
        observations,
        image_size=image_size,
        sample_latent=sample_latent,
    )
    carry = (
        jnp.zeros((num_envs, hidden_dim), dtype=jnp.float32),
        jnp.zeros((num_envs, hidden_dim), dtype=jnp.float32),
    )

    episode_rewards = np.zeros(num_envs, dtype=np.float64)
    active = np.ones(num_envs, dtype=bool)

    # A bit of profiling jit vs non-jit:
    # evaluate_population per-step timings: encode=0.0036s, controller=0.0008s, env_step=0.1165s, world_model=0.3887s
    # evaluate_population per-step timings: encode=0.0018s, controller=0.0009s, env_step=0.1148s, world_model=0.0014s
    for _ in range(max_episode_steps):
        controller_input = jnp.concatenate([z, carry[1]], axis=-1)
        raw_actions = jnp.einsum("bi,bai->ba", controller_input, weights) + bias
        actions = np.asarray(
            squash_car_racing_actions(raw_actions),
            dtype=np.float32,
        )

        observations, rewards, terminated, truncated, _ = env.step(actions)

        episode_rewards += np.asarray(rewards, dtype=np.float64) * active
        done = np.asarray(terminated) | np.asarray(truncated)
        active &= ~done

        action_batch = jnp.asarray(actions, dtype=jnp.float32)[:, None, :]
        carry = jit_world_model(world_model, z, action_batch, carry)

        if not np.any(active):
            break

        z = encode_observations(
            vae,
            observations,
            image_size=image_size,
            sample_latent=sample_latent,
        )

    reward_sums = np.bincount(
        candidate_indices,
        weights=episode_rewards,
        minlength=population_size,
    )
    return reward_sums / episodes_per_candidate


def save_controller(
    path: str | Path,
    parameters: np.ndarray,
    *,
    score: float,
    generation: int,
    z_dim: int,
    hidden_dim: int,
    action_dim: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    controller = CMAESController.from_flat_parameters(
        jnp.asarray(parameters, dtype=jnp.float32),
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )
    np.savez(
        path,
        parameters=np.asarray(parameters, dtype=np.float32),
        weights=np.asarray(controller.weights[...], dtype=np.float32),
        bias=np.asarray(controller.bias[...], dtype=np.float32),
        score=np.asarray(score, dtype=np.float32),
        generation=np.asarray(generation, dtype=np.int32),
        z_dim=np.asarray(z_dim, dtype=np.int32),
        hidden_dim=np.asarray(hidden_dim, dtype=np.int32),
        action_dim=np.asarray(action_dim, dtype=np.int32),
    )


def main() -> None:
    env_id = "CarRacing-v3"
    vae_checkpoint_dir = Path("data/checkpoints/vae")
    world_model_checkpoint_dir = Path("data/checkpoints/world_model")
    output_path = Path("data/checkpoints/controller/best_controller.npz")
    generations = 50
    population_size = 8
    episodes_per_candidate = 4
    max_episode_steps = 1000
    sigma = 0.1
    seed = 0
    image_size = (64, 64)
    image_channels = 3
    latent_dim = 32
    hidden_dim = 256
    action_dim = 3
    num_gaussian_components = 5
    temperature = 1.0
    render_mode = None
    sample_latent = False
    use_mlflow = True

    mlflow_params = {
        "env_id": env_id,
        "vae_checkpoint_dir": str(vae_checkpoint_dir),
        "world_model_checkpoint_dir": str(world_model_checkpoint_dir),
        "output_path": str(output_path),
        "generations": generations,
        "population_size": population_size,
        "episodes_per_candidate": episodes_per_candidate,
        "max_episode_steps": max_episode_steps,
        "sigma": sigma,
        "seed": seed,
        "image_size": image_size,
        "image_channels": image_channels,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "action_dim": action_dim,
        "num_gaussian_components": num_gaussian_components,
        "temperature": temperature,
        "render_mode": render_mode,
        "sample_latent": sample_latent,
    }

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

    parameter_dim = CMAESController.num_parameters(
        z_dim=latent_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
    )
    cma_es = cma.CMAEvolutionStrategy(
        np.zeros(parameter_dim, dtype=np.float64),
        sigma,
        {
            "popsize": population_size,
            "seed": seed,
            "verbose": -9,
        },
    )

    best_score = -np.inf
    best_parameters = np.zeros(parameter_dim, dtype=np.float64)
    env = make_vector_env(
        env_id=env_id,
        num_envs=population_size * episodes_per_candidate,
        render_mode=render_mode,
    )
    try:
        if use_mlflow:
            mlflow.set_experiment("ha_schmidhuber_world_models_2018")
            mlflow.start_run(run_name="controller_cma_es_on_car_racing")
            mlflow.log_params(mlflow_params | {"parameter_dim": parameter_dim})

        for generation in tqdm(range(1, generations + 1), desc="generation"):
            population = cma_es.ask()
            base_seed = seed + generation * population_size * episodes_per_candidate
            rewards = evaluate_population(
                population,
                env=env,
                vae=vae,
                world_model=world_model,
                base_seed=base_seed,
                episodes_per_candidate=episodes_per_candidate,
                max_episode_steps=max_episode_steps,
                image_size=image_size,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                action_dim=action_dim,
                sample_latent=sample_latent,
            )

            cma_es.tell(population, (-rewards).tolist())
            generation_best_idx = int(np.argmax(rewards))
            generation_best_score = float(rewards[generation_best_idx])
            generation_best_parameters = np.asarray(
                population[generation_best_idx], dtype=np.float64
            )
            if generation_best_score > best_score:
                best_score = generation_best_score
                best_parameters = generation_best_parameters.copy()
                save_controller(
                    output_path,
                    best_parameters,
                    score=best_score,
                    generation=generation,
                    z_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    action_dim=action_dim,
                )

            metrics = {
                "reward_best": best_score,
                "reward_generation_best": generation_best_score,
                "reward_mean": float(np.mean(rewards)),
                "reward_std": float(np.std(rewards)),
                "sigma": float(cma_es.sigma),
            }
            tqdm.write(
                "generation "
                f"{generation}: best={metrics['reward_best']:.2f}, "
                f"gen_best={metrics['reward_generation_best']:.2f}, "
                f"mean={metrics['reward_mean']:.2f}, sigma={metrics['sigma']:.4f}"
            )
            if use_mlflow:
                mlflow.log_metrics(metrics, step=generation)

        save_controller(
            output_path,
            best_parameters,
            score=best_score,
            generation=generations,
            z_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        if use_mlflow:
            mlflow.log_artifact(str(output_path), artifact_path="controller")
    finally:
        env.close()
        if use_mlflow and mlflow.active_run() is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
