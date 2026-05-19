from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm
from einops import rearrange

from typing import Tuple
from ha_schmidhuber_world_models_2018.car_racing_dataset import (
    CarRacingDataSource,
    RescaleFrame,
    ResizeFrame,
    split_source,
)
from ha_schmidhuber_world_models_2018.world_model import WorldModel
from ha_schmidhuber_world_models_2018.vae import VAE
from ha_schmidhuber_world_models_2018.train_vae import restore_checkpoint as restore_vae_checkpoint
from ha_schmidhuber_world_models_2018.utils import save_transition_prediction_grid

# Data loading functions
def get_data_loader(
    dataset: grain.MapDataset,
    *,
    batch_size: int,
    image_size: tuple[int, int],
    shuffle: bool,
    seed: int,
) -> grain.MapDataset:
    pipeline = dataset.map(ResizeFrame(image_size=image_size)).map(
        RescaleFrame(scale=255.0)
    )
    if shuffle:
        pipeline = pipeline.shuffle(seed=seed)
    return pipeline.batch(batch_size=batch_size, drop_remainder=True)


def make_checkpoint_manager(
    directory: str | Path, *, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    """Create an Orbax ``CheckpointManager`` rooted at ``directory``."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(directory, options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager, model: WorldModel, step: int
) -> None:
    """Save the model's ``nnx`` state under ``step`` using Orbax."""
    state = nnx.state(model)
    manager.save(step, args=ocp.args.StandardSave(state))


# Loss functions
def world_model_forward_and_loss(
    world_model: WorldModel, z: jax.Array, a: jax.Array, carry: Tuple[jax.Array, jax.Array] | None, labels: jax.Array
) -> tuple[jax.Array, dict[str, jax.Array]]:
    carry, (pi_logits, mu, log_sigma) = world_model(z, a, initial_carry=carry, return_carry=True)

    loss = -world_model.log_prob(labels, (pi_logits, mu, log_sigma)).mean()
    return loss, {
        "loss": loss,
    }


@nnx.jit
def train_step(
    vae: VAE,
    world_model: WorldModel,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    a: jax.Array,
) -> dict[str, jax.Array]:
    z = encode_batch(vae, x) #jnp.asarray(batch["frames"]))

    z_in = z[:, :-1, ...]
    labels = z[:, 1:, ...]
    # a = jnp.asarray(batch["actions"])
    carry = None

    grad_fn = nnx.value_and_grad(world_model_forward_and_loss, has_aux=True)
    (_, metrics), grads = grad_fn(world_model, z_in, a, carry, labels)
    optimizer.update(world_model, grads)
    return metrics


@nnx.jit
def eval_step(
    vae: VAE,
    world_model: WorldModel,
    x: jax.Array,
    a: jax.Array,
) -> dict[str, jax.Array]:
    z = encode_batch(vae, x)

    z_in = z[:, :-1, ...]
    labels = z[:, 1:, ...]
    carry = None

    (_, metrics) = world_model_forward_and_loss(world_model, z_in, a, carry, labels)
    return metrics


def metrics_to_floats(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def log_validation_images(
    vae: VAE,
    world_model: WorldModel,
    validation_frames: np.ndarray,
    validation_actions: np.ndarray,
    step: int | str,
    *,
    max_items: int = 8,
) -> None:
    if validation_frames.ndim != 5:
        raise ValueError(
            "validation_frames must have shape "
            "(batch, seq, height, width, channels); "
            f"got {validation_frames.shape}."
        )
    if validation_actions.ndim != 3:
        raise ValueError(
            "validation_actions must have shape (batch, seq, action_dim); "
            f"got {validation_actions.shape}."
        )

    input_frames = validation_frames[:, :-1]
    next_frames = validation_frames[:, 1:]
    actions = validation_actions[:, : input_frames.shape[1]]

    input_frames_flat = rearrange(input_frames, "b t h w c -> (b t) h w c")
    next_frames_flat = rearrange(next_frames, "b t h w c -> (b t) h w c")
    actions = jnp.asarray(actions)

    z, _, _ = vae.encoder(jnp.asarray(input_frames_flat))
    reconstructions = vae.decoder(z)
    z = rearrange(z, "(b t) d -> b t d", b=validation_frames.shape[0])

    _, mdn_params = world_model(
        z,
        actions,
        initial_carry=None,
        return_carry=True,
    )
    pi_logits, mu, _ = mdn_params
    mixture_weights = jax.nn.softmax(pi_logits, axis=-1)
    predicted_z = jnp.sum(mixture_weights[..., None] * mu, axis=-2)
    predicted_frames = vae.decoder(rearrange(predicted_z, "b t d -> (b t) d"))

    num_items = min(max_items, input_frames_flat.shape[0])
    step_label = f"{step}" if isinstance(step, str) else f"{step:06d}"
    rng = np.random.default_rng(0)
    if num_items <= input_frames.shape[0]:
        batch_indices = rng.choice(input_frames.shape[0], size=num_items, replace=False)
        timestep_indices = rng.integers(input_frames.shape[1], size=num_items)
        logging_indices = batch_indices * input_frames.shape[1] + timestep_indices
    else:
        logging_indices = rng.choice(
            input_frames_flat.shape[0], size=num_items, replace=False
        )

    grid_path = Path(
        f"data/validation_images/transition_prediction_grid_step_{step_label}.png"
    )
    save_transition_prediction_grid(
        np.asarray(input_frames_flat[logging_indices]),
        np.asarray(reconstructions[logging_indices]),
        np.asarray(next_frames_flat[logging_indices]),
        np.asarray(predicted_frames[logging_indices]),
        grid_path,
        title=f"transition predictions (step {step_label})",
    )
    mlflow.log_artifact(str(grid_path), artifact_path="validation_images")


@nnx.jit
def encode_batch(vae: VAE, x: jax.Array) -> jax.Array:
    x = rearrange(x, "... h w c -> (...) h w c")
    z, _, _ = vae.encoder(x)
    z = rearrange(z, "(b t) ... -> b t ...", b=batch_size)
    return z


def validate(
    vae: VAE,
    world_model: WorldModel,
    validation_dataloader: grain.MapDataset,
) -> dict[str, float]:
    aggregated: dict[str, float] = {}

    num_batches = 0
    for batch in validation_dataloader:
        metrics = eval_step(vae, world_model, jnp.asarray(batch["frames"]), jnp.asarray(batch["actions"]))
        metrics_float = metrics_to_floats(metrics)
        for name, value in metrics_float.items():
            aggregated[f"val_{name}"] = aggregated.get(f"val_{name}", 0.0) + value
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataset produced no batches.")
    return {name: value / num_batches for name, value in aggregated.items()}


if __name__ == "__main__":
    source = CarRacingDataSource(
        observations_path="data/100_rolls_sequence/car_racing_observations.npz",
        actions_path="data/100_rolls_sequence/car_racing_actions.npz",
        dtype=np.float32,
    )

    validation_fraction = 0.2
    train_dataset, validation_dataset = split_source(
        source, validation_fraction=validation_fraction, seed=0
    )

    image_size = (64, 64)
    batch_size = 16
    learning_rate = 1e-3
    latent_dim = 32
    num_epochs = 50
    validation_every = 2
    image_channels = int(source.episode_observations[0].shape[-1])

    train_loader = get_data_loader(
        train_dataset,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1,
    )
    validation_loader = get_data_loader(
        validation_dataset,
        batch_size=8, #batch_size,
        image_size=image_size,
        shuffle=False,
        seed=0,
    )
    validation_batch_for_logging = next(iter(validation_loader))
    validation_frames_for_logging = np.asarray(validation_batch_for_logging["frames"])
    validation_actions_for_logging = np.asarray(validation_batch_for_logging["actions"])

    vae = VAE(
        image_channels=image_channels,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0, noise=1),
    )
    vae_checkpoint_dir = Path("data/checkpoints/vae")
    vae_checkpoint_manager = make_checkpoint_manager(vae_checkpoint_dir, max_to_keep=3)
    restore_vae_checkpoint(vae_checkpoint_manager, vae)
    vae_checkpoint_manager.close()

    world_model = WorldModel(
        in_features=latent_dim + 3,
        hidden_features=256,
        out_features=latent_dim,
        num_gaussian_components=5,
        temperature=1.0,
        rngs=nnx.Rngs(2, noise=3),
    )
    optimizer = nnx.Optimizer(world_model, optax.adam(learning_rate), wrt=nnx.Param)
    world_model_checkpoint_dir = Path("data/checkpoints/world_model")
    world_model_checkpoint_manager = make_checkpoint_manager(world_model_checkpoint_dir, max_to_keep=3)

    mlflow.set_experiment("ha_schmidhuber_world_models_2018")
    with mlflow.start_run(run_name="world_model_train_on_car_racing_dataset", log_system_metrics=True):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "train_size": len(train_dataset),
                "validation_size": len(validation_dataset),
                "validation_fraction": validation_fraction,
                "latent_dim": latent_dim,
                "image_height": image_size[0],
                "image_width": image_size[1],
                "image_channels": image_channels,
            }
        )

        global_step = 0
        for epoch in tqdm(range(1, num_epochs + 1), desc="epoch"):
            for batch in train_loader:
                metrics = train_step(
                    vae, world_model, optimizer, jnp.asarray(batch["frames"]), jnp.asarray(batch["actions"])
                )
                train_metrics = metrics_to_floats(metrics)
                global_step += 1
                mlflow.log_metrics(train_metrics, step=global_step)


            if global_step % validation_every == 0:
                loader = get_data_loader(
                    validation_dataset,
                    batch_size=batch_size,
                    image_size=image_size,
                    shuffle=False,
                    seed=0,
                )
                validation_metrics = validate(
                    vae,
                    world_model,
                    loader,
                )
                mlflow.log_metrics(validation_metrics, step=global_step)

                log_validation_images(
                    vae,
                    world_model,
                    validation_frames_for_logging,
                    validation_actions_for_logging,
                    global_step,
                )

            save_checkpoint(world_model_checkpoint_manager, world_model, global_step)

        save_checkpoint(world_model_checkpoint_manager, world_model, global_step)
        world_model_checkpoint_manager.wait_until_finished()
        mlflow.log_artifacts(
            str(world_model_checkpoint_dir), artifact_path="checkpoints/world_model"
        )
        world_model_checkpoint_manager.close()