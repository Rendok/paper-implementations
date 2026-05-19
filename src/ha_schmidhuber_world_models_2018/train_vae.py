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

from ha_schmidhuber_world_models_2018.car_racing_dataset import (
    CarRacingDataSource,
    RescaleFrame,
    ResizeFrame,
    split_source,
)
from ha_schmidhuber_world_models_2018.vae import VAE


def vae_loss(
    model: VAE, x: jax.Array, kl_weight: float
) -> tuple[jax.Array, dict[str, jax.Array]]:
    reconstruction, mu, logvar = model(x)

    reconstruction_loss = jnp.sum((reconstruction - x) ** 2, axis=(1, 2, 3)).mean()
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(1.0 + logvar - jnp.square(mu) - jnp.exp(logvar), axis=-1)
    )
    loss = reconstruction_loss + kl_weight * kl_loss
    return loss, {
        "loss": loss,
        "reconstruction_loss": reconstruction_loss,
        "kl_loss": kl_loss,
    }


@nnx.jit
def train_step(
    model: VAE,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    kl_weight: jax.Array,
) -> dict[str, jax.Array]:
    grad_fn = nnx.value_and_grad(vae_loss, has_aux=True)
    (_, metrics), grads = grad_fn(model, x, kl_weight)
    optimizer.update(model, grads)
    return metrics


@nnx.jit
def eval_step(
    model: VAE, x: jax.Array, kl_weight: jax.Array
) -> dict[str, jax.Array]:
    _, metrics = vae_loss(model, x, kl_weight)
    return metrics


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


def save_input_reconstruction_grid(
    inputs: np.ndarray,
    reconstructions: np.ndarray,
    path: str | Path,
    *,
    title: str | None = None,
) -> None:
    """Save paired ``(input, reconstruction)`` images into a single grid PNG.

    Each cell shows the input on top and its reconstruction directly below, so
    pairs can be compared at a glance.
    """
    if inputs.shape != reconstructions.shape:
        raise ValueError(
            f"inputs {inputs.shape} and reconstructions {reconstructions.shape} "
            "must have the same shape."
        )
    if inputs.ndim != 4:
        raise ValueError(
            "inputs must have shape (batch, height, width, channels); "
            f"got {inputs.shape}."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = inputs.shape[0]
    num_cols = int(np.ceil(np.sqrt(batch_size)))
    num_pair_rows = int(np.ceil(batch_size / num_cols))
    fig, axes = plt.subplots(
        2 * num_pair_rows,
        num_cols,
        figsize=(num_cols * 1.5, num_pair_rows * 3.0),
        squeeze=False,
    )
    for i in range(num_pair_rows * num_cols):
        pair_row, col = divmod(i, num_cols)
        ax_input = axes[2 * pair_row, col]
        ax_recon = axes[2 * pair_row + 1, col]
        ax_input.axis("off")
        ax_recon.axis("off")
        if i < batch_size:
            ax_input.imshow(np.clip(inputs[i], 0.0, 1.0))
            ax_recon.imshow(np.clip(reconstructions[i], 0.0, 1.0))
            if pair_row == 0:
                ax_input.set_title("input", fontsize=8)
            if pair_row == num_pair_rows - 1 or i + num_cols >= batch_size:
                ax_recon.set_xlabel("reconstruction", fontsize=8)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def log_validation_images(
    model: VAE, validation_batch: np.ndarray, step: int | str
) -> None:
    if validation_batch.ndim != 4:
        raise ValueError(
            "validation_batch must have shape (batch, height, width, channels); "
            f"got {validation_batch.shape}."
        )

    reconstruction = np.asarray(model(jnp.asarray(validation_batch))[0])
    step_label = f"{step}" if isinstance(step, str) else f"{step:06d}"

    grid_path = Path(
        f"data/validation_images/reconstruction_grid_step_{step_label}.png"
    )
    save_input_reconstruction_grid(
        validation_batch,
        reconstruction,
        grid_path,
        title=f"inputs vs reconstructions (step {step_label})",
    )
    mlflow.log_artifact(str(grid_path), artifact_path="validation_images")


def metrics_to_floats(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def make_checkpoint_manager(
    directory: str | Path, *, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    """Create an Orbax ``CheckpointManager`` rooted at ``directory``."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(directory, options=options)


def save_checkpoint(
    manager: ocp.CheckpointManager, model: VAE, step: int
) -> None:
    """Save the model's ``nnx`` state under ``step`` using Orbax."""
    state = nnx.state(model)
    manager.save(step, args=ocp.args.StandardSave(state))


def restore_checkpoint(
    manager: ocp.CheckpointManager, model: VAE, *, step: int | None = None
) -> None:
    """Restore model parameters from Orbax in-place. Uses the latest step by default."""
    target_step = manager.latest_step() if step is None else step
    if target_step is None:
        raise FileNotFoundError("No checkpoints found to restore.")
    abstract_state = nnx.state(model)
    restored = manager.restore(
        target_step, args=ocp.args.StandardRestore(abstract_state)
    )
    nnx.update(model, restored)


def validate(
    model: VAE,
    validation_dataset: grain.MapDataset,
    *,
    batch_size: int,
    image_size: tuple[int, int],
    kl_weight: float,
) -> dict[str, float]:
    aggregated = {
        "val_loss": 0.0,
        "val_reconstruction_loss": 0.0,
        "val_kl_loss": 0.0,
    }

    loader = get_data_loader(
        validation_dataset,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=0,
    )
    kl_weight_array = jnp.asarray(kl_weight, dtype=jnp.float32)
    num_batches = 0
    for batch in loader:
        metrics = eval_step(model, jnp.asarray(batch["frame"]), kl_weight_array)
        metrics_float = metrics_to_floats(metrics)
        aggregated["val_loss"] += metrics_float["loss"]
        aggregated["val_reconstruction_loss"] += metrics_float["reconstruction_loss"]
        aggregated["val_kl_loss"] += metrics_float["kl_loss"]
        num_batches += 1

    if num_batches == 0:
        raise ValueError("Validation dataset produced no batches.")
    return {name: value / num_batches for name, value in aggregated.items()}


if __name__ == "__main__":
    source = CarRacingDataSource(
        observations_path="data/100_rolls/car_racing_observations.npy",
        actions_path="data/100_rolls/car_racing_actions.npy",
        dtype=np.float32,
    )

    validation_fraction = 0.2
    train_dataset, validation_dataset = split_source(
        source, validation_fraction=validation_fraction, seed=0
    )

    image_size = (64, 64)
    batch_size = 32
    learning_rate = 3e-4
    kl_weight = 1.0
    latent_dim = 32
    num_epochs = 5
    validation_every = 100
    image_channels = int(source.observations.shape[-1])

    train_loader = get_data_loader(
        train_dataset,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=1,
    )
    validation_loader = get_data_loader(
        validation_dataset,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=0,
    )
    validation_batch_for_logging = np.asarray(
        next(iter(validation_loader))["frame"]
    )

    model = VAE(
        image_channels=image_channels,
        latent_dim=latent_dim,
        rngs=nnx.Rngs(0, noise=1),
    )
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    kl_weight_array = jnp.asarray(kl_weight, dtype=jnp.float32)

    checkpoint_dir = Path("data/checkpoints/vae")
    checkpoint_manager = make_checkpoint_manager(checkpoint_dir, max_to_keep=3)

    mlflow.set_experiment("ha_schmidhuber_world_models_2018")
    with mlflow.start_run(run_name="vae_train_on_car_racing_dataset", log_system_metrics=True):
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "kl_weight": kl_weight,
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
                    model, optimizer, jnp.asarray(batch["frame"]), kl_weight_array
                )
                train_metrics = metrics_to_floats(metrics)
                global_step += 1
                mlflow.log_metrics(train_metrics, step=global_step)

                if global_step % validation_every == 0:
                    validation_metrics = validate(
                        model,
                        validation_dataset,
                        batch_size=batch_size,
                        image_size=image_size,
                        kl_weight=kl_weight,
                    )
                    mlflow.log_metrics(validation_metrics, step=global_step)
                    log_validation_images(
                        model, validation_batch_for_logging, global_step
                    )
            
            save_checkpoint(checkpoint_manager, model, global_step)

        log_validation_images(model, validation_batch_for_logging, "final")
        save_checkpoint(checkpoint_manager, model, global_step)
        checkpoint_manager.wait_until_finished()
        mlflow.log_artifacts(str(checkpoint_dir), artifact_path="checkpoints")
        checkpoint_manager.close()
