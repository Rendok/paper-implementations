import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import grain
import jax
import jax.numpy as jnp
from absl import flags
import functools as ft
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optax
from tqdm import tqdm
from flax import nnx
from jaxtyping import Array, Float, Integer

from datasets import load_dataset
from PIL import Image

from tokenizers.linear_embedder import LinearEmbedder, LinearEmbedderConfig
from models.dit import DiT, DiTConfig
from models.samplers import stochastic_sampler, euler_sampler

# When mp_prefetch spawns workers, grain reads the absl flag
# --grain_enable_multiprocess_worker_profiling (grain/_src/core/profiler.py).
# Launching with plain `python` never parses absl flags, so that read raises
# UnparsedFlagAccessError. Marking them parsed resolves every absl flag to its
# default, which is all grain wants here. Module level, not inside train(), so
# spawned workers re-running this import are covered too.
flags.FLAGS.mark_as_parsed()

# MLflow's default tracking URI is sqlite:///<cwd>/mlflow.db — resolved against
# the *current working directory*. Launching this script from src/training_scripts
# therefore creates a second, empty database there rather than using the one
# `mlflow ui` serves from the repo root, and the run silently never appears in
# the UI. Pin both stores to absolute paths so the launch directory is irrelevant.
REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACT_URI = f"file://{REPO_ROOT / 'mlruns'}"


def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    # The artifact root is likewise resolved against the cwd, but only once — at
    # experiment-creation time — so set it explicitly or sample grids end up
    # beside whichever directory first created the experiment.
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(experiment_name)


@dataclass(slots=True, frozen=True)
class TrainConfig:
    total_steps: int
    warmup_steps: int
    learning_rate: float
    adaptive_grad_clip_threshold: float
    guidance_dropout_prob: float  # CFG
    guidance_scale: float
    dit_config: DiTConfig
    emb_config: LinearEmbedderConfig
    batch_size: int
    log_every: int
    eval_every: int
    eval_batch_size: int
    eval_num_steps: int
    eval_sigma: float
    eval_seed: int


def to_numpy(sample: dict, *, size: int) -> dict:
    # Imagenette images are 160px on the short side with varying width, and a
    # few ImageNet files are grayscale or CMYK, so normalize mode and shape
    # before batching.
    image = sample["image"].convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    image = np.asarray(image, dtype=np.float32) / 255.0
    sample["image"] = image * 2.0 - 1.0  # [-1, 1]
    return sample


def metrics_to_floats(metrics: dict[str, jax.Array]) -> dict[str, float]:
    return {name: float(value) for name, value in metrics.items()}


def config_to_params(train_config: TrainConfig) -> dict[str, object]:
    """Flatten the nested dataclasses into the flat dict mlflow.log_params wants."""
    params: dict[str, object] = {}
    for field in fields(train_config):
        value = getattr(train_config, field.name)
        if field.name in ("dit_config", "emb_config"):
            prefix = field.name.removesuffix("_config")
            for sub in fields(value):
                params[f"{prefix}.{sub.name}"] = str(getattr(value, sub.name))
        else:
            params[field.name] = value
    return params


def save_sample_grid(
    images: np.ndarray,  # (batch, H, W, C) in [0, 1]
    path: Path,
    *,
    title: str | None = None,
    num_cols: int = 4,
) -> None:
    batch_size = images.shape[0]
    num_cols = min(num_cols, batch_size)
    num_rows = int(np.ceil(batch_size / num_cols))

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(num_cols * 1.6, num_rows * 1.6), squeeze=False
    )
    for idx in range(num_rows * num_cols):
        ax = axes[idx // num_cols][idx % num_cols]
        ax.axis("off")
        if idx < batch_size:
            image = images[idx]
            if image.shape[-1] == 1:
                image = image.squeeze(-1)
            ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def log_sample_images(model: nnx.Module, train_config: TrainConfig, step: int) -> None:
    """Sample from the model and log the grid to mlflow as an artifact."""
    batch_size = train_config.eval_batch_size
    samples = euler_sampler(
        model,
        batch_size,
        train_config.emb_config.imgage_size,
        # Cycle through the digits so one grid shows conditioning working.
        jnp.arange(batch_size, dtype=jnp.int32) % train_config.dit_config.num_classes,
        rngs=nnx.Rngs(train_config.eval_seed),
        num_steps=train_config.eval_num_steps,
        sigma=train_config.eval_sigma,
        comp_dtype=train_config.dit_config.comp_dtype,
        guidance_strength=train_config.guidance_scale,
    )

    # Training data lives in [-1, 1]; map back to [0, 1] for display.
    images = np.asarray(jax.device_get(samples), dtype=np.float32)
    images = np.clip((images + 1.0) / 2.0, 0.0, 1.0)

    # TemporaryDirectory deletes the png and its folder on exit, including when
    # log_artifact raises, so nothing accumulates on disk between evaluations.
    with tempfile.TemporaryDirectory() as tmp_dir:
        grid_path = Path(tmp_dir) / f"samples_step_{step:06d}.png"
        save_sample_grid(images, grid_path, title=f"samples (step {step})")
        mlflow.log_artifact(str(grid_path), artifact_path="samples")


def score_matching_loss_fn(
    model: nnx.Module,
    z: Float[Array, "batch H W C"],
    labels: Integer[Array, "batch"],
    rngs: nnx.Rngs,
):
    "Uses reparametrization -beta * score = eps for numerical stability."
    # float32 time: bfloat16 gives only 128 distinct values on [0, 1).
    t = rngs.uniform((z.shape[0],), jnp.float32, 0, 1)
    noise = rngs.normal(z.shape, z.dtype)
    t_b = t.astype(z.dtype)[:, None, None, None]
    x_t = t_b * z + (1 - t_b) * noise

    # eps-prediction: equivalent to sigma^2-weighted score matching, and unlike
    # the raw score target -noise/(1-t) its second moment stays at 1 for all t.
    eps_hat = model(x_t, t, labels)
    loss = jnp.mean(
        optax.l2_loss(eps_hat.astype(jnp.float32), noise.astype(jnp.float32))
    )
    return loss, {"l2_loss": loss, "eps_pred_std": jnp.std(eps_hat.astype(jnp.float32))}


def flow_model_loss_fn(
    model: nnx.Module,
    z: Float[Array, "batch H W C"],
    labels: Integer[Array, "batch"],
    rngs: nnx.Rngs,
    dropout_prob: float,
    empty_token_id: int,
):
    "With CFG."
    t = rngs.uniform((z.shape[0],), jnp.float32, 0, 1)
    noise = rngs.normal(z.shape, z.dtype)
    t_b = t.astype(z.dtype)[:, None, None, None]
    x_t = t_b * z + (1 - t_b) * noise

    p = rngs.uniform((z.shape[0],), jnp.bfloat16, 0, 1)
    labels = jnp.where(p > dropout_prob, labels, empty_token_id)

    u_target = z - noise
    loss = jnp.mean(
        optax.l2_loss(
            model(x_t, t, labels).astype(jnp.float32), u_target.astype(jnp.float32)
        )
    )

    return loss, {"l2_loss": loss}


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    z: Float[Array, "batch H W C"],
    labels: Integer[Array, "batch"],
    rngs: nnx.Rngs,
):
    grads, metrics = nnx.grad(
        ft.partial(
            flow_model_loss_fn,
            dropout_prob=0.3,
            empty_token_id=model.class_embeddings.num_embeddings - 1,
        ),
        has_aux=True,
    )(model, z, labels, rngs)

    grads_leaves = jax.tree_util.tree_leaves(nnx.state(grads))
    print(f"{grads_leaves[0].dtype = }")
    metrics["grad_norm"] = jnp.sqrt(
        sum([jnp.vdot(g, g) for g in grads_leaves if g is not None])
    )

    optimizer.update(model, grads)

    return metrics


def train(train_config):

    ### Dataset ###

    # hf_dataset = load_dataset("ylecun/mnist")  # size=28x28
    # Imagenette: 10 ImageNet classes (tench, English springer, cassette player,
    # chain saw, church, French horn, garbage truck, gas pump, golf ball,
    # parachute), labels 0-9, ~107 MB. imagenet-1k itself can't be fetched per
    # class: its parquet shards mix all 1000 classes, so any class filter still
    # downloads the full ~150 GB.
    hf_dataset = load_dataset("ShaomuTan/imagenette")
    hf_train, hf_test = hf_dataset["train"], hf_dataset["validation"]

    dataset = (
        grain.MapDataset.source(hf_train)
        .shuffle(seed=42)
        .map(ft.partial(to_numpy, size=train_config.emb_config.imgage_size[0]))
        .repeat()
        .to_iter_dataset()
        .batch(train_config.batch_size)
    )

    # print(dataset)

    performance_config = grain.experimental.pick_performance_config(
        ds=dataset, ram_budget_mb=1024, max_workers=None, max_buffer_size=None
    )

    dataset = dataset.mp_prefetch(
        performance_config.multiprocessing_options,
    )

    ### Model ###

    emb = LinearEmbedder(train_config.emb_config, rngs=nnx.Rngs(0))
    dit = DiT(train_config.dit_config, image_embedder=emb, rngs=nnx.Rngs(1))

    ### Optimizer ###

    lr_schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        train_config.learning_rate,
        train_config.warmup_steps,
        train_config.total_steps,
    )

    optimizer = optax.chain(
        optax.adaptive_grad_clip(clipping=train_config.adaptive_grad_clip_threshold),
        optax.adamw(lr_schedule),
    )

    optimizer = nnx.Optimizer(dit, optimizer, wrt=nnx.Param)

    ### Rngs ###

    rngs = nnx.Rngs(0)

    ### Trian ###

    data_iter = iter(dataset)

    setup_mlflow("dit")
    with mlflow.start_run(run_name="dit_imagenet_10_cfg", log_system_metrics=True):
        mlflow.log_params(config_to_params(train_config))

        for i in tqdm(range(1, train_config.total_steps + 1)):
            #     if i == 2:
            #         jax.profiler.start_trace("/tmp/profile-data")

            #     with jax.profiler.StepTraceAnnotation("train", step_num=i):
            #         with jax.profiler.TraceAnnotation("data_load"):
            batch = next(data_iter)
            x = jnp.asarray(batch["image"], dtype=train_config.dit_config.comp_dtype)
            labels = jnp.asarray(batch["label"], dtype=jnp.int32)

            metrics = train_step(dit, optimizer, x, labels, rngs)

            if i % train_config.log_every == 0:
                mlflow.log_metrics(
                    {
                        f"train/{name}": value
                        for name, value in metrics_to_floats(metrics).items()
                    },
                    step=i,
                )

            if i % train_config.eval_every == 0 or i == train_config.total_steps:
                log_sample_images(dit, train_config, i)

        # jax.block_until_ready(metrics)
        # jax.profiler.stop_trace()


if __name__ == "__main__":
    dit_config = DiTConfig(
        num_classes=11,
        max_seq_len=(64 // 4) ** 2,  # (image side / patch_size)^2 tokens
        num_layers=12,
        num_q_heads=12,
        num_kv_heads=4,
        hidden_dim=768,
        comp_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )

    emb_config = LinearEmbedderConfig(
        imgage_size=(64, 64, 3),
        patch_size=4,
        hidden_dim=768,
        comp_dtype=jnp.bfloat16,
        param_dtype=jnp.float32,
    )

    train_config = TrainConfig(
        total_steps=100_000,
        warmup_steps=500,
        learning_rate=3e-4,
        adaptive_grad_clip_threshold=0.01,
        guidance_dropout_prob=0.3,
        guidance_scale=4.0,
        batch_size=128,
        log_every=10,
        eval_every=500,
        eval_batch_size=11,
        eval_num_steps=200,
        eval_sigma=0.1,
        eval_seed=1234,
        dit_config=dit_config,
        emb_config=emb_config,
    )

    # print(train_config)
    train(train_config)
