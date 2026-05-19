from collections.abc import Mapping
from pathlib import Path

import grain.python as grain
import jax.image
import numpy as np


class ResizeFrame(grain.MapTransform):
    """Resize frames in an example to ``image_size`` using bilinear interpolation.

    Works on either a single frame ``(H, W, C)`` or a stack of frames
    ``(..., H, W, C)``, so it can be applied to trajectories.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        *,
        method: str = "bilinear",
        dtype: np.dtype = np.float32,
        frame_keys: tuple[str, ...] = ("frames",),
    ) -> None:
        self.image_size = image_size
        self.method = method
        self.dtype = dtype
        self.frame_keys = frame_keys

    def _resize(self, frames: np.ndarray) -> np.ndarray:
        leading_shape = frames.shape[:-3]
        target_shape = (*leading_shape, *self.image_size, frames.shape[-1])
        if frames.shape == target_shape:
            return np.asarray(frames, dtype=self.dtype)
        resized = jax.image.resize(frames, target_shape, method=self.method)
        return np.asarray(resized, dtype=self.dtype)

    def map(
        self, example: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        result = dict(example)
        for key in self.frame_keys:
            result[key] = self._resize(result[key])
        return result


class RescaleFrame(grain.MapTransform):
    """Rescale frame values to ``[0, 1]`` by dividing by ``scale`` (default 255)."""

    def __init__(
        self,
        *,
        scale: float = 255.0,
        dtype: np.dtype = np.float32,
        frame_keys: tuple[str, ...] = ("frames",),
    ) -> None:
        self.scale = scale
        self.dtype = dtype
        self.frame_keys = frame_keys

    def map(
        self, example: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        result = dict(example)
        divisor = self.dtype(self.scale)
        for key in self.frame_keys:
            result[key] = np.asarray(result[key], dtype=self.dtype) / divisor
        return result


def _episode_index(key: str) -> int:
    return int(key.rsplit("_", 1)[-1])


class CarRacingDataSource:
    """Random-access data source over pre-recorded CarRacing trajectories.

    The data is loaded from per-episode ``.npz`` archives produced by
    ``collect_data.py``. Each archive contains one array per episode, keyed
    ``episode_0``, ``episode_1``, ... Each item returned is a full trajectory
    ``{"frames": (T + 1, H, W, C), "actions": (T, A)}`` for one episode, where
    ``frames[t]`` is the observation seen before applying ``actions[t]`` and
    ``frames[t + 1]`` is the resulting next observation.

    Implements grain's ``RandomAccessDataSource`` protocol (``__len__`` and
    ``__getitem__``) so it can be plugged directly into ``grain.MapDataset``.
    """

    def __init__(
        self,
        observations_path: str | Path,
        actions_path: str | Path,
        *,
        dtype: np.dtype = np.float32,
    ) -> None:
        self.dtype = dtype

        with np.load(observations_path) as obs_npz:
            obs_keys = sorted(obs_npz.files, key=_episode_index)
            self.episode_observations: list[np.ndarray] = [
                np.asarray(obs_npz[k]) for k in obs_keys
            ]
        with np.load(actions_path) as act_npz:
            act_keys = sorted(act_npz.files, key=_episode_index)
            self.episode_actions: list[np.ndarray] = [
                np.asarray(act_npz[k]) for k in act_keys
            ]

        if len(self.episode_observations) != len(self.episode_actions):
            raise ValueError(
                "Mismatched number of episodes between observations "
                f"({len(self.episode_observations)}) and actions "
                f"({len(self.episode_actions)})."
            )
        if not self.episode_observations:
            raise ValueError("No episodes found in the provided archives.")

        episode_lengths: list[int] = []
        for ep_idx, (obs, act) in enumerate(
            zip(self.episode_observations, self.episode_actions)
        ):
            if obs.ndim != 4:
                raise ValueError(
                    f"Episode {ep_idx} observations must have shape "
                    "(num_frames, height, width, channels); "
                    f"got {obs.shape}."
                )
            if act.ndim != 2:
                raise ValueError(
                    f"Episode {ep_idx} actions must have shape "
                    f"(num_actions, action_dim); got {act.shape}."
                )
            if obs.shape[0] != act.shape[0] + 1:
                raise ValueError(
                    f"Episode {ep_idx} expected num_frames == num_actions + 1; "
                    f"got num_frames={obs.shape[0]}, num_actions={act.shape[0]}."
                )
            episode_lengths.append(int(act.shape[0]))

        self.episode_lengths = np.asarray(episode_lengths, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.episode_observations)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        frames = np.asarray(
            self.episode_observations[index], dtype=self.dtype
        ).copy()
        actions = np.asarray(
            self.episode_actions[index], dtype=self.dtype
        ).copy()
        return {"frames": frames, "actions": actions}

    @property
    def num_episodes(self) -> int:
        return len(self.episode_observations)

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.episode_observations[0].shape[1:])

    @property
    def action_shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.episode_actions[0].shape[1:])


def split_source(
    source: CarRacingDataSource,
    validation_fraction: float,
    *,
    seed: int = 0,
) -> tuple[grain.MapDataset, grain.MapDataset]:
    """Deterministically split a data source into train/validation MapDatasets."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    num_items = len(source)
    if num_items < 2:
        raise ValueError("Need at least 2 samples to create train/validation splits.")

    validation_size = max(1, int(num_items * validation_fraction))
    train_size = num_items - validation_size
    if train_size < 1:
        raise ValueError("Training split is empty; collect more data first.")

    shuffled = grain.MapDataset.source(source).shuffle(seed=seed)
    train_dataset = shuffled.slice(slice(0, train_size))
    validation_dataset = shuffled.slice(slice(train_size, num_items))
    return train_dataset, validation_dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    source = CarRacingDataSource(
        observations_path=Path("data/car_racing_observations.npz"),
        actions_path=Path("data/car_racing_actions.npz"),
    )
    print(f"num_episodes={source.num_episodes}")
    print(f"frame_shape={source.frame_shape}, action_shape={source.action_shape}")
    trajectory = source[0]
    print(
        f"trajectory[0]: frames={trajectory['frames'].shape}, "
        f"actions={trajectory['actions'].shape}"
    )

    index_sampler = grain.IndexSampler(
        num_records=8,
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=False,
        seed=0,
    )
    dataset = grain.DataLoader(
        data_source=source,
        sampler=index_sampler,
        operations=[
            ResizeFrame(image_size=(64, 64)),
            RescaleFrame(scale=255.0),
            # grain.Batch(batch_size=2, drop_remainder=True),
        ],
        worker_count=0,
    )

    examples = list(dataset)
    fig, axes = plt.subplots(
        len(examples), 2, figsize=(4, 2 * len(examples)), squeeze=False
    )
    for i, example in enumerate(examples):
        frames = np.clip(example["frames"], 0.0, 1.0)
        actions = example["actions"]
        print(
            f"[{i}] frame={frames[2].shape} next_frame={frames[3].shape} "
            f"action={np.array2string(actions[2], precision=3)}"
        )
        axes[i, 0].imshow(frames[2])
        axes[i, 0].set_title(f"frame\naction={np.array2string(actions[2], precision=2)}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(frames[3])
        axes[i, 1].set_title(f"next_frame")
        axes[i, 1].axis("off")

    output_path = Path("data/examples/car_racing_transitions.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved examples to {output_path}")
