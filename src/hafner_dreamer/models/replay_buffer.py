import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float, UInt8


class SequenceReplayBuffer:
    """Circular episode buffer for RSSM / sequence-model training.

    Stores full episodes as variable-length sequences of
    (image, action, reward, continue) and samples fixed-length
    sub-sequences uniformly at training time.

    Storage layout (numpy, on CPU):
        images   : uint8   (T, H, W, C)   — raw pixel values [0, 255]
        actions  : float32 (T, action_dim)
        rewards  : float32 (T,)
        continues: float32 (T,)            — 1 = episode continues, 0 = done

    ``sample`` normalises images to float32 [0, 1] before returning.
    """

    def __init__(self, capacity: int, rngs: nnx.Rngs) -> None:
        self.capacity = capacity
        self._episodes: list[dict[str, np.ndarray]] = []
        self._write_idx: int = 0
        self.rngs = rngs

    def __len__(self) -> int:
        return len(self._episodes)

    def add_episode(
        self,
        images: np.ndarray,    # (T, H, W, C) uint8
        actions: np.ndarray,   # (T, action_dim) float32
        rewards: np.ndarray,   # (T,) float32
        continues: np.ndarray, # (T,) float32  1=continue, 0=done
    ) -> None:
        """Append one episode; overwrites the oldest on overflow."""
        T = images.shape[0]
        if not (T == actions.shape[0] == rewards.shape[0] == continues.shape[0]):
            raise ValueError(
                f"All arrays must share episode length T; got "
                f"images={images.shape[0]}, actions={actions.shape[0]}, "
                f"rewards={rewards.shape[0]}, continues={continues.shape[0]}"
            )
        ep = {
            "images": np.asarray(images, dtype=np.uint8),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "continues": np.asarray(continues, dtype=np.float32),
        }
        if len(self._episodes) < self.capacity:
            self._episodes.append(ep)
        else:
            self._episodes[self._write_idx] = ep
        self._write_idx = (self._write_idx + 1) % self.capacity

    def sample(
        self, batch_size: int, seq_len: int
    ) -> dict[str, jax.Array]:
        """Sample *batch_size* sub-sequences of length *seq_len*.

        Returns a dict with keys ``images`` (float32 [0,1]),
        ``actions``, ``rewards``, ``continues``.
        Each value has shape ``(batch_size, seq_len, ...)``.
        """
        if not self._episodes:
            raise ValueError("Cannot sample from an empty SequenceReplayBuffer.")
        valid = [ep for ep in self._episodes if ep["images"].shape[0] >= seq_len]
        if not valid:
            raise ValueError(
                f"No stored episode has at least {seq_len} steps. "
                f"Longest episode: {max(ep['images'].shape[0] for ep in self._episodes)}"
            )

        key = self.rngs.noise()
        k1, k2 = jax.random.split(key)
        ep_idx = np.asarray(jax.random.randint(k1, (batch_size,), 0, len(valid)))
        starts = np.array([
            int(jax.random.randint(k2, (), 0, valid[i]["images"].shape[0] - seq_len + 1))
            for i in ep_idx
        ])

        slices = [
            {k: v[starts[b] : starts[b] + seq_len] for k, v in valid[ep_idx[b]].items()}
            for b in range(batch_size)
        ]
        batch = {k: np.stack([s[k] for s in slices]) for k in slices[0]}

        return {
            "images": jnp.asarray(batch["images"], dtype=jnp.float32) / 255.0,
            "actions": jnp.asarray(batch["actions"]),
            "rewards": jnp.asarray(batch["rewards"]),
            "continues": jnp.asarray(batch["continues"]),
        }


if __name__ == "__main__":
    rngs = nnx.Rngs(0, noise=1)
    buf = SequenceReplayBuffer(capacity=10, rngs=rngs)
    assert len(buf) == 0

    rng = np.random.default_rng(42)
    for _ in range(3):
        T = rng.integers(20, 30)
        buf.add_episode(
            images=rng.integers(0, 255, (T, 64, 64, 3), dtype=np.uint8),
            actions=rng.random((T, 3)).astype(np.float32),
            rewards=rng.random((T,)).astype(np.float32),
            continues=np.ones((T,), dtype=np.float32),
        )
    assert len(buf) == 3

    batch = buf.sample(batch_size=4, seq_len=16)
    assert batch["images"].shape == (4, 16, 64, 64, 3)
    assert batch["actions"].shape == (4, 16, 3)
    assert batch["rewards"].shape == (4, 16)
    assert batch["continues"].shape == (4, 16)
    assert float(batch["images"].max()) <= 1.0
    assert float(batch["images"].min()) >= 0.0
    print("SequenceReplayBuffer OK")
