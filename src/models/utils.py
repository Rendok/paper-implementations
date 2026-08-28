import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class PositionEmbedding(nnx.Module):
    """Fixed sinusoidal embeddings for positions or diffusion timesteps."""

    def __init__(
        self,
        hidden_dim: int,
        max_len: int,
        *,
        base: float = 10_000.0,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        assert hidden_dim > 0
        assert max_len > 0
        assert base > 0

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.base = float(base)
        self.dtype = dtype

        self.P = jnp.zeros((1, self.max_len, self.hidden_dim), dtype=dtype)
        X = jnp.arange(self.max_len, dtype=jnp.float32).reshape(-1, 1) / jnp.power(
            self.base,
            jnp.arange(0, self.hidden_dim, 2, dtype=jnp.float32) / self.hidden_dim,
        )
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X).astype(dtype))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X).astype(dtype))

    def __call__(
        self, x: Float[Array, "batch seq hidden"]
    ) -> Float[Array, "batch seq hidden_dim"]:
        return x + self.P[:, : x.shape[1], :]


class TimeEmbedding(nnx.Module):
    """Sinusoidal embedding of a continuous time t in [0, 1], then an MLP.

    Unlike PositionEmbedding there is no lookup table: t is continuous, so the
    sinusoids are evaluated directly.

    t is rescaled by ``scale`` first. With the usual base=10_000 frequencies the
    largest angle is ``t * scale``, so raw t in [0, 1] would keep every feature
    inside the first radian of sin/cos — all monotonic and nearly linear, so
    neighbouring times end up with almost identical embeddings. Multiplying by
    1000 recovers the resolution of the discrete t = 0…1000 convention.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        scale: float = 1000.0,
        base: float = 10_000.0,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        assert hidden_dim > 0 and hidden_dim % 2 == 0  # splits evenly into sin/cos
        assert scale > 0
        assert base > 0

        self.hidden_dim = hidden_dim
        self.scale = float(scale)
        self.dtype = dtype

        self.inv_freq = jnp.power(
            base, -jnp.arange(0, hidden_dim, 2, dtype=jnp.float32) / hidden_dim
        )

    def __call__(self, t: Float[Array, "batch"]) -> Float[Array, "batch hidden_dim"]:
        """Embed t in [0, 1]. A scalar t yields (hidden_dim,), which broadcasts
        against per-sample conditioning such as class embeddings."""
        t = jnp.asarray(t, dtype=jnp.float32)
        angles = t[..., None] * self.scale * self.inv_freq
        features = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return features.astype(self.dtype)


if __name__ == "__main__":
    emb = PositionEmbedding(512, 128)
    x = jnp.ones((8, 32, 512))
    print(emb(x).shape)

    t_emb = TimeEmbedding(512, dtype=jnp.bfloat16)
    print(t_emb(jnp.linspace(0.0, 1.0, 8)).shape)
    print(t_emb(jnp.asarray(0.3)).shape)
