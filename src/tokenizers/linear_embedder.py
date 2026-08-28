import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from typing import Tuple
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class LinearEmbedderConfig:
    imgage_size: Tuple[int, int, int]
    patch_size: int
    hidden_dim: int
    comp_dtype: jnp.dtype
    param_dtype: jnp.dtype


class LinearEmbedder(nnx.Module):
    "Image -> Velocity (unbounded)"
    def __init__(
        self,
        config: LinearEmbedderConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        assert config.imgage_size[0] % config.patch_size == 0
        assert config.imgage_size[1] % config.patch_size == 0
        self.config = config

        self.patch_embeddings = nnx.Conv(
            config.imgage_size[2],
            config.hidden_dim,
            kernel_size=(config.patch_size, config.patch_size),
            strides=(config.patch_size, config.patch_size),
            padding="VALID",
            use_bias=True,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
            rngs=rngs,
        )

        self.norm = nnx.RMSNorm(
            config.hidden_dim,
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
        )

        self.out_proj = nnx.Linear(
            config.hidden_dim,
            config.imgage_size[2] * config.patch_size * config.patch_size,
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
            use_bias=True,
        )

    def __call__(
        self, x: Float[Array, "batch H W C"]
    ) -> Float[Array, "batch H W C"]:
        return self.decode(self.encode(x))

    def encode(
        self, x: Float[Array, "batch H W C"]
    ) -> Float[Array, "batch seq hidden"]:
        return self.patch_embeddings(x).reshape(
            (x.shape[0], -1, self.config.hidden_dim)
        )

    def decode(
        self, x: Float[Array, "batch seq hidden"]
    ) -> Float[Array, "batch H W C"]:
        x = self.out_proj(self.norm(x)).reshape(
            (x.shape[0], *self.config.imgage_size)
        )
        return x


if __name__ == "__main__":
    conf = LinearEmbedderConfig((256, 128, 3), 16, 512, jnp.bfloat16, jnp.bfloat16)
    x = jax.random.normal(jax.random.key(0), (8, 256, 128, 3))

    emb = LinearEmbedder(conf, rngs=nnx.Rngs(1))
    print(nnx.tabulate(emb, x))
