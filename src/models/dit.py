from dataclasses import dataclass

import jax
import jax.numpy as jnp
import functools as ft
from flax import nnx
from jaxtyping import Array, Float, Integer

from models.utils import TimeEmbedding


@dataclass(slots=True, frozen=True)
class DiTConfig:
    num_classes: int
    max_seq_len: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    hidden_dim: int
    comp_dtype: jnp.dtype
    param_dtype: jnp.dtype


class DiTBlock(nnx.Module):
    def __init__(
        self,
        config: DiTConfig,
        *,
        rope: nnx.RoPE,
        rngs: nnx.Rngs,
    ) -> None:
        self.rms_norm1 = nnx.RMSNorm(
            config.hidden_dim,
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
            use_scale=False,
        )

        self.attention = nnx.MultiHeadAttention(
            num_heads=config.num_q_heads,
            in_features=config.hidden_dim,
            qkv_features=config.hidden_dim,
            num_kv_heads=config.num_kv_heads,
            attention_fn=ft.partial(nnx.dot_product_attention_with_rope, rope=rope),
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
            use_bias=False,
            decode=False,
        )

        self.rms_norm2 = nnx.RMSNorm(
            config.hidden_dim,
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
            use_scale=False,
        )

        self.mlp = nnx.Sequential(
            nnx.Linear(
                config.hidden_dim,
                4 * config.hidden_dim,
                rngs=rngs,
                use_bias=False,
                dtype=config.comp_dtype,
                param_dtype=config.param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                4 * config.hidden_dim,
                config.hidden_dim,
                rngs=rngs,
                use_bias=False,
                dtype=config.comp_dtype,
                param_dtype=config.param_dtype,
            ),
        )

        self.adaLN = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                config.hidden_dim,
                6 * config.hidden_dim,
                rngs=rngs,
                use_bias=True,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                dtype=config.comp_dtype,
                param_dtype=config.param_dtype,
            ),
        )

    def __call__(
        self, x: Float[Array, "batch seq hidden"], c: Float[Array, "batch hidden"]
    ) -> Float[Array, "batch seq hidden"]:
        params = self.adaLN(c)
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = jnp.split(params, 6, axis=-1)

        # Attn
        h = self.rms_norm1(x)
        h = h * (1 + scale_attn[:, None]) + shift_attn[:, None]
        h = self.attention(h)
        x = x + gate_attn[:, None] * h

        # MLP
        h = self.rms_norm2(x)
        h = h * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        h = self.mlp(h)
        return x + gate_mlp[:, None] * h


class DiT(nnx.Module):
    def __init__(
        self,
        config: DiTConfig,
        *,
        image_embedder: nnx.Module,
        rngs: nnx.Rngs,
    ) -> None:
        assert config.hidden_dim % config.num_q_heads == 0
        assert config.num_q_heads % config.num_kv_heads == 0

        self.config = config

        self.image_embedder = image_embedder
        self.time_embedder = TimeEmbedding(config.hidden_dim, dtype=config.comp_dtype)

        self.class_embeddings = nnx.Embed(
            config.num_classes,
            config.hidden_dim,
            rngs=rngs,
            dtype=config.comp_dtype,
            param_dtype=config.param_dtype,
        )

        self.rope = nnx.RoPE(
            embedding_size=config.hidden_dim // config.num_q_heads,
            max_seq_len=config.max_seq_len,
        )
        self.blocks = nnx.List(
            [
                DiTBlock(config, rope=self.rope, rngs=rngs)
                for _ in range(config.num_layers)
            ]
        )

    def __call__(
        self,
        x: Float[Array, "batch seq hidden"],
        t: Float[Array, "batch"],
        classes: Integer[Array, "batch"],
    ) -> Float[Array, "batch seq hidden"]:
        classes = self.class_embeddings(classes)
        x = self.image_embedder.encode(x)
        c = self.time_embedder(t) + classes

        for block in self.blocks:
            x = block(x, c)
        
        return self.image_embedder.decode(x)


if __name__ == "__main__":
    from tokenizers.linear_embedder import LinearEmbedder, LinearEmbedderConfig

    rngs = nnx.Rngs(0)
    rope = nnx.RoPE(embedding_size=512 // 4, max_seq_len=128)
    config = DiTConfig(5, 128, 2, 4, 2, 512, jnp.bfloat16, jnp.float32)
    block = DiTBlock(
        config=config,
        rope=rope,
        rngs=rngs,
    )

    emb_config = LinearEmbedderConfig(
        (256, 128, 3), 16, 512, jnp.bfloat16, jnp.bfloat16
    )
    emb = LinearEmbedder(emb_config, rngs=nnx.Rngs(2))

    dit = DiT(config, image_embedder=emb, rngs=rngs)

    x = jax.random.uniform(jax.random.key(1), (8, 256, 128, 3), dtype=jnp.bfloat16)
    # c = jax.random.uniform(jax.random.key(2), (8, 512), dtype=jnp.bfloat16)
    y = nnx.jit(dit)(x, t=jnp.asarray([0.3, 0.2, 0.1, 0.5, 0.2, 0.83, 0.9, 0.34]), classes=jnp.zeros((8,), dtype=jnp.int32))
    print(f"output: {y.shape = }, {y.dtype = }")

    # print(jax.jit(block).trace(x).jaxpr)
