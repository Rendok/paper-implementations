from einops import rearrange
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class MDNHead(nnx.Module):
  def __init__(self, in_features: int, out_features: int, num_gaussian_components: int, *, rngs: nnx.Rngs):
    self.num_gaussian_components = num_gaussian_components
    self.out_features = out_features
    # MDN head: K mixture weights + K diagonal Gaussians each over D output dims (mu and log_sigma).
    self.mdn_heads = nnx.Linear(
        in_features=in_features,
        out_features=num_gaussian_components * (1 + 2 * out_features),
        rngs=rngs,
    )

  def __call__(self, x: Float[Array, "... in_features"]) -> tuple[Float[Array, "... K"], Float[Array, "... K D"], Float[Array, "... K D"]]:
    params = self.mdn_heads(x)
    params = rearrange(
        params,
        "... (k params_per_component) -> ... k params_per_component",
        k=self.num_gaussian_components,
    )
    pi_logits = params[..., 0]
    mu = params[..., 1:1 + self.out_features]
    log_var = params[..., 1 + self.out_features:]
    return pi_logits, mu, log_var


class ResidualDownsampleBlock(nnx.Module):
  """ResNet-style block with optional downsampling on the first conv."""

  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], strides: tuple[int, int], *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(
        in_features=in_features,
        out_features=out_features,
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        rngs=rngs,
    )
    self.conv2 = nnx.Conv(
        in_features=out_features,
        out_features=out_features,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="SAME",
        rngs=rngs,
    )
    self.skip = (
        nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )
        if in_features != out_features or strides != (1, 1)
        else None
    )

  def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch height width channels"]:
    residual = x if self.skip is None else self.skip(x)
    x = nnx.relu(self.conv1(x))
    x = self.conv2(x)
    return nnx.relu(x + residual)


class ResidualUpsampleBlock(nnx.Module):
  """ResNet-style block with optional upsampling on the first transpose conv."""

  def __init__(self, in_features: int, out_features: int, kernel_size: tuple[int, int], strides: tuple[int, int], *, rngs: nnx.Rngs):
    self.conv1 = nnx.ConvTranspose(
        in_features=in_features,
        out_features=out_features,
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        rngs=rngs,
    )
    self.conv2 = nnx.Conv(
        in_features=out_features,
        out_features=out_features,
        kernel_size=kernel_size,
        strides=(1, 1),
        padding="SAME",
        rngs=rngs,
    )
    self.skip = (
        nnx.ConvTranspose(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            strides=strides,
            padding="SAME",
            rngs=rngs,
        )
        if in_features != out_features or strides != (1, 1)
        else None
    )

  def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[Array, "batch height width channels"]:
    residual = x if self.skip is None else self.skip(x)
    x = nnx.relu(self.conv1(x))
    x = self.conv2(x)
    return nnx.relu(x + residual)


class MDNEncoder(nnx.Module):
  """Real image encoder with a mixture-density latent head."""
  latent_dim: int
  def __init__(self, in_dim: int, latent_dim: int, memory_dim: int, num_gaussian_components: int, *, rngs: nnx.Rngs):
    self.conv1 = ResidualDownsampleBlock(in_features=in_dim, out_features=32, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv2 = ResidualDownsampleBlock(in_features=32, out_features=64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv3 = ResidualDownsampleBlock(in_features=64, out_features=128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv4 = ResidualDownsampleBlock(in_features=128, out_features=256, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.mdn_head = MDNHead(in_features=256*4*4, out_features=latent_dim, num_gaussian_components=num_gaussian_components, rngs=rngs)
    self.rngs = rngs

  def __call__(
      self, x: Float[Array, "... height width channels"]) -> tuple[
      Float[Array, "... latent_dim"],
      Float[Array, "... K"],
      Float[Array, "... K latent_dim"],
      Float[Array, "... K latent_dim"],
  ]:
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)

    x = rearrange(x, "... h w c -> ... (h w c)")
    # x = jnp.concatenate([x, h], axis=-1)
    pi_logits, mu, log_var = self.mdn_head(x)
    log_var = jnp.clip(log_var, -4.6, 0.0)  # std in [0.1, 1]

    component_idx = jax.random.categorical(self.rngs.noise(), pi_logits, axis=-1)
    component_idx = component_idx[..., None, None]
    chosen_mu = jnp.take_along_axis(mu, component_idx, axis=-2).squeeze(axis=-2)
    chosen_log_var = jnp.take_along_axis(log_var, component_idx, axis=-2).squeeze(axis=-2)
    eps = jax.random.normal(self.rngs.noise(), chosen_mu.shape)
    z = chosen_mu + jnp.exp(0.5 * chosen_log_var) * eps
    return z, pi_logits, mu, log_var


class MDNDecoder(nnx.Module):
  """A decoder that reconstructs 64x64 RGB frames from latent vectors."""
  def __init__(self, latent_dim: int, out_dim: int, *, rngs: nnx.Rngs):
    self.dense1 = nnx.Linear(in_features=latent_dim, out_features=4*4*256, rngs=rngs)
    self.conv1 = ResidualUpsampleBlock(in_features=256, out_features=128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv2 = ResidualUpsampleBlock(in_features=128, out_features=64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv3 = ResidualUpsampleBlock(in_features=64, out_features=32, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv4 = ResidualUpsampleBlock(in_features=32, out_features=out_dim, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)

  def __call__(
      self, z: Float[Array, "... latent_dim"]
  ) -> Float[Array, "... height width channels"]:
    x = nnx.relu(self.dense1(z))
    x = rearrange(x, "... (h w c) -> ... h w c", h=4, w=4, c=256)

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    return nnx.sigmoid(x)


class VAE(nnx.Module):
    """Encoder + decoder wrapper so the training step has a single model object."""

    def __init__(
        self,
        *,
        image_channels: int,
        latent_dim: int,
        memory_dim: int,
        num_gaussian_components: int = 5,
        rngs: nnx.Rngs,
    ) -> None:
        self.encoder = MDNEncoder(
            in_dim=image_channels,
            latent_dim=latent_dim,
            memory_dim=memory_dim,
            num_gaussian_components=num_gaussian_components,
            rngs=rngs,
        )
        self.decoder = MDNDecoder(
            latent_dim=latent_dim, dout=image_channels, rngs=rngs
        )

    def __call__(
        self, x: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        z, pi_logits, mu, log_var = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, pi_logits, mu, log_var


if __name__ == "__main__":
    mdn_head = MDNHead(in_features=32, out_features=3, num_gaussian_components=10, rngs=nnx.Rngs(0, noise=1))
    print(nnx.tabulate(mdn_head, jnp.ones((1, 32))))
    encoder = MDNEncoder(in_dim=3, latent_dim=32, memory_dim=16, num_gaussian_components=10, rngs=nnx.Rngs(0, noise=1))
    print(nnx.tabulate(encoder, jnp.ones((1, 1, 64, 64, 3)), jnp.ones((1, 1, 16))))
    decoder = MDNDecoder(latent_dim=32, out_dim=3, rngs=nnx.Rngs(0, noise=1))
    print(nnx.tabulate(decoder, jnp.ones((1, 1, 32))))
  