from flax import nnx
from jaxtyping import Array, Float


import jax
import jax.numpy as jnp


class VAEEncoder(nnx.Module):
  """A VAE encoder model from the paper."""
  latent_dim: int
  def __init__(self, din: int, latent_dim: int, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(in_features=din, out_features=32, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv3 = nnx.Conv(in_features=64, out_features=128, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.conv4 = nnx.Conv(in_features=128, out_features=256, kernel_size=(4, 4), strides=(2, 2), rngs=rngs)
    self.mu_proj = nnx.Linear(in_features=256*4*4, out_features=latent_dim, rngs=rngs)
    self.logvar_proj = nnx.Linear(in_features=256*4*4, out_features=latent_dim, rngs=rngs)
    self.rngs = rngs

  def __call__(
      self, x: Float[Array, "batch height width channels"]
  ) -> tuple[
      Float[Array, "batch latent_dim"],
      Float[Array, "batch latent_dim"],
      Float[Array, "batch latent_dim"],
  ]:
    x = self.conv1(x)
    x = nnx.relu(x)
    x = self.conv2(x)
    x = nnx.relu(x)
    x = self.conv3(x)
    x = nnx.relu(x)
    x = self.conv4(x)
    x = nnx.relu(x)

    x = x.reshape((x.shape[0], -1))  # flatten
    mu = self.mu_proj(x)
    logvar = self.logvar_proj(x)

    eps = jax.random.normal(self.rngs.noise(), mu.shape)
    z = mu + jnp.exp(logvar * 0.5) * eps
    return z, mu, logvar


class VAEDecoder(nnx.Module):
  """A decoder that reconstructs 64x64 RGB frames from latent vectors."""
  def __init__(self, latent_dim: int, dout: int, *, rngs: nnx.Rngs):
    self.dense1 = nnx.Linear(in_features=latent_dim, out_features=4*4*256, rngs=rngs)
    self.conv1 = nnx.ConvTranspose(in_features=256, out_features=128, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
    self.conv2 = nnx.ConvTranspose(in_features=128, out_features=64, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
    self.conv3 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)
    self.conv4 = nnx.ConvTranspose(in_features=32, out_features=3, kernel_size=(4, 4), strides=(2, 2), padding="SAME", rngs=rngs)

  def __call__(
      self, z: Float[Array, "batch latent_dim"]
  ) -> Float[Array, "batch height width channels"]:
    x = self.dense1(z)
    x = nnx.relu(x)
    x = x.reshape((x.shape[0], 4, 4, 256))

    x = self.conv1(x)
    x = nnx.relu(x)
    x = self.conv2(x)
    x = nnx.relu(x)
    x = self.conv3(x)
    x = nnx.relu(x)
    x = self.conv4(x)
    return nnx.sigmoid(x)


class VAE(nnx.Module):
    """Encoder + decoder wrapper so the training step has a single model object."""

    def __init__(
        self,
        *,
        image_channels: int,
        latent_dim: int,
        rngs: nnx.Rngs,
    ) -> None:
        self.encoder = VAEEncoder(
            din=image_channels, latent_dim=latent_dim, rngs=rngs
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim, dout=image_channels, rngs=rngs
        )

    def __call__(
        self, x: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        z, mu, logvar = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


if __name__ == "__main__":
  encoder = VAEEncoder(din=3, latent_dim=32, rngs=nnx.Rngs(0, noise=1))
  print(nnx.tabulate(encoder, jnp.ones((1, 64, 64, 3))))
  decoder = VAEDecoder(latent_dim=32, dout=3, rngs=nnx.Rngs(0, noise=1))
  print(nnx.tabulate(decoder, jnp.ones((1, 32))))
  