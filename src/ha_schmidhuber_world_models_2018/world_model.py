from flax import nnx
from jaxtyping import Array, Float, Int
from typing import Optional, Tuple


import jax
import jax.numpy as jnp


Carry = Tuple[Float[Array, "batch hidden"], Float[Array, "batch hidden"]]
MdnParams = Tuple[
    Float[Array, "batch seq K"],
    Float[Array, "batch seq K D"],
    Float[Array, "batch seq K D"],
]


class WorldModel(nnx.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_gaussian_components: int,
        temperature: float,
        rngs: nnx.Rngs,
    ):
        self.temperature = temperature
        self.num_gaussian_components = num_gaussian_components
        self.out_features = out_features
        self.rnn = nnx.RNN(nnx.LSTMCell(in_features=in_features, hidden_features=hidden_features, rngs=rngs))
        # MDN head: K mixture weights + K diagonal Gaussians each over D output dims (mu and log_sigma).
        self.mdn_head = nnx.Linear(
            in_features=hidden_features,
            out_features=num_gaussian_components * (1 + 2 * out_features),
            rngs=rngs,
        )

    def __call__(
        self,
        z: Float[Array, "batch seq z_dim"],
        a: Float[Array, "batch seq a_dim"],
        initial_carry: Optional[Carry] = None,
        return_carry: bool = True,
    ):
        x: Float[Array, "batch seq in_features"] = jnp.concatenate([z, a], axis=-1)
        rnn_out = self.rnn(x, initial_carry=initial_carry, return_carry=return_carry)
        if return_carry:
            carry, h = rnn_out
        else:
            carry, h = None, rnn_out

        pi_logits, mu, log_sigma = self._mdn_params(h)
        if return_carry:
            return carry, (pi_logits, mu, log_sigma)
        return pi_logits, mu, log_sigma

    def _mdn_params(self, h: Float[Array, "batch seq hidden"]) -> MdnParams:
        params = self.mdn_head(h)
        K, D = self.num_gaussian_components, self.out_features
        pi_logits: Float[Array, "batch seq K"] = params[..., :K]
        rest = params[..., K:].reshape(*params.shape[:-1], K, 2 * D)
        mu: Float[Array, "batch seq K D"] = rest[..., :D]
        log_sigma: Float[Array, "batch seq K D"] = rest[..., D:]
        return pi_logits, mu, log_sigma

    def log_prob(
        self,
        z_target: Float[Array, "batch seq D"],
        mdn_params: MdnParams,
    ) -> Float[Array, "batch seq"]:
        pi_logits, mu, log_sigma = mdn_params
        log_pi: Float[Array, "batch seq K"] = jax.nn.log_softmax(pi_logits, axis=-1)
        z_exp: Float[Array, "batch seq 1 D"] = z_target[..., None, :]
        # Per-component diagonal Gaussian log-density summed over D dims -> shape "batch seq K".
        log_gauss_per_dim: Float[Array, "batch seq K D"] = (
            -0.5 * jnp.square((z_exp - mu) * jnp.exp(-log_sigma))
            - log_sigma
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        log_gauss: Float[Array, "batch seq K"] = log_gauss_per_dim.sum(axis=-1)
        return jax.nn.logsumexp(log_pi + log_gauss, axis=-1)

    def sample(
        self,
        mdn_params: MdnParams,
        rngs: nnx.Rngs,
    ) -> Float[Array, "batch seq D"]:
        pi_logits, mu, log_sigma = mdn_params
        # Temperature: sharpens/softens mixture weights and scales noise stddev.
        pi_logits = pi_logits / self.temperature
        sigma: Float[Array, "batch seq K D"] = jnp.exp(log_sigma) * jnp.sqrt(self.temperature)

        # One categorical draw per (batch, seq) over K components.
        idx: Int[Array, "batch seq"] = jax.random.categorical(rngs(), pi_logits, axis=-1)
        idx_exp = idx[..., None, None]
        mu_chosen: Float[Array, "batch seq D"] = jnp.take_along_axis(mu, idx_exp, axis=-2).squeeze(-2)
        sigma_chosen: Float[Array, "batch seq D"] = jnp.take_along_axis(sigma, idx_exp, axis=-2).squeeze(-2)

        eps: Float[Array, "batch seq D"] = jax.random.normal(rngs(), mu_chosen.shape)
        return mu_chosen + sigma_chosen * eps

    def initialize_carry(self, input_shape: Tuple[int, ...], rngs: nnx.Rngs) -> Carry:
        return self.rnn.cell.initialize_carry(input_shape=input_shape, rngs=rngs)


if __name__ == "__main__":
    z_dim = 4 * 4 * 32
    a_dim = 3
    world_model = WorldModel(
        in_features=z_dim + a_dim,
        hidden_features=256,
        out_features=z_dim,
        num_gaussian_components=5,
        temperature=1.0,
        rngs=nnx.Rngs(3),
    )
    z = jnp.ones((2, 3, z_dim))
    a = jnp.ones((2, 3, a_dim))
    init_carry = world_model.initialize_carry(input_shape=(2, 256), rngs=nnx.Rngs(4))
    print(len(init_carry), init_carry[0].shape, init_carry[1].shape)
    carry, (pi_logits, mu, log_sigma) = world_model(z, a, initial_carry=init_carry, return_carry=True)
    print(len(carry), carry[1].shape, pi_logits.shape, mu.shape, log_sigma.shape)

    sampled = world_model.sample((pi_logits, mu, log_sigma), rngs=nnx.Rngs(5))
    print("sampled:", sampled.shape)

    nll = -world_model.log_prob(jnp.zeros((2, 3, z_dim)), (pi_logits, mu, log_sigma))
    print("nll:", nll.shape)
