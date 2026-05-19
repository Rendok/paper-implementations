from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


Carry = Tuple[Float[Array, "... hidden_dim"], Float[Array, "... hidden_dim"]]


class CMAESController(nnx.Module):
    """Linear controller optimized by CMA-ES.

    The policy follows the World Models controller form:

        action = W [z, h] + b

    where ``z`` is the VAE latent and ``h`` is the world-model LSTM hidden
    state from the carry tuple.
    """

    def __init__(
        self,
        *,
        z_dim: int,
        hidden_dim: int,
        action_dim: int,
        weights: Float[Array, "action_dim input_dim"] | None = None,
        bias: Float[Array, "action_dim"] | None = None,
    ) -> None:
        input_dim = z_dim + hidden_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.weights = nnx.Param(
            jnp.zeros((action_dim, input_dim), dtype=jnp.float32)
            if weights is None
            else weights
        )
        self.bias = nnx.Param(
            jnp.zeros((action_dim,), dtype=jnp.float32) if bias is None else bias
        )

    def __call__(
        self,
        z: Float[Array, "... z_dim"],
        carry: Float[Array, "... hidden_dim"],
    ) -> Float[Array, "... action_dim"]:
        assert not isinstance(carry, tuple)
        controller_input = jnp.concatenate([z, carry], axis=-1)
        raw_action = controller_input @ self.weights.value.T + self.bias.value
        return jnp.concatenate(
            [
                jnp.tanh(raw_action[..., :1]),
                jax.nn.sigmoid(raw_action[..., 1:]),
            ],
            axis=-1,
        )

    @classmethod
    def from_flat_parameters(
        cls,
        parameters: Float[Array, "num_parameters"],
        *,
        z_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> "CMAESController":
        """Build a controller from the flat vector used by CMA-ES."""
        input_dim = z_dim + hidden_dim
        expected_parameters = cls.num_parameters(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
        )
        if parameters.size != expected_parameters:
            raise ValueError(
                f"Expected {expected_parameters} controller parameters, "
                f"got {parameters.size}."
            )

        weights_size = action_dim * input_dim
        weights = parameters[:weights_size].reshape(action_dim, input_dim)
        bias = parameters[weights_size:]
        return cls(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            weights=weights,
            bias=bias,
        )

    def to_flat_parameters(self) -> Float[Array, "num_parameters"]:
        """Flatten ``W`` and ``b`` back into CMA-ES parameter-vector form."""
        return jnp.concatenate(
            [self.weights.value.reshape(-1), self.bias.value], axis=0
        )

    @staticmethod
    def num_parameters(*, z_dim: int, hidden_dim: int, action_dim: int) -> int:
        input_dim = z_dim + hidden_dim
        return action_dim * input_dim + action_dim

    @classmethod
    def zeros(
        cls,
        *,
        z_dim: int,
        hidden_dim: int,
        action_dim: int,
    ) -> "CMAESController":
        """Create the usual CMA-ES mean initialization for a linear policy."""
        return cls(z_dim=z_dim, hidden_dim=hidden_dim, action_dim=action_dim)

    @classmethod
    def random(
        cls,
        rngs: nnx.Rngs,
        *,
        z_dim: int,
        hidden_dim: int,
        action_dim: int,
        scale: float = 0.01,
    ) -> "CMAESController":
        """Small random initialization, useful for smoke tests and rollouts."""
        input_dim = z_dim + hidden_dim
        return cls(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            weights=scale * jax.random.normal(rngs(), (action_dim, input_dim)),
            bias=scale * jax.random.normal(rngs(), (action_dim,)),
        )


if __name__ == "__main__":
    rngs = nnx.Rngs(jax.random.PRNGKey(42))
    controller = CMAESController.random(rngs, z_dim=32, hidden_dim=256, action_dim=3)
    print(nnx.tabulate(controller, jnp.ones((1, 32)), jnp.ones((1, 256))))
    # print(controller.weights.value)
    # print(controller.bias.value)
