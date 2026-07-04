import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class Policy(nnx.Module):
    """MLP policy conditioned on RSSM features = concat([deter, stoch])."""

    def __init__(
        self,
        features_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.mlp = nnx.Sequential(
            nnx.Linear(features_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, action_dim, rngs=rngs),
        )

    def __call__(
        self,
        features: Float[Array, "... features_dim"],
    ) -> Float[Array, "... action_dim"]:
        raw_action = self.mlp(features)
        return jnp.concatenate(
            [
                jnp.tanh(raw_action[..., :1]),
                jax.nn.sigmoid(raw_action[..., 1:]),
            ],
            axis=-1,
        )


class Value(nnx.Module):
    """MLP value function conditioned on RSSM features = concat([deter, stoch])."""

    def __init__(
        self,
        features_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.mlp = nnx.Sequential(
            nnx.Linear(features_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        )

    def __call__(
        self,
        features: Float[Array, "... features_dim"],
    ) -> Float[Array, "... 1"]:
        return self.mlp(features)

if __name__ == "__main__":
    features_dim = 16 + 32
    policy = Policy(features_dim=features_dim, hidden_dim=64, action_dim=10, rngs=nnx.Rngs(0, noise=1))
    value = Value(features_dim=features_dim, hidden_dim=64, rngs=nnx.Rngs(0, noise=1))
    print(nnx.tabulate(policy, jnp.ones((1, features_dim))))
    print(nnx.tabulate(value, jnp.ones((1, features_dim))))
