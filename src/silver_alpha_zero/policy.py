from collections.abc import Sequence

import jax
import jax.numpy as jnp
import magiccube
from flax import nnx
from jaxtyping import Array, Float, Int

from mcst import Action, Evaluator, Model, State

# Color letter -> index, using magiccube's Color ordering (R, O, W, Y, B, G) so
# indices match the rest of the codebase.
_COLOR_TO_INDEX = {color.name: color.value for color in magiccube.Color}


def states_to_indices(states: Sequence[str]) -> Int[Array, "batch state_dim"]:
    """Convert a batch of cube facelet strings to an integer index array.

    Each string is a sequence of color letters, e.g. "RGROYYBOBYWRGRWWWGOYGBOB";
    every character is mapped to its color index. Whitespace is ignored.
    """
    indices = [[_COLOR_TO_INDEX[char] for char in "".join(s.split())] for s in states]
    return jnp.asarray(indices, dtype=jnp.int32)


class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, *, rngs: nnx.Rngs):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nnx.MultiHeadAttention(
            in_features=embed_dim,
            num_heads=num_heads,
            decode=False,
            use_bias=False,
            rngs=rngs,
        )
        self.linear1 = nnx.Linear(
            in_features=embed_dim, out_features=4 * embed_dim, use_bias=False, rngs=rngs
        )
        self.linear2 = nnx.Linear(
            in_features=4 * embed_dim, out_features=embed_dim, use_bias=False, rngs=rngs
        )
        self.norm1 = nnx.RMSNorm(embed_dim, rngs=rngs)
        self.norm2 = nnx.RMSNorm(embed_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "... embed_dim"]) -> Float[Array, "... embed_dim"]:
        # Pre-norm self-attention with residual connection.
        x = x + self.attention(self.norm1(x))
        # Pre-norm feed-forward with residual connection.
        x = x + self.linear2(nnx.gelu(self.linear1(self.norm2(x))))
        return x

class Policy(nnx.Module):
    def __init__(
        self,
        num_embeddings: int,
        state_dim: int,
        action_dim: int,
        embed_dim: int,
        num_transformer_blocks: int,
        num_heads: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Each input index (e.g. a sticker color in [0, num_embeddings)) is
        # mapped to a learned embed_dim vector.
        self.embed = nnx.Embed(
            num_embeddings=num_embeddings, features=embed_dim, rngs=rngs
        )
        self.transformer_blocks = nnx.Sequential(
            *[
                TransformerBlock(embed_dim, num_heads, rngs=rngs)
                for _ in range(num_transformer_blocks)
            ]
        )
        self.policy_head = nnx.Linear(
            in_features=embed_dim, out_features=action_dim, rngs=rngs
        )
        self.value_head = nnx.Linear(
            in_features=embed_dim, out_features=1, rngs=rngs
        )

    def __call__(
        self, state: Int[Array, "... state_dim"]
    ) -> tuple[Float[Array, "... action_dim"], Float[Array, "..."]]:
        x = self.embed(state)  # (..., state_dim, embed_dim)
        x = self.transformer_blocks(x)
        x = jnp.mean(x, axis=-2)  # pool over stickers -> (..., embed_dim)
        logits = self.policy_head(x)  # action logits (..., action_dim)
        value = jnp.squeeze(self.value_head(x), axis=-1)  # scalar value (...,)
        return logits, value


@nnx.jit
def _policy_forward(
    policy: Policy, indices: Int[Array, "batch state_dim"]
) -> tuple[Float[Array, "batch action_dim"], Float[Array, "batch"]]:
    """JIT-compiled forward pass. ``nnx.jit`` traces the module's parameters."""
    return policy(indices)


# @nnx.jit
# def _policy_probs(
#     policy: Policy, indices: Int[Array, "batch state_dim"]
# ) -> tuple[Float[Array, "batch action_dim"], Float[Array, "batch"]]:
#     """JIT-compiled forward pass that fuses the softmax over the action head, so
#     the whole forward + normalization runs as a single compiled call."""
#     logits, value = policy(indices)
#     return jax.nn.softmax(logits, axis=-1), value


class PolicyRubikCubeEvaluator(Evaluator):
    """AlphaZero-style evaluator backed by the learned :class:`Policy` network.

    The policy head produces a prior distribution over the model's legal actions
    (softmax over logits) and the value head estimates the expected return of the
    current state.
    """

    def __init__(
        self,
        model: Model,
        policy: Policy,
    ):
        self.model = model
        self.policy = policy

    def evaluate(self, state: State) -> tuple[dict[Action, float], float]:
        legal_actions = self.model.legal_actions(state)
        # The network consumes integer sticker indices, not the State object.
        indices = states_to_indices([state.get()])  # (1, state_dim)
        action_logits, value = _policy_forward(self.policy, indices)
        # Restrict logits to the legal actions (same fixed order as the model)
        # and turn them into a normalized prior distribution.
        logits = action_logits[0, : len(legal_actions)]
        probs = jax.nn.softmax(logits)
        priors = {action: float(prob) for action, prob in zip(legal_actions, probs)}
        return priors, float(value[0])


if __name__ == "__main__":
    assert jnp.array_equal(states_to_indices(["RGROYYBOB"]), jnp.array([[0, 5, 0, 1, 3, 3, 4, 1, 4]]))
    assert jnp.array_equal(states_to_indices(["RGWOYYBOB", "RGROYYBOB"]), jnp.array([[0, 5, 2, 1, 3, 3, 4, 1, 4], [0, 5, 0, 1, 3, 3, 4, 1, 4]]))
    rngs = nnx.Rngs(0)
    
    size = 3
    cube = magiccube.Cube(size)
    cube.scramble(4)
    state = states_to_indices([cube.get()])

    policy = Policy(num_embeddings=len(magiccube.Color), state_dim=len(cube.get()), action_dim=18, embed_dim=128, num_transformer_blocks=2, num_heads=2, rngs=rngs)

    print(nnx.tabulate(policy, state))
