import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int


class ReplayBuffer:
    """Fixed-capacity circular buffer of AlphaZero training examples.

    Each example is a triple ``(state, policy, value)``:
      * ``state``  - integer sticker indices, shape ``(state_dim,)``;
      * ``policy`` - MCTS visit-count distribution over the (fixed-order) action
        set, shape ``(action_dim,)``;
      * ``value``  - scalar value target (the return observed from that state).

    Storage lives in NumPy (cheap in-place circular writes); ``sample`` returns
    JAX arrays placed on the default device.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, rngs: nnx.Rngs):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.uint8)
        self.policies = np.zeros((capacity, action_dim), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.index = 0  # next write position (wraps around)
        self.size = 0  # number of valid entries (<= capacity)
        self.rngs = rngs

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        states: Int[Array, "batch state_dim"],
        policies: Float[Array, "batch action_dim"],
        values: Float[Array, "batch"],
    ) -> None:
        """Append a batch of ``n`` examples, overwriting the oldest on overflow."""
        states = np.asarray(states, dtype=np.uint8)
        policies = np.asarray(policies, dtype=np.float32)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        n = states.shape[0]
        if not (n == policies.shape[0] == values.shape[0]):
            raise ValueError("states, policies and values must share the batch size")
        if n > self.capacity:  # keep only the most recent `capacity` examples
            states, policies, values = states[-self.capacity:], policies[-self.capacity:], values[-self.capacity:]
            n = self.capacity
        
        # Circular write, split into the (up to) two contiguous spans it touches.
        idx = (self.index + np.arange(n)) % self.capacity
        self.states[idx] = states
        self.policies[idx] = policies
        self.values[idx] = values
        self.index = (self.index + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[Int[Array, "batch state_dim"], Float[Array, "batch action_dim"], Float[Array, "batch"]]:
        """Uniformly sample ``batch_size`` examples (with replacement)."""
        if self.size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        idx = jax.random.randint(self.rngs(), (batch_size,), 0, self.size)
        idx = np.asarray(idx)
        return (
            jnp.asarray(self.states[idx]),
            jnp.asarray(self.policies[idx]),
            jnp.asarray(self.values[idx]),
        )


if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    buffer = ReplayBuffer(capacity=5, state_dim=4, action_dim=3, rngs=rngs)
    assert len(buffer) == 0

    # Add 3, then 4 more to force a wraparound (capacity 5).
    buffer.add(np.arange(12).reshape(3, 4), np.ones((3, 3)) / 3, np.array([0.1, 0.2, 0.3]))
    assert len(buffer) == 3
    buffer.add(np.arange(16).reshape(4, 4), np.ones((4, 3)) / 3, np.array([1.0, 1.1, 1.2, 1.3]))
    assert len(buffer) == 5, len(buffer)

    states, policies, values = buffer.sample(batch_size=8)
    assert states.shape == (8, 4)
    assert policies.shape == (8, 3)
    assert values.shape == (8,)
    print("states\n", states)
    print("values", values)
    print("ReplayBuffer OK")
