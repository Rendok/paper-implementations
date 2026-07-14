import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


def _log_one_minus_tanh_sq(u: jax.Array) -> jax.Array:
    """Numerically stable ``log(1 - tanh(u)**2)`` = ``log(sech(u)**2)``.

    The naive form underflows to ``log(0) = -inf`` once ``|u|`` is large.
    Uses the identity ``log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u))``.
    """
    return 2.0 * (jnp.log(2.0) - u - jax.nn.softplus(-2.0 * u))


class Policy(nnx.Module):
    """Squashed-Gaussian MLP policy conditioned on RSSM features.

    A diagonal Gaussian over a *pre-squash* variable ``u``; the action is
    ``a = scale * tanh(u) + bias``, so it is always inside CarRacing's valid
    box — steering ∈ [-1, 1], gas ∈ [0, 1], brake ∈ [0, 1] — and remains a
    differentiable function of the policy parameters everywhere (unlike a
    hard clip, whose gradient dies at the boundary). ``log_prob`` includes
    the tanh change-of-variables correction, keeping the SAC entropy term
    bounded and ``target_entropy = -action_dim`` meaningful.

    The *squashed* (bounded) action is what gets stored in the replay buffer,
    applied to the env, and fed to the critic/world model — there is no
    separate "raw" action to clip anymore.

    Use ``sample`` during training/collection (returns action + log_prob).
    Use ``__call__`` (deterministic squashed mean) for evaluation / greedy
    rollouts.
    """

    # log_var is clamped to keep the pre-squash std in a sane range:
    # [exp(-2), exp(1)] ≈ [0.05, 2.7].
    LOG_VAR_MIN = -4.0
    LOG_VAR_MAX = 1.0

    def __init__(
        self,
        features_dim: int,
        hidden_dim: int,
        action_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.action_dim = action_dim
        # Shared trunk → mean head + state-dependent log_var head.
        self.trunk = nnx.Sequential(
            nnx.Linear(features_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
        )
        self.mu_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.log_var_head = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        self.rngs = rngs

        # Per-dimension affine squash: a = scale*tanh(u) + bias maps ℝ onto
        # [low, high]. Dim 0 (steering) → [-1, 1]; the rest (gas, brake) → [0, 1].
        low = jnp.array([-1.0, 0.0, 0.0])
        high = jnp.ones(action_dim)
        self._action_scale = (high - low) / 2.0
        self._action_bias = (high + low) / 2.0

    def _mu_and_std(
        self, features: Float[Array, "... features_dim"]
    ) -> tuple[jax.Array, jax.Array]:
        h = self.trunk(features)
        mu = self.mu_head(h)
        log_var = jnp.clip(self.log_var_head(h), self.LOG_VAR_MIN, self.LOG_VAR_MAX)
        std = jnp.exp(0.5 * log_var)
        return mu, std

    def _squash(self, u: jax.Array) -> jax.Array:
        return self._action_scale * jnp.tanh(u) + self._action_bias

    def _log_prob_from_u(
        self, u: jax.Array, mu: jax.Array, std: jax.Array
    ) -> jax.Array:
        """log π(a) for the squashed action a = squash(u), given pre-squash u.

        log p_a(a) = log N(u; mu, std) − Σ log|d a/d u|, where
        d a/d u = scale · (1 − tanh(u)²).
        """
        var = std ** 2
        log_prob_u = -0.5 * jnp.sum(
            (u - mu) ** 2 / var + jnp.log(2.0 * jnp.pi * var), axis=-1
        )
        correction = jnp.sum(
            jnp.log(self._action_scale) + _log_one_minus_tanh_sq(u), axis=-1
        )
        return log_prob_u - correction

    def __call__(
        self,
        features: Float[Array, "... features_dim"],
    ) -> Float[Array, "... action_dim"]:
        """Deterministic squashed mean action (for evaluation / greedy rollouts).

        Already inside the valid action box — apply directly to the env.
        """
        mu, _ = self._mu_and_std(features)
        return self._squash(mu)

    def sample(
        self,
        features: Float[Array, "... features_dim"],
        rng_key: jax.Array | None = None,
    ) -> tuple[Float[Array, "... action_dim"], Float[Array, "..."]]:
        """Sample a squashed (bounded) action and return its log-probability.

        Pass an explicit *rng_key* when calling from inside jax.lax.scan to
        avoid mutating the NNX RngCount counter from a traced context
        (same pattern as DynamicPredictor).  If omitted, self.rngs.noise()
        is used (safe outside scan).

        The returned action is already inside the valid action box (tanh
        squashed), so no clipping is needed before applying it to the env.
        """
        mu, std = self._mu_and_std(features)
        key = self.rngs.noise() if rng_key is None else rng_key
        eps = jax.random.normal(key, mu.shape)
        u = mu + std * eps
        action = self._squash(u)
        log_prob = self._log_prob_from_u(u, mu, std)
        return action, log_prob

    def log_prob(
        self,
        actions: Float[Array, "... action_dim"],
        features: Float[Array, "... features_dim"],
    ) -> Float[Array, "..."]:
        """Log-probability of squashed *actions* under the current policy.

        ``actions`` must be bounded actions (as stored in the replay buffer /
        returned by ``sample``). They are inverted through the squash to
        recover the pre-squash ``u`` before evaluating the corrected density.
        """
        mu, std = self._mu_and_std(features)
        # Invert a = scale*tanh(u) + bias, clamping to (-1, 1) so arctanh is finite.
        y = jnp.clip((actions - self._action_bias) / self._action_scale, -1.0 + 1e-6, 1.0 - 1e-6)
        u = jnp.arctanh(y)
        return self._log_prob_from_u(u, mu, std)


class ActionValue(nnx.Module):
    """MLP value function conditioned on RSSM features = concat([deter, stoch])."""

    def __init__(
        self,
        features_dim: int,
        actions_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.mlp = nnx.Sequential(
            nnx.Linear(features_dim + actions_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        )

    def __call__(
        self,
        features: Float[Array, "... features_dim"],
        actions: Float[Array, "... actions_dim"],
    ) -> Float[Array, "... 1"]:
        inputs = jnp.concat([features, actions], axis=-1)
        return self.mlp(inputs)


class Critic(nnx.Module):
    """Twin Q(s, a) critics for SAC (clipped double-Q, Fujimoto et al., 2018).

    Keeping two independently-initialised ``ActionValue`` networks and taking
    the min of their predictions mitigates the Q-value overestimation bias
    that plain single-critic actor-critic methods suffer from.
    """

    def __init__(
        self,
        features_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.q1 = ActionValue(features_dim, action_dim, hidden_dim, rngs=rngs)
        self.q2 = ActionValue(features_dim, action_dim, hidden_dim, rngs=rngs)

    def __call__(
        self,
        features: Float[Array, "... features_dim"],
        actions: Float[Array, "... action_dim"],
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        return self.q1(features, actions)[..., 0], self.q2(features, actions)[..., 0]


class EncoderPolicy(nnx.Module):
    """Combines the RSSM encoder + policy's deterministic mean action into a
    single callable: raw uint8/float frame in, raw action out.

    This is what actually runs behind the inference server during self-play
    collection — workers only ever send frames and get back actions, so they
    never need the RSSM/Policy classes (or the accelerator) themselves. Pass
    ``rssm.encoder`` (not the whole ``RSSM``) so weight syncs stay small —
    the decoder/prior/reward heads aren't needed for acting.

    Construct this by wrapping *existing* ``encoder``/``policy`` instances
    (e.g. on the trainer, for ``extract_weights``) — don't pass an instance
    of this class directly to ``run_inference_server``, since it would need
    to pickle those already-built submodules across the process boundary
    (which can fail, e.g. ``nnx.Sequential`` layers built from bare
    activation-function references). Use ``build_encoder_policy`` as the
    server-side ``model_fn`` instead — it builds fresh submodules in-process
    from plain dimension kwargs; only weights (always plain numpy) cross the
    process boundary afterwards.
    """

    def __init__(self, encoder: nnx.Module, policy: "Policy") -> None:
        self.encoder = encoder
        self.policy = policy

    def __call__(
        self, frame: Float[Array, "... height width channels"]
    ) -> Float[Array, "... action_dim"]:
        frame = frame.astype(jnp.float32) / 255.0
        z, _, _, _ = self.encoder(frame)
        return self.policy(z)


def build_encoder_policy(
    *,
    image_channels: int,
    memory_dim: int,
    stoch_dim: int,
    num_gaussian_components: int,
    policy_hidden_dim: int,
    action_dim: int,
    rngs: nnx.Rngs,
) -> EncoderPolicy:
    """Build a fresh ``EncoderPolicy`` from plain dimension kwargs.

    Picklable-by-reference factory for ``run_inference_server``'s
    ``model_fn`` — pass this function itself (not a pre-built instance) so
    the inference-server process constructs its own encoder/policy weights
    locally; real weights are synced afterwards via ``weight_queue``.
    """
    from encoder_decoder import MDNEncoder  # local import: keep this module RSSM-independent otherwise

    encoder = MDNEncoder(
        in_dim=image_channels,
        latent_dim=stoch_dim,
        memory_dim=memory_dim,
        num_gaussian_components=num_gaussian_components,
        rngs=rngs,
    )
    policy = Policy(
        features_dim=stoch_dim,
        hidden_dim=policy_hidden_dim,
        action_dim=action_dim,
        rngs=rngs,
    )
    return EncoderPolicy(encoder, policy)


class Temperature(nnx.Module):
    """Learnable SAC entropy coefficient α = exp(log_alpha).

    Automatic entropy tuning (Haarnoja et al., 2018): ``log_alpha`` is
    updated so the policy's expected entropy tracks a target entropy
    (heuristically ``-action_dim``), instead of hand-tuning a fixed α.
    """

    def __init__(self, initial_alpha: float = 1.0) -> None:
        self.log_alpha = nnx.Param(jnp.log(jnp.asarray(initial_alpha, dtype=jnp.float32)))

    @property
    def value(self) -> jax.Array:
        return jnp.exp(self.log_alpha[...])


if __name__ == "__main__":
    features_dim = 16 + 32
    policy = Policy(features_dim=features_dim, hidden_dim=64, action_dim=10, rngs=nnx.Rngs(0, noise=1))
    q = ActionValue(features_dim=features_dim, actions_dim=3, hidden_dim=64, rngs=nnx.Rngs(0, noise=1))
    critic = Critic(features_dim=features_dim, action_dim=3, hidden_dim=64, rngs=nnx.Rngs(0, noise=1))
    print(nnx.tabulate(policy, jnp.ones((1, features_dim))))
    print(nnx.tabulate(q, jnp.ones((1, features_dim)), jnp.ones((1, 3))))
    print(nnx.tabulate(critic, jnp.ones((1, features_dim)), jnp.ones((1, 3))))
