from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from encoder_decoder import MDNHead, MDNDecoder, MDNEncoder


Carry = Tuple[Float[Array, "batch deter_dim"], Float[Array, "batch deter_dim"]]
GaussianParams = Tuple[
    Float[Array, "batch seq stoch_dim"],
    Float[Array, "batch seq stoch_dim"],
]


class RewardContinuePredictor(nnx.Module):
    """Joint MLP predicting reward and continue logit from RSSM features (h, z).

    Shared trunk → two independent heads:
      - reward_logit  : scalar; train with MSE against real reward
      - continue_logit: scalar logit; sigmoid → continue prob, train with BCE against (1 - done)
    """

    def __init__(self, features_dim: int, hidden_dim: int, *, rngs: nnx.Rngs) -> None:
        self.trunk = nnx.Linear(features_dim, hidden_dim, rngs=rngs)
        self.reward_head = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.continue_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self, features: Float[Array, "... features_dim"]
    ) -> tuple[Float[Array, "... 1"], Float[Array, "... 1"]]:
        h = nnx.relu(self.trunk(features))
        return self.reward_head(h), self.continue_head(h)


class DynamicPredictor(nnx.Module):
  """Convolutional encoder with a mixture-density latent head."""
  latent_dim: int
  def __init__(self, in_dim: int, stoch_dim: int, memory_dim: int, num_gaussian_components: int, *, rngs: nnx.Rngs):
    self.mdn_head = MDNHead(in_features=memory_dim, out_features=stoch_dim, num_gaussian_components=num_gaussian_components, rngs=rngs)
    self.rngs = rngs

  def __call__(
      self,
      h: Float[Array, "... memory_dim"],
      rng_key: Optional[jax.Array] = None,
  ) -> tuple[
      Float[Array, "... latent_dim"],
      Float[Array, "... K"],
      Float[Array, "... K latent_dim"],
      Float[Array, "... K latent_dim"],
  ]:
    pi_logits, mu, log_var = self.mdn_head(h)
    # Clamp log_var: std in [0.1, 1]. Upper=0 prevents z≈noise (std=10 caused
    # decoder to see pure noise and output the mean image).
    log_var = jnp.clip(log_var, -4.6, 0.0)

    # Allow callers inside jax.lax.scan to supply an explicit key so we do
    # not mutate the NNX RngCount counter from inside a traced context.
    if rng_key is None:
        k1, k2 = jax.random.split(self.rngs.noise())
    else:
        k1, k2 = jax.random.split(rng_key)

    component_idx = jax.random.categorical(k1, pi_logits, axis=-1)
    component_idx = component_idx[..., None, None]
    chosen_mu = jnp.take_along_axis(mu, component_idx, axis=-2).squeeze(axis=-2)
    chosen_log_var = jnp.take_along_axis(log_var, component_idx, axis=-2).squeeze(axis=-2)
    eps = jax.random.normal(k2, chosen_mu.shape)
    z = chosen_mu + jnp.exp(0.5 * chosen_log_var) * eps
    return z, pi_logits, mu, log_var


class RSSM(nnx.Module):
    """Dreamer-style RSSM built on this repo's MDN encoder/decoder.

    - deterministic state h_t from an action-conditioned RNN
    - posterior q(s_t | h_t, o_t) from MDNEncoder(image_t, h_t)
    - prior p(s_t | h_t) from DynamicPredictor(h_t)
    - reconstruction from MDNDecoder(s_t)
    """

    def __init__(
        self,
        image_channels: int,
        action_dim: int,
        memory_dim: int,
        stoch_dim: int,
        num_gaussian_components: int,
        predictor_hidden_dim: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.memory_dim = memory_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.rngs = rngs

        features_dim = memory_dim + stoch_dim

        # RNN receives concat(prev_stoch, action) at each step.
        self.cell = nnx.LSTMCell(in_features=stoch_dim + action_dim, hidden_features=memory_dim, rngs=rngs)
        self.encoder = MDNEncoder(
            in_dim=image_channels,
            latent_dim=stoch_dim,
            memory_dim=memory_dim,
            num_gaussian_components=num_gaussian_components,
            rngs=rngs,
        )
        self.decoder = MDNDecoder(latent_dim=stoch_dim, out_dim=image_channels, rngs=rngs)
        self.prior = DynamicPredictor(
            in_dim=memory_dim,
            stoch_dim=stoch_dim,
            memory_dim=memory_dim,
            num_gaussian_components=num_gaussian_components,
            rngs=rngs,
        )
        self.reward_continue = RewardContinuePredictor(features_dim, predictor_hidden_dim, rngs=rngs)

    def __call__(
        self,
        images: Float[Array, "batch seq height width channels"],
        actions: Float[Array, "batch seq action_dim"],
        initial_carry: Optional[Carry] = None,
        return_carry: bool = True,
    ):
        return self.teacher_forcing_forward(images, actions, initial_carry, return_carry)

    def teacher_forcing_forward(
        self,
        images: Float[Array, "batch seq height width channels"],
        actions: Float[Array, "batch seq action_dim"],
        initial_carry: Optional[Carry] = None,
        return_carry: bool = True,
    ):
        # Step 1 — encode all images to get posterior z_0 … z_{T-1}.
        post_stoch, post_pi_logits, post_mu, post_log_var = self.encoder(images)

        # Step 2 — teacher-forcing RNN: feed posterior z_t and a_t as inputs to
        # compute h_{t+1} and prior p(z_{t+1} | h_{t+1}) for t = 0 … T-2.
        # The LSTM sees real observations (via z) at every step, not its own
        # predictions — this keeps h well-conditioned throughout training.
        carry, (deter, prior_stoch, prior_pi_logits, prior_mu, prior_log_var) = self.latent_forward(
            post_stoch[:, :-1], actions[:, :-1],
            initial_carry=initial_carry, return_carry=return_carry,
        )

        # Step 3 — align posterior z_1 … z_{T-1} with the prior sequence above.
        post_stoch_kl   = post_stoch[:, 1:]       # (B, T-1, D)
        post_pi_kl      = post_pi_logits[:, 1:]   # (B, T-1, K)
        post_mu_kl      = post_mu[:, 1:]           # (B, T-1, K, D)
        post_log_var_kl = post_log_var[:, 1:]      # (B, T-1, K, D)

        # Posterior reconstruction (full T frames) and prior imagination (T-1 frames).
        reconstruction       = self.decoder(post_stoch)    # (B, T,   H, W, C)
        prior_reconstruction = self.decoder(prior_stoch)   # (B, T-1, H, W, C)

        features = jnp.concatenate([deter, post_stoch_kl], axis=-1)  # (B, T-1, H+D)
        reward_logit, continue_logit = self.reward_continue(features)

        outputs = {
            "deter": deter,                          # (B, T-1, memory_dim)
            # prior p(z_t | h_t) for t = 1..T-1
            "prior_pi_logits":      prior_pi_logits,
            "prior_mu":             prior_mu,
            "prior_log_var":        prior_log_var,
            "prior_stoch":          prior_stoch,
            "prior_reconstruction": prior_reconstruction,  # (B, T-1, H, W, C)
            # posterior q(z_t | o_t) for t = 1..T-1  — aligned for KL
            "post_pi_logits":  post_pi_kl,
            "post_mu":         post_mu_kl,
            "post_log_var":    post_log_var_kl,
            "post_stoch":      post_stoch_kl,
            # full posterior (all T frames) — for reconstruction loss only
            "post_stoch_full": post_stoch,
            "features":        features,
            "reconstruction":  reconstruction,       # (B, T, H, W, C)
            "reward_logit":    reward_logit,
            "continue_logit":  continue_logit,
        }
        if return_carry:
            return carry, outputs
        return outputs

    def teacher_forcing_latent_forward(
        self,
        post_stoch: Float[Array, "batch seq stoch_dim"],   # posterior z (teacher-forced)
        actions: Float[Array, "batch seq action_dim"],
        initial_carry: Optional[Carry] = None,
        return_carry: bool = True,
    ):
        """Run the RNN with teacher-forcing: posterior z_t drives each LSTM step.

        At step t (t = 0 … T-2):
            h_{t+1} = LSTM( h_t, concat(post_z_t, a_t) )
            prior_params_{t+1} = DynamicPredictor(h_{t+1})

        Using posterior z as input keeps h well-conditioned throughout training
        (no garbage from an untrained prior feeding back into the LSTM).
        """
        batch = actions.shape[0]
        if initial_carry is None:
            initial_carry = self.initialize_carry(batch, rngs=self.rngs)

        # Pass explicit per-step rng keys so NNX RNG counters are not mutated
        # from inside the traced scan body.
        def rnn_step(lstm_carry, x_t):
            post_z_t, action_t, rng_key_t = x_t
            rnn_input = jnp.concatenate([jax.lax.stop_gradient(post_z_t), action_t], axis=-1)
            lstm_carry, deter_t = self.cell(lstm_carry, rnn_input)
            prior_stoch_t, pi_t, mu_t, log_var_t = self.prior(deter_t, rng_key=rng_key_t)
            return lstm_carry, (deter_t, prior_stoch_t, pi_t, mu_t, log_var_t)

        seq = actions.shape[1]
        step_keys = jax.random.split(self.rngs.noise(), seq)  # (seq, 2)

        # Swap (batch, seq, ...) → (seq, batch, ...) for lax.scan.
        post_stoch_T = jnp.swapaxes(post_stoch, 0, 1)
        actions_T    = jnp.swapaxes(actions, 0, 1)

        lstm_carry, (deter_T, prior_stoch_T, prior_pi_T, prior_mu_T, prior_log_var_T) = jax.lax.scan(
            rnn_step, initial_carry, (post_stoch_T, actions_T, step_keys)
        )

        # Swap back to (batch, seq, ...).
        deter                  = jnp.swapaxes(deter_T,         0, 1)
        prior_stoch_seq        = jnp.swapaxes(prior_stoch_T,   0, 1)
        prior_pi_logits        = jnp.swapaxes(prior_pi_T,      0, 1)
        prior_mu_components    = jnp.swapaxes(prior_mu_T,      0, 1)
        prior_log_var_comps    = jnp.swapaxes(prior_log_var_T, 0, 1)

        carry = lstm_carry if return_carry else None
        return carry, (deter, prior_stoch_seq, prior_pi_logits, prior_mu_components, prior_log_var_comps)

    @staticmethod
    def log_prob(
        z: Float[Array, "... D"],
        pi_logits: Float[Array, "... K"],
        mu: Float[Array, "... K D"],
        log_var: Float[Array, "... K D"],
    ) -> Float[Array, "..."]:
        """MDN log-likelihood log sum_k pi_k * N(z; mu_k, diag(exp(log_var_k))).

        Use for one-sample MC KL estimation:
            kl = log_prob(z_post, *post_params) - log_prob(z_post, *prior_params)
        where z_post is the posterior sample from __call__.
        """
        log_pi = jax.nn.log_softmax(pi_logits, axis=-1)        # (..., K)
        z_exp = z[..., None, :]                                 # (..., 1, D)
        # log N(z; mu_k, diag(exp(log_var_k))) summed over D dims.
        # = -0.5 * sum_d [ log(2π) + log_var_kd + (z_d - mu_kd)² * exp(-log_var_kd) ]
        log_gauss = -0.5 * (
            jnp.log(2.0 * jnp.pi) + log_var
            + jnp.square(z_exp - mu) * jnp.exp(-log_var)
        ).sum(axis=-1)                                          # (..., K)
        return jax.nn.logsumexp(log_pi + log_gauss, axis=-1)   # (...)

    def initialize_carry(self, batch_size: int, *, rngs: nnx.Rngs) -> Carry:
        return self.cell.initialize_carry(
            input_shape=(batch_size, self.stoch_dim + self.action_dim), rngs=rngs
        )


if __name__ == "__main__":
    rssm = RSSM(
        image_channels=3,
        action_dim=6,
        memory_dim=256,
        stoch_dim=32,
        num_gaussian_components=10,
        rngs=nnx.Rngs(0, noise=1),
    )
    images = jnp.ones((2, 5, 64, 64, 3))
    actions = jnp.ones((2, 5, 6))
    carry = rssm.initialize_carry(batch_size=2, rngs=nnx.Rngs(2))
    carry, out = rssm(images, actions, initial_carry=carry, return_carry=True)
    print("carry:", len(carry), carry[0].shape, carry[1].shape)
    print(
        "out shapes — deter:", out["deter"].shape,
        "post_stoch:", out["post_stoch"].shape,
        "features:", out["features"].shape,
        "recon:", out["reconstruction"].shape,
        "reward:", out["reward_logit"].shape,
        "continue:", out["continue_logit"].shape,
    )

    # One-sample MC KL: E_{z~q}[log q(z) - log p(z)]
    z = out["post_stoch"]
    log_q = RSSM.log_prob(z, out["post_pi_logits"], out["post_mu"], out["post_log_var"])
    log_p = RSSM.log_prob(z, out["prior_pi_logits"], out["prior_mu"], out["prior_log_var"])
    mc_kl = log_q - log_p  # (batch, seq)
    print("mc_kl shape:", mc_kl.shape, "  mean:", float(mc_kl.mean()))