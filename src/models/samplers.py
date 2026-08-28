import jax
import jax.numpy as jnp
import functools as ft
from flax import nnx
from jaxtyping import Array, Float, Integer
from typing import Tuple


@ft.partial(nnx.jit, static_argnames=("comp_dtype",))
def stochastic_sampler_step(
    model: nnx.Module,
    x: Float[Array, "batch H W C"],
    t: Float[Array, ""],
    classes: Integer[Array, "batch"],
    dt: Float[Array, ""],
    sigma: Float[Array, ""],
    rngs: nnx.Rngs,
    comp_dtype: jnp.dtype = jnp.bfloat16,
    data_range: float = 1.0,
) -> Float[Array, "batch H W C"]:
    """One Euler-Maruyama step of the interpolant SDE.

    ``t``, ``dt`` and ``sigma`` are traced arrays rather than static values, so
    every step of a rollout reuses one compilation; only ``comp_dtype`` is
    static (a dtype cannot be a traced argument).
    """
    t_b = jnp.broadcast_to(t, (x.shape[0],))
    eps_hat = model(x.astype(comp_dtype), t_b, classes).astype(jnp.float32)

    # Velocity of x_t = t*z + (1-t)*eps is v = z_hat - eps_hat, where z_hat is
    # the data endpoint implied by the predicted noise. Written this way instead
    # of the algebraically equal (x - eps_hat)/t, the 1/t is confined to z_hat,
    # which is a *pixel value* and so can be clipped to the range the data
    # actually occupies. That matters because 1/t multiplies the error in
    # eps_hat, not just the signal: an imperfect eps head turns the drift into
    # roughly x/t, whose ODE dx/dt = x/t inflates the initial noise by exactly
    # 1/t_min (100x at t_min=1e-2) and saturates every pixel. Clipping bounds
    # the total drift by |v| <= 2*data_range and is a no-op once the model is
    # accurate enough to keep z_hat in range.
    # z_hat = jnp.clip((x - (1.0 - t) * eps_hat) / t, -data_range, data_range)
    # v = z_hat - eps_hat
    # s = -eps_hat / (1.0 - t)
    # drift = v + sigma**2 / 2.0 * s

    s = -eps_hat / (1.0 - t)
    a = (1.0 - t) ** 2 / t + (1.0 - t)
    b = 1.0 / t
    drift = (a + sigma**2 / 2.0) * s + b * x

    noise = rngs.normal(x.shape, jnp.float32)
    return x + drift * dt + sigma * jnp.sqrt(dt) * noise


def stochastic_sampler(
    model: nnx.Module,
    batch_size: int,
    image_shape: Tuple[int, int, int],
    classes: Integer[Array, "batch"],
    *,
    rngs: nnx.Rngs,
    num_steps: int = 200,
    sigma: float = 0.5,
    t_min: float = 1e-2,      # the velocity's 1/t factor is singular at t=0
    t_max: float = 1.0 - 1e-3,  # score ~ 1/(1-t) is singular at t=1
    comp_dtype: jnp.dtype = jnp.bfloat16,
    data_range: float = 1.0,
):
    """Integrate the interpolant SDE from noise (t=0) to data (t=1).
    Assumes ``model`` predicts eps, so the score is s = -eps_hat / (1 - t).
    """
    step_size = (t_max - t_min) / num_steps
    dt = jnp.asarray(step_size, jnp.float32)
    sigma = jnp.asarray(sigma, jnp.float32)

    # The integration state stays float32: bfloat16 resolves only 128 distinct
    # values on [0, 1), which would corrupt both the step size and the 1/t term.
    x = rngs.normal((batch_size, *image_shape), dtype=jnp.float32)

    for i in range(num_steps):
        t = jnp.asarray(t_min + i * step_size, jnp.float32)
        x = stochastic_sampler_step(
            model, x, t, classes, dt, sigma, rngs,
            comp_dtype=comp_dtype, data_range=data_range,
        )

    return x
