"""Batched inference server for the AlphaZero target policy.

Runs in its own process and owns the *target* copy of the policy on the GPU.
Self-play workers (separate processes) send single-state evaluation requests; the
server collects whatever requests are pending, runs them through the network as
one batch, and routes each result back to the requesting worker. The trainer
pushes fresh weights every N steps via ``weight_queue`` (only the latest update
is applied). This decouples MCTS tree work (cheap, parallel, CPU-bound in the
workers) from network evaluation (batched on the accelerator).

Request format (on ``request_queue``):  ``(worker_id, request_id, state_indices)``
  * ``state_indices`` - ``np.ndarray[state_dim]`` of integer sticker colors.
Response format (on ``response_queues[worker_id]``):  ``(request_id, probs, value)``
  * ``probs`` - ``np.ndarray[action_dim]`` softmax over the full action head;
  * ``value`` - python float.
"""

from __future__ import annotations

import queue as pyqueue
from multiprocessing import Event as MpEvent
from multiprocessing import Queue as MpQueue


def _apply_weights(nnx, policy, pure_dict: dict) -> None:
    state = nnx.state(policy)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(policy, state)


def run_inference_server(
    policy_config: dict,
    request_queue: "MpQueue",
    response_queues: "list[MpQueue]",
    weight_queue: "MpQueue",
    stop_event: "MpEvent",
    ready_event: "MpEvent",
    max_batch: int,
    poll_timeout: float = 0.005,
) -> None:
    """Serve batched policy evaluations until ``stop_event`` is set.

    Blocks once on ``weight_queue`` for the initial weights so the target starts
    identical to the trainer's live policy, then signals ``ready_event``.
    """
    import os

    # Two JAX processes (this server + the trainer) share one GPU, so neither may
    # greedily preallocate the whole device.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from policy import Policy

    policy = Policy(rngs=nnx.Rngs(policy_config.get("seed", 0)), **_policy_kwargs(policy_config))

    # Pre-split the module: ``nnx.jit`` re-flattens the whole module graph on every
    # call (~ms of pure-Python overhead), which dominates this hot loop. Instead we
    # split once into a static ``graphdef`` + array ``state`` and drive a plain
    # ``jax.jit`` function, re-splitting only when weights change.
    @partial(jax.jit, static_argnums=0)
    def forward(graphdef, state, indices):
        logits, value = nnx.merge(graphdef, state)(indices)
        return jax.nn.softmax(logits, axis=-1), value

    # Block for the initial weights pushed by the trainer, then announce readiness.
    _apply_weights(nnx, policy, weight_queue.get())
    graphdef, state = nnx.split(policy)
    ready_event.set()

    while not stop_event.is_set():
        # Apply only the most recent pending weight update.
        latest_weights = None
        try:
            while True:
                latest_weights = weight_queue.get_nowait()
        except pyqueue.Empty:
            pass
        if latest_weights is not None:
            _apply_weights(nnx, policy, latest_weights)
            graphdef, state = nnx.split(policy)

        # Block briefly for the first request, then greedily drain the rest so we
        # evaluate as large a batch as is currently available.
        try:
            batch = [request_queue.get(timeout=poll_timeout)]
        except pyqueue.Empty:
            continue
        while len(batch) < max_batch:
            try:
                batch.append(request_queue.get_nowait())
            except pyqueue.Empty:
                break

        # batch is a list of (worker_id, request_id, state_indices). Its length
        # is data-dependent (1..max_batch), which would make XLA recompile per
        # distinct size. Pad up to the next power-of-two bucket (capped at
        # max_batch) so only ~log2(max_batch) fixed shapes are ever compiled.
        n = len(batch)
        bucket = min(1 << (n - 1).bit_length(), max_batch)
        states_np = np.stack([item[2] for item in batch])
        if bucket > n:
            padding = np.zeros((bucket - n, states_np.shape[1]), dtype=states_np.dtype)
            states_np = np.concatenate([states_np, padding])

        states = jnp.asarray(states_np, dtype=jnp.int32)
        probs, values = forward(graphdef, state, states)
        # Drop the padded rows before responding. Transfer to host *first*, then
        # slice in NumPy: slicing the device arrays with the data-dependent count
        # ``n`` triggers a fresh ``dynamic_slice`` XLA compile for every distinct
        # batch size (1..max_batch), which trickles compilation cost throughout
        # serving and stalls the workers waiting on responses.
        probs = np.asarray(probs)[:n]
        values = np.asarray(values)[:n]

        for (worker_id, request_id, _), prob_row, value in zip(batch, probs, values):
            response_queues[worker_id].put((request_id, prob_row, float(value)))


def _policy_kwargs(policy_config: dict) -> dict:
    """Subset of ``policy_config`` accepted by the ``Policy`` constructor."""
    keys = (
        "num_embeddings",
        "state_dim",
        "action_dim",
        "embed_dim",
        "num_transformer_blocks",
        "num_heads",
    )
    return {key: policy_config[key] for key in keys}
