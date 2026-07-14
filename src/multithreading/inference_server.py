"""Generic batched inference server for any Flax NNX model.

Runs in its own process and owns one "live" copy of a model on the
accelerator. Worker processes each send single-example evaluation requests;
the server drains whatever requests are currently pending, evaluates them as
one batch, and routes each result back to the requesting worker. A trainer
can push fresh weights at any time via ``weight_queue`` (only the latest
pending update is ever applied).

Unlike a model-specific server, this module has **no built-in knowledge** of
input/output shapes, dtypes, or semantics (e.g. "logits" + "value"). All of
that is supplied by the caller:

* ``model_fn``/``model_kwargs`` - a picklable recipe for building the model
  *inside* the server process: ``model_fn(rngs=..., **model_kwargs)``.
  ``model_fn`` is typically an ``nnx.Module`` subclass, or a small top-level
  factory function; it must **not** be a closure over an already-built
  model/submodules — those can embed things (e.g. raw activation-function
  references picked up by ``nnx.Sequential``) that don't survive pickling
  across the ``spawn`` process boundary. Constructing fresh and syncing
  weights separately (via ``weight_queue``, always plain numpy) sidesteps
  that entirely — only ``model_fn`` (by reference) and ``model_kwargs``
  (plain primitives) ever need to pickle cleanly.
* ``postprocess_fn``  - turns the model's raw output into whatever should be
  sent back to workers (e.g. softmax the logits). Runs inside ``jax.jit``,
  so it must be built from JAX ops only.

This lets the same server drive a policy+value net, a plain classifier, a
Q-network taking ``(state, action)``, etc.

Request format (on ``request_queue``): ``QueueData(worker_id, request_id, data)``
  * ``data`` - one array per positional model argument: a single
    ``np.ndarray`` for single-input models, or a tuple of arrays for models
    that take several positional inputs (e.g. a Q-network's
    ``(state, action)``). Every request must supply the same shape/dtype
    (and, if a tuple, the same number of arrays).
Response format (on ``response_queues[worker_id]``): ``(request_id, *outputs)``
  * ``outputs`` - whatever ``postprocess_fn`` returned for that row (or the
    model's raw output(s), if ``postprocess_fn`` is omitted).

Every batch is padded up to exactly ``max_batch`` (never a smaller bucket),
so ``forward`` only ever gets called with one fixed input shape and XLA only
ever compiles a single trace, at the cost of always computing a full
``max_batch``-sized batch even when fewer requests are pending.
"""

from __future__ import annotations

import queue as pyqueue
from dataclasses import dataclass
from multiprocessing import Event as MpEvent
from multiprocessing import Queue as MpQueue
from typing import Any, Callable


@dataclass
class QueueData:
    """Request payload sent on ``request_queue``.

    ``data`` is a single ``np.ndarray`` for single-input models, or a tuple
    of ``np.ndarray`` (one per positional model argument) for models that
    take several inputs, e.g. ``(state, action)`` for a Q-network.
    """

    worker_id: int
    request_id: int
    data: Any  # np.ndarray | tuple[np.ndarray, ...]


def _apply_weights(nnx, model, pure_dict: dict) -> None:
    # Restrict to nnx.Param: other state (e.g. an nnx.Rngs counter used by a
    # stochastic layer) is local to this process and must keep evolving on
    # its own — see the ``forward``/``state`` threading below — not get
    # reset to whatever the trainer's counter happened to read at push time.
    state = nnx.state(model, nnx.Param)
    state.replace_by_pure_dict(pure_dict)
    nnx.update(model, state)


def extract_weights(model: "Any") -> dict:
    """Pure-numpy nested dict of ``model``'s ``nnx.Param``s, safe to send
    over a multiprocessing queue (call this on the trainer side, e.g.
    ``weight_queue.put(extract_weights(model))``)."""
    import jax
    import numpy as np
    from flax import nnx

    pure = nnx.state(model, nnx.Param).to_pure_dict()
    return jax.tree.map(lambda x: np.asarray(x), pure)


def run_inference_server(
    model_fn: Callable[..., Any],
    model_kwargs: dict,
    request_queue: "MpQueue",
    response_queues: "list[MpQueue]",
    weight_queue: "MpQueue",
    stop_event: "MpEvent",
    ready_event: "MpEvent",
    max_batch: int,
    *,
    postprocess_fn: Callable[..., Any] | None = None,
    poll_timeout: float = 0.005,
    seed: int = 0,
) -> None:
    """Serve batched evaluations of a model until ``stop_event`` is set.

    Builds the model *inside this process* via
    ``model_fn(rngs=nnx.Rngs(seed, noise=seed + 1), **model_kwargs)``, then
    blocks once on ``weight_queue`` for the initial weights so it starts
    identical to the trainer's live model, then signals ``ready_event``. The
    model's actual initial random weights don't matter — they're overwritten
    immediately by that first weight push.

    ``postprocess_fn(*raw_outputs) -> Any`` post-processes the model's raw
    output (``model(*inputs)``, unpacked if it was a tuple). Return a single
    array or a tuple of arrays; if omitted, the raw output is forwarded
    unchanged.
    """
    import os

    # Multiple JAX processes (this server, the trainer, …) may share one GPU,
    # so none of them may greedily preallocate the whole device.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    from functools import partial

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    model = model_fn(rngs=nnx.Rngs(seed, noise=seed + 1), **model_kwargs)

    if postprocess_fn is None:
        postprocess_fn = lambda *outputs: outputs if len(outputs) > 1 else outputs[0]

    # Pre-split the module: ``nnx.jit`` re-flattens the whole module graph on every
    # call (~ms of pure-Python overhead), which dominates this hot loop. Instead we
    # split once into a static ``graphdef`` + array ``state`` and drive a plain
    # ``jax.jit`` function, re-splitting only when weights change.
    #
    # ``state`` is threaded back out and reassigned after every call (see the
    # loop below) — not just discarded — so any internal mutable state the
    # model owns (e.g. an ``nnx.Rngs`` counter used for stochastic layers)
    # actually advances call-to-call, exactly as it would under plain eager
    # calls or ``nnx.jit``. Models with no internal state are unaffected.
    @partial(jax.jit, static_argnums=0)
    def forward(graphdef, state, *inputs):
        model = nnx.merge(graphdef, state)
        outputs = model(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return postprocess_fn(*outputs), nnx.state(model)

    # Block for the initial weights pushed by the trainer, then announce readiness.
    _apply_weights(nnx, model, weight_queue.get())
    graphdef, state = nnx.split(model)
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
            _apply_weights(nnx, model, latest_weights)
            graphdef, state = nnx.split(model)

        # Block briefly for the first request, then greedily drain the rest so we
        # evaluate as large a batch as is currently available.
        try:
            batch: "list[QueueData]" = [request_queue.get(timeout=poll_timeout)]
        except pyqueue.Empty:
            continue
        while len(batch) < max_batch:
            try:
                batch.append(request_queue.get_nowait())
            except pyqueue.Empty:
                break

        n = len(batch)
        multi_input = isinstance(batch[0].data, tuple)
        num_inputs = len(batch[0].data) if multi_input else 1

        # Always pad to the fixed max_batch shape (never a smaller bucket) so
        # ``forward`` only ever sees one input shape per input array — XLA
        # compiles exactly one trace, at the cost of always running a full
        # max_batch-sized forward pass even when fewer requests are pending.
        input_arrays = []
        for i in range(num_inputs):
            rows = [item.data[i] if multi_input else item.data for item in batch]
            arr = np.stack(rows)
            if n < max_batch:
                pad_shape = (max_batch - n,) + arr.shape[1:]
                arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)])
            input_arrays.append(jnp.asarray(arr))

        outputs, state = forward(graphdef, state, *input_arrays)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # Drop the padded rows before responding. Transfer to host *first*, then
        # slice in NumPy: slicing the device arrays with the data-dependent count
        # ``n`` triggers a fresh ``dynamic_slice`` XLA compile for every distinct
        # batch size (1..max_batch), which trickles compilation cost throughout
        # serving and stalls the workers waiting on responses.
        outputs_np = [np.asarray(out)[:n] for out in outputs]

        for row_idx, item in enumerate(batch):
            row = tuple(out[row_idx] for out in outputs_np)
            response_queues[item.worker_id].put((item.request_id, *row))
