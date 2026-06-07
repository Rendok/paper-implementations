import os

# Trainer and inference server are two JAX processes sharing one GPU, so disable
# greedy preallocation before JAX is imported anywhere in this process.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import multiprocessing as mp
import queue as pyqueue

import jax
import mlflow
import numpy as np
import optax
import magiccube

from flax import nnx
from jaxtyping import Array, Float, Int

from policy import Policy
from replay_buffer import ReplayBuffer
from rubik_cube_solver import RubikCubeModel, RubikCubeState
from inference_server import run_inference_server
from self_play_worker import run_worker


def _policy_loss(
    policy: Policy,
    states: Int[Array, "batch state_dim"],
    target_actions: Float[Array, "batch action_dim"],
    target_values: Float[Array, "batch"],
) -> tuple[Float[Array, ""], tuple[Float[Array, ""], Float[Array, ""]]]:
    action_logits, value = policy(states)
    # Policy head: cross-entropy against the MCTS visit distribution.
    policy_ce = optax.safe_softmax_cross_entropy(action_logits, target_actions).mean()
    # Value head: MSE against the observed (discounted) return.
    value_mse = optax.l2_loss(value, target_values).mean()
    loss = policy_ce + value_mse
    return loss, (policy_ce, value_mse)


@nnx.jit
def _train_step(
    policy: Policy,
    optimizer: nnx.Optimizer,
    states: Int[Array, "batch state_dim"],
    target_actions: Float[Array, "batch action_dim"],
    target_values: Float[Array, "batch"],
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """One JIT-compiled gradient step; mutates policy and optimizer in place."""
    (loss, (policy_ce, value_mse)), grads = nnx.value_and_grad(_policy_loss, has_aux=True)(
        policy, states, target_actions, target_values
    )
    optimizer.update(policy, grads)
    return loss, policy_ce, value_mse


def _extract_weights(policy: Policy) -> dict:
    """Pure-numpy nested dict of the policy's parameters, safe to send over a
    multiprocessing queue to the inference server."""
    pure = nnx.state(policy).to_pure_dict()
    return jax.tree.map(lambda x: np.asarray(x), pure)


def train_policy(
    policy: Policy,
    optimizer: nnx.Optimizer,
    replay_buffer: ReplayBuffer,
    config: dict,
    policy_config: dict,
):
    """Asynchronous AlphaZero training.

    Spawns one inference-server process (the target policy) and ``num_workers``
    self-play processes. The trainer ingests their trajectories into the replay
    buffer, runs gradient steps on the live policy, and pushes refreshed target
    weights to the server every ``target_sync_steps`` steps.

    Training is throttled by ``replay_ratio``: each newly ingested state adds
    ``replay_ratio / batch_size`` to a train budget, and a gradient step consumes
    one unit. This keeps optimization paced with self-play instead of starving
    the inference server by oversampling a small buffer.
    """
    ctx = mp.get_context("spawn")
    request_queue = ctx.Queue()
    weight_queue = ctx.Queue()
    result_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(config["num_workers"])]
    stop_event = ctx.Event()
    ready_event = ctx.Event()

    sp_config = {
        "cube_size": config["cube_size"],
        "scramble_depth": config["scramble_depth"],
        "max_trajectory": config["max_trajectory"],
        "num_simulation_rollouts": config["num_simulation_rollouts"],
        "tau": config["tau"],
        "c_puct": config["c_puct"],
        "discount": config["discount"],
        "seed_base": config.get("seed_base", 0),
    }

    server = ctx.Process(
        target=run_inference_server,
        args=(
            policy_config,
            request_queue,
            response_queues,
            weight_queue,
            stop_event,
            ready_event,
            config["max_inference_batch"],
        ),
        daemon=True,
    )
    server.start()

    # Seed the server with the live policy's current weights and wait until it is
    # serving before launching workers.
    weight_queue.put(_extract_weights(policy))
    ready_event.wait()

    workers = [
        ctx.Process(
            target=run_worker,
            args=(i, request_queue, response_queues[i], result_queue, stop_event, sp_config),
            daemon=True,
        )
        for i in range(config["num_workers"])
    ]
    for worker in workers:
        worker.start()

    recent_lengths: list[float] = []
    recent_solved: list[float] = []
    train_step = 0
    train_budget = 0.0
    try:
        while train_step < config["total_training_steps"]:
            # Drain freshly produced self-play episodes into the replay buffer.
            new_examples = 0
            for _ in range(config["ingest_budget"]):
                try:
                    states, policies, values = result_queue.get(timeout=0.01)
                except pyqueue.Empty:
                    break
                replay_buffer.add(states, policies, values)
                new_examples += len(values)
                recent_lengths.append(float(len(values)))
                recent_solved.append(float(len(values) > 0 and float(values[-1]) > 0.0))

            if new_examples > 0:
                train_budget += new_examples * config["replay_ratio"] / config["batch_size"]

            # Wait for enough warmup data before taking gradient steps.
            if replay_buffer.size < config["batch_size"] * 10:
                mlflow.log_metrics(
                    {
                        "replay_buffer/size": float(replay_buffer.size),
                        "train/train_budget": train_budget,
                    },
                    step=train_step,
                )
                continue

            if train_budget < 1.0:
                continue

            states, target_actions, target_values = replay_buffer.sample(config["batch_size"])
            loss, policy_ce, value_mse = _train_step(
                policy, optimizer, states, target_actions, target_values
            )
            train_budget -= 1.0
            mlflow.log_metrics(
                {
                    "train/loss": float(loss),
                    "train/policy_cross_entropy": float(policy_ce),
                    "train/value_mse": float(value_mse),
                    "train/train_budget": train_budget,
                },
                step=train_step,
            )
            if recent_lengths:
                mlflow.log_metrics(
                    {
                        "self_play/avg_trajectory_length": float(np.mean(recent_lengths)),
                        "self_play/solved_rate": float(np.mean(recent_solved)),
                        "self_play/episodes": float(len(recent_lengths)),
                        "self_play/new_examples": float(new_examples),
                        "replay_buffer/size": float(replay_buffer.size),
                    },
                    step=train_step,
                )
                recent_lengths.clear()
                recent_solved.clear()

            train_step += 1
            if train_step % config["target_sync_steps"] == 0:
                weight_queue.put(_extract_weights(policy))
    finally:
        stop_event.set()
        for worker in workers:
            worker.terminate()
        server.terminate()
        for worker in workers:
            worker.join(timeout=5)
        server.join(timeout=5)

    return policy


if __name__ == "__main__":
    """Examples with batch_size=32:

    replay_ratio=1.0 → 1 grad step per 32 new states (≈1 step per full batch of fresh data)
    replay_ratio=0.5 → 1 grad step per 64 new states (train half as often)
    replay_ratio=2.0 → 2 grad steps per 32 new states"""

    config = {
        "batch_size": 32,
        "total_training_steps": 1000,
        "cube_size": 2,
        "scramble_depth": 2,
        "max_trajectory": 10,
        "num_simulation_rollouts": 500,
        "tau": 1.0,
        "c_puct": 5.0,
        "discount": 0.99,
        "num_workers": 32,
        "target_sync_steps": 10,
        "max_inference_batch": 64,
        "ingest_budget": 64,
        "replay_ratio": 0.5,
        "seed_base": 0,
    }
    rules_model = RubikCubeModel(config["cube_size"])

    cube = magiccube.Cube(config["cube_size"])
    action_dim = len(rules_model.legal_actions(RubikCubeState(cube)))
    state_dim = len(cube.get())

    policy_config = {
        "num_embeddings": len(magiccube.Color),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "embed_dim": 128,
        "num_transformer_blocks": 8,
        "num_heads": 2,
        "seed": 0,
    }
    policy = Policy(rngs=nnx.Rngs(policy_config["seed"]), **{k: policy_config[k] for k in ("num_embeddings", "state_dim", "action_dim", "embed_dim", "num_transformer_blocks", "num_heads")})

    optimizer = nnx.Optimizer(policy, optax.adam(learning_rate=0.001), wrt=nnx.Param)
    replay_buffer = ReplayBuffer(capacity=2048, state_dim=state_dim, action_dim=action_dim, rngs=nnx.Rngs(1))

    mlflow.set_experiment("silver_alpha_zero")
    with mlflow.start_run(run_name="rubik_cube_alpha_zero", log_system_metrics=True):
        mlflow.log_params(config)
        train_policy(policy, optimizer, replay_buffer, config, policy_config)
