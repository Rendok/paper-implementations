import os
from pathlib import Path

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
import orbax.checkpoint as ocp

from flax import nnx
from jaxtyping import Array, Float, Int

from policy import Policy
from replay_buffer import ReplayBuffer
from rubik_cube_solver import RubikCubeModel, RubikCubeState
from inference_server import run_inference_server
from self_play_worker import run_worker


class AsyncEval:
    """Fire-and-forget greedy (``tau=0``) evaluation on the self-play workers.

    ``dispatch`` broadcasts an eval request and returns immediately. Workers run
    their share of the trials between self-play episodes and stream results back
    on ``eval_result_queue``. ``collect`` is polled (non-blocking) each training
    iteration and returns aggregated metrics only once every trial for the
    in-flight request has arrived, so gradient steps never stall on evaluation.
    """

    def __init__(
        self,
        command_queues: list,
        eval_result_queue,
        eval_trials: int,
        eval_seed_base: int,
    ):
        self._command_queues = command_queues
        self._eval_result_queue = eval_result_queue
        self._eval_trials = eval_trials
        self._eval_seed_base = eval_seed_base
        self._sync_id: int | None = None
        self._lengths: list[int] = []
        self._solved: list[float] = []

    @property
    def busy(self) -> bool:
        return self._sync_id is not None

    def dispatch(self, sync_id: int) -> None:
        """Broadcast an eval request to all workers (returns immediately)."""
        if self.busy:
            return  # an eval is still running; don't overlap requests
        self._sync_id = sync_id
        self._lengths.clear()
        self._solved.clear()
        for command_queue in self._command_queues:
            command_queue.put(("eval", sync_id, self._eval_seed_base, self._eval_trials))

    def collect(self) -> tuple[int, dict[str, float]] | None:
        """Drain finished trials; return ``(step, metrics)`` once all arrive."""
        if not self.busy:
            return None
        while True:
            try:
                sync_id, length, solved = self._eval_result_queue.get_nowait()
            except pyqueue.Empty:
                break
            if sync_id != self._sync_id:
                continue  # stale result from a previous request
            self._lengths.append(length)
            self._solved.append(float(solved))

        if len(self._lengths) < self._eval_trials:
            return None

        step = self._sync_id
        metrics = {
            "eval/greedy_avg_trajectory_length": float(np.mean(self._lengths)),
            "eval/greedy_solved_rate": float(np.mean(self._solved)),
        }
        self._sync_id = None
        return step, metrics


def _max_trajectory_for_depth(depth: int) -> int:
    return depth * 3


def _max_trajectory_schedule(scramble_depth_schedule: dict[int, int]) -> dict[int, int]:
    return {
        step: _max_trajectory_for_depth(depth)
        for step, depth in scramble_depth_schedule.items()
    }


def _set_scramble_depth(command_queues: list, depth: int, train_step: int) -> None:
    """Push curriculum settings to all self-play workers and log them."""
    max_trajectory = _max_trajectory_for_depth(depth)
    for command_queue in command_queues:
        command_queue.put(("set_scramble_depth", depth, max_trajectory))
    mlflow.log_metric("curriculum/scramble_depth", depth, step=train_step)
    mlflow.log_metric("curriculum/max_trajectory", max_trajectory, step=train_step)
    print(
        f"train_step {train_step}: scramble_depth -> {depth}, "
        f"max_trajectory -> {max_trajectory}"
    )


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
) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
    """One JIT-compiled gradient step; mutates policy and optimizer in place."""
    (loss, (policy_ce, value_mse)), grads = nnx.value_and_grad(_policy_loss, has_aux=True)(
        policy, states, target_actions, target_values
    )
    # Pre-clip global gradient norm, so the clip threshold can be set from data.
    grad_norm = optax.global_norm(grads)
    optimizer.update(policy, grads)
    return loss, policy_ce, value_mse, grad_norm


def _extract_weights(policy: Policy) -> dict:
    """Pure-numpy nested dict of the policy's parameters, safe to send over a
    multiprocessing queue to the inference server."""
    pure = nnx.state(policy).to_pure_dict()
    return jax.tree.map(lambda x: np.asarray(x), pure)


def _create_checkpoint_manager(
    directory: str | Path, *, max_to_keep: int = 5
) -> ocp.CheckpointManager:
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True)
    return ocp.CheckpointManager(directory, options=options)


def _save_policy_checkpoint(
    manager: ocp.CheckpointManager, policy: Policy, step: int
) -> Path:
    """Persist the live policy (same weights pushed to the target) at ``step``."""
    state = nnx.state(policy)
    manager.save(step, args=ocp.args.StandardSave(state))
    manager.wait_until_finished()
    return Path(manager.directory) / str(step)


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
    eval_result_queue = ctx.Queue()
    response_queues = [ctx.Queue() for _ in range(config["num_workers"])]
    command_queues = [ctx.Queue() for _ in range(config["num_workers"])]
    stop_event = ctx.Event()
    ready_event = ctx.Event()
    checkpoint_manager = _create_checkpoint_manager(
        config["checkpoint_dir"],
        max_to_keep=config.get("checkpoint_max_to_keep", 5),
    )

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
            args=(
                i,
                request_queue,
                response_queues[i],
                result_queue,
                command_queues[i],
                eval_result_queue,
                stop_event,
                sp_config,
                config["num_workers"],
            ),
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
    scramble_depth_schedule = config.get("scramble_depth_schedule", {})

    # async_eval = AsyncEval(
    #     command_queues,
    #     eval_result_queue,
    #     config["eval_trials"],
    #     config["eval_seed_base"],
    # )
    # async_eval.dispatch(sync_id=train_step)  # baseline eval at step 0
    try:
        while train_step < config["total_training_steps"]:
            # Pick up any finished evaluation without blocking gradient steps.
            # done = async_eval.collect()
            # if done is not None:
            #     eval_step, eval_metrics = done
            #     mlflow.log_metrics(eval_metrics, step=eval_step)

            if train_step in scramble_depth_schedule:
                _set_scramble_depth(
                    command_queues,
                    scramble_depth_schedule[train_step],
                    train_step,
                )

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
            loss, policy_ce, value_mse, grad_norm = _train_step(
                policy, optimizer, states, target_actions, target_values
            )
            train_budget -= 1.0
            mlflow.log_metrics(
                {
                    "train/loss": float(loss),
                    "train/policy_cross_entropy": float(policy_ce),
                    "train/value_mse": float(value_mse),
                    "train/grad_norm": float(grad_norm),
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
                checkpoint_path = _save_policy_checkpoint(
                    checkpoint_manager, policy, train_step
                )
                # async_eval.dispatch(sync_id=train_step)
                print(f"saved target checkpoint step {train_step} -> {checkpoint_path}")
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
        "batch_size": 32, # training
        "max_inference_batch": 64,
        "total_training_steps": 2000,
        "cube_size": 2,
        "scramble_depth": 4,
        "scramble_depth_schedule": {
            # 150: 4,
            750: 8,
        },
        "max_trajectory": _max_trajectory_for_depth(4),
        "num_simulation_rollouts": 1000,
        "tau": 1.0,
        "c_puct": 1.5,
        "discount": 0.99,
        "num_workers": 64,
        "target_sync_steps": 60,
        "eval_trials": 32,
        "eval_seed_base": 100_000,
        "ingest_budget": 64,
        "replay_ratio": 1.0,
        "seed_base": 0,
        "grad_clip_norm": 10.0,  # loose guard for spikes; tune via train/grad_norm
        "buffer_capacity": 2**13,
        "checkpoint_dir": "checkpoints/silver_alpha_zero",
        "checkpoint_max_to_keep": 5,
    }
    rules_model = RubikCubeModel(config["cube_size"])

    cube = magiccube.Cube(config["cube_size"])
    action_dim = len(rules_model.legal_actions(RubikCubeState(cube)))
    state_dim = len(cube.get())

    policy_config = {
        "num_embeddings": len(magiccube.Color),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "embed_dim": 256,
        "num_transformer_blocks": 16,
        "num_heads": 4,
        "seed": 0,
    }
    policy = Policy(rngs=nnx.Rngs(policy_config["seed"]), **{k: policy_config[k] for k in ("num_embeddings", "state_dim", "action_dim", "embed_dim", "num_transformer_blocks", "num_heads")})

    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,            # Learning rate at start of warmup
        peak_value=1e-4,           # Maximum learning rate
        warmup_steps=50,         # Steps to reach peak value
        decay_steps=config["total_training_steps"],         # Steps for the cosine decay phase
        end_value=0.0,             # Final learning rate
    )
    # tx = optax.chain(
    #     optax.clip_by_global_norm(config["grad_clip_norm"]),
    #     optax.adamw(learning_rate=schedule_fn),
    # )
    tx = optax.adamw(learning_rate=schedule_fn)
    optimizer = nnx.Optimizer(policy, tx, wrt=nnx.Param)
    replay_buffer = ReplayBuffer(capacity=config["buffer_capacity"], state_dim=state_dim, action_dim=action_dim, rngs=nnx.Rngs(1))

    mlflow.set_experiment("silver_alpha_zero")
    with mlflow.start_run(run_name="rubik_cube_alpha_zero", log_system_metrics=True):
        schedule = config.pop("scramble_depth_schedule", {})
        max_trajectory_schedule = _max_trajectory_schedule(schedule)
        mlflow.log_params(config)
        mlflow.log_param("scramble_depth_schedule", str(schedule))
        mlflow.log_param("max_trajectory_schedule", str(max_trajectory_schedule))
        config["scramble_depth_schedule"] = schedule
        train_policy(policy, optimizer, replay_buffer, config, policy_config)
