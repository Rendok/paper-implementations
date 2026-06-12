"""Self-play worker process for parallel AlphaZero data generation.

Each worker runs MCTS self-play episodes on the CPU. Instead of holding a copy of
the network, it uses :class:`RemoteEvaluator`, which forwards every leaf
evaluation to the shared :mod:`inference_server` and blocks for the batched
result. Completed episodes are pushed to ``result_queue`` as numpy
``(states, policies, values)`` triples for the trainer to drop into the replay
buffer.

Workers never touch the accelerator: GPU access is disabled at process start so
the device is reserved for the inference server and the trainer.
"""

from __future__ import annotations

import magiccube
import numpy as np

from mcst import Action, Evaluator, Model, State

# Sticker color letter -> integer index (magiccube's Color ordering), matching
# the embedding indices the network was trained on. Pure-python, no JAX.
_COLOR_TO_INDEX = {color.name: color.value for color in magiccube.Color}


def state_to_indices(state_string: str) -> np.ndarray:
    """Convert a cube facelet string to an ``int8`` index array (no JAX)."""
    return np.fromiter(
        (_COLOR_TO_INDEX[char] for char in "".join(state_string.split())),
        dtype=np.uint8,
    )


class RemoteEvaluator(Evaluator):
    """Evaluator that delegates to the batched inference server over queues.

    The worker is single-threaded, so at most one request is outstanding at a
    time; responses are matched by ``request_id`` only as a safety check.
    """

    def __init__(self, model: Model, worker_id: int, request_queue, response_queue):
        self.model = model
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_request_id = 0

    def evaluate(self, state: State) -> tuple[dict[Action, float], float]:
        legal_actions = self.model.legal_actions(state)
        request_id = self._next_request_id
        self._next_request_id += 1

        self.request_queue.put((self.worker_id, request_id, state_to_indices(state.get())))
        while True:
            response_id, probs, value = self.response_queue.get()
            if response_id == request_id:
                break
            print(f"Received wrong response: Response ID: {response_id}, Request ID: {request_id}")

        # Restrict to legal actions (same fixed order) and renormalize.
        # row = np.asarray(probs[: len(legal_actions)], dtype=np.float64)
        # total = row.sum()
        # All actions are legal for now.
        assert probs.shape[0] == len(legal_actions)

        # row = row / total if total > 0 else np.full(len(legal_actions), 1.0 / len(legal_actions))
        priors = {action: float(prob) for action, prob in zip(legal_actions, probs)}
        return priors, float(value)


def _self_play_episode(model, mcts, sp_config: dict, rng: np.random.Generator):
    """Play one episode; return ``(states, policies, values)`` or ``None`` if the
    scramble produced an already-solved cube (no training signal)."""
    from rubik_cube_solver import RubikCubeState  # cached; imported off the GPU

    cube = magiccube.Cube(sp_config["cube_size"])
    cube.scramble(sp_config["scramble_depth"])
    state = RubikCubeState(cube)

    legal_actions = model.legal_actions(state)
    action_to_index = {action: i for i, action in enumerate(legal_actions)}

    state_strings: list[str] = []
    policies: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(sp_config["max_trajectory"]):
        if state.is_terminal():
            break
        best_action, action_distribution = mcts.search(
            state, sp_config["num_simulation_rollouts"], sp_config["tau"]
        )
        policy_vec = np.zeros(len(legal_actions), dtype=np.float32)
        for action, prob in action_distribution.items():
            policy_vec[action_to_index[action]] = prob
        state_strings.append(state.get())
        policies.append(policy_vec)

        # action = legal_actions[rng.choice(len(legal_actions), p=policy_vec)]
        state = model.step(state, best_action)
        rewards.append(state.reward())

    if not state_strings:
        return None

    # Value target: discounted future return, unrolled from the back.
    values = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + mcts.discount * g
        values[t] = g

    states = np.stack([state_to_indices(s) for s in state_strings]).astype(np.uint8)
    return states, np.asarray(policies, dtype=np.float32), values


def _greedy_eval_episode(model, mcts, sp_config: dict, seed: int) -> tuple[int, bool]:
    """Play one greedy (``tau=0``) episode with a fixed scramble seed.

    Returns ``(trajectory_length, solved)``. Unsolved runs report
    ``max_trajectory`` as the length.
    """
    import random

    from rubik_cube_solver import RubikCubeState

    random.seed(seed)
    cube = magiccube.Cube(sp_config["cube_size"])
    cube.scramble(sp_config["scramble_depth"])
    state = RubikCubeState(cube)
    if state.is_terminal():
        return 0, True

    max_trajectory = sp_config["max_trajectory"]
    steps = 0
    for _ in range(max_trajectory):
        if state.is_terminal():
            break
        action, _ = mcts.search(state, sp_config["num_simulation_rollouts"], tau=0.0)
        state = model.step(state, action)
        steps += 1

    solved = state.is_terminal() and state.reward() > 0
    return (steps if solved else max_trajectory), solved


def run_worker(
    worker_id: int,
    request_queue,
    response_queue,
    result_queue,
    command_queue,
    eval_result_queue,
    stop_event,
    sp_config: dict,
    num_workers: int,
) -> None:
    """Generate self-play episodes, or run greedy eval when commanded."""
    import os
    import queue as pyqueue

    # Keep the GPU reserved for the server/trainer; this process is CPU-only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["JAX_PLATFORMS"] = "cpu"

    from mcst import MonteCarloTreeSearch
    from rubik_cube_solver import RubikCubeModel

    model = RubikCubeModel(sp_config["cube_size"])
    evaluator = RemoteEvaluator(model, worker_id, request_queue, response_queue)
    mcts = MonteCarloTreeSearch(
        model, evaluator, c_puct=sp_config["c_puct"], discount=sp_config["discount"]
    )
    rng = np.random.default_rng(sp_config.get("seed_base", 0) + worker_id)

    while not stop_event.is_set():
        try:
            cmd = command_queue.get_nowait()
        except pyqueue.Empty:
            cmd = None

        if cmd is not None:
            kind = cmd[0]
            if kind == "eval":
                _, sync_id, eval_seed_base, eval_trials = cmd
                for trial in range(worker_id, eval_trials, num_workers):
                    length, solved = _greedy_eval_episode(
                        model, mcts, sp_config, eval_seed_base + trial
                    )
                    eval_result_queue.put((sync_id, length, solved))
            elif kind == "set_scramble_depth":
                sp_config["scramble_depth"] = cmd[1]
                if len(cmd) > 2:
                    sp_config["max_trajectory"] = cmd[2]
            continue

        trajectory = _self_play_episode(model, mcts, sp_config, rng)
        if trajectory is not None:
            result_queue.put(trajectory)
