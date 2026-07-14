"""Self-play worker process for parallel CarRacing-v3 episode collection.

Each worker owns its own ``gymnasium`` environment and never touches the
accelerator or the RSSM/Policy weights directly: it converts each frame into
a raw action by sending it to the shared batched :mod:`inference_server`
(via :class:`RemotePolicy`) and blocking for the response. This mirrors
``silver_alpha_zero``'s ``RemoteEvaluator`` pattern, except there is no local
game model needed — ``gymnasium`` owns all the CarRacing dynamics.

Workers sit idle until the trainer sends a ``("collect", num_episodes)``
command on their ``command_queue``; they then collect exactly that many
episodes, push each one onto the shared ``result_queue`` tagged with
``worker_id``, and go back to waiting. This lets the trainer block until a
whole batch of fresh episodes has arrived before starting a training phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Expose the shared inference-server module (sibling package under src/).
sys.path.insert(0, str(Path(__file__).parent.parent / "multithreading"))

import numpy as np
from PIL import Image

from inference_server import QueueData

IMAGE_SIZE = (64, 64)  # MDNEncoder architecture is fixed to 64×64 input


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize a (H, W, 3) uint8 frame to IMAGE_SIZE uint8."""
    img = Image.fromarray(frame).resize(IMAGE_SIZE, Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class RemotePolicy:
    """Client for the shared inference server.

    Turns a frame into a bounded action via one blocking request/response
    round trip. The worker is single-threaded, so at most one request is
    outstanding at a time; responses are matched by ``request_id`` as a
    safety check.
    """

    def __init__(self, worker_id: int, request_queue, response_queue) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self._next_request_id = 0

    def act(self, frame: np.ndarray) -> np.ndarray:
        request_id = self._next_request_id
        self._next_request_id += 1
        self.request_queue.put(QueueData(self.worker_id, request_id, frame))
        while True:
            response_id, action = self.response_queue.get()
            if response_id == request_id:
                return np.asarray(action, dtype=np.float32)
            print(
                f"[worker {self.worker_id}] stale response id={response_id}, "
                f"expected {request_id} — discarding"
            )


def collect_episode(
    env,
    remote_policy: RemotePolicy,
    *,
    action_repeat: int = 2,
    max_steps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect one episode using the remote policy.

    At each step: preprocess the latest frame, query the remote policy for a
    bounded (tanh-squashed) action, repeat the action *action_repeat* times,
    and record the result. Exploration comes from the SAC policy's own
    stochasticity (max-entropy objective) — no separate ε-greedy noise. The
    action stored in the buffer is exactly the one applied to the env
    (already in-bounds), so the critic/world model see the same bounded
    actions the policy produces.

    Returns
    -------
    images   : uint8  (T, 64, 64, 3)
    actions  : float32 (T, action_dim)
    rewards  : float32 (T,)
    continues: float32 (T,)   1 = episode not done, 0 = done
    """
    obs, _ = env.reset()
    images, actions, rewards, continues = [], [], [], []

    done = False
    step = 0
    while not done and step < max_steps:
        frame = preprocess_frame(obs)

        action = remote_policy.act(frame)

        total_reward = 0.0
        last_obs = obs
        for _ in range(action_repeat):
            last_obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            done = terminated or truncated
            if done:
                break

        images.append(frame)
        actions.append(action)
        rewards.append(float(total_reward))
        continues.append(0.0 if done else 1.0)
        obs = last_obs
        step += 1

    return (
        np.stack(images),
        np.stack(actions),
        np.array(rewards, dtype=np.float32),
        np.array(continues, dtype=np.float32),
    )


def run_worker(
    worker_id: int,
    command_queue,
    request_queue,
    response_queue,
    result_queue,
    stop_event,
    sp_config: dict,
) -> None:
    """Collect episodes on command until ``stop_event`` is set.

    Blocks on ``command_queue`` for ``("collect", num_episodes)``; runs that
    many episodes (each pushed individually onto ``result_queue`` as
    ``(worker_id, images, actions, rewards, continues)``), then waits for the
    next command. Never touches the accelerator — the model lives in the
    inference server; this process only runs the CPU-bound gym env + PIL
    preprocessing.
    """
    import os
    import queue as pyqueue

    # Keep the GPU reserved for the inference server/trainer; this process is
    # CPU-only (it never imports jax).
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import gymnasium as gym

    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)
    remote_policy = RemotePolicy(worker_id, request_queue, response_queue)

    try:
        while not stop_event.is_set():
            try:
                cmd = command_queue.get(timeout=0.1)
            except pyqueue.Empty:
                continue

            kind, *args = cmd
            if kind == "collect":
                (num_episodes,) = args
                for _ in range(num_episodes):
                    if stop_event.is_set():
                        break
                    images, actions, rewards, continues = collect_episode(
                        env, remote_policy,
                        action_repeat=sp_config["action_repeat"],
                        max_steps=sp_config["max_episode_steps"],
                    )
                    result_queue.put((worker_id, images, actions, rewards, continues))
            elif kind == "stop":
                break
    finally:
        env.close()
