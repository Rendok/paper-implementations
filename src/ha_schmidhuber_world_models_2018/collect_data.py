import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np


OUTPUT_PATH = Path("data/car_racing_observations.npz")
ACTIONS_PATH = Path("data/car_racing_actions.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to collect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("num_episodes must be at least 1.")

    env = gym.make("CarRacing-v3", render_mode="rgb_array", continuous=True)

    episode_observations: list[np.ndarray] = []
    episode_actions: list[np.ndarray] = []
    episode_rewards: list[float] = []

    try:
        for episode_idx in range(args.num_episodes):
            observation, _ = env.reset()
            obs_trajectory: list[np.ndarray] = [observation]
            act_trajectory: list[np.ndarray] = []
            episode_reward = 0.0

            episode_over = False
            while not episode_over:
                action = env.action_space.sample()
                act_trajectory.append(np.asarray(action, dtype=np.float32))
                observation, reward, terminated, truncated, _ = env.step(action)
                obs_trajectory.append(observation)
                episode_reward += reward
                episode_over = terminated or truncated

            episode_observations.append(np.stack(obs_trajectory))
            episode_actions.append(np.stack(act_trajectory))
            episode_rewards.append(episode_reward)
            print(
                f"Collected episode {episode_idx + 1}/{args.num_episodes} "
                f"with {len(act_trajectory)} steps and reward {episode_reward:.2f}"
            )
    finally:
        env.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    obs_archive = {f"episode_{i}": traj for i, traj in enumerate(episode_observations)}
    act_archive = {f"episode_{i}": traj for i, traj in enumerate(episode_actions)}
    np.savez(OUTPUT_PATH, **obs_archive)
    np.savez(ACTIONS_PATH, **act_archive)

    total_observations = sum(traj.shape[0] for traj in episode_observations)
    total_actions = sum(traj.shape[0] for traj in episode_actions)
    print(
        f"Saved {total_observations} observations across {args.num_episodes} "
        f"trajectories to {OUTPUT_PATH}"
    )
    print(
        f"Saved {total_actions} actions across {args.num_episodes} "
        f"trajectories to {ACTIONS_PATH}"
    )
    print(f"Episode lengths (steps): {[traj.shape[0] for traj in episode_actions]}")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    main()