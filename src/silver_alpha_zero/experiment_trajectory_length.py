"""Experiment: average solution length vs number of simulations.

Sweeps several scramble depths and simulation budgets. For each setting it runs
`solve_rubik_cube` (iterative MCTS, one search per move) over many random
scrambles and averages the resulting solution length, then plots one line per
scramble depth.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # silence per-solve progress bars

import random
from pathlib import Path
from statistics import mean

import magiccube

from mcst import MonteCarloTreeSearch
from rubik_cube_solver import (
    RandomRolloutRubikCubeEvaluator,
    HeuristicRubikCubeEvaluator,
    RubikCubeModel,
    RubikCubeState,
    solve_rubik_cube,
)


def average_solution_length(
    model: RubikCubeModel,
    mcts: MonteCarloTreeSearch,
    size: int,
    scramble_depth: int,
    num_simulations: int,
    trials: int,
    base_seed: int,
    max_trajectory: int,
) -> float:
    """Mean solution length over `trials` random scrambles of a fixed depth.

    The scramble for trial `t` is fixed by `base_seed + t`, so the same set of
    scrambles is reused across simulation budgets (lower variance along x).
    Unsolved runs contribute `max_trajectory`.
    """
    lengths: list[int] = []
    for t in range(trials):
        seed = base_seed + t
        random.seed(seed)  # makes cube.scramble reproducible
        cube = magiccube.Cube(size)
        cube.scramble(scramble_depth)

        # model = RubikCubeModel(size)
        # evaluator = RandomRolloutRubikCubeEvaluator(model, rng=random.Random(seed))
        # evaluator = HeuristicRubikCubeEvaluator(model)
        # mcts = MonteCarloTreeSearch(model, evaluator)

        actions = solve_rubik_cube(
            RubikCubeState(cube),
            model,
            mcts,
            max_trajectory=max_trajectory,
            num_simulation_rollouts=num_simulations,
        )
        lengths.append(len(actions))
    return mean(lengths)


def run_experiment(
    size: int,
    scramble_depths: list[int],
    simulation_counts: list[int],
    trials: int,
    base_seed: int,
    max_trajectory: int,
) -> dict[int, list[float]]:
    model = RubikCubeModel(size)
    evaluator = RandomRolloutRubikCubeEvaluator(model, rng=random.Random(0)) # seed
    mcts = MonteCarloTreeSearch(model, evaluator)
    results: dict[int, list[float]] = {}
    for depth in scramble_depths:
        row: list[float] = []
        for num_simulations in simulation_counts:
            avg = average_solution_length(
                model, mcts, size, depth, num_simulations, trials, base_seed, max_trajectory
            )
            row.append(avg)
            print(f"depth={depth} sims={num_simulations:>4} -> avg_len={avg:.2f}")
        results[depth] = row
    return results


def plot_results(
    results: dict[int, list[float]],
    simulation_counts: list[int],
    path: str | Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for depth, lengths in results.items():
        ax.plot(simulation_counts, lengths, marker="o", label=f"scramble depth {depth}")
    ax.set_xlabel("number of MCTS simulations (per move)")
    ax.set_ylabel("average solution length")
    ax.set_title("MCTS solution length vs simulations")
    ax.grid(True, alpha=0.3)
    ax.legend(title="scramble")
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


if __name__ == "__main__":
    size = 2
    scramble_depths = [2, 4, 8, 16]
    simulation_counts = [10, 25, 50, 100]
    trials = 16
    base_seed = 0
    max_trajectory = 30

    results = run_experiment(
        size, scramble_depths, simulation_counts, trials, base_seed, max_trajectory
    )
    out = plot_results(
        results, simulation_counts, Path(__file__).parent / "trajectory_length_vs_simulations.png"
    )
    print(f"saved {out}")
