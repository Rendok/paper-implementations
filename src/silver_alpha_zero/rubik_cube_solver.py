import math
import random
from flax import nnx
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import magiccube
import orbax.checkpoint as ocp

from cube_viz import save_trajectory_gif
from mcst import Action, Evaluator, Model, MonteCarloTreeSearch, State
from policy import Policy, PolicyRubikCubeEvaluator


# One face per axis (R: x, U: y, F: z). Combined with per-layer turns this
# generates every single-layer move on a cube of any size, with no duplication
# across axes. Slice (M/E/S) and whole-cube (X/Y/Z) moves are intentionally
# excluded: they are decomposable into these turns and never shorten a solution.
_AXIS_FACES = ("R", "U", "F")
_TURNS = ("", "'", "2")  # 90 clockwise, 90 counter-clockwise, 180


@dataclass(frozen=True)
class RubikCubeAction(Action):
    """A Rubik's Cube move in magiccube notation, e.g. "R", "U'", "2R2"."""

    notation: str

    def __post_init__(self) -> None:
        notation = self.notation.strip()
        if not notation:
            raise ValueError("RubikCubeAction notation cannot be empty")
        object.__setattr__(self, "notation", notation)

    def __str__(self) -> str:
        return self.notation


class RubikCubeState(State):
    def __init__(self, cube: magiccube.Cube):
        self.cube = cube

    def get(self) -> str:
        return self.cube.get()

    def is_terminal(self) -> bool:
        return self.cube.is_done()

    def reward(self) -> float:
        return 1.0 if self.cube.is_done() else 0.0

    def __hash__(self) -> int:
        return hash((self.cube.size, self.cube.get()))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RubikCubeState):
            return NotImplemented
        return self.cube.size == other.cube.size and self.cube.get() == other.cube.get()

    def __str__(self) -> str:
        return str(self.cube)


class RubikCubeModel(Model):
    """MDP dynamics for a single NxNxN Rubik's Cube.

    The cube size is fixed at construction; the action set (every layer turn is
    always legal) depends only on that size, so it is built once.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._actions = self._build_actions(size)

    def legal_actions(self, state: State) -> list[Action]:
        return self._actions

    @staticmethod
    def _build_actions(size: int) -> list[Action]:
        actions: list[Action] = []
        for face in _AXIS_FACES:
            for layer in range(1, size + 1):
                prefix = "" if layer == 1 else str(layer)
                for turn in _TURNS:
                    actions.append(RubikCubeAction(f"{prefix}{face}{turn}"))
        return actions

    def step(self, state: State, action: Action) -> State:
        assert isinstance(state, RubikCubeState)
        next_cube = magiccube.Cube(self.size, state=state.cube.get())
        next_cube.rotate(str(action))
        return RubikCubeState(next_cube)


class RandomRolloutRubikCubeEvaluator(Evaluator):
    """Baseline evaluator: uniform action priors and a random-rollout value.

    This stands in for the AlphaZero network. Random rollouts essentially never
    solve a scrambled cube, so the value is almost always 0; swap this out for a
    learned model to get a useful signal.
    """

    def __init__(
        self,
        model: Model,
        rollout_depth: int = 30,
        discount: float = 0.99,
        rng: random.Random | None = None,
    ):
        self.model = model
        self.rollout_depth = rollout_depth
        self.discount = discount
        self.rng = rng or random.Random()

    def evaluate(self, state: State) -> tuple[dict[Action, float], float]:
        actions = self.model.legal_actions(state)
        prior = 1.0 / len(actions) if actions else 0.0
        priors = {action: prior for action in actions}
        return priors, self._rollout(state)

    def _rollout(self, state: State) -> float:
        if state.is_terminal():
            return state.reward()
        current = state
        for depth in range(self.rollout_depth):
            actions = self.model.legal_actions(current)
            if not actions:
                break
            current = self.model.step(current, self.rng.choice(actions))
            if current.is_terminal():
                return self.discount**depth * current.reward()
        return 0.0


class HeuristicRubikCubeEvaluator(Evaluator):
    """Priors favor moves that reduce the Hamming distance to a solved cube.

    The Hamming distance counts misplaced stickers: per face, the number of
    stickers whose color differs from that face's dominant color. It is 0 for
    any solved orientation (a 2x2 has no fixed centers, so orientation is free).

    Action priors are a softmax over the negative distance of the *resulting*
    state (lower distance => higher probability); the value is a bounded,
    decreasing function of the current state's distance.
    """

    def __init__(self, model: Model, temperature: float = 2.0):
        self.model = model
        self.temperature = temperature

    def evaluate(self, state: State) -> tuple[dict[Action, float], float]:
        actions = self.model.legal_actions(state)
        if not actions:
            return {}, self._value(self._hamming(state))
        next_distances = [self._hamming(self.model.step(state, action)) for action in actions]
        priors = self._softmax(actions, next_distances)
        return priors, self._value(self._hamming(state))

    def _softmax(self, actions: list[Action], distances: list[int]) -> dict[Action, float]:
        logits = [-d / self.temperature for d in distances]
        shift = max(logits)  # for numerical stability
        weights = [math.exp(logit - shift) for logit in logits]
        total = sum(weights)
        return {action: weight / total for action, weight in zip(actions, weights)}

    @staticmethod
    def _value(distance: int) -> float:
        # 1.0 when solved (distance 0), decreasing toward 0 as the cube gets
        # further from solved; matches the reward scale (solved == 1.0).
        return 1.0 / (1.0 + distance)

    @staticmethod
    def _hamming(state: State) -> int:
        assert isinstance(state, RubikCubeState)
        misplaced = 0
        for face in state.cube.get_all_faces().values():
            stickers = [color for row in face for color in row]
            dominant = max(set(stickers), key=stickers.count)
            misplaced += sum(1 for color in stickers if color != dominant)
        return misplaced


def solve_rubik_cube(initial_state: RubikCubeState, model: Model, mcts: MonteCarloTreeSearch, max_trajectory: int = 500, num_simulation_rollouts: int = 200) -> list[Action]:
    actions = []
    state = initial_state
    for _ in tqdm(range(max_trajectory)):
        if state.is_terminal():
            break
        action, _ = mcts.search(state, num_simulation_rollouts)
        actions.append(action)
        state = model.step(state, action)
    return actions


def load_policy_checkpoint(
    policy: Policy,
    checkpoint_dir: str | Path,
    step: int | None = None,
) -> int:
    """Restore ``policy`` weights from an Orbax checkpoint saved by ``train_policy``.

    Returns the checkpoint step that was loaded. Uses the latest step when
    ``step`` is ``None``.
    """
    directory = Path(checkpoint_dir).expanduser().resolve()
    manager = ocp.CheckpointManager(
        directory,
        options=ocp.CheckpointManagerOptions(create=True),
    )
    target_step = manager.latest_step() if step is None else step
    if target_step is None:
        raise FileNotFoundError(f"No checkpoints found in {directory}")
    restored = manager.restore(
        target_step,
        args=ocp.args.StandardRestore(nnx.state(policy)),
    )
    nnx.update(policy, restored)
    return target_step


if __name__ == "__main__":
    size = 2
    checkpoint_dir = Path(__file__).resolve().parents[2] / "checkpoints/silver_alpha_zero"
    checkpoint_step = None  # latest; set e.g. 60 to load a specific step

    cube = magiccube.Cube(size)
    cube.scramble(5)
    initial_state = RubikCubeState(cube)
    print("scrambled:")
    print(initial_state)
    print("state_dim:", len(initial_state.get()))
    

    model = RubikCubeModel(size)
    action_dim = len(model.legal_actions(initial_state))
    print("action_dim:", action_dim)

    policy = Policy(
        num_embeddings=len(magiccube.Color),
        state_dim=len(cube.get()),
        action_dim=action_dim,
        embed_dim=256,
        num_transformer_blocks=16,
        num_heads=4,
        rngs=nnx.Rngs(0),
    )
    loaded_step = load_policy_checkpoint(policy, checkpoint_dir, step=checkpoint_step)
    print(f"loaded checkpoint step {loaded_step} from {checkpoint_dir}")

    evaluator = PolicyRubikCubeEvaluator(model, policy)
    mcts = MonteCarloTreeSearch(model, evaluator, c_puct=1.5)

    max_trajectory = 500
    num_simulation_rollouts = 200
    actions = solve_rubik_cube(initial_state, model, mcts, max_trajectory, num_simulation_rollouts)

    print("trajectory:", " ".join(str(a) for a in actions) or "(empty)")
    here = Path(__file__).parent
    print(f"saved {save_trajectory_gif(initial_state.cube, actions, here / 'cube_solution.gif', mode='2d')}")
    print(f"saved {save_trajectory_gif(initial_state.cube, actions, here / 'cube_solution_3d.gif', mode='3d')}")
