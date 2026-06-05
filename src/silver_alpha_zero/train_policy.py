import mlflow
import numpy as np
import optax
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import magiccube

from rubik_cube_solver import RubikCubeModel, RubikCubeState
from policy import Policy, states_to_indices, PolicyRubikCubeEvaluator
from replay_buffer import ReplayBuffer
from flax import nnx
from tqdm import tqdm
from jaxtyping import Array, Float, Int
from mcst import Model, State, Action, MonteCarloTreeSearch


def self_play(
    rules_model: Model,
    mcts: MonteCarloTreeSearch,
    cube_size: int,
    scramble_depth: int,
    max_trajectory: int,
    num_simulation_rollouts: int,
    tau: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Play one self-play episode and return AlphaZero training targets.

    Returns ``(states, policies, values)`` aligned per visited state:
      * ``states``   - integer sticker indices, shape ``(T, state_dim)``;
      * ``policies`` - MCTS visit distribution over the fixed legal-action order,
        shape ``(T, action_dim)``;
      * ``values``   - discounted return from each state, shape ``(T,)``.

    Moves are sampled from the MCTS visit distribution (temperature ``tau``) for
    exploration, not played greedily.
    """
    rng = rng or np.random.default_rng()
    cube = magiccube.Cube(cube_size)
    cube.scramble(scramble_depth)
    state = RubikCubeState(cube)

    # Fixed action ordering shared with the policy network's output head.
    legal_actions = rules_model.legal_actions(state)
    action_to_index = {action: i for i, action in enumerate(legal_actions)}

    state_strings: list[str] = []
    policies: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in tqdm(range(max_trajectory)):
        if state.is_terminal():
            break

        _, action_distribution = mcts.search(state, num_simulation_rollouts, tau)

        # Record the training target for this state: the visit distribution as a
        # fixed-order probability vector.
        policy_vec = np.zeros(len(legal_actions), dtype=np.float32)
        for action, prob in action_distribution.items():
            policy_vec[action_to_index[action]] = prob
        state_strings.append(state.get())
        policies.append(policy_vec)

        # Sample the next move from the visit distribution for exploration,
        # reusing policy_vec (float64 + renormalize so choice's sum check passes).
        # probs = policy_vec.astype(np.float64)
        # probs /= probs.sum()
        action = legal_actions[rng.choice(len(legal_actions), p=policy_vec)]
        state = rules_model.step(state, action)
        rewards.append(state.reward())

    if not state_strings:  # already solved at scramble time
        return (
            np.zeros((0, len(state.get())), dtype=np.int32),
            np.zeros((0, len(legal_actions)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    # Value target for each state is the discounted future return.
    values = np.zeros(len(rewards), dtype=np.float32)
    g = 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + mcts.discount * g
        values[t] = g

    states = np.asarray(states_to_indices(state_strings), dtype=np.int32)
    return states, np.asarray(policies, dtype=np.float32), values


def policy_loss(
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



def train_policy(rules_model: Model, mcts: MonteCarloTreeSearch, policy: Policy, optimizer: nnx.Optimizer, replay_buffer: ReplayBuffer, config: dict):
    # target_policy = nnx.clone(policy)
    train_step = 0
    while train_step < config['total_training_steps']:
        trajectory_lengths: list[float] = []
        solved_flags: list[float] = []
        for _ in range(config['num_self_play_steps_per_epoch']):
            states, policies, values = self_play(
                rules_model,
                mcts,
                config['cube_size'],
                config['scramble_depth'],
                config['max_trajectory'],
                config['num_simulation_rollouts'],
            )
            if len(states) > 0:
                replay_buffer.add(states, policies, values)
            trajectory_lengths.append(float(len(values)))
            solved_flags.append(float(len(values) > 0 and float(values[-1]) > 0.0))

        # Self-play stats averaged over this epoch's episodes, logged against the
        # current train step so they share the x-axis with the training metrics.
        mlflow.log_metrics(
            {
                "self_play/avg_trajectory_length": float(np.mean(trajectory_lengths)),
                "self_play/solved_rate": float(np.mean(solved_flags)),
                "replay_buffer/size": float(replay_buffer.size),
            },
            step=train_step,
        )

        if 10 * replay_buffer.size < config['batch_size']:
            continue

        for _ in range(config['num_training_steps_per_epoch']):
            if train_step >= config['total_training_steps']:
                break
            states, target_actions, target_values = replay_buffer.sample(config['batch_size'])
            (loss, (policy_ce, value_mse)), grads = nnx.value_and_grad(policy_loss, has_aux=True)(
                policy, states, target_actions, target_values
            )
            optimizer.update(policy, grads)
            mlflow.log_metrics(
                {
                    "train/loss": float(loss),
                    "train/policy_cross_entropy": float(policy_ce),
                    "train/value_mse": float(value_mse),
                },
                step=train_step,
            )
            train_step += 1

    return policy


if __name__ == "__main__":
    config = {
        'batch_size': 32,
        'total_training_steps': 100,
        'num_self_play_steps_per_epoch': 4,
        'num_training_steps_per_epoch': 2,
        'cube_size': 2,
        'scramble_depth': 2,
        'max_trajectory': 10,
        'num_simulation_rollouts': 200,
    }
    rules_model = RubikCubeModel(config['cube_size'])

    cube = magiccube.Cube(config['cube_size'])
    action_dim = len(rules_model.legal_actions(RubikCubeState(cube)))
    state_dim = len(cube.get())

    policy = Policy(num_embeddings=len(magiccube.Color), state_dim=state_dim, action_dim=action_dim, embed_dim=128, num_transformer_blocks=8, num_heads=2, rngs=nnx.Rngs(0))

    evaluator = PolicyRubikCubeEvaluator(rules_model, policy)
    mcts = MonteCarloTreeSearch(rules_model, evaluator, c_puct=5)

    optimizer = nnx.Optimizer(policy, optax.adam(learning_rate=0.001), wrt=nnx.Param)
    replay_buffer = ReplayBuffer(capacity=500, state_dim=state_dim, action_dim=action_dim, rngs=nnx.Rngs(1))

    mlflow.set_experiment("silver_alpha_zero")
    with mlflow.start_run(run_name="rubik_cube_alpha_zero", log_system_metrics=True):
        mlflow.log_params(config)
        train_policy(rules_model, mcts, policy, optimizer, replay_buffer, config)