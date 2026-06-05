from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import math


class State(ABC):
    """Abstract base class for a state in the Monte Carlo Tree Search."""

    @abstractmethod
    def get(self) -> str: # TODO: is it necessary to serialize the state?
        """Return a serializable representation of the state."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the state is terminal."""

    @abstractmethod
    def reward(self) -> float:
        """Return the immediate reward of *reaching* this state."""
        
    # identity for use as a dict/tree key:
    @abstractmethod
    def __hash__(self) -> int: ...
    @abstractmethod
    def __eq__(self, other) -> bool: ...


class Action(ABC):
    """Abstract base class for an action in the Monte Carlo Tree Search."""
    pass


class Model(ABC):
    """The MDP dynamics. Pure, ideally stateless."""
    @abstractmethod
    def legal_actions(self, state: State) -> list[Action]: ...
    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: State) -> tuple[dict[Action, float], float]:
        """Returns (prior probabilities over legal actions, value estimate)."""


@dataclass
class Node:
    """A node in the Monte Carlo Tree Search.

    Nodes are shared across the search (one per state), so the per-edge prior
    P(s, a) lives on the parent in `priors`, not on the (shared) child node.
    """
    state: State
    visit_count: int = 0                # N
    value_sum: float = 0.0              # W
    children: dict[Action, "Node"] = field(default_factory=dict)
    priors: dict[Action, float] = field(default_factory=dict)  # P(s, a) per edge

    @property
    def q(self) -> float:               # Q = W / N
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def puct(self, prior: float, sqrt_parent_visit_count: float, c_puct: float) -> float:
        return self.q + c_puct * prior * sqrt_parent_visit_count / (1 + self.visit_count)

    def is_expanded(self) -> bool:
        return bool(self.children)


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search algorithm."""
    def __init__(self, model: Model, evaluator: Evaluator, c_puct: float = 1.4, discount: float = 0.99):
        self.model = model
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.discount = discount

    def search(
        self, root_state: State, num_iterations: int, tau: float = 1.0
    ) -> tuple[Action, dict[Action, float]]:
        # Transposition table: one Node per state, so states reachable through
        # different paths share the same Node (the tree becomes a DAG). Reset per
        # search so statistics are not carried across different roots.
        self._nodes: dict[State, Node] = {}
        root = self._node(root_state)
        for _ in range(num_iterations):
            path = self._select(root)
            leaf = path[-1]
            value = self._expand_and_evaluate(leaf)
            self._backup(path, value)
        return self._best_action(root), self._action_distribution(root, tau)

    def _node(self, state: State) -> Node:
        """Return the shared Node for `state`, creating it on first encounter."""
        node = self._nodes.get(state)
        if node is None:
            node = Node(state)
            self._nodes[state] = node
        return node

    def _select(self, root: Node) -> list[Node]:
        """Walk down via PUCT to a leaf, stopping if a transposition revisits a
        state already on the path (the DAG can contain cycles)."""
        trajectory = [root]
        visited = {root.state}
        while trajectory[-1].is_expanded() and not trajectory[-1].state.is_terminal():
            child = self._select_step(trajectory[-1])
            if child.state in visited:  # cycle back to an ancestor; stop here
                break
            trajectory.append(child)
            visited.add(child.state)
        return trajectory

    def _select_step(self, node: Node) -> Node:
        c = self.c_puct
        sqrt_n = math.sqrt(node.visit_count)
        action = max(
            node.children,
            key=lambda a: node.children[a].puct(node.priors[a], sqrt_n, c),
        )
        return node.children[action]

    def _expand_and_evaluate(self, leaf: Node) -> float:
        # A terminal state has no children to expand: its reward is the leaf
        # value that backup bootstraps from and propagates up the path.
        if leaf.state.is_terminal():
            return leaf.state.reward()
        # The leaf can already be expanded if selection stopped on a cycle; in
        # that case bootstrap from its current estimate instead of re-expanding.
        if leaf.is_expanded():
            return leaf.q
        
        priors, value = self.evaluator.evaluate(leaf.state)
        leaf.priors = dict(priors)
        leaf.children = {
            action: self._node(self.model.step(leaf.state, action)) for action in priors
        }
        return value

    def _backup(self, path: list[Node], value: float) -> None:
        # Single-agent (MDP) backup: bootstrap from the leaf's value estimate and
        # walk up, discounting and adding the reward collected at each transition.
        # `value` is the future return from the leaf (0 if terminal); the reward
        # for *reaching* a node is added as we move to its parent. No sign flip:
        # unlike two-player AlphaZero, the value is not negated between levels.
        path[-1].visit_count += 1
        path[-1].value_sum += value
        for node in reversed(path[:-1]):
            value = node.state.reward() + self.discount * value
            node.visit_count += 1
            node.value_sum += value

    def _best_action(self, node: Node) -> Action:
        return max(node.children, key=lambda a: node.children[a].visit_count)

    def _action_distribution(self, node: Node, tau: float = 1.0) -> dict[Action, float]:
        """Visit-count policy with temperature: pi(a | s) proportional to
        N(s, a)^(1/tau).

        tau = 1 makes pi proportional to the visit counts; tau -> 0 collapses to
        a greedy (one-hot on the most-visited action) policy; larger tau flattens
        the distribution toward uniform.
        """
        actions = list(node.children)
        if not actions:
            return {}
        counts = [node.children[a].visit_count for a in actions]
        max_count = max(counts)
        if max_count == 0:  # no visits recorded yet -> uniform over legal actions
            return {a: 1.0 / len(actions) for a in actions}
        if tau == 0:  # greedy: split mass over the most-visited action(s)
            winners = [a for a, c in zip(actions, counts) if c == max_count]
            return {a: (1.0 / len(winners) if a in winners else 0.0) for a in actions}
        # Normalize counts by their max before exponentiating so that 1/tau being
        # large cannot overflow; the ratios lie in [0, 1].
        weights = [(c / max_count) ** (1.0 / tau) for c in counts]
        total = sum(weights)
        return {a: w / total for a, w in zip(actions, weights)}

    def _best_trajectory(self, root: Node) -> list[Action]:
        # Principal variation: from the root, greedily follow the most-visited
        # child until a terminal state, an unexpanded leaf, or a revisited state.
        actions: list[Action] = []
        node = root
        visited = {node.state}
        while node.is_expanded() and not node.state.is_terminal():
            action = self._best_action(node)
            child = node.children[action]
            if child.state in visited:  # cycle; stop to keep the path finite
                break
            actions.append(action)
            node = child
            visited.add(child.state)
        return actions