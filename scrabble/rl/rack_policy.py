"""Train a small policy to copy optimal Rack Scrabble play.

A position is three things: the letters on the rack, the letters in the bag,
and how many turns are left. Order does not matter, so the network sees how
many of each letter are in those lists, plus a one-hot of the turns left.

Labels come from the exact dynamic program. The loss is cross-entropy against
its optimal actions. If several actions tie, any of them is a correct label.
Evaluation checks held-out positions and full games against the same solver.

This file does not train on import. Run it with:

    python -m scrabble.rl.rack_policy

Loss and accuracy are logged to the Weights & Biases project ``rack-scrabble``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import wandb
from torch import nn

from scrabble.games.rack_scrabble import (
    COPIES_PER_LETTER,
    HORIZON,
    LETTERS,
    N_ACTIONS,
    N_LETTERS,
    RACK_SIZE,
    RackScrabbleEnv,
    RackScrabbleSolver,
)

# Rack counts, bag counts, then one slot per possible turns-remaining value.
INPUT_DIM = N_LETTERS + N_LETTERS + HORIZON
ILLEGAL_LOGIT = -1.0e9
_LETTER_INDEX = {letter: index for index, letter in enumerate(LETTERS)}


@dataclass(frozen=True)
class Example:
    """One decision, in the form the policy reads and the solver labels."""

    rack: tuple[str, ...]
    bag: tuple[str, ...]
    turns_remaining: int
    legal: tuple[bool, ...]
    optimal: tuple[bool, ...]


class Policy(nn.Module):
    """One hidden layer from the position vector to the 16 action logits."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.hidden = nn.Linear(INPUT_DIM, hidden)
        self.actions = nn.Linear(hidden, N_ACTIONS)

    def forward(self, features: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.hidden(features))
        logits = self.actions(hidden)
        return logits.masked_fill(~legal, ILLEGAL_LOGIT)


def letter_list(counts) -> tuple[str, ...]:
    """Expand letter counts into a sorted list, such as ``(2, 1, 0, 0)`` -> ``('A', 'A', 'B')``."""
    return tuple(
        letter
        for letter, count in zip(LETTERS, counts, strict=True)
        for _ in range(int(count))
    )


def encode(rack: tuple[str, ...], bag: tuple[str, ...], turns_remaining: int) -> np.ndarray:
    """Pack a position into the network input.

    Layout: rack counts of A B C D, bag counts of A B C D, then a one-hot of
    turns remaining. One turn left uses the first of those six slots.
    """
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    for letter in rack:
        features[_LETTER_INDEX[letter]] += 1.0
    for letter in bag:
        features[N_LETTERS + _LETTER_INDEX[letter]] += 1.0
    if 1 <= turns_remaining <= HORIZON:
        features[2 * N_LETTERS + turns_remaining - 1] = 1.0
    return features


def build_examples(solver: RackScrabbleSolver) -> list[Example]:
    """Label every legal position with the solver's optimal actions."""
    examples = []
    for rack, bag in _count_pairs():
        for turns_remaining in range(1, HORIZON + 1):
            decision = solver.decide(rack, bag, HORIZON - turns_remaining)
            if decision.action is None:
                continue
            legal = tuple(q is not None for q in decision.q_values)
            optimal = tuple(
                q is not None and q == decision.value for q in decision.q_values
            )
            examples.append(
                Example(
                    letter_list(rack),
                    letter_list(bag),
                    turns_remaining,
                    legal,
                    optimal,
                )
            )
    return examples


def split_examples(
    examples: list[Example],
    test_fraction: float,
    seed: int,
) -> tuple[list[Example], list[Example]]:
    """Hold out whole rack/bag pairs, so every turn count of a pair stays together."""
    keys = sorted({(example.rack, example.bag) for example in examples})
    order = np.arange(len(keys))
    np.random.default_rng(seed).shuffle(order)
    n_test = max(1, int(round(len(keys) * test_fraction)))
    test_keys = {keys[index] for index in order[:n_test]}
    train = []
    test = []
    for example in examples:
        if (example.rack, example.bag) in test_keys:
            test.append(example)
        else:
            train.append(example)
    return train, test


def tensors(
    examples: list[Example],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = np.stack(
        [encode(example.rack, example.bag, example.turns_remaining) for example in examples]
    )
    legal = np.asarray([example.legal for example in examples], dtype=np.bool_)
    optimal = np.asarray([example.optimal for example in examples], dtype=np.bool_)
    return torch.from_numpy(features), torch.from_numpy(legal), torch.from_numpy(optimal)


def imitation_loss(logits: torch.Tensor, optimal: torch.Tensor) -> torch.Tensor:
    """Cross-entropy. Tied solver actions share the probability target equally."""
    target = optimal.float()
    target = target / target.sum(dim=1, keepdim=True)
    log_probs = torch.log_softmax(logits, dim=1)
    # Illegal logits are -1e9, so their log-probability is -inf. Drop those
    # terms; their target is zero and they are not a training signal.
    picked = torch.where(target > 0, target * log_probs, torch.zeros_like(log_probs))
    return -picked.sum(dim=1).mean()


def train(
    policy: Policy,
    features: torch.Tensor,
    legal: torch.Tensor,
    optimal: torch.Tensor,
    test_features: torch.Tensor,
    test_legal: torch.Tensor,
    test_optimal: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> None:
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n_rows = features.shape[0]
    for epoch in range(epochs):
        order = torch.randperm(n_rows)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            logits = policy(features[batch], legal[batch])
            loss = imitation_loss(logits, optimal[batch])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        train_loss = total_loss / n_batches
        train_accuracy = optimal_accuracy(policy, features, legal, optimal)
        test_accuracy = optimal_accuracy(policy, test_features, test_legal, test_optimal)
        wandb.log(
            {
                "train/loss": train_loss,
                "train/accuracy": train_accuracy,
                "test/accuracy": test_accuracy,
            },
            step=epoch + 1,
        )
        print(
            f"epoch {epoch + 1:3d}  "
            f"loss {train_loss:.4f}  "
            f"train acc {train_accuracy:.3f}  "
            f"test acc {test_accuracy:.3f}"
        )


@torch.no_grad()
def optimal_accuracy(
    policy: Policy,
    features: torch.Tensor,
    legal: torch.Tensor,
    optimal: torch.Tensor,
) -> float:
    """Fraction of positions whose chosen action is one of the solver's best."""
    choice = policy(features, legal).argmax(dim=1)
    hit = optimal[torch.arange(choice.shape[0]), choice]
    return float(hit.float().mean())


def evaluate_games(
    policy: Policy,
    solver: RackScrabbleSolver,
    n_games: int,
    seed: int,
) -> dict[str, float]:
    """Play the same deals with the policy and the solver.

    Solver expected is the best possible average score on those openings.
    The solver column is one sampled play of that optimal policy.
    """
    dealer = RackScrabbleEnv(n_games, seed)
    dealer.reset()
    openings = dealer.state.copy()
    expected = np.array([float(solver.decide_obs(row).value) for row in openings])
    policy_score = _rollout(lambda env: _choose_policy(policy, env), openings, seed)
    solver_score = _rollout(lambda env: _choose_solver(solver, env), openings, seed)
    metrics = {
        "games/policy": float(policy_score.mean()),
        "games/solver": float(solver_score.mean()),
        "games/solver_expected": float(expected.mean()),
    }
    print(
        f"games {n_games}  "
        f"policy {metrics['games/policy']:.2f}  "
        f"solver {metrics['games/solver']:.2f}  "
        f"solver expected {metrics['games/solver_expected']:.2f}"
    )
    return metrics


def _choose_policy(policy: Policy, env: RackScrabbleEnv) -> np.ndarray:
    features = np.stack(
        [
            encode(
                letter_list(env.state[index, :N_LETTERS]),
                letter_list(env.state[index, N_LETTERS : N_LETTERS * 2]),
                HORIZON - int(env.state[index, -1]),
            )
            for index in range(env.n_envs)
        ]
    )
    legal = torch.from_numpy(np.array(env.mask))
    with torch.no_grad():
        choice = policy(torch.from_numpy(features), legal).argmax(dim=1)
    return choice.numpy().astype(np.int64)


def _choose_solver(solver: RackScrabbleSolver, env: RackScrabbleEnv) -> np.ndarray:
    actions = np.zeros(env.n_envs, dtype=np.int64)
    for index in range(env.n_envs):
        if env.terminated[index]:
            continue
        actions[index] = solver.best_action(
            env.state[index, :N_LETTERS],
            env.state[index, N_LETTERS : N_LETTERS * 2],
            int(env.state[index, -1]),
        )
    return actions


def _rollout(choose, openings: np.ndarray, seed: int) -> np.ndarray:
    env = RackScrabbleEnv(openings.shape[0], seed)
    for index in range(openings.shape[0]):
        env.set_state(
            openings[index, :N_LETTERS],
            openings[index, N_LETTERS : N_LETTERS * 2],
            turn=int(openings[index, -1]),
            index=index,
        )
    score = np.zeros(openings.shape[0], dtype=np.int64)
    for _ in range(HORIZON):
        if env.terminated.all():
            break
        _, rewards, _, _ = env.step(choose(env))
        score += rewards.astype(np.int64)
    return score


def _count_pairs():
    for rack in product(range(RACK_SIZE + 1), repeat=N_LETTERS):
        if sum(rack) == 0 or sum(rack) > RACK_SIZE:
            continue
        bag_ranges = [range(COPIES_PER_LETTER - count + 1) for count in rack]
        for bag in product(*bag_ranges):
            yield rack, bag


def main() -> None:
    seed = 0
    hidden = 64
    epochs = 40
    batch_size = 1024
    lr = 1e-3
    test_fraction = 0.2
    n_games = 512

    torch.manual_seed(seed)
    wandb.init(
        project="rack-scrabble",
        name="imitate-dp",
        config={
            "seed": seed,
            "hidden": hidden,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "test_fraction": test_fraction,
            "n_games": n_games,
        },
    )
    try:
        print("solving the game")
        solver = RackScrabbleSolver()
        print("labeling positions from the solver")
        examples = build_examples(solver)
        train_examples, test_examples = split_examples(examples, test_fraction, seed)
        print(f"{len(train_examples)} train positions, {len(test_examples)} test positions")
        wandb.config.update(
            {"n_train": len(train_examples), "n_test": len(test_examples)},
            allow_val_change=True,
        )

        policy = Policy(hidden)
        features, legal, optimal = tensors(train_examples)
        test_features, test_legal, test_optimal = tensors(test_examples)
        train(
            policy,
            features,
            legal,
            optimal,
            test_features,
            test_legal,
            test_optimal,
            epochs,
            batch_size,
            lr,
        )

        policy.eval()
        accuracy = optimal_accuracy(policy, test_features, test_legal, test_optimal)
        print(f"test acc {accuracy:.3f}")
        wandb.log(evaluate_games(policy, solver, n_games, seed + 1))
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
