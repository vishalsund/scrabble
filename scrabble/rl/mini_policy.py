"""Train a flat MLP to play Mini-Scrabble against the greedy baseline.

The board is one-hot and flattened. There is no solver, so the update is
REINFORCE on finished games: a win is +1, a loss is -1, and a draw is 0,
from the seat the network is playing. The other seat is always greedy.
Illegal actions are masked out.

The network runs on the MLX GPU. This file does not train on import. Run it
with:

    python -m scrabble.rl.mini_policy

Loss, entropy, and win rate against greedy are logged to the Weights &
Biases project ``mini-scrabble``.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import wandb

from scrabble.games.mini_scrabble import (
    INPUT_DIM,
    N_ACTIONS,
    PASS,
    PLAYER,
    MiniScrabbleEnv,
    encode_env,
)
from scrabble.rl.mini_greedy import choose

ILLEGAL_LOGIT = -1.0e9
_ILLEGAL = mx.array(ILLEGAL_LOGIT, dtype=mx.float32)


class Policy(nn.Module):
    """Two hidden layers from the flattened state to one logit per action."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def __call__(self, features: mx.array, legal: mx.array) -> mx.array:
        return mx.where(legal, self.net(features), _ILLEGAL)

    def inputs(self, env: MiniScrabbleEnv, rows: np.ndarray) -> tuple[np.ndarray, ...]:
        features = np.ascontiguousarray(encode_env(env.state, env.score)[rows])
        legal = np.array(env.mask[rows])
        return features, legal


def use_gpu() -> str:
    """Pin MLX to the Apple GPU and return the device name."""
    mx.set_default_device(mx.gpu)
    probe = mx.ones((64, 64))
    mx.eval(probe @ probe)
    device = mx.default_device()
    if device.type != mx.gpu:
        raise RuntimeError(f"MLX default device is {device}, not the GPU")
    return str(mx.device_info()["device_name"])


def train(policy: nn.Module, name: str, config: dict | None = None) -> None:
    """REINFORCE against greedy, logged to the ``mini-scrabble`` project."""
    device_name = use_gpu()
    settings = {
        "seed": 0,
        "updates": 1000,
        "n_games": 128,
        "max_steps": 80,
        "lr": 3e-4,
        "entropy_coef": 0.01,
        "eval_games": 128,
        "eval_seed": 10_000,
        "opponent": "greedy",
        "n_actions": N_ACTIONS,
        "framework": "mlx",
        "device": str(mx.default_device()),
        "device_name": device_name,
    }
    settings.update(config or {})
    seed = int(settings["seed"])
    updates = int(settings["updates"])
    n_games = int(settings["n_games"])
    max_steps = int(settings["max_steps"])
    lr = float(settings["lr"])
    entropy_coef = float(settings["entropy_coef"])
    eval_games = int(settings["eval_games"])
    eval_seed = int(settings["eval_seed"])

    print(f"MLX GPU: {device_name} ({mx.default_device()})")
    mx.random.seed(seed)
    wandb.init(project="mini-scrabble", name=name, config=settings)
    wandb.define_metric("update")
    wandb.define_metric("*", step_metric="update")
    try:
        opt = optim.Adam(learning_rate=lr)
        env = MiniScrabbleEnv(n_games, seed)
        env.reset()
        for update in range(1, updates + 1):
            if update > 1:
                env.reset()
            metrics = _update(policy, opt, env, max_steps, entropy_coef)
            metrics.update(_evaluate(policy, eval_games, eval_seed))
            metrics["update"] = update
            print(
                f"update {update:04d}  loss {metrics['loss']:.3f}  "
                f"entropy {metrics['entropy']:.3f}  "
                f"vs greedy {metrics['win_rate']:.3f} "
                f"(draw {metrics['draw_rate']:.3f}, diff {metrics['mean_score_diff']:.2f})"
            )
            wandb.log(metrics)
    finally:
        wandb.finish()


def main() -> None:
    train(Policy(128), "mlp-vs-greedy", {"hidden": 128, "input_dim": INPUT_DIM})


def _update(
    policy: nn.Module,
    opt: optim.Optimizer,
    env: MiniScrabbleEnv,
    max_steps: int,
    entropy_coef: float,
) -> dict[str, float]:
    """Play a batch against greedy and apply one REINFORCE step.

    Even envs seat the network first. Odd envs seat greedy first. Only the
    network's decisions are trained.
    """
    seat = np.arange(env.n_envs) % 2
    visits: list[list[tuple[tuple[np.ndarray, ...], int]]] = [[] for _ in range(env.n_envs)]
    passes = 0
    decisions = 0

    for _ in range(max_steps):
        active = ~env.terminated
        if not active.any():
            break
        actions = np.zeros(env.n_envs, dtype=np.int64)
        mine = active & (env.state[:, PLAYER] == seat)
        if mine.any():
            index = np.flatnonzero(mine)
            inputs = policy.inputs(env, index)
            chosen = _sample(policy, inputs)
            actions[index] = chosen
            passes += int(np.count_nonzero(chosen == PASS))
            decisions += int(index.size)
            for local, row in enumerate(index):
                visit = tuple(field[local] for field in inputs)
                visits[int(row)].append((visit, int(chosen[local])))
        theirs = active & ~mine
        if theirs.any():
            actions[theirs] = choose(env)[theirs]
        env.step(actions)

    played_rows = [row for row, steps in enumerate(visits) if steps]
    returns: list[float] = []
    lengths: list[int] = []
    gaps: list[int] = []
    samples: list[tuple[np.ndarray, ...]] = []
    taken: list[int] = []
    wins = 0
    draws = 0
    for row in played_rows:
        gap = int(env.score[row, seat[row]]) - int(env.score[row, 1 - int(seat[row])])
        outcome = _sign(gap)
        wins += int(outcome > 0)
        draws += int(outcome == 0)
        lengths.append(len(visits[row]))
        gaps.append(gap)
        for visit, action in visits[row]:
            samples.append(visit)
            taken.append(action)
            returns.append(outcome)

    packed = _pack(samples)
    advantage = np.asarray(returns, dtype=np.float32)
    advantage = advantage - float(advantage.mean())
    loss = _apply(
        policy,
        opt,
        packed,
        mx.array(np.asarray(taken, dtype=np.int32)),
        mx.array(advantage),
        entropy_coef,
    )
    entropy = _entropy(policy, packed)
    played = len(lengths)
    return {
        "loss": loss,
        "entropy": entropy,
        "mean_return": float(np.mean(returns)),
        "train_win_rate": wins / played,
        "train_draw_rate": draws / played,
        "train_score_diff": float(np.mean(gaps)),
        "game_length": float(np.mean(lengths)),
        "pass_rate": passes / max(decisions, 1),
    }


def _evaluate(policy: nn.Module, n_games: int, seed: int) -> dict[str, float]:
    """Argmax network against greedy, on a fixed set of deals."""
    env = MiniScrabbleEnv(n_games, seed)
    env.reset()
    seat = np.arange(n_games) % 2
    for _ in range(80):
        active = ~env.terminated
        if not active.any():
            break
        actions = np.zeros(n_games, dtype=np.int64)
        mine = active & (env.state[:, PLAYER] == seat)
        if mine.any():
            index = np.flatnonzero(mine)
            actions[index] = _sample(policy, policy.inputs(env, index), greedy=True)
        theirs = active & ~mine
        if theirs.any():
            actions[theirs] = choose(env)[theirs]
        env.step(actions)
    diff = _score_diff(env.score, seat)
    return {
        "win_rate": float(np.mean(diff > 0)),
        "draw_rate": float(np.mean(diff == 0)),
        "mean_score_diff": float(np.mean(diff)),
    }


def _sample(policy: nn.Module, inputs: tuple[np.ndarray, ...], greedy: bool = False) -> np.ndarray:
    logits = policy(*[mx.array(field) for field in inputs])
    choice = mx.argmax(logits, axis=-1) if greedy else mx.random.categorical(logits)
    mx.eval(choice)
    return np.asarray(choice, dtype=np.int64)


def _pack(samples: list[tuple[np.ndarray, ...]]) -> tuple[mx.array, ...]:
    return tuple(
        mx.array(np.stack([sample[field] for sample in samples]))
        for field in range(len(samples[0]))
    )


def _apply(
    policy: nn.Module,
    opt: optim.Optimizer,
    packed: tuple[mx.array, ...],
    actions: mx.array,
    advantage: mx.array,
    entropy_coef: float,
) -> float:
    def loss_fn(model, inputs, taken, adv):
        logits = model(*inputs)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        chosen = mx.take_along_axis(logp, taken[:, None], axis=1).reshape(-1)
        probs = mx.softmax(logits, axis=-1)
        entropy = -(probs * mx.log(mx.clip(probs, 1e-12, None))).sum(axis=-1)
        return -(chosen * adv).mean() - entropy_coef * entropy.mean()

    grad_fn = nn.value_and_grad(policy, loss_fn)
    loss, grads = grad_fn(policy, packed, actions, advantage)
    grads, _ = optim.clip_grad_norm(grads, 1.0)
    opt.update(policy, grads)
    mx.eval(loss, policy.parameters(), opt.state)
    return float(loss)


def _entropy(policy: nn.Module, packed: tuple[mx.array, ...]) -> float:
    logits = policy(*packed)
    probs = mx.softmax(logits, axis=-1)
    entropy = -(probs * mx.log(mx.clip(probs, 1e-12, None))).sum(axis=-1).mean()
    mx.eval(entropy)
    return float(entropy)


def _score_diff(score: np.ndarray, seat: np.ndarray) -> np.ndarray:
    """Points for the network's seat minus points for greedy."""
    own = np.where(seat == 0, score[:, 0], score[:, 1])
    opp = np.where(seat == 0, score[:, 1], score[:, 0])
    return own.astype(np.int64) - opp.astype(np.int64)


def _sign(gap: int) -> float:
    if gap > 0:
        return 1.0
    if gap < 0:
        return -1.0
    return 0.0


if __name__ == "__main__":
    main()
