"""One REINFORCE step against the greedy baseline."""

import mlx.core as mx
import mlx.optimizers as optim

from scrabble.games.mini_scrabble import MiniScrabbleEnv
from scrabble.rl.mini_policy import Policy, _update, use_gpu


def test_one_update_against_greedy_is_finite() -> None:
    use_gpu()
    mx.random.seed(0)
    policy = Policy(hidden=32)
    opt = optim.Adam(learning_rate=3e-4)
    env = MiniScrabbleEnv(4, seed=0)
    env.reset()
    metrics = _update(policy, opt, env, max_steps=80, entropy_coef=0.01)
    assert metrics["loss"] == metrics["loss"]
    assert 0.0 <= metrics["train_win_rate"] <= 1.0
    assert metrics["game_length"] > 0
    assert mx.default_device().type == mx.gpu
