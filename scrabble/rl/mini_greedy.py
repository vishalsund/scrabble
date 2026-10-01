"""Greedy Mini-Scrabble: play the legal move with the highest immediate score.

Pass scores 0. A legal placement scores at least 2, so this passes only when
the rack cannot play. Ties go to the lowest action id.

This file does not evaluate on import. Run it with:

    python -m scrabble.rl.mini_greedy
"""

from __future__ import annotations

import numpy as np
import wandb
from numba import njit, prange

from scrabble.games.mini_scrabble import N_ACTIONS, PASS, PLAYER, MiniScrabbleEnv
from scrabble.games.mini_scrabble._engine import _placement_score


@njit(parallel=True, cache=True)
def _choose(state, mask, actions):
    for i in prange(state.shape[0]):
        best = PASS
        best_score = 0
        for action in range(N_ACTIONS):
            if not mask[i, action]:
                continue
            score = 0 if action == PASS else _placement_score(state[i], action)
            if score > best_score:
                best_score = score
                best = action
        actions[i] = best


def choose(env: MiniScrabbleEnv) -> np.ndarray:
    """One greedy action per env. Finished games are left at pass."""
    actions = np.zeros(env.n_envs, dtype=np.int64)
    _choose(env.state, env.mask, actions)
    return actions


def evaluate(n_games: int = 256, seed: int = 0) -> dict[str, float]:
    """Greedy against uniform legal play, alternating which seat it holds."""
    env = MiniScrabbleEnv(n_games, seed)
    env.reset()
    seat = np.arange(n_games) % 2
    passes = 0
    decisions = 0
    for _ in range(80):
        active = ~env.terminated
        if not active.any():
            break
        actions = np.zeros(n_games, dtype=np.int64)
        mine = active & (env.state[:, PLAYER] == seat)
        if mine.any():
            chosen = choose(env)
            actions[mine] = chosen[mine]
            passes += int(np.count_nonzero(actions[mine] == PASS))
            decisions += int(mine.sum())
        theirs = active & ~mine
        if theirs.any():
            actions[theirs] = env.sample_legal_actions()[theirs]
        env.step(actions)
    diff = np.where(seat == 0, env.score[:, 0] - env.score[:, 1], env.score[:, 1] - env.score[:, 0])
    return {
        "win_rate": float(np.mean(diff > 0)),
        "draw_rate": float(np.mean(diff == 0)),
        "mean_score_diff": float(np.mean(diff)),
        "pass_rate": passes / max(decisions, 1),
    }


def main() -> None:
    n_games = 256
    seed = 0
    wandb.init(
        project="mini-scrabble",
        name="greedy",
        config={"seed": seed, "n_games": n_games, "opponent": "uniform-legal"},
    )
    try:
        metrics = evaluate(n_games, seed)
        print(
            f"vs random  {metrics['win_rate']:.3f}  "
            f"(draw {metrics['draw_rate']:.3f}, diff {metrics['mean_score_diff']:.2f}, "
            f"pass {metrics['pass_rate']:.3f})"
        )
        wandb.log(metrics)
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
