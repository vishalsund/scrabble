"""Throughput of the vectorized step loop.

Run with ``python -m scrabble.games.mini_scrabble.bench``.
"""

from __future__ import annotations

import time

from scrabble.games.mini_scrabble.env import MiniScrabbleEnv


def bench(n_envs: int = 256, rounds: int = 20, seed: int = 0) -> float:
    """Env-steps per second, including legal-action sampling and redeals."""
    env = MiniScrabbleEnv(n_envs, seed)
    env.reset()
    for _ in range(3):
        env.step(env.sample_legal_actions())
        env.reset_done()

    start = time.perf_counter()
    for _ in range(rounds):
        env.step(env.sample_legal_actions())
        env.reset_done()
    elapsed = time.perf_counter() - start
    return n_envs * rounds / elapsed


def main() -> None:
    n_envs = 256
    rounds = 20
    rate = bench(n_envs=n_envs, rounds=rounds)
    print(f"{rate:,.0f} env-steps/s  ({n_envs} envs x {rounds} steps)")


if __name__ == "__main__":
    main()
