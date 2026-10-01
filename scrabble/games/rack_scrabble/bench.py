"""Throughput of the vectorized step loop.

Run with ``python -m scrabble.games.rack_scrabble.bench``.
"""

from __future__ import annotations

import time

from scrabble.games.rack_scrabble.env import RackScrabbleEnv


def bench(n_envs: int = 4096, rounds: int = 200, seed: int = 0) -> float:
    """Env-steps per second, including legal-action sampling and redeals."""
    env = RackScrabbleEnv(n_envs, seed)
    env.reset()
    for _ in range(5):
        env.step(env.sample_legal_actions())
        env.reset_done()

    start = time.perf_counter()
    for _ in range(rounds):
        env.step(env.sample_legal_actions())
        env.reset_done()
    elapsed = time.perf_counter() - start
    return n_envs * rounds / elapsed


def main() -> None:
    n_envs = 4096
    rounds = 200
    rate = bench(n_envs=n_envs, rounds=rounds)
    print(f"{rate:,.0f} env-steps/s  ({n_envs} envs x {rounds} steps)")


if __name__ == "__main__":
    main()
