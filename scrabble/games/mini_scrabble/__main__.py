"""Play one random Mini-Scrabble game. Run with ``python -m scrabble.games.mini_scrabble``."""

import numpy as np

from scrabble.games.mini_scrabble.rules import (
    BAG,
    LETTERS,
    N_LETTERS,
    N_SQUARES,
    PASS,
    PASSES,
    PLAYER,
    RACK,
    action_name,
    format_board,
)
from scrabble.games.mini_scrabble.env import MiniScrabbleEnv


def _tiles(counts: np.ndarray) -> str:
    tiles = [
        letter
        for letter, count in zip(LETTERS, counts, strict=True)
        for _ in range(int(count))
    ]
    return " ".join(tiles) if tiles else "(empty)"


def main() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    env.reset()
    print("Mini-Scrabble v1  seed=0")
    turn = 1
    while not env.terminated[0]:
        player = int(env.state[0, PLAYER])
        print(f"Turn {turn}  player {player + 1}  passes {int(env.state[0, PASSES])}")
        print(format_board(env.state[0, :N_SQUARES]))
        print(f"  P1 {_tiles(env.state[0, RACK : RACK + N_LETTERS])}")
        print(f"  P2 {_tiles(env.state[0, RACK + N_LETTERS : RACK + 2 * N_LETTERS])}")
        print(f"  bag {_tiles(env.state[0, BAG : BAG + N_LETTERS])}")
        action = int(env.sample_legal_actions()[0])
        _, rewards, _, _ = env.step(np.array([action], dtype=np.int64))
        label = "pass" if action == PASS else action_name(action)
        print(f"  play {label}  +{int(rewards[0])}")
        print(f"  score {int(env.score[0, 0])}-{int(env.score[0, 1])}")
        turn += 1
    print(format_board(env.state[0, :N_SQUARES]))
    p0 = int(env.score[0, 0])
    p1 = int(env.score[0, 1])
    if p0 == p1:
        result = "draw"
    elif p0 > p1:
        result = "player 1 wins"
    else:
        result = "player 2 wins"
    print(f"Final {p0}-{p1}  {result}")


if __name__ == "__main__":
    main()
