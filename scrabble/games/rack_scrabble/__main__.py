"""Play one random Rack Scrabble game. Run with ``python -m scrabble.games.rack_scrabble``."""

import numpy as np

from scrabble.games.rack_scrabble.actions import ACTION_NAMES, LETTERS
from scrabble.games.rack_scrabble.env import RackScrabbleEnv


def _tiles(counts: np.ndarray) -> str:
    tiles = [
        letter
        for letter, count in zip(LETTERS, counts, strict=True)
        for _ in range(int(count))
    ]
    return " ".join(tiles) if tiles else "(empty)"


def main() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.reset()
    print("Rack Scrabble  seed=0")
    turn = 1
    while not env.terminated[0]:
        print(f"Turn {turn}")
        print(f"  rack: {_tiles(env.state[0, :4])}")
        print(f"  bag:  {_tiles(env.state[0, 4:8])}")
        action = int(env.sample_legal_actions()[0])
        _, rewards, _, _ = env.step(np.array([action], dtype=np.int64))
        print(f"  play: {ACTION_NAMES[action]}  +{int(rewards[0])}")
        turn += 1
    print(f"Leftover: {_tiles(env.state[0, :4])}  (worth 0)")
    print(f"Score: {int(env.score[0])}")


if __name__ == "__main__":
    main()
