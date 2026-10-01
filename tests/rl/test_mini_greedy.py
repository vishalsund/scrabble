"""The greedy baseline plays the highest-scoring legal move."""

import numpy as np

from scrabble.games.mini_scrabble import MiniScrabbleEnv
from scrabble.games.mini_scrabble.rules import PASS, RACK, placement_score
from scrabble.rl.mini_greedy import choose


def _best_score(env: MiniScrabbleEnv) -> tuple[int, int]:
    board = env.state[0, :25]
    rack = env.state[0, RACK : RACK + 4]
    opened = bool(env.state[0, -1])
    best_action = PASS
    best = 0
    for action in np.flatnonzero(env.mask[0]):
        score = 0 if int(action) == PASS else placement_score(board, rack, opened, int(action))
        if score > best:
            best = score
            best_action = int(action)
    return best_action, best


def test_three_distinct_letters_score_more_than_a_pair() -> None:
    env = MiniScrabbleEnv(1)
    board = np.zeros(25, dtype=np.int64)
    racks = np.array([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=np.int64)
    bag = np.array([4, 4, 4, 6], dtype=np.int64)
    env.set_state(board, racks, bag)
    action = int(choose(env)[0])
    expected, score = _best_score(env)
    assert action == expected
    assert score == 3


def test_a_repeated_pair_cannot_outscore_two_tiles() -> None:
    env = MiniScrabbleEnv(1)
    board = np.zeros(25, dtype=np.int64)
    racks = np.array([[2, 1, 0, 0], [2, 1, 0, 0]], dtype=np.int64)
    bag = np.array([2, 4, 6, 6], dtype=np.int64)
    env.set_state(board, racks, bag)
    action = int(choose(env)[0])
    expected, score = _best_score(env)
    assert action == expected
    assert score == 2


def test_an_empty_rack_passes() -> None:
    env = MiniScrabbleEnv(1)
    board = np.zeros(25, dtype=np.int64)
    board[12] = 1
    racks = np.array([[0, 0, 0, 0], [3, 0, 0, 0]], dtype=np.int64)
    bag = np.array([2, 6, 6, 6], dtype=np.int64)
    env.set_state(board, racks, bag)
    assert int(env.mask[0].sum()) == 1
    assert int(choose(env)[0]) == PASS
