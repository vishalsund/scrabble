"""The 16 legal Rack Scrabble patterns.

An action is an unordered set of tiles. It is legal when the rack has at
least ``COST[action]`` of each letter. Patterns outside this table, including
two of one letter plus a third different letter, are not actions.
"""

import numpy as np

LETTERS = ("A", "B", "C", "D")
N_LETTERS = 4
RACK_SIZE = 3
COPIES_PER_LETTER = 4
HORIZON = 6
N_ACTIONS = 16
OBS_SIZE = N_LETTERS * 2 + 1

# Rows follow ACTION_NAMES. Columns are A, B, C, D.
COST = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [2, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 2, 0],
        [0, 0, 0, 2],
        [3, 0, 0, 0],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 3],
        [1, 1, 1, 0],
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ],
    dtype=np.uint8,
)

SCORE = np.array(
    [1, 1, 1, 1, 4, 4, 4, 4, 12, 12, 12, 12, 6, 6, 6, 6],
    dtype=np.int16,
)

ACTION_NAMES = (
    "A",
    "B",
    "C",
    "D",
    "AA",
    "BB",
    "CC",
    "DD",
    "AAA",
    "BBB",
    "CCC",
    "DDD",
    "ABC",
    "ABD",
    "ACD",
    "BCD",
)
