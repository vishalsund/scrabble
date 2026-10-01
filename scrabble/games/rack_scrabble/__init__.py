"""Rack Scrabble, the game an agent plays.

One player, four letters, a rack of three, and six turns. There is no board.
:class:`RackScrabbleEnv` steps the game. :class:`RackScrabbleSolver` is the
exact best action for labeling and evaluation. Keep learning algorithms
outside this package.
"""

from scrabble.games.rack_scrabble.actions import (
    ACTION_NAMES,
    COPIES_PER_LETTER,
    COST,
    HORIZON,
    LETTERS,
    N_ACTIONS,
    N_LETTERS,
    OBS_SIZE,
    RACK_SIZE,
    SCORE,
)
from scrabble.games.rack_scrabble.env import RackScrabbleEnv
from scrabble.games.rack_scrabble.solver import Decision, RackScrabbleSolver

__all__ = [
    "ACTION_NAMES",
    "COPIES_PER_LETTER",
    "COST",
    "Decision",
    "HORIZON",
    "LETTERS",
    "N_ACTIONS",
    "N_LETTERS",
    "OBS_SIZE",
    "RACK_SIZE",
    "RackScrabbleEnv",
    "RackScrabbleSolver",
    "SCORE",
]
