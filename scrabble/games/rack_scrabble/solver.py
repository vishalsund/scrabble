"""Exact best play for Rack Scrabble.

Each position is solved once from the last turn backward and stored, which is
memoized dynamic programming over a finite tree. A position is the rack
counts, the bag counts, and how many turns are already completed. Points
already scored do not change the best action.

``value`` is the expected score still to be earned. Draws are averaged
exactly, as rationals, over the physical tiles left in the bag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from typing import NamedTuple

import numpy as np

from scrabble.games.rack_scrabble.actions import (
    COPIES_PER_LETTER,
    COST,
    HORIZON,
    N_ACTIONS,
    N_LETTERS,
    OBS_SIZE,
    RACK_SIZE,
    SCORE,
)

_FACT = [math.factorial(n) for n in range(COPIES_PER_LETTER * N_LETTERS + 1)]
_COMB = [
    [math.comb(have, take) for take in range(have + 1)]
    for have in range(COPIES_PER_LETTER + 1)
]


class _Jump(NamedTuple):
    score: int
    total: int
    children: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class Decision:
    """Best action in one position, and the value of every legal action.

    ``action`` is the lowest-index action that achieves ``value``. It is
    ``None`` when the rack is empty or the horizon is over. ``q_values[a]``
    is the expected remaining score if ``a`` is played next and later turns
    are played optimally. Illegal actions are ``None``.
    """

    action: int | None
    value: Fraction
    q_values: tuple[Fraction | None, ...]


class RackScrabbleSolver:
    """Optimal policy for every reachable Rack Scrabble position.

    The table is built once. Later lookups do not resimulate the game.
    """

    def __init__(self) -> None:
        self._positions = list(_positions())
        self._index = {
            _pos_key(rack, bag): index
            for index, (rack, bag) in enumerate(self._positions)
        }
        self._legal: list[tuple[int, ...]] = []
        self._jumps: list[tuple[_Jump, ...]] = []
        for rack, bag in self._positions:
            actions = tuple(
                action
                for action in range(N_ACTIONS)
                if _fits(rack, action)
            )
            self._legal.append(actions)
            self._jumps.append(
                tuple(_jump(self._index, rack, bag, action) for action in actions)
            )

        width = len(self._positions)
        self._num = [[0] * width for _ in range(HORIZON)]
        self._den = [[1] * width for _ in range(HORIZON)]
        self._best = [[-1] * width for _ in range(HORIZON)]
        self._solve()

    @property
    def n_states(self) -> int:
        """Positions stored in the table, across every decision turn."""
        return len(self._positions) * HORIZON

    def decide(self, rack: np.ndarray, bag: np.ndarray, turn: int) -> Decision:
        """Return the exact best action and action values for this position."""
        rack_counts, bag_counts, turn_index = _parse_position(rack, bag, turn)
        if turn_index >= HORIZON or sum(rack_counts) == 0:
            return Decision(None, Fraction(0), (None,) * N_ACTIONS)
        pos = self._index[_pos_key(rack_counts, bag_counts)]
        q_values: list[Fraction | None] = [None] * N_ACTIONS
        for action, jump in zip(self._legal[pos], self._jumps[pos], strict=True):
            num, den = self._q_pair(turn_index, jump)
            q_values[action] = Fraction(num, den)
        action = self._best[turn_index][pos]
        value = Fraction(self._num[turn_index][pos], self._den[turn_index][pos])
        return Decision(action, value, tuple(q_values))

    def decide_obs(self, obs: np.ndarray) -> Decision:
        """``decide`` for one env observation row."""
        row = np.asarray(obs)
        if row.shape != (OBS_SIZE,):
            raise ValueError(f"obs must have shape {(OBS_SIZE,)}, got {row.shape}")
        if not np.issubdtype(row.dtype, np.integer):
            raise ValueError("obs must be integers")
        return self.decide(
            row[:N_LETTERS],
            row[N_LETTERS : N_LETTERS * 2],
            int(row[-1]),
        )

    def best_action(self, rack: np.ndarray, bag: np.ndarray, turn: int) -> int:
        """Lowest-index optimal action. Raises when the game is already over."""
        rack_counts, bag_counts, turn_index = _parse_position(rack, bag, turn)
        if turn_index >= HORIZON or sum(rack_counts) == 0:
            raise ValueError("position has no legal action")
        pos = self._index[_pos_key(rack_counts, bag_counts)]
        return self._best[turn_index][pos]

    def _solve(self) -> None:
        for turn in range(HORIZON - 1, -1, -1):
            nums = self._num[turn]
            dens = self._den[turn]
            bests = self._best[turn]
            for pos, actions in enumerate(self._legal):
                if not actions:
                    continue
                best_action = -1
                best_num = 0
                best_den = 1
                for action_index, jump in enumerate(self._jumps[pos]):
                    q_num, q_den = self._q_pair(turn, jump)
                    if best_action < 0 or q_num * best_den > best_num * q_den:
                        best_action = actions[action_index]
                        best_num = q_num
                        best_den = q_den
                nums[pos] = best_num
                dens[pos] = best_den
                bests[pos] = best_action

    def _q_pair(self, turn: int, jump: _Jump) -> tuple[int, int]:
        if turn + 1 >= HORIZON:
            return jump.score, 1
        acc_num = 0
        acc_den = 1
        child_num = self._num[turn + 1]
        child_den = self._den[turn + 1]
        for child, ways in jump.children:
            acc_num, acc_den = _add(
                acc_num,
                acc_den,
                ways * child_num[child],
                child_den[child],
            )
        mixed = jump.score * acc_den * jump.total + acc_num
        return _norm(mixed, acc_den * jump.total)


def _solve_jump_children(
    index: dict[int, int],
    rack_after: tuple[int, ...],
    bag: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], int]:
    need = RACK_SIZE - sum(rack_after)
    bag_size = sum(bag)
    if need <= 0 or bag_size == 0:
        return ((index[_pos_key(rack_after, bag)], 1),), 1
    if bag_size <= need:
        filled = tuple(rack_after[letter] + bag[letter] for letter in range(N_LETTERS))
        empty = (0,) * N_LETTERS
        return ((index[_pos_key(filled, empty)], 1),), 1
    total = _FACT[bag_size] // _FACT[bag_size - need]
    children = []
    for drawn, ways in _draw_outcomes(bag, need):
        child_rack = tuple(rack_after[letter] + drawn[letter] for letter in range(N_LETTERS))
        child_bag = tuple(bag[letter] - drawn[letter] for letter in range(N_LETTERS))
        children.append((index[_pos_key(child_rack, child_bag)], ways))
    return tuple(children), total


def _jump(
    index: dict[int, int],
    rack: tuple[int, ...],
    bag: tuple[int, ...],
    action: int,
) -> _Jump:
    cost = COST[action]
    rack_after = tuple(rack[letter] - int(cost[letter]) for letter in range(N_LETTERS))
    children, total = _solve_jump_children(index, rack_after, bag)
    return _Jump(int(SCORE[action]), total, children)


def _positions() -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    positions = []
    for rack in product(range(RACK_SIZE + 1), repeat=N_LETTERS):
        if sum(rack) > RACK_SIZE:
            continue
        bag_ranges = [range(COPIES_PER_LETTER - count + 1) for count in rack]
        for bag in product(*bag_ranges):
            positions.append((rack, bag))
    return positions


def _fits(rack: tuple[int, ...], action: int) -> bool:
    cost = COST[action]
    return all(rack[letter] >= int(cost[letter]) for letter in range(N_LETTERS))


def _pos_key(rack: tuple[int, ...], bag: tuple[int, ...]) -> int:
    key = 0
    for count in rack:
        key = (key << 2) | count
    for count in bag:
        key = (key << 3) | count
    return key


@lru_cache(maxsize=None)
def _draw_outcomes(
    bag: tuple[int, ...],
    k: int,
) -> tuple[tuple[tuple[int, ...], int], ...]:
    """Multisets of ``k`` tiles and how many ordered draws produce each one."""
    found: list[tuple[int, ...]] = []

    def walk(letter: int, left: int, taken: tuple[int, ...]) -> None:
        if letter == N_LETTERS - 1:
            if left <= bag[letter]:
                found.append(taken + (left,))
            return
        for take in range(min(bag[letter], left) + 1):
            walk(letter + 1, left - take, taken + (take,))

    walk(0, k, ())
    weighted = []
    for drawn in found:
        ways = _FACT[k]
        for have, take in zip(bag, drawn, strict=True):
            ways *= _COMB[have][take]
        weighted.append((drawn, ways))
    total = sum(ways for _, ways in weighted)
    if total != _FACT[sum(bag)] // _FACT[sum(bag) - k]:
        raise RuntimeError("draw weights do not cover the bag")
    return tuple(weighted)


def _norm(num: int, den: int) -> tuple[int, int]:
    if num == 0:
        return 0, 1
    divisor = math.gcd(num, den)
    return num // divisor, den // divisor


def _add(n1: int, d1: int, n2: int, d2: int) -> tuple[int, int]:
    if n1 == 0:
        return _norm(n2, d2)
    if n2 == 0:
        return n1, d1
    divisor = math.gcd(d1, d2)
    num = n1 * (d2 // divisor) + n2 * (d1 // divisor)
    den = d1 * (d2 // divisor)
    return _norm(num, den)


def _parse_position(
    rack: np.ndarray,
    bag: np.ndarray,
    turn: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int]:
    rack_counts = _parse_counts(rack, "rack", RACK_SIZE)
    bag_counts = _parse_counts(bag, "bag", COPIES_PER_LETTER)
    if sum(rack_counts) > RACK_SIZE:
        raise ValueError(f"rack cannot hold more than {RACK_SIZE} tiles")
    overflow = zip(rack_counts, bag_counts, strict=True)
    if any(left + right > COPIES_PER_LETTER for left, right in overflow):
        raise ValueError("rack plus bag exceeds the copies of a letter")
    return rack_counts, bag_counts, _parse_turn(turn)


def _parse_turn(turn: int) -> int:
    if isinstance(turn, bool):
        raise ValueError(f"turn must be an int in [0, {HORIZON}]")
    if isinstance(turn, np.integer):
        value = int(turn)
    elif isinstance(turn, int):
        value = turn
    else:
        raise ValueError(f"turn must be an int in [0, {HORIZON}]")
    if not 0 <= value <= HORIZON:
        raise ValueError(f"turn must be an int in [0, {HORIZON}]")
    return value


def _parse_counts(value: np.ndarray, name: str, per_letter_max: int) -> tuple[int, int, int, int]:
    arr = np.asarray(value)
    if arr.shape != (N_LETTERS,):
        raise ValueError(f"{name} must have shape {(N_LETTERS,)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integers")
    counts = [int(count) for count in arr.tolist()]
    if any(count < 0 for count in counts):
        raise ValueError(f"{name} counts must be non-negative")
    if any(count > per_letter_max for count in counts):
        raise ValueError("tile counts exceed the supply")
    letter_a, letter_b, letter_c, letter_d = counts
    return letter_a, letter_b, letter_c, letter_d
