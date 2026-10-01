"""Numba step loop for Mini-Scrabble.

Training code should use :class:`~scrabble.games.mini_scrabble.MiniScrabbleEnv`
instead of these kernels. Placement legality here matches
:func:`scrabble.games.mini_scrabble.rules.placement_score`.
"""

import numpy as np
from numba import njit, prange

from scrabble.games.mini_scrabble.rules import (
    AXIS,
    BAG,
    BOARD_SIZE,
    CENTER,
    COPIES_PER_LETTER,
    COST,
    GAPS,
    LEGAL,
    LETTERS_AT,
    N_ACTIONS,
    N_GAPS,
    N_LETTERS,
    N_TILES,
    OBS_SIZE,
    OPENED,
    PASS,
    PASSES,
    PLAYER,
    RACK,
    RACK_SIZE,
    SQUARES,
)

_BOARD = BOARD_SIZE


@njit
def _rand_u64(rng, i):
    rng[i] = np.uint64(rng[i] + np.uint64(0x9E3779B97F4A7C15))
    z = rng[i]
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@njit
def _rack_count(state_row, player):
    total = 0
    base = RACK + player * N_LETTERS
    for letter in range(N_LETTERS):
        total += int(state_row[base + letter])
    return total


@njit
def _bag_count(state_row):
    total = 0
    for letter in range(N_LETTERS):
        total += int(state_row[BAG + letter])
    return total


@njit
def _sample_from_bag(state_row, total, rng, i):
    pick = _rand_u64(rng, i) % np.uint64(total)
    acc = np.uint64(0)
    for letter in range(N_LETTERS):
        acc += np.uint64(state_row[BAG + letter])
        if pick < acc:
            return letter
    return 0


@njit
def _refill(state_row, player, rng, i):
    """Draw until this rack has 3 tiles or the bag is empty."""
    base = RACK + player * N_LETTERS
    rack_n = _rack_count(state_row, player)
    while rack_n < RACK_SIZE:
        bag_n = _bag_count(state_row)
        if bag_n == 0:
            return
        letter = _sample_from_bag(state_row, bag_n, rng, i)
        state_row[BAG + letter] -= np.uint8(1)
        state_row[base + letter] += np.uint8(1)
        rack_n += 1


@njit
def _letter_at(state_row, action, count, pos):
    for slot in range(count):
        if SQUARES[action, slot] == pos:
            return LETTERS_AT[action, slot]
    value = int(state_row[pos])
    if value == 0:
        return -1
    return value - 1


@njit
def _run(state_row, action, count, row, col, horizontal):
    if horizontal:
        start = col
        while start > 0 and _letter_at(state_row, action, count, row * _BOARD + start - 1) >= 0:
            start -= 1
        stop = col
        while (
            stop < _BOARD - 1
            and _letter_at(state_row, action, count, row * _BOARD + stop + 1) >= 0
        ):
            stop += 1
        code = 0
        for cursor in range(start, stop + 1):
            code = code * N_LETTERS + _letter_at(state_row, action, count, row * _BOARD + cursor)
        return stop - start + 1, code
    start = row
    while start > 0 and _letter_at(state_row, action, count, (start - 1) * _BOARD + col) >= 0:
        start -= 1
    stop = row
    while (
        stop < _BOARD - 1
        and _letter_at(state_row, action, count, (stop + 1) * _BOARD + col) >= 0
    ):
        stop += 1
    code = 0
    for cursor in range(start, stop + 1):
        code = code * N_LETTERS + _letter_at(state_row, action, count, cursor * _BOARD + col)
    return stop - start + 1, code


@njit
def _score_run(length, code):
    if length <= 1:
        return 0
    if length > 4:
        return -1
    if not LEGAL[length, code]:
        return -1
    return length


@njit
def _placement_score(state_row, action):
    """Points for a non-pass action, or -1 when the placement is illegal."""
    count = N_TILES[action]
    player = int(state_row[PLAYER])
    base = RACK + player * N_LETTERS
    for letter in range(N_LETTERS):
        if int(state_row[base + letter]) < COST[action, letter]:
            return -1
    for slot in range(count):
        if int(state_row[SQUARES[action, slot]]) != 0:
            return -1
    n_gaps = N_GAPS[action]
    for slot in range(n_gaps):
        if int(state_row[GAPS[action, slot]]) == 0:
            return -1
    if int(state_row[OPENED]) == 0:
        if count < 2:
            return -1
        covers = False
        for slot in range(count):
            if SQUARES[action, slot] == CENTER:
                covers = True
        if not covers:
            return -1
    else:
        touches = False
        for slot in range(count):
            pos = SQUARES[action, slot]
            row = pos // _BOARD
            col = pos % _BOARD
            if row > 0 and int(state_row[pos - _BOARD]) != 0:
                touches = True
            if row < _BOARD - 1 and int(state_row[pos + _BOARD]) != 0:
                touches = True
            if col > 0 and int(state_row[pos - 1]) != 0:
                touches = True
            if col < _BOARD - 1 and int(state_row[pos + 1]) != 0:
                touches = True
        if not touches:
            return -1

    origin = SQUARES[action, 0]
    row = origin // _BOARD
    col = origin % _BOARD
    horizontal = AXIS[action] == 0
    length, code = _run(state_row, action, count, row, col, horizontal)
    total = _score_run(length, code)
    if total < 0:
        return -1
    for slot in range(count):
        pos = SQUARES[action, slot]
        length, code = _run(
            state_row,
            action,
            count,
            pos // _BOARD,
            pos % _BOARD,
            not horizontal,
        )
        gained = _score_run(length, code)
        if gained < 0:
            return -1
        total += gained
    if total == 0:
        return -1
    return total


@njit
def _fill_mask_row(state_row, is_done, mask_row):
    if is_done:
        for action in range(N_ACTIONS):
            mask_row[action] = False
        return
    mask_row[PASS] = True
    for action in range(1, N_ACTIONS):
        mask_row[action] = _placement_score(state_row, action) >= 0


@njit
def _ended_by_empty_rack(state_row):
    if _bag_count(state_row) != 0:
        return False
    return _rack_count(state_row, 0) == 0 or _rack_count(state_row, 1) == 0


@njit
def _deal_row(state_row, score_row, terminated, invalid, mask_row, rng, i):
    for col in range(OBS_SIZE):
        state_row[col] = 0
    for letter in range(N_LETTERS):
        state_row[BAG + letter] = COPIES_PER_LETTER
    score_row[0] = 0
    score_row[1] = 0
    terminated[i] = False
    invalid[i] = False
    # Fair coin for the starter, then three tiles for them, then three for the other.
    starter = int(_rand_u64(rng, i) % np.uint64(2))
    state_row[PLAYER] = starter
    _refill(state_row, starter, rng, i)
    _refill(state_row, 1 - starter, rng, i)
    _fill_mask_row(state_row, False, mask_row)


@njit(parallel=True, cache=True)
def reseed(rng, seed):
    base = np.uint64(seed)
    stride = np.uint64(0x9E3779B97F4A7C15)
    for i in prange(rng.shape[0]):
        rng[i] = base + np.uint64(i) * stride + np.uint64(1)


@njit(parallel=True, cache=True)
def reset_all(state, score, terminated, invalid, mask, rng):
    for i in prange(state.shape[0]):
        _deal_row(state[i], score[i], terminated, invalid, mask[i], rng, i)


@njit(parallel=True, cache=True)
def reset_done(state, score, terminated, invalid, mask, rng):
    for i in prange(state.shape[0]):
        if terminated[i]:
            _deal_row(state[i], score[i], terminated, invalid, mask[i], rng, i)


@njit(parallel=True, cache=True)
def apply_mask(state, terminated, mask):
    for i in prange(state.shape[0]):
        _fill_mask_row(state[i], terminated[i], mask[i])


@njit(parallel=True, cache=True)
def sample_actions(mask, rng, actions):
    """Write a uniform legal action for each env into ``actions``."""
    for i in prange(mask.shape[0]):
        n_legal = 0
        for action in range(N_ACTIONS):
            if mask[i, action]:
                n_legal += 1
        if n_legal == 0:
            actions[i] = 0
            continue
        pick = int(_rand_u64(rng, i) % np.uint64(n_legal))
        seen = 0
        chosen = 0
        for action in range(N_ACTIONS):
            if mask[i, action]:
                if seen == pick:
                    chosen = action
                    break
                seen += 1
        actions[i] = chosen


@njit
def _apply_placement(state_row, score_row, player, action, gained):
    count = N_TILES[action]
    base = RACK + player * N_LETTERS
    for slot in range(count):
        state_row[SQUARES[action, slot]] = np.uint8(LETTERS_AT[action, slot] + 1)
        state_row[base + LETTERS_AT[action, slot]] -= np.uint8(1)
    score_row[player] += gained
    state_row[PASSES] = 0
    state_row[OPENED] = 1


@njit(parallel=True, cache=True)
def step(state, score, terminated, invalid, mask, rewards, rng, actions):
    """Play ``actions``. Illegal actions set ``invalid`` and change nothing."""
    for i in prange(state.shape[0]):
        rewards[i] = 0
        if terminated[i]:
            invalid[i] = False
            continue
        action = actions[i]
        if action < 0 or action >= N_ACTIONS:
            invalid[i] = True
            continue
        if action == PASS:
            invalid[i] = False
            state[i, PASSES] = np.uint8(int(state[i, PASSES]) + 1)
            if int(state[i, PASSES]) >= 2:
                terminated[i] = True
            else:
                state[i, PLAYER] = np.uint8(1 - int(state[i, PLAYER]))
            _fill_mask_row(state[i], terminated[i], mask[i])
            continue
        gained = _placement_score(state[i], action)
        if gained < 0:
            invalid[i] = True
            continue
        invalid[i] = False
        player = int(state[i, PLAYER])
        _apply_placement(state[i], score[i], player, action, gained)
        rewards[i] = gained
        _refill(state[i], player, rng, i)
        if _ended_by_empty_rack(state[i]):
            terminated[i] = True
        else:
            state[i, PLAYER] = np.uint8(1 - player)
        _fill_mask_row(state[i], terminated[i], mask[i])
