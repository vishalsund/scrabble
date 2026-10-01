"""Numba step loop for Rack Scrabble.

Training code should use :class:`~scrabble.games.rack_scrabble.RackScrabbleEnv`
instead of these kernels.
"""

import numpy as np
from numba import njit, prange

from scrabble.games.rack_scrabble.actions import (
    COPIES_PER_LETTER,
    COST,
    HORIZON,
    N_ACTIONS,
    N_LETTERS,
    RACK_SIZE,
    SCORE,
)

_TURN = N_LETTERS * 2
_BAG = N_LETTERS


@njit
def _rand_u64(rng, i):
    rng[i] = np.uint64(rng[i] + np.uint64(0x9E3779B97F4A7C15))
    z = rng[i]
    z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> 31)


@njit
def _rack_count(state_row):
    total = 0
    for letter in range(N_LETTERS):
        total += int(state_row[letter])
    return total


@njit
def _sample_from_bag(state_row, total, rng, i):
    pick = _rand_u64(rng, i) % np.uint64(total)
    acc = np.uint64(0)
    for letter in range(N_LETTERS):
        acc += np.uint64(state_row[_BAG + letter])
        if pick < acc:
            return letter
    return 0


@njit
def _refill(state_row, rng, i):
    """Draw until the rack has 3 tiles or the bag is empty."""
    rack_n = _rack_count(state_row)
    while rack_n < RACK_SIZE:
        bag_n = 0
        for letter in range(N_LETTERS):
            bag_n += int(state_row[_BAG + letter])
        if bag_n == 0:
            return
        letter = _sample_from_bag(state_row, bag_n, rng, i)
        state_row[_BAG + letter] -= np.uint8(1)
        state_row[letter] += np.uint8(1)
        rack_n += 1


@njit
def _fill_mask_row(state_row, is_done, mask_row):
    for action in range(N_ACTIONS):
        if is_done:
            mask_row[action] = False
            continue
        legal = True
        for letter in range(N_LETTERS):
            if state_row[letter] < COST[action, letter]:
                legal = False
                break
        mask_row[action] = legal


@njit
def _deal_row(state_row, score, terminated, invalid, mask_row, rng, i):
    for letter in range(N_LETTERS):
        state_row[letter] = 0
        state_row[_BAG + letter] = COPIES_PER_LETTER
    state_row[_TURN] = 0
    score[i] = 0
    terminated[i] = False
    invalid[i] = False
    _refill(state_row, rng, i)
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
        _deal_row(state[i], score, terminated, invalid, mask[i], rng, i)


@njit(parallel=True, cache=True)
def reset_done(state, score, terminated, invalid, mask, rng):
    for i in prange(state.shape[0]):
        if terminated[i]:
            _deal_row(state[i], score, terminated, invalid, mask[i], rng, i)


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
        legal = True
        for letter in range(N_LETTERS):
            if state[i, letter] < COST[action, letter]:
                legal = False
                break
        if not legal:
            invalid[i] = True
            continue
        invalid[i] = False
        for letter in range(N_LETTERS):
            state[i, letter] -= COST[action, letter]
        gained = SCORE[action]
        rewards[i] = gained
        score[i] += gained
        state[i, _TURN] = np.uint8(int(state[i, _TURN]) + 1)
        # The sixth action ends the game before a refill. An empty rack
        # ends it on an earlier turn.
        if int(state[i, _TURN]) < HORIZON:
            _refill(state[i], rng, i)
            if _rack_count(state[i]) == 0:
                terminated[i] = True
        else:
            terminated[i] = True
        _fill_mask_row(state[i], terminated[i], mask[i])
