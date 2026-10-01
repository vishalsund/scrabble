"""Vectorized Mini-Scrabble v1 environment."""

from __future__ import annotations

import numpy as np

from scrabble.games.mini_scrabble import _engine
from scrabble.games.mini_scrabble.rules import (
    BAG,
    BOARD,
    COPIES_PER_LETTER,
    N_ACTIONS,
    N_LETTERS,
    N_PLAYERS,
    N_SQUARES,
    OBS_SIZE,
    OPENED,
    PASSES,
    PLAYER,
    RACK,
    RACK_SIZE,
)


class MiniScrabbleEnv:
    """Many independent Mini-Scrabble games stepped together.

    ``state`` columns are the 25 board cells (0 empty, 1–4 for A–D), both
    racks, the bag, the player to move, the consecutive-pass count, and
    whether any placement has opened the board. ``score`` is the two
    accumulated totals. Both are visible; nothing else is.

    The step reward is the points just scored by the player who moved.
    Passes score 0. Leftover tiles are worth nothing, and the env does not
    add a win or loss when the game ends. Summing a player's rewards is
    that player's score.

    ``reset`` deals from the ongoing random stream. Pass ``seed`` to start
    that stream over. ``reset_done`` redeals only finished games.

    Returned arrays are this object's buffers. The next call overwrites them.
    """

    def __init__(self, n_envs: int, seed: int = 0) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
            raise ValueError("n_envs must be a positive int")
        self.n_envs = n_envs
        self._seed = _as_seed(seed)
        self.state = np.zeros((n_envs, OBS_SIZE), dtype=np.uint8)
        self.score = np.zeros((n_envs, N_PLAYERS), dtype=np.int16)
        self.terminated = np.ones(n_envs, dtype=np.bool_)
        self.invalid = np.zeros(n_envs, dtype=np.bool_)
        self.mask = np.zeros((n_envs, N_ACTIONS), dtype=np.bool_)
        self.rewards = np.zeros(n_envs, dtype=np.int16)
        self._actions = np.zeros(n_envs, dtype=np.int64)
        self.rng = np.empty(n_envs, dtype=np.uint64)
        _engine.reseed(self.rng, self._seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Deal a new game in every env. Returns ``(state, legal_mask)``."""
        if seed is not None:
            self._seed = _as_seed(seed)
            _engine.reseed(self.rng, self._seed)
        self.rewards.fill(0)
        _engine.reset_all(
            self.state,
            self.score,
            self.terminated,
            self.invalid,
            self.mask,
            self.rng,
        )
        return self.state, self.mask

    def reset_done(self) -> tuple[np.ndarray, np.ndarray]:
        """Redeal finished games and leave the others mid-episode."""
        _engine.reset_done(
            self.state,
            self.score,
            self.terminated,
            self.invalid,
            self.mask,
            self.rng,
        )
        return self.state, self.mask

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Play one action per env.

        Returns ``(state, rewards, terminated, legal_mask)``. An illegal
        action sets ``invalid`` for that env and leaves the position
        unchanged. A finished env ignores the action until it is dealt again.
        After a non-terminal move the opponent is to play. A move that ends
        the game leaves the player who just acted as the player to move.
        """
        actions = np.asarray(actions)
        if actions.shape != (self.n_envs,):
            raise ValueError(
                f"actions must have shape {(self.n_envs,)}, got {actions.shape}"
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("actions must be integers")
        if actions.dtype != np.int64 or not actions.flags.c_contiguous:
            actions = np.ascontiguousarray(actions, dtype=np.int64)
        _engine.step(
            self.state,
            self.score,
            self.terminated,
            self.invalid,
            self.mask,
            self.rewards,
            self.rng,
            actions,
        )
        return self.state, self.rewards, self.terminated, self.mask

    def sample_legal_actions(self) -> np.ndarray:
        """Pick a uniform legal action for each env."""
        _engine.sample_actions(self.mask, self.rng, self._actions)
        return self._actions

    def set_state(
        self,
        board: np.ndarray,
        racks: np.ndarray,
        bag: np.ndarray,
        player: int = 0,
        passes: int = 0,
        score: tuple[int, int] | np.ndarray = (0, 0),
        index: int | None = None,
    ) -> None:
        """Place one env, or every env, at an exact position.

        ``board`` is 25 cells, 0 empty and 1–4 for A–D. ``racks`` has shape
        ``(2, 4)``. The board, both racks, and the bag must use each letter
        exactly six times. Does not touch the random stream. ``index=None``
        writes all envs.
        """
        board = _cells(board)
        racks = _racks(racks)
        bag = _counts(bag, "bag")
        if np.any(_letter_counts(board, racks, bag) != COPIES_PER_LETTER):
            raise ValueError("board, racks, and bag must account for every tile")
        player = _small_int(player, "player", 0, 1)
        passes = _small_int(passes, "passes", 0, 2)
        totals = _score_pair(score)

        rows = self._rows(index)
        opened = bool(np.any(board != 0))
        bag_n = int(bag.sum())
        empty_rack = int(racks[0].sum()) == 0 or int(racks[1].sum()) == 0
        done = passes >= 2 or (bag_n == 0 and empty_rack)
        for row in rows:
            self.state[row, BOARD:N_SQUARES] = board
            self.state[row, RACK : RACK + N_PLAYERS * N_LETTERS] = racks.reshape(-1)
            self.state[row, BAG : BAG + N_LETTERS] = bag
            self.state[row, PLAYER] = player
            self.state[row, PASSES] = passes
            self.state[row, OPENED] = opened
            self.score[row] = totals
            self.invalid[row] = False
            self.terminated[row] = done
        _engine.apply_mask(self.state, self.terminated, self.mask)

    def _rows(self, index: int | None) -> range | tuple[int, ...]:
        if index is None:
            return range(self.n_envs)
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("index must be an int")
        if index < 0 or index >= self.n_envs:
            raise IndexError(f"index {index} is outside 0..{self.n_envs - 1}")
        return (index,)


def _as_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise ValueError("seed must be an int in [0, 2**64)")
    return seed


def _small_int(value: int, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an int in [{low}, {high}]")
    return value


def _cells(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (N_SQUARES,):
        raise ValueError(f"board must have shape {(N_SQUARES,)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("board must be integers")
    out = arr.astype(np.int64, copy=False)
    if np.any((out < 0) | (out > N_LETTERS)):
        raise ValueError("board cells must be 0 (empty) or 1–4 (A–D)")
    return out


def _counts(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (N_LETTERS,):
        raise ValueError(f"{name} must have shape {(N_LETTERS,)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integers")
    out = arr.astype(np.int64, copy=False)
    if np.any(out < 0) or np.any(out > COPIES_PER_LETTER):
        raise ValueError(f"{name} counts must be in [0, {COPIES_PER_LETTER}]")
    return out


def _racks(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (N_PLAYERS, N_LETTERS):
        raise ValueError(f"racks must have shape {(N_PLAYERS, N_LETTERS)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("racks must be integers")
    out = arr.astype(np.int64, copy=False)
    if np.any(out < 0) or np.any(out > COPIES_PER_LETTER):
        raise ValueError(f"rack counts must be in [0, {COPIES_PER_LETTER}]")
    if np.any(out.sum(axis=1) > RACK_SIZE):
        raise ValueError(f"a rack cannot hold more than {RACK_SIZE} tiles")
    return out


def _letter_counts(board: np.ndarray, racks: np.ndarray, bag: np.ndarray) -> np.ndarray:
    counts = racks[0] + racks[1] + bag
    for cell in board:
        if cell:
            counts[int(cell) - 1] += 1
    return counts


def _score_pair(value: tuple[int, int] | np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (N_PLAYERS,):
        raise ValueError(f"score must have shape {(N_PLAYERS,)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("score must be integers")
    out = arr.astype(np.int64, copy=False)
    limit = int(np.iinfo(np.int16).max)
    if np.any(out < 0) or np.any(out > limit):
        raise ValueError("scores must be non-negative ints that fit in int16")
    return out
