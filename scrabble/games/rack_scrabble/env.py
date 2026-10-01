"""Vectorized Rack Scrabble environment."""

from __future__ import annotations

import numpy as np

from scrabble.games.rack_scrabble import _engine
from scrabble.games.rack_scrabble.actions import (
    COPIES_PER_LETTER,
    HORIZON,
    N_ACTIONS,
    N_LETTERS,
    OBS_SIZE,
    RACK_SIZE,
)


class RackScrabbleEnv:
    """Many independent Rack Scrabble games stepped together.

    Observation columns are four rack counts, four bag counts, and the
    number of turns completed. The step reward is the score of the pattern
    just played. Summing those rewards is the final score: leftover tiles
    are worth zero, and the env does not add another reward when the game ends.

    ``reset`` deals from the ongoing random stream. Pass ``seed`` to start
    that stream over. ``reset_done`` redeals only finished games, which is
    the call a training loop wants between steps.

    Returned arrays are this object's buffers. The next call overwrites them.
    """

    def __init__(self, n_envs: int, seed: int = 0) -> None:
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
            raise ValueError("n_envs must be a positive int")
        self.n_envs = n_envs
        self._seed = _as_seed(seed)
        self.state = np.zeros((n_envs, OBS_SIZE), dtype=np.uint8)
        self.score = np.zeros(n_envs, dtype=np.int16)
        self.terminated = np.ones(n_envs, dtype=np.bool_)
        self.invalid = np.zeros(n_envs, dtype=np.bool_)
        self.mask = np.zeros((n_envs, N_ACTIONS), dtype=np.bool_)
        self.rewards = np.zeros(n_envs, dtype=np.int16)
        self._actions = np.zeros(n_envs, dtype=np.int64)
        self.rng = np.empty(n_envs, dtype=np.uint64)
        _engine.reseed(self.rng, self._seed)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Deal a new game in every env. Returns ``(obs, legal_mask)``."""
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

        Returns ``(obs, rewards, terminated, legal_mask)``. An action the
        current rack cannot play sets ``invalid`` for that env and leaves
        its tiles, score, and turn unchanged. A finished env ignores the
        action until it is dealt again.
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
        rack: np.ndarray,
        bag: np.ndarray,
        turn: int = 0,
        score: int = 0,
        index: int | None = None,
    ) -> None:
        """Place one env, or every env, at an exact position.

        Does not touch the random stream. ``index=None`` writes all envs.
        """
        rack = _counts(rack, "rack")
        bag = _counts(bag, "bag")
        if int(rack.sum()) > RACK_SIZE:
            raise ValueError(f"rack cannot hold more than {RACK_SIZE} tiles")
        if np.any(rack > RACK_SIZE) or np.any(bag > COPIES_PER_LETTER):
            raise ValueError("tile counts exceed the supply")
        if np.any(rack + bag > COPIES_PER_LETTER):
            raise ValueError("rack plus bag exceeds the copies of a letter")
        turn_ok = isinstance(turn, int) and not isinstance(turn, bool)
        if not turn_ok or not 0 <= turn <= HORIZON:
            raise ValueError(f"turn must be an int in [0, {HORIZON}]")
        max_score = int(np.iinfo(np.int16).max)
        score_ok = isinstance(score, int) and not isinstance(score, bool)
        if not score_ok or not 0 <= score <= max_score:
            raise ValueError("score must be a non-negative int that fits in int16")

        rows = self._rows(index)
        done = turn >= HORIZON or int(rack.sum()) == 0
        for row in rows:
            self.state[row, :N_LETTERS] = rack
            self.state[row, N_LETTERS : N_LETTERS * 2] = bag
            self.state[row, OBS_SIZE - 1] = turn
            self.score[row] = score
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


def _counts(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != (N_LETTERS,):
        raise ValueError(f"{name} must have shape {(N_LETTERS,)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must be integers")
    out = arr.astype(np.int64, copy=False)
    if np.any(out < 0):
        raise ValueError(f"{name} counts must be non-negative")
    return out
