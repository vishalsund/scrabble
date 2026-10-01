"""Shared Mini-Scrabble state, moves, and transitions.

The environment, a model, and these helpers all use the same action ids.
:class:`Move` is that action. :class:`State` is the position from the player
about to move. :func:`apply_move_without_draw` is the position after the
tiles and the score change, before a refill and before the turn swaps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from scrabble.games.mini_scrabble import _engine
from scrabble.games.mini_scrabble.rules import (
    BAG,
    BOARD_SIZE,
    COPIES_PER_LETTER,
    LETTERS,
    LETTERS_AT,
    N_ACTIONS,
    N_LETTERS,
    N_SQUARES,
    N_TILES,
    OBS_SIZE,
    OPENED,
    PASS,
    PASSES,
    PLAYER,
    RACK,
    RACK_SIZE,
    SQUARES,
    action_for,
    placement_score,
)

# Board one-hot (empty, A, B, C, D), both racks, the bag, score difference, passes.
_CELL = N_LETTERS + 1
GLOBAL_DIM = N_LETTERS * 3 + 2
INPUT_DIM = N_SQUARES * _CELL + GLOBAL_DIM
_LETTER_INDEX = {letter: index for index, letter in enumerate(LETTERS)}

_scratch_state = np.zeros((1, OBS_SIZE), dtype=np.uint8)
_scratch_done = np.zeros(1, dtype=np.bool_)
_scratch_mask = np.zeros((1, N_ACTIONS), dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class Move:
    """One action. Placements are ``(row, col, letter)`` with 0-based squares.

    Pass has no placements. The same tiles in any order are one move: the
    squares are stored top-to-bottom, left-to-right.
    """

    placements: tuple[tuple[int, int, str], ...] = ()
    is_pass: bool = False

    def __post_init__(self) -> None:
        if self.is_pass:
            if self.placements:
                raise ValueError("a pass cannot place tiles")
            return
        if not 1 <= len(self.placements) <= RACK_SIZE:
            raise ValueError("a placement uses 1, 2, or 3 tiles")
        ordered = tuple(sorted(self.placements))
        squares: set[tuple[int, int]] = set()
        for row, col, letter in ordered:
            if not 0 <= row < BOARD_SIZE or not 0 <= col < BOARD_SIZE:
                raise ValueError("rows and columns are 0–4")
            if letter not in _LETTER_INDEX:
                raise ValueError("tiles are A, B, C, and D")
            if (row, col) in squares:
                raise ValueError("two tiles cannot share a square")
            squares.add((row, col))
        object.__setattr__(self, "placements", ordered)

    def action_id(self) -> int:
        """Index in the environment's action table."""
        if self.is_pass:
            return PASS
        try:
            return action_for(
                [(row, col, _LETTER_INDEX[letter]) for row, col, letter in self.placements]
            )
        except KeyError:
            raise ValueError("tiles must lie in one row or one column") from None


PASS_MOVE = Move(is_pass=True)


def _move_from_action(action: int) -> Move:
    count = int(N_TILES[action])
    if count == 0:
        return PASS_MOVE
    cells = []
    for slot in range(count):
        row, col = divmod(int(SQUARES[action, slot]), BOARD_SIZE)
        cells.append((row, col, LETTERS[int(LETTERS_AT[action, slot])]))
    return Move(tuple(cells))


MOVES: tuple[Move, ...] = tuple(_move_from_action(action) for action in range(N_ACTIONS))


@dataclass(frozen=True, slots=True)
class State:
    """The game from the player about to move.

    ``board`` has 25 cells, ``"."`` or ``"A"``–``"D"``, row by row.
    ``rack`` and ``opponent_rack`` are letters. ``bag`` is counts of A–D.
    ``score_diff`` is this player's score minus the opponent's.
    ``passes`` is the consecutive-pass count.
    """

    board: tuple[str, ...]
    rack: tuple[str, ...]
    opponent_rack: tuple[str, ...]
    bag: tuple[int, int, int, int]
    score_diff: int
    passes: int

    def __post_init__(self) -> None:
        if len(self.board) != N_SQUARES:
            raise ValueError("the board has 25 squares")
        if len(self.bag) != N_LETTERS:
            raise ValueError("the bag has four letter counts")
        if self.passes < 0:
            raise ValueError("passes cannot be negative")


def deal(seed: int | random.Random) -> State:
    """Deal an opening. The player to move drew first. Same seed, same deal."""
    rng = _rng(seed)
    bag: tuple[int, ...] = (COPIES_PER_LETTER,) * N_LETTERS
    bag, mine = draw_tiles(bag, RACK_SIZE, rng)
    bag, theirs = draw_tiles(bag, RACK_SIZE, rng)
    return State(
        board=tuple("." for _ in range(N_SQUARES)),
        rack=tuple(sorted(mine)),
        opponent_rack=tuple(sorted(theirs)),
        bag=bag,
        score_diff=0,
        passes=0,
    )


def score_placement(
    board: tuple[str, ...] | np.ndarray,
    placements: Move | tuple[tuple[int, int, str], ...],
) -> int:
    """Points scored by placing those tiles on ``board``.

    The rack is not an input: the points are the lengths of the words the
    tiles create or extend. An illegal placement raises ``ValueError``.
    """
    move = placements if isinstance(placements, Move) else Move(tuple(placements))
    if move.is_pass:
        raise ValueError("a pass scores nothing")
    values = _board_values(board)
    rack = np.zeros(N_LETTERS, dtype=np.int64)
    for _, _, letter in move.placements:
        rack[_LETTER_INDEX[letter]] += 1
    opened = bool(np.any(values != 0))
    score = placement_score(values, rack, opened, move.action_id())
    if score < 0:
        raise ValueError("that placement is illegal")
    return int(score)


def legal_moves(board: tuple[str, ...] | np.ndarray, rack: tuple[str, ...]) -> tuple[Move, ...]:
    """Every legal placement for this rack, including pass."""
    values = _board_values(board)
    counts = _count_letters(rack)
    _scratch_state.fill(0)
    _scratch_state[0, :N_SQUARES] = values
    _scratch_state[0, RACK : RACK + N_LETTERS] = counts
    _scratch_state[0, OPENED] = np.uint8(np.any(values != 0))
    _scratch_done[0] = False
    _engine.apply_mask(_scratch_state, _scratch_done, _scratch_mask)
    return tuple(MOVES[int(action)] for action in np.flatnonzero(_scratch_mask[0]))


def apply_move_without_draw(state: State, move: Move) -> State:
    """Place tiles and update the score. The bag, the refill, and the turn stay put."""
    if move.is_pass:
        return State(
            state.board,
            state.rack,
            state.opponent_rack,
            state.bag,
            state.score_diff,
            state.passes + 1,
        )
    points = _points(state, move)
    board = list(state.board)
    rack = list(state.rack)
    for row, col, letter in move.placements:
        try:
            rack.remove(letter)
        except ValueError:
            raise ValueError(f"the rack has no {letter}") from None
        board[row * BOARD_SIZE + col] = letter
    return State(
        tuple(board),
        tuple(rack),
        state.opponent_rack,
        state.bag,
        state.score_diff + points,
        0,
    )


def apply_move(state: State, move: Move, rng: random.Random | int) -> State:
    """Play a move: afterstate, refill if tiles were placed, then the opponent unless the game ended."""
    if result(state) is not None:
        raise ValueError("the game is over")
    after = apply_move_without_draw(state, move)
    if not move.is_pass:
        after = refill(after, rng)
    if result(after) is not None:
        return after
    return _swap(after)


def result(state: State) -> str | None:
    """``"win"``, ``"loss"``, or ``"draw"`` for the player to move, or ``None``.

    The game is over when two passes are in a row, or the bag is empty and
    either rack is empty. On a finished game the player to move is the one
    who just acted.
    """
    bag_empty = sum(state.bag) == 0
    someone_empty = len(state.rack) == 0 or len(state.opponent_rack) == 0
    if state.passes < 2 and not (bag_empty and someone_empty):
        return None
    if state.score_diff > 0:
        return "win"
    if state.score_diff < 0:
        return "loss"
    return "draw"


def draw_tiles(
    bag: tuple[int, ...] | list[int],
    n: int,
    rng: random.Random | int,
) -> tuple[tuple[int, int, int, int], tuple[str, ...]]:
    """Draw up to ``n`` tiles without replacement.

    Each remaining tile is equally likely, so three A's and one B make A
    three times as likely as B. An integer ``rng`` is a seed. A
    ``random.Random`` is advanced and can be reused.
    """
    stream = _rng(rng)
    counts = [int(count) for count in bag]
    if len(counts) != N_LETTERS:
        raise ValueError("the bag has four letter counts")
    drawn: list[str] = []
    for _ in range(n):
        total = sum(counts)
        if total == 0:
            break
        pick = stream.randrange(total)
        acc = 0
        for index, count in enumerate(counts):
            acc += count
            if pick < acc:
                counts[index] -= 1
                drawn.append(LETTERS[index])
                break
    return (counts[0], counts[1], counts[2], counts[3]), tuple(drawn)


def refill(state: State, rng: random.Random | int) -> State:
    """Draw until the player to move has three tiles or the bag is empty."""
    need = RACK_SIZE - len(state.rack)
    if need <= 0:
        return state
    bag, drawn = draw_tiles(state.bag, need, rng)
    return State(
        state.board,
        tuple(sorted((*state.rack, *drawn))),
        state.opponent_rack,
        bag,
        state.score_diff,
        state.passes,
    )


def encode(state: State) -> np.ndarray:
    """Flatten one state the way :func:`encode_env` flattens a batch."""
    board = _board_values(state.board).reshape(1, N_SQUARES)
    rack = np.asarray(_count_letters(state.rack), dtype=np.int64).reshape(1, N_LETTERS)
    opponent = np.asarray(_count_letters(state.opponent_rack), dtype=np.int64).reshape(1, N_LETTERS)
    bag = np.asarray(state.bag, dtype=np.int64).reshape(1, N_LETTERS)
    return encode_batch(
        board,
        rack,
        opponent,
        bag,
        np.asarray([state.score_diff], dtype=np.float32),
        np.asarray([state.passes], dtype=np.float32),
    )[0]


def encode_spatial(state: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split :func:`encode_env` into a board image and the other features.

    The board is ``(n, 5, 5, 5)`` in channels-first order: empty, A, B, C, D.
    Globals are the current rack, the opponent rack, the bag, the score
    difference divided by 20, and the pass count.
    """
    flat = encode_env(state, score)
    board = flat[:, : N_SQUARES * _CELL].reshape(-1, BOARD_SIZE, BOARD_SIZE, _CELL)
    board = np.ascontiguousarray(np.transpose(board, (0, 3, 1, 2)))
    globals_ = np.ascontiguousarray(flat[:, N_SQUARES * _CELL :])
    return board, globals_


def encode_env(state: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Flatten env rows from each player-to-move's point of view."""
    player = state[:, PLAYER].astype(np.int64)
    rack0 = state[:, RACK : RACK + N_LETTERS].astype(np.int64)
    rack1 = state[:, RACK + N_LETTERS : RACK + 2 * N_LETTERS].astype(np.int64)
    current = np.where(player[:, None] == 0, rack0, rack1)
    opponent = np.where(player[:, None] == 0, rack1, rack0)
    diff = np.where(player == 0, score[:, 0] - score[:, 1], score[:, 1] - score[:, 0])
    return encode_batch(
        state[:, :N_SQUARES].astype(np.int64),
        current,
        opponent,
        state[:, BAG : BAG + N_LETTERS].astype(np.int64),
        diff.astype(np.float32),
        state[:, PASSES].astype(np.float32),
    )


def encode_batch(
    board: np.ndarray,
    rack: np.ndarray,
    opponent: np.ndarray,
    bag: np.ndarray,
    score_diff: np.ndarray,
    passes: np.ndarray,
) -> np.ndarray:
    """Stack the state fields into one float vector per row."""
    n = board.shape[0]
    features = np.zeros((n, INPUT_DIM), dtype=np.float32)
    one_hot = np.eye(_CELL, dtype=np.float32)[board.astype(np.int64, copy=False)]
    features[:, : N_SQUARES * _CELL] = one_hot.reshape(n, N_SQUARES * _CELL)
    offset = N_SQUARES * _CELL
    features[:, offset : offset + N_LETTERS] = rack
    offset += N_LETTERS
    features[:, offset : offset + N_LETTERS] = opponent
    offset += N_LETTERS
    features[:, offset : offset + N_LETTERS] = bag
    features[:, -2] = score_diff / 20.0
    features[:, -1] = passes
    return features


def from_env(state_row: np.ndarray, score_row: np.ndarray) -> State:
    """Read one env row as a :class:`State` for the player to move."""
    player = int(state_row[PLAYER])
    board = tuple(
        "." if int(value) == 0 else LETTERS[int(value) - 1] for value in state_row[:N_SQUARES]
    )
    rack = _letters(state_row[RACK + player * N_LETTERS : RACK + (player + 1) * N_LETTERS])
    other = 1 - player
    opponent = _letters(state_row[RACK + other * N_LETTERS : RACK + (other + 1) * N_LETTERS])
    bag = tuple(int(state_row[BAG + index]) for index in range(N_LETTERS))
    diff = int(score_row[player] - score_row[other])
    return State(board, rack, opponent, bag, diff, int(state_row[PASSES]))


def _points(state: State, move: Move) -> int:
    values = _board_values(state.board)
    rack = np.asarray(_count_letters(state.rack), dtype=np.int64)
    opened = any(cell != "." for cell in state.board)
    score = placement_score(values, rack, opened, move.action_id())
    if score < 0:
        raise ValueError("that placement is illegal")
    return int(score)


def _swap(state: State) -> State:
    return State(
        state.board,
        state.opponent_rack,
        state.rack,
        state.bag,
        -state.score_diff,
        state.passes,
    )


def _rng(rng: random.Random | int) -> random.Random:
    if isinstance(rng, random.Random):
        return rng
    return random.Random(int(rng))


def _board_values(board: tuple[str, ...] | np.ndarray) -> np.ndarray:
    if isinstance(board, np.ndarray):
        values = np.asarray(board, dtype=np.int64).reshape(N_SQUARES)
        return values
    if len(board) != N_SQUARES:
        raise ValueError("the board has 25 squares")
    values = np.zeros(N_SQUARES, dtype=np.int64)
    for index, cell in enumerate(board):
        if cell == ".":
            continue
        if cell not in _LETTER_INDEX:
            raise ValueError("tiles are A, B, C, and D")
        values[index] = _LETTER_INDEX[cell] + 1
    return values


def _count_letters(letters: tuple[str, ...]) -> tuple[int, int, int, int]:
    counts = [0, 0, 0, 0]
    for letter in letters:
        counts[_LETTER_INDEX[letter]] += 1
    return (counts[0], counts[1], counts[2], counts[3])


def _letters(counts) -> tuple[str, ...]:
    return tuple(
        letter
        for letter, count in zip(LETTERS, counts, strict=True)
        for _ in range(int(count))
    )
