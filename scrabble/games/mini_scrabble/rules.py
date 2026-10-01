"""Mini-Scrabble v1 rules, dictionary, and the flat action table.

An action is either pass (id 0) or a placement of one, two, or three tiles on
one row or one column. The table lists every geometric placement, including
ones the current board will reject. Legality (ownership, empty squares,
filled gaps, the opening rule, connection, and the dictionary) is decided
at step time.

:func:`placement_score` is the readable statement of a placement. The Numba
step loop in ``_engine`` is the same check and has to stay in step with it.
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

LETTERS = ("A", "B", "C", "D")
N_LETTERS = 4
BOARD_SIZE = 5
N_SQUARES = BOARD_SIZE * BOARD_SIZE
CENTER = 2 * BOARD_SIZE + 2  # (3, 3) in 1-based coordinates
RACK_SIZE = 3
COPIES_PER_LETTER = 6
N_PLAYERS = 2

# state columns: 25 board cells, then each player's four counts, the bag,
# whose turn it is, consecutive passes, and whether the board has been opened.
BOARD = 0
RACK = N_SQUARES
BAG = RACK + N_PLAYERS * N_LETTERS
PLAYER = BAG + N_LETTERS
PASSES = PLAYER + 1
OPENED = PASSES + 1
OBS_SIZE = OPENED + 1

PASS = 0

# Word code is a base-4 integer. Length 5 is never indexed: a run of 5 is illegal.
LEGAL = np.zeros((5, 256), dtype=np.bool_)


def _code(letters: tuple[int, ...]) -> int:
    value = 0
    for letter in letters:
        value = value * N_LETTERS + letter
    return value


for _a, _b in product(range(N_LETTERS), repeat=2):
    LEGAL[2, _code((_a, _b))] = True

for _letters in product(range(N_LETTERS), repeat=3):
    if len(set(_letters)) in (1, 3):
        LEGAL[3, _code(_letters)] = True

for _letters in product(range(N_LETTERS), repeat=4):
    if len(set(_letters)) == 4:
        LEGAL[4, _code(_letters)] = True


def _square(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def _gaps(squares: list[int], axis: int) -> list[int]:
    """Squares strictly between the new tiles. Those must already be full."""
    if axis == 0:
        row = squares[0] // BOARD_SIZE
        cols = [square % BOARD_SIZE for square in squares]
        return [
            _square(row, col)
            for col in range(min(cols), max(cols) + 1)
            if _square(row, col) not in squares
        ]
    col = squares[0] % BOARD_SIZE
    rows = [square // BOARD_SIZE for square in squares]
    return [
        _square(row, col)
        for row in range(min(rows), max(rows) + 1)
        if _square(row, col) not in squares
    ]


_squares: list[list[int]] = [[]]
_letters: list[list[int]] = [[]]
_axes: list[int] = [0]
_index: dict[tuple[tuple[int, int], ...], int] = {}


def _add(cells: list[tuple[int, int]], letters: tuple[int, ...], axis: int) -> None:
    squares = [_square(row, col) for row, col in cells]
    key = tuple(sorted(zip(squares, letters, strict=True)))
    _index[key] = len(_squares)
    _squares.append(squares)
    _letters.append(list(letters))
    _axes.append(axis)


for _row, _col, _letter in product(range(BOARD_SIZE), range(BOARD_SIZE), range(N_LETTERS)):
    _add([(_row, _col)], (_letter,), 0)

for _row, (_c0, _c1), _pair in product(
    range(BOARD_SIZE), combinations(range(BOARD_SIZE), 2), product(range(N_LETTERS), repeat=2)
):
    _add([(_row, _c0), (_row, _c1)], _pair, 0)

for _col, (_r0, _r1), _pair in product(
    range(BOARD_SIZE), combinations(range(BOARD_SIZE), 2), product(range(N_LETTERS), repeat=2)
):
    _add([(_r0, _col), (_r1, _col)], _pair, 1)

for _row, _cols, _triple in product(
    range(BOARD_SIZE), combinations(range(BOARD_SIZE), 3), product(range(N_LETTERS), repeat=3)
):
    _add([(_row, _col) for _col in _cols], _triple, 0)

for _col, _rows, _triple in product(
    range(BOARD_SIZE), combinations(range(BOARD_SIZE), 3), product(range(N_LETTERS), repeat=3)
):
    _add([(_row, _col) for _row in _rows], _triple, 1)

N_ACTIONS = len(_squares)
N_TILES = np.array([len(squares) for squares in _squares], dtype=np.int64)
SQUARES = np.full((N_ACTIONS, RACK_SIZE), -1, dtype=np.int64)
LETTERS_AT = np.zeros((N_ACTIONS, RACK_SIZE), dtype=np.int64)
GAPS = np.full((N_ACTIONS, RACK_SIZE), -1, dtype=np.int64)
N_GAPS = np.zeros(N_ACTIONS, dtype=np.int64)
AXIS = np.array(_axes, dtype=np.int64)
COST = np.zeros((N_ACTIONS, N_LETTERS), dtype=np.int64)

for _action, (_sq, _let) in enumerate(zip(_squares, _letters, strict=True)):
    for _slot, (_pos, _letter) in enumerate(zip(_sq, _let, strict=True)):
        SQUARES[_action, _slot] = _pos
        LETTERS_AT[_action, _slot] = _letter
        COST[_action, _letter] += 1
    _gap_list = _gaps(_sq, int(AXIS[_action])) if _sq else []
    N_GAPS[_action] = len(_gap_list)
    for _slot, _pos in enumerate(_gap_list):
        GAPS[_action, _slot] = _pos


def is_legal_word(word: str) -> bool:
    """True when ``word`` is one of the 68 dictionary entries."""
    if len(word) < 2 or len(word) > 4:
        return False
    letters = []
    for char in word:
        if char not in LETTERS:
            return False
        letters.append(LETTERS.index(char))
    return bool(LEGAL[len(word), _code(tuple(letters))])


def action_for(cells: list[tuple[int, int, int]]) -> int:
    """Id of a placement. Each cell is ``(row, col, letter)`` with 0-based indexes.

    An empty list is pass. Raises ``KeyError`` when the cells are not one
    collinear placement from the action table.
    """
    if not cells:
        return PASS
    key = tuple(sorted((_square(row, col), letter) for row, col, letter in cells))
    return _index[key]


def action_name(action: int) -> str:
    """``pass`` or ``A(3,3)+B(3,4)`` using 1-based board coordinates."""
    count = int(N_TILES[action])
    if count == 0:
        return "pass"
    parts = []
    for slot in range(count):
        row, col = divmod(int(SQUARES[action, slot]), BOARD_SIZE)
        letter = LETTERS[int(LETTERS_AT[action, slot])]
        parts.append(f"{letter}({row + 1},{col + 1})")
    return "+".join(parts)


def parse_board(text: str) -> np.ndarray:
    """Parse five lines of five ``.`` or ``A``–``D`` cells into a length-25 board."""
    rows = text.strip().splitlines()
    if len(rows) != BOARD_SIZE:
        raise ValueError(f"board must have {BOARD_SIZE} rows")
    board = np.zeros(N_SQUARES, dtype=np.int64)
    for row, line in enumerate(rows):
        cells = line.split()
        if len(cells) != BOARD_SIZE:
            raise ValueError(f"row {row + 1} must have {BOARD_SIZE} cells")
        for col, cell in enumerate(cells):
            if cell == ".":
                continue
            if cell not in LETTERS:
                raise ValueError(f"unknown cell {cell!r}")
            board[_square(row, col)] = LETTERS.index(cell) + 1
    return board


def format_board(board: np.ndarray) -> str:
    """Five lines of five cells, ``.`` for empty."""
    lines = []
    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            value = int(board[_square(row, col)])
            cells.append("." if value == 0 else LETTERS[value - 1])
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _letter_at(
    board: np.ndarray,
    squares: np.ndarray,
    letters: np.ndarray,
    count: int,
    pos: int,
) -> int:
    """Letter 0–3 on ``pos`` after the placement, or -1 when the square is empty."""
    for slot in range(count):
        if int(squares[slot]) == pos:
            return int(letters[slot])
    value = int(board[pos])
    if value == 0:
        return -1
    return value - 1


def _run(
    board: np.ndarray,
    squares: np.ndarray,
    letters: np.ndarray,
    count: int,
    row: int,
    col: int,
    horizontal: bool,
) -> tuple[int, int]:
    """Length and base-4 code of the maximal run through ``(row, col)``."""
    if horizontal:
        start = col
        while start > 0 and _letter_at(board, squares, letters, count, _square(row, start - 1)) >= 0:
            start -= 1
        stop = col
        while stop < BOARD_SIZE - 1 and _letter_at(
            board, squares, letters, count, _square(row, stop + 1)
        ) >= 0:
            stop += 1
        code = 0
        for cursor in range(start, stop + 1):
            code = code * N_LETTERS + _letter_at(
                board, squares, letters, count, _square(row, cursor)
            )
        return stop - start + 1, code
    start = row
    while start > 0 and _letter_at(board, squares, letters, count, _square(start - 1, col)) >= 0:
        start -= 1
    stop = row
    while stop < BOARD_SIZE - 1 and _letter_at(
        board, squares, letters, count, _square(stop + 1, col)
    ) >= 0:
        stop += 1
    code = 0
    for cursor in range(start, stop + 1):
        code = code * N_LETTERS + _letter_at(board, squares, letters, count, _square(cursor, col))
    return stop - start + 1, code


def _score_run(length: int, code: int) -> int:
    """Points from one run, or -1 when that run is not a legal word."""
    if length <= 1:
        return 0
    if length > 4:
        return -1
    if not LEGAL[length, code]:
        return -1
    return length


def placement_score(board: np.ndarray, rack: np.ndarray, opened: bool, action: int) -> int:
    """Points scored by a placement, or -1 when it is illegal.

    Pass is not a placement. A legal placement scores at least 2: the sum of
    the lengths of the horizontal and vertical words it creates or extends.
    """
    count = int(N_TILES[action])
    squares = SQUARES[action]
    letters = LETTERS_AT[action]
    for letter in range(N_LETTERS):
        if int(rack[letter]) < int(COST[action, letter]):
            return -1
    for slot in range(count):
        if int(board[int(squares[slot])]) != 0:
            return -1
    for slot in range(int(N_GAPS[action])):
        if int(board[int(GAPS[action, slot])]) == 0:
            return -1
    if not opened:
        if count < 2:
            return -1
        covers_center = False
        for slot in range(count):
            if int(squares[slot]) == CENTER:
                covers_center = True
        if not covers_center:
            return -1
    else:
        touches = False
        for slot in range(count):
            pos = int(squares[slot])
            row, col = divmod(pos, BOARD_SIZE)
            if row > 0 and int(board[pos - BOARD_SIZE]) != 0:
                touches = True
            if row < BOARD_SIZE - 1 and int(board[pos + BOARD_SIZE]) != 0:
                touches = True
            if col > 0 and int(board[pos - 1]) != 0:
                touches = True
            if col < BOARD_SIZE - 1 and int(board[pos + 1]) != 0:
                touches = True
        if not touches:
            return -1

    origin = int(squares[0])
    row, col = divmod(origin, BOARD_SIZE)
    horizontal = int(AXIS[action]) == 0
    length, code = _run(board, squares, letters, count, row, col, horizontal)
    total = _score_run(length, code)
    if total < 0:
        return -1
    for slot in range(count):
        pos = int(squares[slot])
        cross_row, cross_col = divmod(pos, BOARD_SIZE)
        length, code = _run(board, squares, letters, count, cross_row, cross_col, not horizontal)
        gained = _score_run(length, code)
        if gained < 0:
            return -1
        total += gained
    if total == 0:
        return -1
    return total
