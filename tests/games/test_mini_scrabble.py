"""Rules and stepping for Mini-Scrabble v1."""

from itertools import product

import numpy as np
import pytest

from scrabble.games.mini_scrabble import (
    BAG,
    CENTER,
    COPIES_PER_LETTER,
    LETTERS,
    N_ACTIONS,
    N_LETTERS,
    N_SQUARES,
    OBS_SIZE,
    OPENED,
    PASS,
    PASSES,
    PLAYER,
    RACK,
    RACK_SIZE,
    MiniScrabbleEnv,
    action_for,
    is_legal_word,
    parse_board,
)
from scrabble.games.mini_scrabble.rules import (
    GAPS,
    LETTERS_AT,
    N_GAPS,
    N_TILES,
    SQUARES,
    placement_score,
)


def _act(*cells: tuple[str, int, int]) -> int:
    """Placement id. Each cell is ``(letter, row, col)`` in 1-based coordinates."""
    return action_for(
        [(row - 1, col - 1, LETTERS.index(letter)) for letter, row, col in cells]
    )


def _auto(board: np.ndarray, rack0: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Put leftover tiles on player 2's rack, then in the bag."""
    rack0_arr = np.asarray(rack0, dtype=np.int64)
    used = np.zeros(N_LETTERS, dtype=np.int64)
    for cell in board:
        if cell:
            used[int(cell) - 1] += 1
    remain = COPIES_PER_LETTER - used - rack0_arr
    assert np.all(remain >= 0)
    rack1 = np.zeros(N_LETTERS, dtype=np.int64)
    bag = remain.copy()
    room = RACK_SIZE
    for letter in range(N_LETTERS):
        take = min(int(bag[letter]), room)
        rack1[letter] = take
        bag[letter] -= take
        room -= take
    return board.astype(np.int64), np.stack([rack0_arr, rack1]), bag


def _dump(
    board: np.ndarray,
    blocked: set[int],
    rack0: list[int],
    rack1: list[int],
    bag: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Place whatever copies are still unused onto squares that are not blocked."""
    board = board.astype(np.int64).copy()
    rack0_arr = np.asarray(rack0, dtype=np.int64)
    rack1_arr = np.asarray(rack1, dtype=np.int64)
    bag_arr = np.asarray(bag, dtype=np.int64)
    used = rack0_arr + rack1_arr + bag_arr
    for cell in board:
        if cell:
            used[int(cell) - 1] += 1
    need = COPIES_PER_LETTER - used
    assert np.all(need >= 0)
    slots = [pos for pos in range(N_SQUARES) if board[pos] == 0 and pos not in blocked]
    assert int(need.sum()) <= len(slots)
    cursor = 0
    for letter, count in enumerate(need):
        for _ in range(int(count)):
            board[slots[cursor]] = letter + 1
            cursor += 1
    return board, np.stack([rack0_arr, rack1_arr]), bag_arr


def _oracle(env: MiniScrabbleEnv, index: int = 0) -> np.ndarray:
    board = env.state[index, :N_SQUARES].astype(np.int64)
    player = int(env.state[index, PLAYER])
    start = RACK + player * N_LETTERS
    rack = env.state[index, start : start + N_LETTERS].astype(np.int64)
    opened = bool(env.state[index, OPENED])
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    mask[PASS] = True
    for action in range(1, N_ACTIONS):
        mask[action] = placement_score(board, rack, opened, action) >= 0
    return mask


def _totals(env: MiniScrabbleEnv) -> np.ndarray:
    counts = np.zeros((env.n_envs, N_LETTERS), dtype=np.int64)
    board = env.state[:, :N_SQUARES]
    for letter in range(N_LETTERS):
        counts[:, letter] += np.count_nonzero(board == letter + 1, axis=1)
        counts[:, letter] += env.state[:, RACK + letter]
        counts[:, letter] += env.state[:, RACK + N_LETTERS + letter]
        counts[:, letter] += env.state[:, BAG + letter]
    return counts


@pytest.mark.parametrize(
    "word",
    ["AA", "AB", "DC", "AAA", "ABC", "DBA", "ABCD", "DCAB", "CADB"],
)
def test_dictionary_accepts_the_stated_words(word: str) -> None:
    assert is_legal_word(word)


@pytest.mark.parametrize("word", ["A", "AAB", "ABA", "BCC", "AABC", "ABCA", "ABCDA"])
def test_dictionary_rejects_the_stated_non_words(word: str) -> None:
    assert not is_legal_word(word)


def test_dictionary_has_68_words() -> None:
    words = [
        "".join(letters)
        for length in (2, 3, 4)
        for letters in product(LETTERS, repeat=length)
        if is_legal_word("".join(letters))
    ]
    assert len(words) == 68
    assert len({word for word in words if len(word) == 2}) == 16
    assert len({word for word in words if len(word) == 3}) == 28
    assert len({word for word in words if len(word) == 4}) == 24


def test_action_table_is_every_collinear_placement_plus_pass() -> None:
    assert N_ACTIONS == 1 + 100 + 1600 + 6400
    assert action_for([]) == PASS
    seen: set[tuple[tuple[int, int], ...]] = set()
    for action in range(N_ACTIONS):
        count = int(N_TILES[action])
        assert count <= RACK_SIZE
        if count == 0:
            assert action == PASS
            continue
        rows = set()
        cols = set()
        cells = []
        occupied = set()
        for slot in range(count):
            pos = int(SQUARES[action, slot])
            letter = int(LETTERS_AT[action, slot])
            row, col = divmod(pos, 5)
            rows.add(row)
            cols.add(col)
            cells.append((row, col, letter))
            occupied.add(pos)
        assert len(rows) == 1 or len(cols) == 1
        key = tuple(sorted((cell[0] * 5 + cell[1], cell[2]) for cell in cells))
        assert key not in seen
        seen.add(key)
        assert action_for(cells) == action
        for slot in range(int(N_GAPS[action])):
            assert int(GAPS[action, slot]) not in occupied


def test_reset_deals_both_racks_and_matches_a_repeated_seed() -> None:
    first = MiniScrabbleEnv(16, seed=1)
    second = MiniScrabbleEnv(16, seed=1)
    state, mask = first.reset()
    other, other_mask = second.reset()

    assert state.shape == (16, OBS_SIZE)
    assert mask.shape == (16, N_ACTIONS)
    assert np.all(state[:, :N_SQUARES] == 0)
    assert np.all(state[:, RACK : RACK + N_LETTERS].sum(axis=1) == RACK_SIZE)
    assert np.all(state[:, RACK + N_LETTERS : RACK + 2 * N_LETTERS].sum(axis=1) == RACK_SIZE)
    assert np.all(state[:, BAG : BAG + N_LETTERS].sum(axis=1) == 18)
    assert np.all(state[:, PASSES] == 0)
    assert np.all(state[:, OPENED] == 0)
    assert set(np.unique(state[:, PLAYER]).tolist()) == {0, 1}
    assert np.all(_totals(first) == COPIES_PER_LETTER)
    assert np.all(first.score == 0)
    assert not first.terminated.any()
    assert mask[:, PASS].all()
    np.testing.assert_array_equal(state, other)
    np.testing.assert_array_equal(mask, other_mask)
    np.testing.assert_array_equal(first.score, second.score)

    first.step(first.sample_legal_actions())
    first.reset(seed=1)
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_opening_move_must_cover_the_center_with_at_least_two_tiles() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board, racks, bag = _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0])
    env.set_state(board, racks, bag)
    assert int(env.state[0, OPENED]) == 0
    assert bool(env.mask[0, PASS])
    assert not bool(env.mask[0, _act(("A", 3, 3))])
    assert not bool(env.mask[0, _act(("A", 1, 1), ("B", 1, 2))])
    assert not bool(env.mask[0, _act(("A", 3, 1), ("A", 3, 3), ("B", 3, 5))])
    opening = _act(("A", 3, 2), ("B", 3, 3), ("C", 3, 4))
    assert bool(env.mask[0, opening])

    _, rewards, terminated, _ = env.step(np.array([opening]))
    assert int(rewards[0]) == 3
    assert int(env.score[0, 0]) == 3
    assert int(env.score[0, 1]) == 0
    assert not bool(terminated[0])
    assert int(env.state[0, OPENED]) == 1
    assert int(env.state[0, PLAYER]) == 1
    assert int(env.state[0, CENTER]) == LETTERS.index("B") + 1


def test_a_pass_before_the_opening_keeps_the_center_rule() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board, racks, bag = _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0])
    racks = racks.copy()
    racks[1] = [1, 1, 1, 0]
    bag = COPIES_PER_LETTER - racks[0] - racks[1]
    env.set_state(board, racks, bag)

    env.step(np.array([PASS]))
    assert int(env.state[0, PLAYER]) == 1
    assert int(env.state[0, PASSES]) == 1
    assert int(env.state[0, OPENED]) == 0
    assert not bool(env.terminated[0])
    assert not bool(env.mask[0, _act(("A", 3, 3))])
    opening = _act(("A", 3, 3), ("B", 3, 4))
    assert bool(env.mask[0, opening])
    _, rewards, _, _ = env.step(np.array([opening]))
    assert int(rewards[0]) == 2
    assert int(env.score[0, 1]) == 2
    assert int(env.state[0, PASSES]) == 0
    assert int(env.state[0, PLAYER]) == 0


def test_two_passes_end_the_game_and_a_placement_resets_the_count() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board, racks, bag = _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0])
    env.set_state(board, racks, bag, score=(5, 3))
    before = env.state.copy()

    env.step(np.array([PASS]))
    env.step(np.array([PASS]))
    assert bool(env.terminated[0])
    assert int(env.state[0, PASSES]) == 2
    assert int(env.state[0, PLAYER]) == 1
    assert int(env.rewards[0]) == 0
    np.testing.assert_array_equal(env.score[0], [5, 3])
    np.testing.assert_array_equal(env.state[0, :N_SQUARES], before[0, :N_SQUARES])
    np.testing.assert_array_equal(env.state[0, RACK:BAG], before[0, RACK:BAG])
    assert not env.mask[0].any()


def test_pass_does_not_refill() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board, racks, bag = _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 0, 0])
    env.set_state(board, racks, bag)
    rack = env.state[0, RACK : RACK + N_LETTERS].copy()
    bag_before = env.state[0, BAG : BAG + N_LETTERS].copy()

    env.step(np.array([PASS]))
    np.testing.assert_array_equal(env.state[0, RACK : RACK + N_LETTERS], rack)
    np.testing.assert_array_equal(env.state[0, BAG : BAG + N_LETTERS], bag_before)
    assert int(env.state[0, PASSES]) == 1


def test_extending_abc_scores_the_whole_word_and_a_repeat_is_illegal() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . . . . .
        A B C . .
        . . . . .
        . . . . .
        """
    )
    laid, racks, bag = _auto(board, [0, 0, 0, 1])
    env.set_state(laid, racks, bag, score=(4, 9))
    extend = _act(("D", 3, 4))
    assert bool(env.mask[0, extend])
    _, rewards, terminated, _ = env.step(np.array([extend]))
    assert int(rewards[0]) == 4
    np.testing.assert_array_equal(env.score[0], [8, 9])
    assert not bool(terminated[0])

    env.set_state(*_auto(board, [0, 0, 1, 0]), score=(4, 9))
    repeat = _act(("C", 3, 4))
    assert not bool(env.mask[0, repeat])
    before = env.state.copy()
    _, rewards, _, _ = env.step(np.array([repeat]))
    assert bool(env.invalid[0])
    assert int(rewards[0]) == 0
    np.testing.assert_array_equal(env.score[0], [4, 9])
    np.testing.assert_array_equal(env.state, before)


def test_aab_is_illegal_even_though_its_pairs_are_legal() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(*_auto(np.zeros(N_SQUARES, dtype=np.int64), [2, 1, 0, 0]))
    assert not bool(env.mask[0, _act(("A", 3, 2), ("A", 3, 3), ("B", 3, 4))])
    assert bool(env.mask[0, _act(("A", 3, 3), ("A", 3, 4))])
    assert bool(env.mask[0, _act(("A", 3, 3), ("B", 3, 4))])


def test_crossing_words_score_and_unchanged_words_do_not() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . . A B .
        . . C . .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [0, 0, 0, 1]), score=(10, 7))
    _, rewards, _, _ = env.step(np.array([_act(("D", 3, 4))]))
    assert int(rewards[0]) == 4
    np.testing.assert_array_equal(env.score[0], [14, 7])


def test_a_vertical_play_scores_its_main_word_and_both_crosses() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . A . . .
        . B . . .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [0, 0, 1, 1]))
    _, rewards, _, _ = env.step(np.array([_act(("C", 2, 3), ("D", 3, 3))]))
    assert int(rewards[0]) == 6


def test_an_isolated_direction_is_allowed_when_the_other_word_is_legal() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . A . . .
        . B . . .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [0, 0, 1, 0]))
    _, rewards, _, _ = env.step(np.array([_act(("C", 3, 3))]))
    assert int(rewards[0]) == 2
    assert not bool(env.invalid[0])


def test_existing_tiles_may_fill_a_gap_and_an_empty_gap_may_not() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . . . . .
        . B . . .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [1, 0, 1, 0]))
    bridged = _act(("A", 3, 1), ("C", 3, 3))
    assert bool(env.mask[0, bridged])
    _, rewards, _, _ = env.step(np.array([bridged]))
    assert int(rewards[0]) == 3

    env.set_state(*_auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 0, 1, 0]))
    assert not bool(env.mask[0, _act(("A", 3, 1), ("C", 3, 3))])


def test_diagonal_contact_and_overwrites_are_illegal() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . . . . .
        . . A . .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [0, 1, 0, 0]))
    assert not bool(env.mask[0, _act(("B", 4, 4))])
    assert not bool(env.mask[0, _act(("B", 3, 3))])
    assert bool(env.mask[0, _act(("B", 3, 4))])


def test_a_five_letter_run_is_illegal() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board = parse_board(
        """
        . . . . .
        . . . . .
        A B C D .
        . . . . .
        . . . . .
        """
    )
    env.set_state(*_auto(board, [1, 0, 0, 0]))
    assert not bool(env.mask[0, _act(("A", 3, 5))])


def test_mask_matches_the_written_rules() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    positions = [
        _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0]),
        _auto(
            parse_board(
                """
                . . . . .
                . . A B .
                . . C . .
                . . . . .
                . . . . .
                """
            ),
            [0, 0, 0, 1],
        ),
        _auto(
            parse_board(
                """
                . . . . .
                . . . . .
                . B . . .
                . . . . .
                . . . . .
                """
            ),
            [2, 0, 1, 0],
        ),
    ]
    for board, racks, bag in positions:
        env.set_state(board, racks, bag)
        np.testing.assert_array_equal(env.mask[0], _oracle(env))


def test_a_short_bag_is_drawn_until_it_is_empty() -> None:
    board = np.zeros(N_SQUARES, dtype=np.int64)
    board[0 * 5 + 2] = LETTERS.index("B") + 1
    board[1 * 5 + 2] = LETTERS.index("B") + 1
    board[2 * 5 + 1] = LETTERS.index("A") + 1
    blocked = {
        2 * 5 + 0,
        2 * 5 + 2,
        2 * 5 + 3,
        2 * 5 + 4,
        3 * 5 + 2,
        1 * 5 + 3,
        3 * 5 + 3,
        1 * 5 + 4,
        3 * 5 + 4,
    }
    laid, racks, bag = _dump(
        board,
        blocked,
        [0, 1, 1, 1],
        [1, 1, 1, 0],
        [1, 0, 0, 1],
    )
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(laid, racks, bag)
    action = _act(("B", 3, 3), ("C", 3, 4), ("D", 3, 5))
    assert bool(env.mask[0, action])
    _, rewards, terminated, _ = env.step(np.array([action]))

    assert int(rewards[0]) == 7
    assert not bool(terminated[0])
    assert int(env.state[0, PLAYER]) == 1
    np.testing.assert_array_equal(env.state[0, RACK : RACK + N_LETTERS], [1, 0, 0, 1])
    assert int(env.state[0, BAG : BAG + N_LETTERS].sum()) == 0
    assert int(env.state[0, RACK + N_LETTERS : RACK + 2 * N_LETTERS].sum()) == 3


def test_empty_bag_and_an_empty_rack_ends_without_penalizing_leftovers() -> None:
    board = np.zeros(N_SQUARES, dtype=np.int64)
    board[CENTER] = LETTERS.index("A") + 1
    blocked = {2 * 5 + 1, 2 * 5 + 3, 2 * 5 + 4, 1 * 5 + 3, 3 * 5 + 3}
    laid, racks, bag = _dump(board, blocked, [0, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0])
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(laid, racks, bag, score=(6, 4))
    _, rewards, terminated, _ = env.step(np.array([_act(("B", 3, 4))]))

    assert int(rewards[0]) == 2
    assert bool(terminated[0])
    assert int(env.state[0, PLAYER]) == 0
    np.testing.assert_array_equal(env.score[0], [8, 4])
    assert int(env.state[0, RACK : RACK + N_LETTERS].sum()) == 0
    assert int(env.state[0, RACK + N_LETTERS : RACK + 2 * N_LETTERS].sum()) == 3
    assert not env.mask[0].any()


def test_an_empty_bag_alone_does_not_end_the_game() -> None:
    board = np.zeros(N_SQUARES, dtype=np.int64)
    board[CENTER] = LETTERS.index("A") + 1
    blocked = {2 * 5 + 1, 2 * 5 + 3, 2 * 5 + 4, 1 * 5 + 3, 3 * 5 + 3}
    laid, racks, bag = _dump(board, blocked, [0, 1, 1, 1], [1, 1, 1, 0], [0, 0, 0, 0])
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(laid, racks, bag)
    _, rewards, terminated, _ = env.step(np.array([_act(("B", 3, 4))]))

    assert int(rewards[0]) == 2
    assert not bool(terminated[0])
    assert int(env.state[0, PLAYER]) == 1
    assert int(env.state[0, RACK : RACK + N_LETTERS].sum()) == 2
    assert int(env.state[0, BAG : BAG + N_LETTERS].sum()) == 0


def test_one_draw_from_two_a_and_one_c_tracks_tile_odds() -> None:
    board = np.zeros(N_SQUARES, dtype=np.int64)
    board[CENTER] = LETTERS.index("A") + 1
    blocked = {2 * 5 + 1, 2 * 5 + 3, 2 * 5 + 4, 1 * 5 + 3, 3 * 5 + 3}
    laid, racks, bag = _dump(board, blocked, [0, 1, 1, 1], [0, 0, 0, 0], [2, 0, 1, 0])
    n_envs = 2_000
    env = MiniScrabbleEnv(n_envs, seed=11)
    env.set_state(laid, racks, bag)
    env.step(np.full(n_envs, _act(("B", 3, 4)), dtype=np.int64))

    drew_a = env.state[:, RACK] == 1
    drew_c = env.state[:, RACK + 2] == 2
    assert np.all(env.state[:, RACK + 1] == 0)
    assert np.all(env.state[:, RACK + 3] == 1)
    assert int(drew_a.sum() + drew_c.sum()) == n_envs
    assert abs(float(drew_a.mean()) - 2 / 3) < 0.04


def test_illegal_action_does_not_change_the_position() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(*_auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0]), score=(5, 1))
    before = env.state.copy()
    illegal = _act(("D", 3, 3), ("D", 3, 4))

    _, rewards, _, _ = env.step(np.array([illegal]))
    assert bool(env.invalid[0])
    assert int(rewards[0]) == 0
    np.testing.assert_array_equal(env.score[0], [5, 1])
    np.testing.assert_array_equal(env.state, before)

    _, rewards, _, _ = env.step(np.array([N_ACTIONS]))
    assert bool(env.invalid[0])
    np.testing.assert_array_equal(env.state, before)

    _, rewards, _, _ = env.step(np.array([-1]))
    assert bool(env.invalid[0])
    np.testing.assert_array_equal(env.state, before)


def test_step_on_a_finished_game_is_a_noop() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    env.set_state(*_auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0]), passes=2, score=(2, 2))
    assert bool(env.terminated[0])
    before = env.state.copy()

    _, rewards, terminated, _ = env.step(np.array([PASS]))
    assert int(rewards[0]) == 0
    assert not bool(env.invalid[0])
    assert bool(terminated[0])
    np.testing.assert_array_equal(env.score[0], [2, 2])
    np.testing.assert_array_equal(env.state, before)


def test_there_is_no_turn_limit() -> None:
    env = MiniScrabbleEnv(1, seed=2)
    env.reset()
    for _ in range(8):
        legal = np.flatnonzero(env.mask[0])
        action = int(legal[legal > 0][0])
        env.step(np.array([action]))
        assert not bool(env.invalid[0])
        assert not bool(env.terminated[0])
    assert int(env.state[0, :N_SQUARES].astype(bool).sum()) >= 8


def test_same_seed_replays_the_same_games() -> None:
    def rollout(seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        env = MiniScrabbleEnv(4, seed=seed)
        env.reset()
        states = [env.state.copy()]
        scores = [env.score.copy()]
        for _ in range(12):
            env.step(env.sample_legal_actions().copy())
            states.append(env.state.copy())
            scores.append(env.score.copy())
        return states, scores

    again = rollout(123)
    repeated = rollout(123)
    other = rollout(456)
    for left, right in zip(again[0], repeated[0], strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(again[1], repeated[1], strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(again[0][0], other[0][0])


def test_random_games_conserve_tiles_and_end() -> None:
    n_envs = 8
    env = MiniScrabbleEnv(n_envs, seed=7)
    env.reset()
    gained = np.zeros((n_envs, 2), dtype=np.int64)

    for _ in range(80):
        assert np.all(_totals(env) == COPIES_PER_LETTER)
        if env.terminated.all():
            break
        active = ~env.terminated
        assert env.mask[active, PASS].all()
        actions = env.sample_legal_actions().copy()
        assert np.all(env.mask[np.arange(n_envs)[active], actions[active]])
        players = env.state[:, PLAYER].astype(np.int64).copy()
        before = env.score.copy()
        _, rewards, _, _ = env.step(actions)
        assert not env.invalid.any()
        for index in range(n_envs):
            if not active[index]:
                assert int(rewards[index]) == 0
                continue
            player = int(players[index])
            gained[index, player] += int(rewards[index])
            assert int(env.score[index, player]) == int(before[index, player]) + int(rewards[index])
            assert int(env.score[index, 1 - player]) == int(before[index, 1 - player])
            if not env.terminated[index] and int(actions[index]) != PASS:
                rack_n = int(env.state[index, RACK + player * N_LETTERS : RACK + (player + 1) * N_LETTERS].sum())
                bag_n = int(env.state[index, BAG : BAG + N_LETTERS].sum())
                assert rack_n == RACK_SIZE or bag_n == 0

    assert env.terminated.all()
    np.testing.assert_array_equal(env.score, gained)
    assert np.all(_totals(env) == COPIES_PER_LETTER)
    ended_by_passes = env.state[:, PASSES] >= 2
    bag_empty = env.state[:, BAG : BAG + N_LETTERS].sum(axis=1) == 0
    rack_empty = (env.state[:, RACK : RACK + N_LETTERS].sum(axis=1) == 0) | (
        env.state[:, RACK + N_LETTERS : RACK + 2 * N_LETTERS].sum(axis=1) == 0
    )
    assert np.all(ended_by_passes | (bag_empty & rack_empty))


def test_reset_done_redeals_only_finished_games() -> None:
    env = MiniScrabbleEnv(2, seed=3)
    env.reset()
    env.set_state(*_auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0]), passes=1, score=(4, 1), index=0)

    env.step(np.array([PASS, PASS]))
    assert bool(env.terminated[0])
    assert not bool(env.terminated[1])
    survivor = env.state[1].copy()
    survivor_score = env.score[1].copy()

    env.reset_done()
    assert not bool(env.terminated[0])
    assert int(env.state[0, OPENED]) == 0
    assert int(env.score[0, 0]) == 0
    assert int(env.state[0, RACK : RACK + N_LETTERS].sum()) == 3
    np.testing.assert_array_equal(env.state[1], survivor)
    np.testing.assert_array_equal(env.score[1], survivor_score)


def test_set_state_rejects_impossible_positions() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    board, racks, bag = _auto(np.zeros(N_SQUARES, dtype=np.int64), [1, 1, 1, 0])
    extra = bag.copy()
    extra[0] += 1
    with pytest.raises(ValueError):
        env.set_state(board, racks, extra)
    with pytest.raises(ValueError):
        env.set_state(board, np.array([[4, 0, 0, 0], [0, 0, 0, 0]]), bag)
    with pytest.raises(ValueError):
        env.set_state(board, racks, bag, player=2)
    with pytest.raises(ValueError):
        env.set_state(board, racks, bag, passes=3)


def test_step_rejects_a_bad_action_shape() -> None:
    env = MiniScrabbleEnv(1, seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([0, 0]))
