"""State, moves, and transitions for Mini-Scrabble."""

import random

import numpy as np
import pytest

from scrabble.games.mini_scrabble import MiniScrabbleEnv, N_ACTIONS, PASS
from scrabble.games.mini_scrabble.play import (
    GLOBAL_DIM,
    INPUT_DIM,
    MOVES,
    PASS_MOVE,
    Move,
    State,
    apply_move,
    apply_move_without_draw,
    deal,
    draw_tiles,
    encode,
    encode_env,
    encode_spatial,
    from_env,
    legal_moves,
    result,
    score_placement,
)
from scrabble.games.mini_scrabble.rules import LETTERS


def _empty() -> tuple[str, ...]:
    return tuple("." for _ in range(25))


def _with(cells: dict[tuple[int, int], str]) -> tuple[str, ...]:
    board = list(_empty())
    for (row, col), letter in cells.items():
        board[row * 5 + col] = letter
    return tuple(board)


def test_moves_match_the_environment_action_table() -> None:
    assert len(MOVES) == N_ACTIONS
    assert MOVES[PASS] == PASS_MOVE
    assert PASS_MOVE.action_id() == PASS
    for action, move in enumerate(MOVES):
        assert move.action_id() == action
    flipped = Move(placements=((2, 3, "B"), (2, 2, "A")))
    assert flipped.placements == ((2, 2, "A"), (2, 3, "B"))


def test_score_placement_counts_created_words() -> None:
    empty = _empty()
    assert score_placement(empty, ((2, 2, "A"), (2, 3, "B"))) == 2
    board = _with({(1, 2): "A", (1, 3): "B", (2, 2): "C"})
    assert score_placement(board, ((2, 3, "D"),)) == 4
    extended = _with({(2, 0): "A", (2, 1): "B", (2, 2): "C"})
    assert score_placement(extended, ((2, 3, "D"),)) == 4
    try:
        score_placement(empty, ((2, 1, "A"), (2, 2, "A"), (2, 3, "B")))
    except ValueError as exc:
        assert "illegal" in str(exc)
    else:
        raise AssertionError("AAB should be illegal")


def test_legal_moves_match_the_environment_mask() -> None:
    env = MiniScrabbleEnv(1, seed=1)
    env.reset()
    state = from_env(env.state[0], env.score[0])
    moves = legal_moves(state.board, state.rack)
    ids = {move.action_id() for move in moves}
    assert ids == set(np.flatnonzero(env.mask[0]).tolist())
    assert PASS_MOVE in moves
    assert all(move.is_pass or len(move.placements) >= 2 for move in moves)
    one = Move(placements=((2, 2, state.rack[0]),))
    assert one not in moves


def test_afterstate_stops_before_the_draw_and_the_swap() -> None:
    state = deal(1)
    move = next(item for item in legal_moves(state.board, state.rack) if len(item.placements) == 2)
    after = apply_move_without_draw(state, move)
    assert after.bag == state.bag
    assert after.opponent_rack == state.opponent_rack
    assert after.passes == 0
    assert len(after.rack) == 1
    assert after.score_diff == score_placement(state.board, move)
    for row, col, letter in move.placements:
        assert after.board[row * 5 + col] == letter

    played = apply_move(state, move, 0)
    assert played.rack == state.opponent_rack
    assert len(played.opponent_rack) == 3
    assert sum(played.bag) == sum(state.bag) - 2
    assert played.score_diff == -after.score_diff
    assert result(played) is None


def test_draw_is_deterministic_and_without_replacement() -> None:
    first = draw_tiles((2, 0, 1, 0), 3, 11)
    assert draw_tiles((2, 0, 1, 0), 3, 11) == first
    assert first[0] == (0, 0, 0, 0)
    assert sorted(first[1]) == ["A", "A", "C"]

    rng = random.Random(4)
    bag = (3, 1, 0, 0)
    seen = []
    for _ in range(4):
        bag, tiles = draw_tiles(bag, 1, rng)
        seen.extend(tiles)
    assert sorted(seen) == ["A", "A", "A", "B"]
    assert bag == (0, 0, 0, 0)


def test_two_passes_draw_and_an_empty_rack_can_win() -> None:
    state = deal(2)
    once = apply_move(state, PASS_MOVE, 0)
    assert once.passes == 1
    assert once.rack == state.opponent_rack
    done = apply_move(once, PASS_MOVE, 0)
    assert done.passes == 2
    assert done.rack == once.rack
    assert result(done) == "draw"
    with pytest.raises(ValueError):
        apply_move(done, PASS_MOVE, 0)

    ending = State(
        _with({(2, 2): "A"}),
        ("B",),
        ("C", "C", "C"),
        (0, 0, 0, 0),
        3,
        0,
    )
    finished = apply_move(ending, Move(placements=((2, 3, "B"),)), 0)
    assert finished.rack == ()
    assert finished.bag == (0, 0, 0, 0)
    assert finished.score_diff == 5
    assert result(finished) == "win"


def test_encoding_matches_the_environment_view() -> None:
    env = MiniScrabbleEnv(4, seed=3)
    env.reset()
    batch = encode_env(env.state, env.score)
    assert batch.shape == (4, INPUT_DIM)
    for index in range(4):
        state = from_env(env.state[index], env.score[index])
        np.testing.assert_array_equal(encode(state), batch[index])
        assert sum(state.bag) + len(state.rack) + len(state.opponent_rack) == 24
        assert set(state.rack).issubset(set(LETTERS))
    board, globals_ = encode_spatial(env.state, env.score)
    assert board.shape == (4, 5, 5, 5)
    assert globals_.shape == (4, GLOBAL_DIM)
    flat_board = np.transpose(board, (0, 2, 3, 1)).reshape(4, -1)
    np.testing.assert_array_equal(flat_board, batch[:, : flat_board.shape[1]])
    np.testing.assert_array_equal(globals_, batch[:, flat_board.shape[1] :])
