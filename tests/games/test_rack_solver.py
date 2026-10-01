"""Exact values for the Rack Scrabble dynamic program."""

import itertools
from fractions import Fraction
from functools import cache

import numpy as np
import pytest

from scrabble.games.rack_scrabble import (
    COPIES_PER_LETTER,
    COST,
    HORIZON,
    N_ACTIONS,
    N_LETTERS,
    RACK_SIZE,
    SCORE,
    RackScrabbleEnv,
    RackScrabbleSolver,
)


@pytest.fixture(scope="module")
def solver() -> RackScrabbleSolver:
    return RackScrabbleSolver()


def _racks():
    for counts in itertools.product(range(RACK_SIZE + 1), repeat=N_LETTERS):
        if sum(counts) <= RACK_SIZE:
            yield counts


def _count_positions() -> int:
    total = 0
    for rack in _racks():
        bags = 1
        for count in rack:
            bags *= COPIES_PER_LETTER - count + 1
        total += bags
    return total


@cache
def _empty_bag_best(rack: tuple[int, ...], turns_left: int) -> tuple[int | None, int]:
    """Independent oracle for positions whose bag is empty."""
    if turns_left <= 0 or sum(rack) == 0:
        return None, 0
    best_action = None
    best_value = -1
    for action in range(N_ACTIONS):
        cost = COST[action]
        if any(rack[letter] < int(cost[letter]) for letter in range(N_LETTERS)):
            continue
        left = tuple(rack[letter] - int(cost[letter]) for letter in range(N_LETTERS))
        _, future = _empty_bag_best(left, turns_left - 1)
        value = int(SCORE[action]) + future
        if best_action is None or value > best_value:
            best_action = action
            best_value = value
    return best_action, best_value


def _rollout(choose, rack, bag, turn) -> int:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state(rack, bag, turn=turn)
    total = 0
    while not env.terminated[0]:
        action = choose(env)
        _, rewards, _, _ = env.step(np.array([action]))
        total += int(rewards[0])
    return total


def test_table_covers_every_legal_position(solver: RackScrabbleSolver) -> None:
    assert solver.n_states == _count_positions() * HORIZON


def test_finished_positions_have_no_action(solver: RackScrabbleSolver) -> None:
    done = solver.decide([1, 1, 1, 0], [1, 1, 1, 1], turn=HORIZON)
    empty = solver.decide([0, 0, 0, 0], [4, 4, 4, 4], turn=2)
    assert done.action is None
    assert empty.action is None
    assert done.value == 0
    assert empty.value == 0
    assert done.q_values == (None,) * N_ACTIONS
    with pytest.raises(ValueError):
        solver.best_action([1, 0, 0, 0], [0, 0, 0, 0], turn=HORIZON)


def test_final_turn_plays_the_highest_immediate_score(solver: RackScrabbleSolver) -> None:
    for rack in _racks():
        if sum(rack) == 0:
            continue
        decision = solver.decide(rack, (0, 0, 0, 0), turn=HORIZON - 1)
        with_tiles = tuple(COPIES_PER_LETTER - count for count in rack)
        other = solver.decide(rack, with_tiles, turn=HORIZON - 1)
        action, value = _empty_bag_best(rack, 1)
        assert decision.action == other.action == action
        assert decision.value == other.value == value
        assert decision.q_values[action] == value


def test_empty_bag_matches_an_independent_search(solver: RackScrabbleSolver) -> None:
    for rack in _racks():
        for turn in range(HORIZON):
            decision = solver.decide(rack, (0, 0, 0, 0), turn=turn)
            action, value = _empty_bag_best(rack, HORIZON - turn)
            assert decision.action == action
            assert decision.value == value
            live = [q for q in decision.q_values if q is not None]
            if live:
                assert decision.value == max(live)


def test_one_draw_keeps_the_two_thirds_tile_weights(solver: RackScrabbleSolver) -> None:
    # Rack ABC, bag A A C, two turns left. Playing A draws one tile.
    # P(A)=2/3 leaves ABC, which scores 6 next. P(C)=1/3 leaves BCC, which scores 4.
    decision = solver.decide([1, 1, 1, 0], [2, 0, 1, 0], turn=HORIZON - 2)
    assert decision.q_values[0] == Fraction(19, 3)
    assert decision.q_values[1] == 5
    assert decision.q_values[2] == Fraction(17, 3)
    assert decision.q_values[12] == 10
    assert decision.action == 12
    assert decision.value == 10
    assert [index for index, q in enumerate(decision.q_values) if q is not None] == [
        0,
        1,
        2,
        12,
    ]


def test_holding_a_single_beats_cashing_the_pair(solver: RackScrabbleSolver) -> None:
    # Rack A A C, bag A A, three turns left.
    # AA scores 4 now and 9 in total. A or C scores 1 now and 14 in total.
    rack = [2, 0, 1, 0]
    bag = [2, 0, 0, 0]
    decision = solver.decide(rack, bag, turn=3)
    assert decision.q_values[0] == 14
    assert decision.q_values[2] == 14
    assert decision.q_values[4] == 9
    assert decision.action == 0
    assert decision.value == 14

    def optimal(env: RackScrabbleEnv) -> int:
        return solver.best_action(
            env.state[0, :4],
            env.state[0, 4:8],
            int(env.state[0, 8]),
        )

    def myopic(env: RackScrabbleEnv) -> int:
        chosen = 0
        chosen_score = -1
        for action in range(N_ACTIONS):
            if not env.mask[0, action]:
                continue
            score = int(SCORE[action])
            if score > chosen_score:
                chosen = action
                chosen_score = score
        return chosen

    assert _rollout(optimal, rack, bag, 3) == 14
    assert _rollout(myopic, rack, bag, 3) == 9


def test_decide_obs_reads_an_environment_row(solver: RackScrabbleSolver) -> None:
    obs = np.array([2, 0, 1, 0, 2, 0, 0, 0, 3], dtype=np.uint8)
    assert solver.decide_obs(obs) == solver.decide(obs[:4], obs[4:8], 3)
    assert solver.best_action(obs[:4], obs[4:8], np.uint8(3)) == 0


def test_decide_rejects_an_impossible_supply(solver: RackScrabbleSolver) -> None:
    with pytest.raises(ValueError):
        solver.decide([4, 0, 0, 0], [0, 0, 0, 0], turn=0)
    with pytest.raises(ValueError):
        solver.decide([1, 0, 0, 0], [4, 0, 0, 0], turn=0)
