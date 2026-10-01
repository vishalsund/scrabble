"""Rules and stepping for Rack Scrabble."""

import numpy as np
import pytest

from scrabble.games.rack_scrabble import (
    ACTION_NAMES,
    COPIES_PER_LETTER,
    COST,
    HORIZON,
    LETTERS,
    N_ACTIONS,
    N_LETTERS,
    RACK_SIZE,
    SCORE,
    RackScrabbleEnv,
)


def _legal(env: RackScrabbleEnv, index: int = 0) -> set[int]:
    return set(np.flatnonzero(env.mask[index]).tolist())


def test_action_table_matches_the_four_patterns() -> None:
    assert COST.shape == (N_ACTIONS, N_LETTERS)
    assert list(SCORE) == [1, 1, 1, 1, 4, 4, 4, 4, 12, 12, 12, 12, 6, 6, 6, 6]
    assert len(ACTION_NAMES) == N_ACTIONS
    for name, cost in zip(ACTION_NAMES, COST, strict=True):
        assert int(cost.sum()) == len(name)
        for letter, count in zip(LETTERS, cost, strict=True):
            assert int(count) == name.count(letter)
    for cost in COST:
        two_plus_one_other = int(cost.sum()) == 3 and int(cost.max()) == 2
        assert not two_plus_one_other


def test_reset_deals_three_tiles_and_matches_a_repeated_seed() -> None:
    first = RackScrabbleEnv(32, seed=1)
    second = RackScrabbleEnv(32, seed=1)
    obs, mask = first.reset()
    other, _ = second.reset()

    assert obs.shape == (32, 9)
    assert mask.shape == (32, N_ACTIONS)
    assert np.all(obs[:, :4].sum(axis=1) == 3)
    assert np.all(obs[:, 4:8].sum(axis=1) == 13)
    assert np.all(obs[:, :4] + obs[:, 4:8] == COPIES_PER_LETTER)
    assert np.all(obs[:, 8] == 0)
    assert not first.terminated.any()
    assert np.all(first.score == 0)
    np.testing.assert_array_equal(obs, other)
    for env_index in range(32):
        for action in range(N_ACTIONS):
            legal = bool(np.all(obs[env_index, :4] >= COST[action]))
            assert bool(mask[env_index, action]) is legal

    first.step(first.sample_legal_actions())
    first.reset(seed=1)
    np.testing.assert_array_equal(first.state, second.state)
    np.testing.assert_array_equal(first.mask, second.mask)


def test_rack_aab_allows_only_singles_and_the_pair() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([2, 1, 0, 0], [2, 3, 4, 4])
    assert _legal(env) == {0, 1, 4}


def test_rack_abc_allows_singles_and_three_distinct() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([1, 1, 1, 0], [3, 3, 3, 4])
    assert _legal(env) == {0, 1, 2, 12}


def test_rack_aaa_allows_single_pair_and_triple() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([3, 0, 0, 0], [1, 4, 4, 4])
    assert _legal(env) == {0, 4, 8}


def test_pair_scores_four_then_draws_the_only_tiles_left() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([2, 1, 0, 0], [0, 0, 2, 0])
    _, rewards, terminated, mask = env.step(np.array([4]))

    assert int(rewards[0]) == 4
    assert int(env.score[0]) == 4
    assert not bool(terminated[0])
    assert int(env.state[0, 8]) == 1
    np.testing.assert_array_equal(env.state[0, :4], [0, 1, 2, 0])
    np.testing.assert_array_equal(env.state[0, 4:8], [0, 0, 0, 0])
    assert _legal(env) == {1, 2, 6}
    assert set(np.flatnonzero(mask[0]).tolist()) == {1, 2, 6}


def test_triple_scores_twelve_and_draws_only_what_the_bag_has() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([3, 0, 0, 0], [1, 0, 0, 0])
    _, rewards, terminated, _ = env.step(np.array([8]))

    assert int(rewards[0]) == 12
    assert not bool(terminated[0])
    assert int(env.state[0, 8]) == 1
    np.testing.assert_array_equal(env.state[0, :4], [1, 0, 0, 0])
    np.testing.assert_array_equal(env.state[0, 4:8], [0, 0, 0, 0])
    assert _legal(env) == {0}

    _, rewards, terminated, _ = env.step(np.array([0]))
    assert int(rewards[0]) == 1
    assert int(env.score[0]) == 13
    assert bool(terminated[0])
    assert int(env.state[0, 8]) == 2
    assert int(env.state[0, :8].sum()) == 0


def test_three_distinct_on_the_final_turn_does_not_refill() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([1, 1, 1, 0], [1, 1, 1, 1], turn=5, score=10)
    _, rewards, terminated, mask = env.step(np.array([12]))

    assert int(rewards[0]) == 6
    assert int(env.score[0]) == 16
    assert bool(terminated[0])
    assert int(env.state[0, 8]) == HORIZON
    np.testing.assert_array_equal(env.state[0, :4], [0, 0, 0, 0])
    np.testing.assert_array_equal(env.state[0, 4:8], [1, 1, 1, 1])
    assert not mask[0].any()


def test_final_single_leaves_the_rest_of_the_rack_undrawn() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([1, 1, 0, 0], [2, 0, 0, 0], turn=5)
    _, rewards, terminated, _ = env.step(np.array([0]))

    assert int(rewards[0]) == 1
    assert bool(terminated[0])
    np.testing.assert_array_equal(env.state[0, :4], [0, 1, 0, 0])
    np.testing.assert_array_equal(env.state[0, 4:8], [2, 0, 0, 0])


def test_empty_rack_after_a_nonfinal_play_ends_the_game() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([1, 0, 0, 0], [0, 0, 0, 0])
    assert _legal(env) == {0}
    _, rewards, terminated, mask = env.step(np.array([0]))

    assert int(rewards[0]) == 1
    assert bool(terminated[0])
    assert int(env.state[0, 8]) == 1
    assert int(env.state[0, :8].sum()) == 0
    assert not mask[0].any()


def test_illegal_action_does_not_change_the_position() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([2, 1, 0, 0], [2, 2, 4, 4], turn=2, score=5)
    before = env.state.copy()

    _, rewards, _, _ = env.step(np.array([5]))
    assert bool(env.invalid[0])
    assert int(rewards[0]) == 0
    assert int(env.score[0]) == 5
    np.testing.assert_array_equal(env.state, before)

    _, rewards, _, _ = env.step(np.array([99]))
    assert bool(env.invalid[0])
    np.testing.assert_array_equal(env.state, before)

    _, rewards, _, _ = env.step(np.array([-1]))
    assert bool(env.invalid[0])
    np.testing.assert_array_equal(env.state, before)

    _, rewards, _, _ = env.step(np.array([4]))
    assert not bool(env.invalid[0])
    assert int(rewards[0]) == 4
    assert int(env.score[0]) == 9


def test_step_on_a_finished_game_is_a_noop() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.set_state([1, 0, 0, 0], [0, 0, 0, 0])
    env.step(np.array([0]))
    assert bool(env.terminated[0])
    before = env.state.copy()
    score = int(env.score[0])

    _, rewards, terminated, _ = env.step(np.array([0]))
    assert int(rewards[0]) == 0
    assert not bool(env.invalid[0])
    assert bool(terminated[0])
    assert int(env.score[0]) == score
    np.testing.assert_array_equal(env.state, before)


def test_one_draw_from_two_a_and_one_c_tracks_tile_odds() -> None:
    # A full rack opens one space when a single tile is played, so this is one draw.
    n_envs = 12_000
    env = RackScrabbleEnv(n_envs, seed=11)
    env.set_state([1, 1, 0, 1], [2, 0, 1, 0])
    env.step(np.zeros(n_envs, dtype=np.int64))

    drew_a = env.state[:, 0] == 1
    drew_c = env.state[:, 2] == 1
    assert np.all(env.state[:, 1] == 1)
    assert np.all(env.state[:, 3] == 1)
    assert int(drew_a.sum() + drew_c.sum()) == n_envs
    assert abs(float(drew_a.mean()) - 2 / 3) < 0.03


def test_same_seed_replays_the_same_games() -> None:
    def rollout(seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        env = RackScrabbleEnv(8, seed=seed)
        env.reset()
        states = [env.state.copy()]
        rewards = []
        for _ in range(HORIZON):
            actions = env.sample_legal_actions().copy()
            _, reward, _, _ = env.step(actions)
            states.append(env.state.copy())
            rewards.append(reward.copy())
        return states, rewards

    again = rollout(123)
    repeated = rollout(123)
    other = rollout(456)
    for left, right in zip(again[0], repeated[0], strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(again[1], repeated[1], strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(again[0][0], other[0][0])


def test_random_games_conserve_tiles_and_sum_to_the_score() -> None:
    n_envs = 64
    env = RackScrabbleEnv(n_envs, seed=7)
    env.reset()
    spent = np.zeros((n_envs, N_LETTERS), dtype=np.int64)
    total_reward = np.zeros(n_envs, dtype=np.int64)

    for _ in range(HORIZON):
        rack = env.state[:, :N_LETTERS].astype(np.int64)
        bag = env.state[:, N_LETTERS : N_LETTERS * 2].astype(np.int64)
        assert np.all(rack + bag + spent == COPIES_PER_LETTER)
        if env.terminated.all():
            break
        active = ~env.terminated
        actions = env.sample_legal_actions().copy()
        assert np.all(env.mask[np.arange(n_envs)[active], actions[active]])
        _, rewards, terminated, _ = env.step(actions)
        assert not env.invalid.any()
        ongoing = ~terminated
        rack_n = env.state[ongoing, :N_LETTERS].sum(axis=1)
        bag_n = env.state[ongoing, N_LETTERS : N_LETTERS * 2].sum(axis=1)
        assert np.all((rack_n == RACK_SIZE) | (bag_n == 0))
        for env_index in range(n_envs):
            if not active[env_index]:
                assert int(rewards[env_index]) == 0
                continue
            action = int(actions[env_index])
            spent[env_index] += COST[action]
            total_reward[env_index] += int(rewards[env_index])
            assert int(rewards[env_index]) == int(SCORE[action])

    assert env.terminated.all()
    assert np.all(env.state[:, 8] <= HORIZON)
    np.testing.assert_array_equal(env.score, total_reward)
    rack = env.state[:, :N_LETTERS].astype(np.int64)
    bag = env.state[:, N_LETTERS : N_LETTERS * 2].astype(np.int64)
    assert np.all(rack + bag + spent == COPIES_PER_LETTER)


def test_reset_done_redeals_only_finished_games() -> None:
    env = RackScrabbleEnv(2, seed=3)
    env.reset()
    env.set_state([1, 0, 0, 0], [1, 0, 0, 0], turn=5, score=4, index=0)
    env.set_state([1, 1, 1, 0], [3, 3, 3, 4], turn=0, score=2, index=1)

    env.step(np.array([0, 0]))
    assert bool(env.terminated[0])
    assert not bool(env.terminated[1])
    survivor = env.state[1].copy()
    survivor_score = int(env.score[1])

    env.reset_done()
    assert not env.terminated.any()
    assert int(env.state[0, 8]) == 0
    assert int(env.score[0]) == 0
    assert int(env.state[0, :4].sum()) == 3
    assert int(env.state[0, 4:8].sum()) == 13
    assert np.all(env.state[0, :4] + env.state[0, 4:8] == COPIES_PER_LETTER)
    np.testing.assert_array_equal(env.state[1], survivor)
    assert int(env.score[1]) == survivor_score


def test_set_state_rejects_impossible_counts() -> None:
    env = RackScrabbleEnv(1, seed=0)
    with pytest.raises(ValueError):
        env.set_state([4, 0, 0, 0], [0, 0, 0, 0])
    with pytest.raises(ValueError):
        env.set_state([1, 0, 0, 0], [4, 0, 0, 0])


def test_step_rejects_a_bad_action_shape() -> None:
    env = RackScrabbleEnv(1, seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([0, 0]))
