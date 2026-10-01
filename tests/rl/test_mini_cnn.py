"""The CNN policy scores actions from a board image and an action embedding."""

import mlx.core as mx
import numpy as np

from scrabble.games.mini_scrabble import GLOBAL_DIM, N_ACTIONS
from scrabble.rl.mini_cnn import ConvPolicy
from scrabble.rl.mini_policy import ILLEGAL_LOGIT


def test_illegal_actions_are_masked_and_legal_ones_are_scored() -> None:
    policy = ConvPolicy()
    board = mx.zeros((2, 5, 5, 5))
    board[..., 0] = 1.0
    globals_ = mx.zeros((2, GLOBAL_DIM))
    legal = mx.zeros((2, N_ACTIONS), dtype=mx.bool_)
    legal[:, 0] = True
    legal[0, 12] = True

    logits = np.array(policy(board, globals_, legal))
    allowed = np.array(legal)

    assert logits.shape == (2, N_ACTIONS)
    assert np.all(logits[~allowed] == ILLEGAL_LOGIT)
    assert logits[0, 0] != ILLEGAL_LOGIT
    assert logits[0, 12] != ILLEGAL_LOGIT
    assert logits[1, 12] == ILLEGAL_LOGIT
