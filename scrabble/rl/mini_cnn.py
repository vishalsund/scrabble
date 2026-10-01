"""Train a CNN board encoder plus an MLP over the other features.

The board stays a 5×5 image. Two convolutions read it, and a separate MLP
reads the racks, the bag, the score difference, and the pass count. Those
two vectors are joined into one state vector. Each action has an embedding,
and the logit is the dot product of that embedding with the state vector.
Illegal actions are masked out.

The network runs on the MLX GPU. The training loop is the same REINFORCE run
against greedy as the flat MLP. This file does not train on import. Run it
with:

    python -m scrabble.rl.mini_cnn
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from scrabble.games.mini_scrabble import (
    GLOBAL_DIM,
    N_ACTIONS,
    N_SQUARES,
    MiniScrabbleEnv,
    encode_spatial,
)
from scrabble.rl.mini_policy import _ILLEGAL, train

_CHANNELS = 5


class ConvPolicy(nn.Module):
    """Score every action from a spatial board encoding and an action embedding.

    ``board`` is NHWC: batch, row, column, then empty/A/B/C/D.
    """

    def __init__(self, width: int = 64, channels: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(_CHANNELS, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.board_head = nn.Linear(channels * N_SQUARES, width)
        self.global_head = nn.Sequential(
            nn.Linear(GLOBAL_DIM, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.ReLU(),
        )
        self.action_embed = nn.Embedding(N_ACTIONS, width)
        self.action_embed.weight = mx.random.normal(
            shape=(N_ACTIONS, width), scale=width**-0.5
        )

    def __call__(self, board: mx.array, globals_: mx.array, legal: mx.array) -> mx.array:
        spatial = self.conv(board)
        spatial = mx.reshape(spatial, (spatial.shape[0], -1))
        state = self.combine(
            mx.concatenate([nn.relu(self.board_head(spatial)), self.global_head(globals_)], axis=-1)
        )
        logits = state @ self.action_embed.weight.T
        return mx.where(legal, logits, _ILLEGAL)

    def inputs(self, env: MiniScrabbleEnv, rows: np.ndarray) -> tuple[np.ndarray, ...]:
        board, globals_ = encode_spatial(env.state, env.score)
        board = np.ascontiguousarray(np.transpose(board[rows], (0, 2, 3, 1)))
        globals_ = np.ascontiguousarray(globals_[rows])
        legal = np.array(env.mask[rows])
        return board, globals_, legal


def main() -> None:
    width = 64
    channels = 32
    train(
        ConvPolicy(width, channels),
        "cnn-vs-greedy",
        {"width": width, "conv_channels": channels, "global_dim": GLOBAL_DIM},
    )


if __name__ == "__main__":
    main()
