"""The network, in floats. src/nnue/network.hpp is the same thing in integers.

Architecture: HalfKP -> 256 per perspective -> 512 -> 32 -> 32 -> 1.

Two things here exist only because of what happens after training:

  - the activation is a clipped ReLU on [0, 1], not a plain ReLU, because the
    engine stores activations as uint8 in [0, 127] and anything above that
    would be clamped there anyway. Training with the clamp means the network
    learns inside the range it will actually run in.

  - the hidden weights are clipped after every step. int8 at a scale of 64
    holds 127/64 and no more, so a weight that grows past that is a weight the
    engine cannot represent. Clipping during training costs a little accuracy;
    clipping afterwards costs whatever the network had learned to do with
    those weights.
"""

from __future__ import annotations

import torch
from torch import nn

from . import features

FT_DIM = 256
L1_OUT = 32
L2_OUT = 32

# What the engine divides by to turn the network's output into a score. The
# network learns a win rate, and 600 is the conventional constant relating a
# shogi score to one.
PONANZA_CONSTANT = 600.0

# int8 weights at a scale of 64 for the hidden layers, and at 9600/127 for the
# output layer.
HIDDEN_WEIGHT_LIMIT = 127.0 / 64.0
OUTPUT_WEIGHT_LIMIT = 127.0 / (9600.0 / 127.0)


class HalfKP(nn.Module):
    def __init__(self, ft_dim: int = FT_DIM, l1_out: int = L1_OUT, l2_out: int = L2_OUT):
        super().__init__()
        self.ft_dim = ft_dim
        # EmbeddingBag rather than a Linear over 125388 inputs: only 38 of
        # them are ever on, and summing 38 rows is the whole computation. A
        # dense matmul would be three thousand times the work for the same
        # answer.
        #
        # sparse=True matters as much as the EmbeddingBag itself. A batch
        # touches at most 38 rows per position; a dense gradient would build
        # and then update all 32 million of them on every step, which is a
        # 128MB write per step to change a few thousand numbers. It is why
        # train.py runs two optimizers.
        self.ft = nn.EmbeddingBag(features.FEATURE_DIMS, ft_dim, mode="sum", sparse=True)
        self.ft_bias = nn.Parameter(torch.zeros(ft_dim))
        self.l1 = nn.Linear(2 * ft_dim, l1_out)
        self.l2 = nn.Linear(l1_out, l2_out)
        self.out = nn.Linear(l2_out, 1)

        # Small, because the accumulator adds 38 of them together and the
        # clamp at 1.0 would otherwise saturate every unit from the first step.
        nn.init.normal_(self.ft.weight, std=0.01)

    def forward(self, black: torch.Tensor, white: torch.Tensor, stm: torch.Tensor) -> torch.Tensor:
        """black, white: (B, 38) int64 feature indices. stm: (B,) float, 1 for white."""
        black_half = self.ft(black) + self.ft_bias
        white_half = self.ft(white) + self.ft_bias

        # The side to move goes first, which is the only thing telling the
        # network whose turn it is. The engine concatenates the same way.
        stm = stm.unsqueeze(1)
        own = black_half * (1.0 - stm) + white_half * stm
        opponent = black_half * stm + white_half * (1.0 - stm)

        x = torch.cat([own, opponent], dim=1).clamp(0.0, 1.0)
        x = self.l1(x).clamp(0.0, 1.0)
        x = self.l2(x).clamp(0.0, 1.0)
        return self.out(x).squeeze(1)

    @torch.no_grad()
    def clip_weights(self) -> None:
        """Keep every weight inside what the quantized engine can hold."""
        self.l1.weight.clamp_(-HIDDEN_WEIGHT_LIMIT, HIDDEN_WEIGHT_LIMIT)
        self.l2.weight.clamp_(-HIDDEN_WEIGHT_LIMIT, HIDDEN_WEIGHT_LIMIT)
        self.out.weight.clamp_(-OUTPUT_WEIGHT_LIMIT, OUTPUT_WEIGHT_LIMIT)


def loss(
    prediction: torch.Tensor,
    score: torch.Tensor,
    result: torch.Tensor,
    stm: torch.Tensor,
    lambda_score: float = 0.7,
) -> torch.Tensor:
    """Squared error against a blend of the search score and the game result.

    Both targets are win rates rather than scores, which is what makes the
    blend meaningful: a score of +2000 and a score of +3000 are almost the
    same position to play, and training on the difference between them wastes
    the network's capacity on the part of the range where it does not matter.

    `lambda_score` weights the search score against the result. Leaning on the
    score is what makes the first generation of a net trainable at all — the
    result of one game is a very noisy label — and leaning on the result is
    what eventually lets a net become better than the search that taught it.
    """
    sign = 1.0 - 2.0 * stm  # scores and results are stored from black
    predicted_win = torch.sigmoid(prediction)
    score_win = torch.sigmoid(score * sign / PONANZA_CONSTANT)
    result_win = (result * sign + 1.0) / 2.0
    target = lambda_score * score_win + (1.0 - lambda_score) * result_win
    return ((predicted_win - target) ** 2).mean()
