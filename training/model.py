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

# Engine points per unit of the network's output. The network learns a win
# rate and this is the constant relating a shogi score to one: the target is
# sigmoid(score / PONANZA_CONSTANT).
#
# 600, the conventional value, and raising it has been tried and lost.
#
# The case for raising it was good on paper. At 600 a score of 3000 is a win
# rate of 0.9933 and 8000 is 0.9999985, so nothing asks the network to tell
# them apart and it learns to stop at about 2600 -- while the search spends
# 37% of its static evaluations past 3000. Raising the constant does fix that:
# at 1800 the median output where the label is past 3000 goes from 937 to 1829,
# against the hand-written evaluation's 3837.
#
# It also costs resolution at the other end, where a score of 600 falls from a
# win rate of 0.731 to 0.583, and the residual where |label| < 600 goes from
# 708 to 1242. Three matches of 400 games, same data and same steps, one
# constant apart, said that side of the trade is the one that matters:
#
#     C 600 vs C1200   262-136-2   +113 Elo   LOS 100%
#     C1200 vs C1800   252-141-7    +99 Elo   LOS 100%
#     C 600 vs C1800   273-127-0   +133 Elo   LOS 100%
#
# Monotone, and every one of them the opposite of what the residual in the
# tail suggested. Do not raise this again without a match to go with it.
# docs/nnue-search-distribution.md has the whole measurement.
#
# Nothing in the engine reads this. quantize.py folds it into the output
# layer, so networks built at different values still play each other, which is
# what made those three matches possible on one binary.
PONANZA_CONSTANT = 600.0

# int8 weights at a scale of 64 for the hidden layers, and at
# PONANZA_CONSTANT * 16 / 127 for the output layer. Derived rather than
# written out, so that changing the constant cannot leave this behind: a stale
# limit here clips the output weights to the wrong range and the model quietly
# trains against a ceiling the engine does not have.
HIDDEN_WEIGHT_LIMIT = 127.0 / 64.0
OUTPUT_WEIGHT_LIMIT = 127.0 / (PONANZA_CONSTANT * 16.0 / 127.0)


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


# Where the score term stops being quadratic, in the network's own output
# units (one unit is PONANZA_CONSTANT engine points). 1.0 puts the knee at a
# whole constant's worth of error, so the deciding band -- where the labels are
# a fraction of a unit and the win-rate term is already doing the work -- stays
# quadratic, and the tail, where the labels run to 25 units, is linear and
# cannot dominate the gradient.
SCORE_HUBER_DELTA = 1.0

# Labels past this are not scores, they are mate announcements, and asking a
# static evaluation to regress +32000 teaches it nothing. luna-datagen already
# drops them, but a file labelled from a search dump has plenty, so the term
# clamps rather than trusting its input.
SCORE_CLAMP = 16000.0


def loss(
    prediction: torch.Tensor,
    score: torch.Tensor,
    result: torch.Tensor,
    stm: torch.Tensor,
    lambda_score: float = 0.7,
    score_weight: float = 0.0,
) -> torch.Tensor:
    """Squared error against a blend of the search score and the game result,
    optionally plus a term on the score itself.

    Both targets are win rates rather than scores, which is what makes the
    blend meaningful: a score of +2000 and a score of +3000 are almost the
    same position to play, and training on the difference between them wastes
    the network's capacity on the part of the range where it does not matter.

    `lambda_score` weights the search score against the result. Leaning on the
    score is what makes the first generation of a net trainable at all — the
    result of one game is a very noisy label — and leaning on the result is
    what eventually lets a net become better than the search that taught it.
    Samples marked dataset.RESULT_UNKNOWN have no game to weigh, and are
    trained at an effective lambda of 1 whatever this says.

    `score_weight` adds a Huber term on the raw output against the raw score.
    It defaults to 0, which is exactly the loss the first three generations
    were trained with, down to the arithmetic.

    Why the extra term exists. "+2000 and +3000 are the same position" is true
    of the move to play and false of the number the search needs: alpha-beta
    compares evaluations against margins, and a sigmoid that has saturated
    returns nearly the same number for positions hundreds of points apart. The
    third generation stopped at an output of 3335 where the hand-written
    evaluation reaches 10228, because sigmoid(3335/600) is 0.9959 and there is
    no gradient past it — and 37% of what the search evaluates lives out there.

    The other way to reach that range is to raise PONANZA_CONSTANT, which
    stretches the sigmoid at both ends at once. It was tried and it lost three
    matches out of three, because what it gives up is the deciding band.

    This term is not free either -- it costs the deciding band too -- but it
    buys far more with it. On the same 391k slice, 30k steps, measured on the
    tree the search walks:

        w=0      tail  936   deciding  711
        w=0.003  tail 1127   deciding  816
        w=0.01   tail 1968   deciding  989
        w=0.03   tail 2438   deciding 1087
        C=1200   tail 1514   deciding  997     <- the constant, for comparison
        C=1800   tail 1829   deciding 1242
        hand     tail 3837   deciding  653

    At the same cost in the deciding band, w=0.01 reaches further into the tail
    than C=1200 does (1968 against 1514) and predicts the label better overall
    (residual 1984 against 2223). Which weight is actually worth playing is a
    question for a match, not for this table; C=1200 looked reasonable here too.
    """
    sign = 1.0 - 2.0 * stm  # scores and results are stored from black
    stm_score = score * sign

    predicted_win = torch.sigmoid(prediction)
    score_win = torch.sigmoid(stm_score / PONANZA_CONSTANT)
    result_win = (result * sign + 1.0) / 2.0

    # A sample carrying dataset.RESULT_UNKNOWN came out of a search tree and
    # has no game around it. Blending a result into its target would teach the
    # network that every position the search ever looked at was a draw, which
    # for a won position is a lie of half a win rate. Those samples are trained
    # on the score alone; the rest are unchanged, so a file without the
    # sentinel gives the same arithmetic this function had before.
    blend = torch.where(result.abs() > 1.0, torch.ones_like(result), result.new_full((), lambda_score))
    target = blend * score_win + (1.0 - blend) * result_win
    total = ((predicted_win - target) ** 2).mean()

    if score_weight:
        # Both sides in output units, so the weight is a pure ratio and does
        # not have to absorb a factor of PONANZA_CONSTANT.
        wanted = stm_score.clamp(-SCORE_CLAMP, SCORE_CLAMP) / PONANZA_CONSTANT
        total = total + score_weight * torch.nn.functional.huber_loss(
            prediction, wanted, delta=SCORE_HUBER_DELTA
        )
    return total
