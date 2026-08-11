"""A fixed set of unseen positions, measured during training.

The first generation of a net was trained without this. Its training loss fell
by a factor of a thousand and looked like a success; the net then lost 399 of
400 games. Everything that showed why had to be measured afterwards, by hand,
from outside. docs/nnue-gen1.md is the record.

Two numbers come out of here:

  - `val_loss`, the same loss train.py optimizes, on data the run has never
    trained on. Comparable to the training loss printed beside it.

  - `val_resid`, the standard deviation of (predicted score - label score) in
    the engine's own units. This one is comparable to something outside the
    training run: training/baseline.py puts the hand-written evaluation on the
    same holdout, and a network that cannot beat what it scores is not worth
    the 3.3x it costs the search to run. The pass mark belongs to a depth --
    473 on depth-4 data -- so measure it again whenever the datagen depth
    changes.

The holdout has to come from its own datagen run with a different --seed.
Splitting one file would put positions from the same game on both sides, and
consecutive positions in a game are nearly the same position: the split would
leak, and val_loss would follow the training loss down.

    luna-datagen --out data/holdout.bin --games 60 --depth 6 --seed 999

Nothing here touches any random number generator, and it runs under no_grad.
That is what keeps a validated run bit-identical to the same run without
validation -- and to itself after a stop and a resume.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from . import dataset, model as model_module


def _blocks(total: int, wanted: int, count: int = 16) -> list[tuple[int, int]]:
    """`count` evenly spread (start, length) ranges covering `wanted` samples.

    Contiguous, because reading is a seek per range and a stride would be a
    seek per sample. Spread, because the front of the file is the first games
    of the run and nothing else.
    """
    wanted = min(wanted, total)
    count = max(1, min(count, wanted))
    per = wanted // count
    if per == 0:
        return [(0, wanted)]
    stride = total // count
    out = []
    for i in range(count):
        start = min(i * stride, total - per)
        out.append((start, per))
    return out


class Holdout:
    """Unseen samples, converted once and kept on the device."""

    def __init__(
        self,
        paths: Sequence[str],
        batch_size: int,
        samples: int,
        to_device: Callable[[dataset.Batch, torch.device], tuple],
        device: torch.device,
    ):
        files = dataset.SampleFiles(paths)
        chunks = [files.read(start, length) for start, length in _blocks(len(files), samples)]
        raw = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

        self.count = len(raw)
        self.paths = list(files.paths)
        # Converted here rather than per measurement: the set never changes, so
        # the transfer is worth doing once. `to_device` is train.py's own, so
        # the holdout cannot drift from the training path in how it reads a
        # record.
        # Every sample is used, short last batch included. The training loader
        # drops its remainder because a step on a short batch is a step with a
        # different amount of gradient in it; here there is no step, the loss
        # is averaged by sample count, and dropping the tail would only mean
        # measuring on less. It also keeps a holdout smaller than one training
        # batch -- 60 games is about 7000 samples, the default batch is 8192 --
        # from being refused outright.
        self.batches = [
            to_device(dataset.to_batch(raw[at : min(at + batch_size, self.count)]), device)
            for at in range(0, self.count, batch_size)
        ]
        if not self.batches:
            raise ValueError(f"{', '.join(self.paths)}: no samples to measure on")

    @torch.no_grad()
    def measure(self, net: torch.nn.Module, lambda_score: float) -> dict:
        was_training = net.training
        net.eval()
        total_loss = 0.0
        total_n = 0
        square_sum = 0.0
        error_sum = 0.0
        for black, white, stm, score, result in self.batches:
            prediction = net(black, white, stm)
            loss = model_module.loss(prediction, score, result, stm, lambda_score)

            # The label is the search score from the side to move; the network
            # predicts it divided by the ponanza constant. Undo that and the
            # residual is in the engine's units, where a pawn is 90.
            sign = 1.0 - 2.0 * stm
            error = prediction * model_module.PONANZA_CONSTANT - score * sign

            n = len(stm)
            total_loss += float(loss) * n
            total_n += n
            error_sum += float(error.sum())
            square_sum += float((error * error).sum())
        if was_training:
            net.train()

        mean = error_sum / total_n
        variance = max(0.0, square_sum / total_n - mean * mean)
        return {
            "val_loss": total_loss / total_n,
            "val_resid": variance**0.5,
            "val_bias": mean,
            "val_samples": total_n,
        }
