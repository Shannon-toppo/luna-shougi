"""What score does an evaluation have to beat? Measure the engine on a holdout.

    python -m training.baseline --engine build-msvc/Release/luna-shougi.exe \
        --data data/holdout6.bin --sfen data/holdout6.sfen

train.py prints `val_resid` while it trains: how far the network's predicted
score sits from the label, in the engine's units. On its own that number says
nothing. It becomes a pass mark when the same measurement is taken on the
hand-written evaluation, because a network that predicts the teacher's own
labels worse than the teacher does has nothing to give the search -- and it
costs the search 3.3x to run. That is what the first generation was, and the
match that proved it took half an hour to find out what this finds in three
minutes. docs/nnue-gen1.md is the record.

**The pass mark belongs to a depth, not to the project.** The label is a
search score, so the deeper the datagen run that made the holdout, the further
the label sits from any static evaluation and the larger every residual here.
A number measured on depth-4 data means nothing to a run trained on depth-6
data. Re-measure whenever the depth changes.

--net measures a network instead, through the engine. Training's own
val_resid is computed on the float model in Python; this one goes through
quantization and the engine's integer arithmetic, so the two agreeing is a
check that nothing was lost on the way out. training/verify.py is the strict
form of that check; this is the one that says whether it mattered.

Both of those are residuals against a label, which is a spread, and a spread
is nearly blind to a constant offset between the model and the network: gen1
scored 604 here against 600 in training while the two sat 34.6 points apart.
training/drift.py measures that offset directly.

numpy only, no torch: the labels come out of the .bin and the scores come out
of the engine.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import dataset, verify


def spread(total: int, wanted: int) -> np.ndarray:
    """`wanted` indices spread over `total`, in order.

    Evenly spread rather than the first N: the front of a datagen file is the
    opening moves of the first games and nothing else.
    """
    if wanted >= total:
        return np.arange(total)
    return np.linspace(0, total - 1, wanted).astype(np.int64)


# Where a residual is worth measuring. The loss trains on win rates through
# sigmoid(score / 600), which is a deliberate statement that +1500 and +3000
# are the same position to play; a network trained that way compresses the top
# of its range on purpose. A single residual over every position is therefore
# mostly a measurement of the part the training was told to ignore -- on
# depth-6 data a third of the holdout sits past 1200 -- and it ranks a network
# below the evaluation it beats everywhere a game is still undecided.
BANDS = ((0, 300), (300, 600), (600, 1200), (1200, None))

# The band the pass mark is read from: close enough that the evaluation still
# decides the game, wide enough to be most of the holdout.
DECIDING = 600


def statistics(predicted: np.ndarray, label: np.ndarray) -> dict:
    error = predicted - label
    predicted_sd = float(predicted.std())
    label_sd = float(label.std())
    covariance = float(((predicted - predicted.mean()) * (label - label.mean())).mean())
    deciding = np.abs(label) < DECIDING
    return {
        "n": len(label),
        "corr": covariance / (predicted_sd * label_sd) if predicted_sd and label_sd else math.nan,
        "resid": float(error.std()),
        "resid_deciding": float(error[deciding].std()),
        "n_deciding": int(deciding.sum()),
        "bias": float(error.mean()),
        "predicted_sd": predicted_sd,
        "label_sd": label_sd,
        "bands": [
            (lo, hi, int(mask.sum()), float(error[mask].std()))
            for lo, hi in BANDS
            for mask in [(np.abs(label) >= lo) & (np.abs(label) < (hi if hi else np.inf))]
            if mask.any()
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.baseline", description=__doc__)
    parser.add_argument("--engine", required=True, help="path to luna-shougi")
    parser.add_argument("--data", required=True, help="the .bin holding the labels")
    parser.add_argument("--sfen", required=True,
                        help="the .sfen written beside it by --sfen-out, one line per sample")
    parser.add_argument("--net", help="measure this network instead of the hand-written evaluation")
    parser.add_argument("--positions", type=int, default=5000, help="how many to measure")
    args = parser.parse_args(argv)

    files = dataset.SampleFiles([args.data])
    sfens = verify.load_sfens(args.sfen, None)
    if len(sfens) != len(files):
        # They are written by the same loop over the same positions, so a
        # mismatch means they came from different runs. Comparing a score to
        # another position's label produces a plausible-looking number.
        print(
            f"{args.data} has {len(files):,} samples and {args.sfen} has {len(sfens):,} lines; "
            "they are not from the same datagen run",
            file=sys.stderr,
        )
        return 1

    index = spread(len(files), args.positions)
    samples = np.concatenate([files.read(int(i), 1) for i in index])

    # The label is the search score from the side to move, which is what the
    # engine's eval prints. The file stores it from black.
    sign = np.where(samples["side_to_move"] == 0, 1, -1)
    label = samples["score"].astype(np.int64) * sign

    engine = verify.Engine(args.engine)
    engine.handshake()
    if args.net:
        print(engine.load_network(args.net))
    predicted = np.empty(len(index), dtype=np.int64)
    kind = ""
    for at, i in enumerate(index):
        predicted[at], kind = engine.evaluate_either(sfens[int(i)])
    engine.close()

    result = statistics(predicted, label)
    print(f"{kind} evaluation on {result['n']:,} held-out positions from {args.data}")
    print(f"  corr with label   {result['corr']:+.3f}")
    print(f"  bias              {result['bias']:+.0f}")
    print(f"  spread            predicted {result['predicted_sd']:.0f}, label {result['label_sd']:.0f}")
    print("  resid sd by |label|:")
    for lo, hi, n, sd in result["bands"]:
        band = f"{lo}-{hi}" if hi else f"{lo}+"
        print(f"    {band:>10} {n:>6}  {sd:>6.0f}")
    print(f"  resid sd, |label| < {DECIDING}   {result['resid_deciding']:.0f}"
          f"   <- the floor to clear ({result['n_deciding']:,} positions)")
    print(f"  resid sd, everything      {result['resid']:.0f}"
          f"   (dominated by the band the loss ignores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
