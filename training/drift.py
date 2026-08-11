"""How far the quantized network drifted from the float model it came from.

    python -m training.drift --checkpoint runs/gen1/checkpoint.pt \
        --net nets/gen1.nnue --sfen data/holdout.sfen

training/verify.py checks the quantized network against the quantized network:
Python's integer arithmetic against the engine's, and they have to agree
exactly. That is the right test for the arithmetic and it is blind to the
question here, because a quantization that is wrong in the same way on both
sides passes it. gen1 agreed exactly on 5000 positions while sitting 34.6
points below its own float model.

This is the other comparison: the float model against the integers, where the
answer cannot be exact and the useful test is on the mean.

**The mean is the number that fails this check, not the spread.** A spread of a
few tens of points is int8 doing its job -- the engine's score is a rounded
version of the model's and rounding is not free. A non-zero *mean* is a
different thing. The evaluation is from the side to move, so negamax flips a
constant's sign every ply, and a constant offset lands as a tempo bonus at
every node in the tree. It changes the moves chosen. See
quantize.correct_biases for where such a constant comes from and what takes it
out again.

The check runs against Python's integer forward pass by default, which needs
no engine and no build. That is only legitimate because verify.py pins that
pass to src/nnue/network.cpp; --engine does it end to end instead, and the two
should print the same numbers.

Held-out positions, please. quantize.correct_biases fits its correction on
training positions, and measuring a fitted mean where it was fitted says
nothing:

    luna-datagen --out data/holdout.bin --sfen-out data/holdout.sfen \
        --games 60 --depth 6 --seed 999
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from . import checkpoint, features, model as model_module, quantize, verify


def positions(sfens: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """HalfKP indices for the SFENs that hold a full complement of pieces."""
    bona, king_black, king_white, side, kept = [], [], [], [], []
    for sfen in sfens:
        pieces, black_king, white_king, stm = features.parse_sfen(sfen)
        if len(pieces) != features.ACTIVE_FEATURES:
            continue
        bona.append(pieces)
        king_black.append(black_king)
        king_white.append(white_king)
        side.append(stm)
        kept.append(sfen)
    if not kept:
        raise ValueError("none of the positions had all 38 pieces")
    black, white = features.halfkp_indices(
        np.stack(bona), np.array(king_black), np.array(king_white)
    )
    return black, white, np.array(side, dtype=np.int64), kept


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.drift", description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="the checkpoint.pt the net came from")
    parser.add_argument("--net", required=True, help="the .nnue file that will play")
    parser.add_argument("--sfen", required=True,
                        help="held-out SFENs, one per line (luna-datagen --sfen-out)")
    parser.add_argument("--positions", type=int, default=800, help="how many to measure")
    parser.add_argument("--engine", help="score through this engine instead of Python's "
                                         "integer forward pass, end to end")
    parser.add_argument("--max-mean", type=float, default=8.0,
                        help="fail if the mean drift exceeds this many engine points. "
                             "A pawn is 90, so the default is under a tenth of one")
    parser.add_argument("--show", type=int, default=3, help="how many worst positions to print")
    args = parser.parse_args(argv)

    sfens = verify.load_sfens(args.sfen, None)
    # Spread rather than the first N: the front of a datagen file is the
    # opening moves of the first games and nothing else.
    if args.positions and args.positions < len(sfens):
        sfens = [sfens[int(i)] for i in np.linspace(0, len(sfens) - 1, args.positions)]
    black, white, stm, sfens = positions(sfens)

    state = checkpoint.load(args.checkpoint)
    net = model_module.HalfKP()
    net.load_state_dict(state["model"])
    net.eval()
    with torch.no_grad():
        predicted = net(
            torch.from_numpy(black.astype(np.int64)),
            torch.from_numpy(white.astype(np.int64)),
            torch.from_numpy(stm.astype(np.float32)),
        ).numpy()
    # The model learns a score divided by the ponanza constant; the engine
    # prints the score. Undo that and the two are in the same units.
    float_score = model_module.PONANZA_CONSTANT * predicted.astype(np.float64)

    if args.engine:
        engine = verify.Engine(args.engine)
        try:
            engine.handshake()
            print(engine.load_network(args.net))
            quantized = np.array([engine.evaluate(sfen)[0] for sfen in sfens], dtype=np.float64)
        finally:
            engine.close()
        source = args.engine
    else:
        quantized = quantize.evaluate(
            quantize.read(args.net), black, white, stm
        ).astype(np.float64)
        source = "training.quantize (verify.py is what says this matches the engine)"

    drift = quantized - float_score
    mean = float(drift.mean())
    print(f"{args.net} against {args.checkpoint}, step {state.get('step', '?')}")
    print(f"  {len(sfens):,} positions from {args.sfen}, scored by {source}")
    print(f"  mean      {mean:+.1f}   <- the number that matters")
    print(f"  sd        {drift.std():.1f}")
    print(f"  mean|d|   {np.abs(drift).mean():.1f}")
    print(f"  max|d|    {np.abs(drift).max():.0f}")

    for index in np.argsort(-np.abs(drift))[: args.show]:
        print(f"  worst {drift[index]:+7.0f}  engine {quantized[index]:+6.0f}  "
              f"model {float_score[index]:+7.1f}  {sfens[index]}")

    if abs(mean) > args.max_mean:
        print(
            f"FAILED: the quantized network is offset from the model by {mean:+.1f} points, "
            f"over the {args.max_mean:.0f} allowed. This is a constant seen from the side to "
            "move, so the search reads it as a tempo term at every node. Export with "
            "--calibration so quantize.correct_biases can take it out."
        )
        return 1
    print(f"OK: within {args.max_mean:.0f} points of the float model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
