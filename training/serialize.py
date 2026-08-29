"""Turn a training checkpoint into a .nnue file the engine can read.

    python -m training.serialize runs/net1/checkpoint.pt nets/gen1.nnue \
        --calibration data/gen1.bin

train.py --export does this automatically as it goes; this is for taking a
particular kept checkpoint, after a run has finished, and turning that one
into a network to measure.

--calibration is the run's own training data, and it is worth giving. Rounding
the hidden layers to int8 leaves each of them a fixed offset, which lands on
the engine's score as a constant of either sign and around thirty points;
quantize.correct_biases measures that offset on these positions and takes it
out of the integer biases. Without it the network still works, it is just
shifted by a number nobody chose. training/drift.py is what measures the
result.
"""

from __future__ import annotations

import argparse
import sys

import torch

from . import checkpoint, dataset, quantize


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.serialize", description=__doc__)
    parser.add_argument("checkpoint", help="a checkpoint.pt written by training.train")
    parser.add_argument("output", help="where to write the .nnue file")
    parser.add_argument("--calibration", nargs="+",
                        help="the .bin files the run trained on, for the bias correction")
    parser.add_argument("--calibration-samples", type=int, default=4096,
                        help="how many positions to measure the correction on")
    parser.add_argument("--scale-generation", type=int,
                        default=quantize.CURRENT_SCALE_GENERATION,
                        help="which quantization scales to write. 0 is what every checkpoint "
                             "trained before L1 was split onto its own scale was clipped "
                             "against; exporting one of those at the default clips its L1 "
                             "weights, and the warning below is what says so")
    args = parser.parse_args(argv)

    state = checkpoint.load(args.checkpoint)
    net = checkpoint.build_model(state)
    net.eval()

    with torch.no_grad():
        quantized, clipped = quantize.quantize(net, args.scale_generation)
    print(f"step {state.get('step', '?')} -> {args.output} "
          f"(scale generation {args.scale_generation}, L1 at {quantized.l1_weight_scale})")

    if args.calibration:
        batch = dataset.sample_evenly(args.calibration, args.calibration_samples)
        report = quantize.correct_biases(
            quantized, net, batch.black, batch.white, batch.side_to_move
        )
        print(
            f"bias correction on {len(batch):,} positions from "
            f"{', '.join(args.calibration)}: mean error {report['before']:+.1f} -> "
            f"{report['after']:+.1f} engine points"
        )
    else:
        print("no --calibration: the biases are uncorrected and the network is offset "
              "from the model it came from by a number nobody measured", file=sys.stderr)

    quantize.write(quantized, args.output)

    total = sum(clipped.values())
    if total:
        # Weight clipping during training should keep this at zero. That it
        # is not means the engine will compute something the trainer never
        # measured, so it is worth saying out loud rather than logging.
        print(f"warning: {total} weights did not fit the quantization: {clipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
