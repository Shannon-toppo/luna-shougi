"""Turn a training checkpoint into a .nnue file the engine can read.

    python -m training.serialize runs/net1/checkpoint.pt nets/gen1.nnue

train.py --export does this automatically as it goes; this is for taking a
particular kept checkpoint, after a run has finished, and turning that one
into a network to measure.
"""

from __future__ import annotations

import argparse
import sys

import torch

from . import checkpoint, model as model_module, quantize


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.serialize", description=__doc__)
    parser.add_argument("checkpoint", help="a checkpoint.pt written by training.train")
    parser.add_argument("output", help="where to write the .nnue file")
    args = parser.parse_args(argv)

    state = checkpoint.load(args.checkpoint)
    net = model_module.HalfKP()
    net.load_state_dict(state["model"])
    net.eval()

    with torch.no_grad():
        quantized, clipped = quantize.quantize(net)
    quantize.write(quantized, args.output)

    total = sum(clipped.values())
    print(f"step {state.get('step', '?')} -> {args.output}")
    if total:
        # Weight clipping during training should keep this at zero. That it
        # is not means the engine will compute something the trainer never
        # measured, so it is worth saying out loud rather than logging.
        print(f"warning: {total} weights did not fit the quantization: {clipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
