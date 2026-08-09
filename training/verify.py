"""Check that the engine and the trainer agree, number for number.

    python -m training.verify --engine build-msvc/Release/luna-shougi.exe \
        --net nets/gen1.nnue --sfen data/gen1.sfen

The two sides of this project compute the same network twice: once in Python
with numpy, once in C++ with AVX2 or NEON. If they disagree, everything
measured during training was measured on a network that is not the one
playing, and neither side can tell on its own — a quantization bug does not
crash, it just makes the engine a bit worse than the trainer thought.

So: the same positions through both, and the scores have to be identical. Not
close. The arithmetic is integer on both sides and there is nothing in it that
could legitimately round differently.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import features, quantize

# Enough shapes of position to catch a feature that is only wrong for one kind
# of piece: an empty hand and a full one, promoted pieces, kings on their
# starting squares and kings that have run, and both sides to move.
DEFAULT_SFENS = [
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
    "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1",
    "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL b GR5pnsg 1",
    "8l/1l+R2P3/p2pBG1pp/kps1p4/Nn1P2G2/P1P1P2PP/1PS6/1KSG3+r1/LN2+p3L w Sbgn3p 124",
    "9/9/9/4k4/9/4K4/9/9/9 b 2R2B4G4S4N4L18P 1",
    "9/9/9/4k4/9/4K4/9/9/9 w 2r2b4g4s4n4l18p 1",
    "+B+B5nl/3k2g2/2n1pp1p1/p1pp2p1p/9/P1PPP1P1P/2N2PS2/3K1G3/L4G1NL b RGS4Prsl 1",
]


class Engine:
    """A USI engine on the other end of a pipe, driven one command at a time."""

    def __init__(self, path: str, extra_args: list[str] | None = None):
        # str(Path(...)) so that a path typed with forward slashes still finds
        # the executable on Windows.
        self.process = subprocess.Popen(
            [str(Path(path)), *(extra_args or [])],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
        )

    def send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def read_lines(self, until) -> list[str]:
        assert self.process.stdout is not None
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("the engine stopped answering")
            line = line.rstrip("\r\n")
            lines.append(line)
            if until(line):
                return lines

    def handshake(self) -> None:
        self.send("usi")
        self.read_lines(lambda line: line == "usiok")
        self.send("isready")
        self.read_lines(lambda line: line == "readyok")

    def load_network(self, path: str) -> str:
        self.send(f"setoption name EvalFile value {path}")
        # The engine answers a load with one info string; asking isready after
        # it gives a line to read up to whatever it said.
        self.send("isready")
        lines = self.read_lines(lambda line: line == "readyok")
        for line in lines:
            if "failed" in line:
                raise RuntimeError(line)
        return next((line for line in lines if line.startswith("info string eval")), "")

    def evaluate(self, sfen: str) -> tuple[int, str]:
        self.send(f"position sfen {sfen}")
        self.send("eval")
        line = self.read_lines(lambda line: line.startswith("info string eval"))[-1]
        tokens = line.split()
        if len(tokens) < 5 or tokens[3] != "nnue":
            raise RuntimeError(f"the engine is not using a network: {line}")
        simd = tokens[-1].strip("()")
        return int(tokens[4]), simd

    def close(self) -> None:
        try:
            self.send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def load_sfens(path: str | None, limit: int | None) -> list[str]:
    if path is None:
        return DEFAULT_SFENS
    lines = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return lines[:limit] if limit else lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.verify", description=__doc__)
    parser.add_argument("--engine", required=True, help="path to luna-shougi")
    parser.add_argument("--net", help="the .nnue file both sides should read")
    parser.add_argument("--random-net", metavar="PATH",
                        help="write an untrained network there and check that one instead. "
                             "Needs no trained net and no torch, which is what makes this "
                             "runnable in CI on every push.")
    parser.add_argument("--seed", type=int, default=0, help="seed for --random-net")
    parser.add_argument("--sfen", help="file of SFENs, one per line (luna-datagen --sfen-out)")
    parser.add_argument("--positions", type=int, default=2000, help="how many SFENs to check")
    parser.add_argument("--show", type=int, default=5, help="how many mismatches to print")
    args = parser.parse_args(argv)

    if not args.net and not args.random_net:
        parser.error("one of --net or --random-net is required")

    sfens = load_sfens(args.sfen, args.positions)
    print(f"{len(sfens)} positions")

    if args.random_net:
        # Written out and read back rather than used in memory, so that the
        # file format is part of what gets checked.
        quantize.write(quantize.random_net(args.seed), args.random_net)
        args.net = args.random_net
        print(f"checking an untrained network: {args.net}")
    net = quantize.read(args.net)
    engine = Engine(args.engine)
    mismatches = 0
    simd = "?"
    try:
        engine.handshake()
        print(engine.load_network(str(Path(args.net).resolve())))

        for index, sfen in enumerate(sfens):
            bona, king_black, king_white, side = features.parse_sfen(sfen)
            if len(bona) != features.ACTIVE_FEATURES:
                print(f"skipping a position with {len(bona)} pieces: {sfen}")
                continue
            black, white = features.halfkp_indices(
                bona[None, :], np.array([king_black]), np.array([king_white])
            )
            expected = int(quantize.evaluate(net, black, white, np.array([side]))[0])
            got, simd = engine.evaluate(sfen)
            if expected != got:
                mismatches += 1
                if mismatches <= args.show:
                    print(f"mismatch at {index}: python {expected}, engine {got}\n  {sfen}")
    finally:
        engine.close()

    print(f"engine kernels: {simd}")
    if mismatches:
        print(f"FAILED: {mismatches} of {len(sfens)} positions disagree")
        return 1
    print(f"OK: all {len(sfens)} positions agree exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
