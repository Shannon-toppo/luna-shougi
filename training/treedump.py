"""Collect the positions a search really evaluates, by playing out real
positions with the engine's EvalDump switched on.

    python -m training.treedump --out data/tree-gen2 --net nets/gen2.nnue \
        --openings data/holdout6.sfen --games 40 --moves 6 --depth 6

Why this is not another datagen holdout. A self-play file holds the positions
that were *reached*: one per move, quiet, never in check, and filtered again by
--score-limit. The search evaluates the positions it *considered*, which is
four or five orders of magnitude more of them and is made mostly of quiescence
stand-pats in the middle of exchanges. docs/nnue-gen2.md is the record of three
attempts to approximate the second set by reconstructing the first; all three
disagreed with what happened on the board.

The tree an engine walks depends on the evaluation driving it, so --net
matters. gen2's search and the hand-written search do not visit the same
positions, and the question worth asking is how gen2 does on the tree gen2
itself walks.

The engine is deterministic, so playing from the start position would give the
same game every time however many were asked for. Variety comes from the
opening positions instead: `--games` of them, spread evenly through
`--openings`, each played out for `--moves` moves.

Writes `<out>.tsv` (the dump, every column) and `<out>.sfen` (a subsample, one
SFEN per line) ready for:

    luna-datagen --out <out>.bin --sfen-out <out>.labelled.sfen \
        --label <out>.sfen --depth 6 --concurrency 16

and then training/baseline.py on that pair.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from .verify import Engine, load_sfens


# One evaluation in this many is written down. A depth-6 search takes about
# 17,000 static evaluations per move, so recording all of them would produce
# tens of millions of lines for a few dozen positions and nothing would be
# gained: the sample is what gets labelled, and it is drawn down to a few
# thousand at the end regardless.
DEFAULT_STRIDE = 64


def spread(total: int, wanted: int) -> list[int]:
    """`wanted` indices spread evenly over `total`, in order.

    The same reasoning as training/baseline.py: the front of a datagen file is
    the opening moves of its first games and nothing else.
    """
    if wanted >= total:
        return list(range(total))
    return [round(i * (total - 1) / (wanted - 1)) for i in range(wanted)] if wanted > 1 else [0]


def play_from(engine: Engine, sfen: str, depth: int, moves: int) -> int:
    """Plays `moves` moves out of `sfen`, returning how many it managed."""
    played: list[str] = []
    for _ in range(moves):
        suffix = " moves " + " ".join(played) if played else ""
        engine.send(f"position sfen {sfen}{suffix}")
        engine.send(f"go depth {depth}")
        # Read through to the bestmove before sending anything else: a command
        # arriving mid-search is read as a stop and cuts the tree short, which
        # still answers the go and quietly ruins the dump (docs/nnue-gen2.md,
        # trap 6).
        line = engine.read_lines(lambda text: text.startswith("bestmove"))[-1]
        best = line.split()[1]
        if best in ("resign", "win", "none"):
            break
        played.append(best)
    return len(played)


def summarise(tsv: Path) -> tuple[int, Counter]:
    sites: Counter = Counter()
    total = 0
    with tsv.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            sites[line.split("\t", 1)[0]] += 1
            total += 1
    return total, sites


def subsample(tsv: Path, out: Path, sites_out: Path, wanted: int, seed: int) -> int:
    """`wanted` SFENs drawn uniformly from the dump, in the order they appear.

    Uniform over records rather than over distinct positions on purpose: a
    position the search looked at forty times is forty times as much of the
    work the evaluation is being asked to get right.

    Writes the site of each one alongside, in step. luna-datagen --label keeps
    the order of its input, so the two stay lined up with the labels as long
    as it reports nothing skipped -- which for a dump it never should, every
    position in a search tree having all forty pieces on it.
    """
    rows = []
    with tsv.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) == 5:
                rows.append((fields[0], fields[4]))

    rng = random.Random(seed)
    keep = sorted(rng.sample(range(len(rows)), min(wanted, len(rows))))
    out.write_text("".join(rows[i][1] + "\n" for i in keep), encoding="utf-8")
    sites_out.write_text("".join(rows[i][0] + "\n" for i in keep), encoding="utf-8")
    return len(keep)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.treedump", description=__doc__)
    parser.add_argument("--engine", default="build-msvc/Release/luna-shougi.exe")
    parser.add_argument("--out", required=True, help="path prefix for .tsv and .sfen")
    parser.add_argument("--net", help="play with this network instead of the hand-written eval")
    parser.add_argument("--openings", default="data/holdout6.sfen",
                        help="positions to play out of, one SFEN per line")
    parser.add_argument("--games", type=int, default=40, help="how many of them to use")
    parser.add_argument("--moves", type=int, default=6, help="moves to play from each")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help=f"record one evaluation in every N (default {DEFAULT_STRIDE})")
    parser.add_argument("--positions", type=int, default=5000,
                        help="how many SFENs to subsample for labelling")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    openings = load_sfens(args.openings, None)
    chosen = [openings[i] for i in spread(len(openings), args.games)]
    print(f"{len(chosen)} openings out of {len(openings):,} in {args.openings}")

    tsv = Path(str(args.out) + ".tsv")
    sfen = Path(str(args.out) + ".sfen")
    sites = Path(str(args.out) + ".sites")
    tsv.parent.mkdir(parents=True, exist_ok=True)

    engine = Engine(args.engine)
    try:
        engine.handshake()
        if args.net:
            # Checked rather than assumed: a failed load leaves the
            # hand-written evaluation in place, and the dump would then be of
            # a tree nobody asked about (docs/nnue-gen1.md, trap 1).
            loaded = engine.load_network(str(Path(args.net).resolve()))
            print(loaded)
            if "nnue" not in loaded:
                raise RuntimeError(f"the network did not load: {loaded}")
        else:
            print("hand-written evaluation")

        engine.send(f"setoption name EvalDumpEvery value {args.stride}")
        engine.send(f"setoption name EvalDump value {tsv.resolve()}")
        engine.send("isready")
        for line in engine.read_lines(lambda text: text == "readyok"):
            if "failed" in line:
                raise RuntimeError(line)
            if line.startswith("info string eval dump"):
                print(line)

        for index, opening in enumerate(chosen, start=1):
            engine.send("usinewgame")
            played = play_from(engine, opening, args.depth, args.moves)
            if index % 10 == 0 or index == len(chosen):
                print(f"  {index}/{len(chosen)} openings, {played} moves from the last")

        engine.send("setoption name EvalDump value ")
        engine.send("isready")
        for line in engine.read_lines(lambda text: text == "readyok"):
            if line.startswith("info string eval dump off"):
                print(line)
    finally:
        engine.close()

    total, sites = summarise(tsv)
    if total == 0:
        print(f"nothing was recorded in {tsv}", file=sys.stderr)
        return 1
    print(f"\n{total:,} records in {tsv}  (one in {args.stride} of the evaluations taken)")
    for site, count in sites.most_common():
        print(f"  {site:>10} {count:>9,}  {100 * count / total:5.1f}%")

    kept = subsample(tsv, sfen, sites, args.positions, args.seed)
    print(f"\n{kept:,} sampled into {sfen}, sites into {sites}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
