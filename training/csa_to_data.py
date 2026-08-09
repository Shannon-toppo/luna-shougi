"""Turn floodgate CSA game records into the same .bin samples luna-datagen writes.

    python -m training.csa_to_data --out data/floodgate.bin --sfen-out data/floodgate.sfen \
        wdoor2024/*.csa

Why bother, when luna-datagen exists: the first network has to be trained on
something, and a network trained on games its own hand-written evaluation
played is a network that can learn that evaluation's mistakes and not much
else. Human and strong-engine games give it somewhere to start. After that,
self-play at depth is better data and this script stops being interesting.

What a CSA record does not have is a score for each position, only the result
of the game. The score field is written as 0, so files made here have to be
trained with --lambda-score 0, and must not be mixed into one run with
luna-datagen files, which do have scores. Two runs, or two stages, instead.
"""

from __future__ import annotations

import argparse
import glob
import struct
import sys
from pathlib import Path

from . import features

SQUARE_NB = 81

# CSA piece codes. The move records the piece as it stands *after* the move,
# so a promotion arrives as the promoted code and no promotion flag is needed.
_CSA_PIECES = {
    "FU": 1,   # 歩
    "KY": 2,   # 香
    "KE": 3,   # 桂
    "GI": 4,   # 銀
    "KA": 5,   # 角
    "HI": 6,   # 飛
    "KI": 7,   # 金
    "OU": 8,   # 玉
    "TO": 9,   # と
    "NY": 10,  # 成香
    "NK": 11,  # 成桂
    "NG": 12,  # 成銀
    "UM": 13,  # 馬
    "RY": 14,  # 龍
}

_KING = 8
_PROMOTE = 8

# Same order src/nnue/features.cpp builds hands in.
_HAND_ORDER = [6, 5, 7, 4, 3, 2, 1]

_BOARD_BASE_OF = {
    1: features.F_PAWN,
    2: features.F_LANCE,
    3: features.F_KNIGHT,
    4: features.F_SILVER,
    5: features.F_BISHOP,
    6: features.F_ROOK,
    7: features.F_GOLD,
    9: features.F_GOLD,
    10: features.F_GOLD,
    11: features.F_GOLD,
    12: features.F_GOLD,
    13: features.F_HORSE,
    14: features.F_DRAGON,
}

_HAND_BASE_OF = {
    1: (features.F_HAND_PAWN, features.E_HAND_PAWN),
    2: (features.F_HAND_LANCE, features.E_HAND_LANCE),
    3: (features.F_HAND_KNIGHT, features.E_HAND_KNIGHT),
    4: (features.F_HAND_SILVER, features.E_HAND_SILVER),
    5: (features.F_HAND_BISHOP, features.E_HAND_BISHOP),
    6: (features.F_HAND_ROOK, features.E_HAND_ROOK),
    7: (features.F_HAND_GOLD, features.E_HAND_GOLD),
}

_SFEN_LETTERS = {
    1: "P", 2: "L", 3: "N", 4: "S", 5: "B", 6: "R", 7: "G", 8: "K",
    9: "+P", 10: "+L", 11: "+N", 12: "+S", 13: "+B", 14: "+R",
}


def _square(file: int, rank: int) -> int:
    return (rank - 1) * 9 + (9 - file)


def _raw(piece_type: int) -> int:
    return piece_type & ~_PROMOTE if piece_type >= 9 else piece_type


class Board:
    """Just enough of a position to follow a game that is already known legal.

    No move generation and no legality checking: a CSA record says what was
    played, and re-deciding whether it was legal would only be a second
    opinion about a game that already happened.
    """

    def __init__(self):
        self.squares = [0] * SQUARE_NB  # (colour << 4) | piece type
        self.hands = {(c, pt): 0 for c in (0, 1) for pt in _HAND_ORDER}
        self.side = 0

    @classmethod
    def hirate(cls) -> "Board":
        board = cls()
        # 香桂銀金玉金銀桂香, file 9 through file 1.
        back = (2, 3, 4, 7, 8, 7, 4, 3, 2)
        for i, piece_type in enumerate(back):
            file = 9 - i
            board.squares[_square(file, 1)] = (1 << 4) | piece_type
            board.squares[_square(file, 9)] = piece_type
        board.squares[_square(8, 2)] = (1 << 4) | 6
        board.squares[_square(2, 2)] = (1 << 4) | 5
        board.squares[_square(8, 8)] = 5
        board.squares[_square(2, 8)] = 6
        for file in range(1, 10):
            board.squares[_square(file, 3)] = (1 << 4) | 1
            board.squares[_square(file, 7)] = 1
        return board

    def apply(self, colour: int, from_sq: int, to_sq: int, piece_type: int) -> None:
        captured = self.squares[to_sq]
        if captured:
            self.hands[(colour, _raw(captured & 15))] += 1
        if from_sq < 0:
            self.hands[(colour, piece_type)] -= 1
        else:
            self.squares[from_sq] = 0
        self.squares[to_sq] = (colour << 4) | piece_type
        self.side = 1 - colour

    def bona(self):
        """(list, black king, white king) in the engine's numbering and order."""
        pieces = []
        king = [None, None]
        for sq in range(SQUARE_NB):
            piece = self.squares[sq]
            if piece == 0:
                continue
            colour, piece_type = piece >> 4, piece & 15
            if piece_type == _KING:
                king[colour] = sq
                continue
            base = _BOARD_BASE_OF[piece_type]
            pieces.append(base + (0 if colour == 0 else SQUARE_NB) + sq)
        for colour in (0, 1):
            for piece_type in _HAND_ORDER:
                friendly, enemy = _HAND_BASE_OF[piece_type]
                base = friendly if colour == 0 else enemy
                for i in range(self.hands[(colour, piece_type)]):
                    pieces.append(base + i)
        return pieces, king[0], king[1]

    def sfen(self, move_number: int) -> str:
        rows = []
        for rank in range(1, 10):
            row = ""
            empty = 0
            for file in range(9, 0, -1):
                piece = self.squares[_square(file, rank)]
                if piece == 0:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                letters = _SFEN_LETTERS[piece & 15]
                row += letters if (piece >> 4) == 0 else letters.lower()
            if empty:
                row += str(empty)
            rows.append(row)

        hand = ""
        for colour in (0, 1):
            for piece_type in _HAND_ORDER:
                count = self.hands[(colour, piece_type)]
                if not count:
                    continue
                if count > 1:
                    hand += str(count)
                letter = _SFEN_LETTERS[piece_type]
                hand += letter if colour == 0 else letter.lower()
        return (
            "/".join(rows)
            + (" b " if self.side == 0 else " w ")
            + (hand if hand else "-")
            + f" {move_number}"
        )


# Results as CSA writes them, from the point of view of the side to move when
# the marker appears.
_LOSS_FOR_MOVER = {"%TORYO", "%TSUMI", "%ILLEGAL_MOVE", "%TIME_UP", "%+ILLEGAL_ACTION"}
_WIN_FOR_MOVER = {"%KACHI"}
_DRAW = {"%SENNICHITE", "%HIKIWAKE", "%JISHOGI", "%MAX_MOVES"}


def parse_game(text: str, want_sfen: bool = False):
    """(board states, result). Result is +1 black won, 0 draw, -1 white won.

    Returns (None, None) for a record this script will not use: a handicap
    game, an aborted one, or one whose starting position is spelled out with
    P1..P9 lines rather than PI.
    """
    board = None
    positions = []
    result = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("'", "N", "V", "$", "T")):
            continue

        if line.startswith("PI"):
            if len(line) > 2:
                return None, None  # a handicap game; different starting material
            board = Board.hirate()
            continue
        if line.startswith("P"):
            # An explicit board. Rare on floodgate and not worth a second
            # parser; skipping a few records costs nothing.
            return None, None
        if line in ("+", "-"):
            if board is None:
                board = Board.hirate()
            board.side = 0 if line == "+" else 1
            positions.append(
                (board.side, *board.bona(), board.sfen(len(positions) + 1) if want_sfen else None)
            )
            continue

        if line.startswith("%"):
            marker = line.split(",")[0]
            mover = board.side if board is not None else 0
            if marker in _LOSS_FOR_MOVER:
                result = -1 if mover == 0 else 1
            elif marker in _WIN_FOR_MOVER:
                result = 1 if mover == 0 else -1
            elif marker in _DRAW:
                result = 0
            break

        if line[0] in "+-" and len(line) >= 7:
            if board is None:
                return None, None
            colour = 0 if line[0] == "+" else 1
            try:
                from_file, from_rank = int(line[1]), int(line[2])
                to_file, to_rank = int(line[3]), int(line[4])
                piece_type = _CSA_PIECES[line[5:7]]
            except (ValueError, KeyError):
                return None, None
            from_sq = -1 if from_file == 0 else _square(from_file, from_rank)
            board.apply(colour, from_sq, _square(to_file, to_rank), piece_type)
            positions.append(
                (board.side, *board.bona(), board.sfen(len(positions) + 1) if want_sfen else None)
            )

    if result is None or board is None:
        return None, None
    return positions, result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="training.csa_to_data", description=__doc__)
    parser.add_argument("files", nargs="+", help=".csa files or globs")
    parser.add_argument("--out", required=True, help="where to write the samples")
    parser.add_argument("--sfen-out", help="also write the SFEN of every sample")
    parser.add_argument("--skip-plies", type=int, default=16,
                        help="opening moves to leave out; they are all the same few positions")
    parser.add_argument("--drop-draws", action="store_true",
                        help="leave out drawn games, whose label says nothing")
    args = parser.parse_args(argv)

    paths = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])

    out = open(args.out, "wb")
    sfen_out = open(args.sfen_out, "w", encoding="utf-8") if args.sfen_out else None
    games = 0
    skipped = 0
    written = 0

    try:
        for path in paths:
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
            except OSError as error:
                print(f"skipping {path}: {error}", file=sys.stderr)
                skipped += 1
                continue

            positions, result = parse_game(text, want_sfen=sfen_out is not None)
            if positions is None:
                skipped += 1
                continue
            if args.drop_draws and result == 0:
                skipped += 1
                continue
            games += 1

            for ply, (side, bona, king_black, king_white, sfen) in enumerate(positions):
                if ply < args.skip_plies:
                    continue
                if len(bona) != features.ACTIVE_FEATURES or king_black is None or king_white is None:
                    continue
                record = struct.pack(
                    f"<{features.ACTIVE_FEATURES}HBBBbhH",
                    *bona,
                    king_black,
                    king_white,
                    side,
                    result,
                    0,  # no score: a game record does not have one
                    min(ply + 1, 0xFFFF),
                )
                out.write(record)
                if sfen_out is not None:
                    sfen_out.write(sfen + "\n")
                written += 1
    finally:
        out.close()
        if sfen_out:
            sfen_out.close()

    print(f"{games} games used, {skipped} skipped, {written} positions written to {args.out}")
    if written:
        print("these samples have no score; train them with --lambda-score 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
