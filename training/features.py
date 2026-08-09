"""HalfKP feature indexing, kept in step with src/nnue/features.hpp.

Every constant here has a twin in the C++ header. They are not generated from
one another on purpose: two independent statements of the same numbering, and
a cross-check (training/verify.py) that fails loudly when they drift, is more
useful than one statement nobody ever reads.
"""

from __future__ import annotations

import numpy as np

SQUARE_NB = 81

# Hand slots. Each piece held lights up its own feature, so a hand of three
# pawns is three consecutive indices, not one index with a count.
F_HAND_PAWN = 0
E_HAND_PAWN = 19
F_HAND_LANCE = 38
E_HAND_LANCE = 43
F_HAND_KNIGHT = 48
E_HAND_KNIGHT = 53
F_HAND_SILVER = 58
E_HAND_SILVER = 63
F_HAND_GOLD = 68
E_HAND_GOLD = 73
F_HAND_BISHOP = 78
E_HAND_BISHOP = 81
F_HAND_ROOK = 84
E_HAND_ROOK = 87
BONA_HAND_END = 90

# Board slots, 81 apiece. The four small promoted pieces share the gold range.
F_PAWN = BONA_HAND_END
E_PAWN = F_PAWN + 81
F_LANCE = E_PAWN + 81
E_LANCE = F_LANCE + 81
F_KNIGHT = E_LANCE + 81
E_KNIGHT = F_KNIGHT + 81
F_SILVER = E_KNIGHT + 81
E_SILVER = F_SILVER + 81
F_GOLD = E_SILVER + 81
E_GOLD = F_GOLD + 81
F_BISHOP = E_GOLD + 81
E_BISHOP = F_BISHOP + 81
F_HORSE = E_BISHOP + 81
E_HORSE = F_HORSE + 81
F_ROOK = E_HORSE + 81
E_ROOK = F_ROOK + 81
F_DRAGON = E_ROOK + 81
E_DRAGON = F_DRAGON + 81
BONA_END = E_DRAGON + 81  # 1548

FEATURE_DIMS = SQUARE_NB * BONA_END  # 125388

# 40 pieces less the two kings, every one of them either on the board or in a
# hand. A played position always has exactly this many active features.
ACTIVE_FEATURES = 38

_HAND_RANGES = [
    (F_HAND_PAWN, E_HAND_PAWN),
    (F_HAND_LANCE, E_HAND_LANCE),
    (F_HAND_KNIGHT, E_HAND_KNIGHT),
    (F_HAND_SILVER, E_HAND_SILVER),
    (F_HAND_GOLD, E_HAND_GOLD),
    (F_HAND_BISHOP, E_HAND_BISHOP),
    (F_HAND_ROOK, E_HAND_ROOK),
]

_BOARD_BASES = [
    F_PAWN,
    F_LANCE,
    F_KNIGHT,
    F_SILVER,
    F_GOLD,
    F_BISHOP,
    F_HORSE,
    F_ROOK,
    F_DRAGON,
]


def _build_inverse() -> np.ndarray:
    """The same piece seen from the other side: colours swap, squares rotate."""
    table = np.zeros(BONA_END, dtype=np.int32)
    for friendly, enemy in _HAND_RANGES:
        width = enemy - friendly
        for i in range(width):
            table[friendly + i] = enemy + i
            table[enemy + i] = friendly + i
    for base in _BOARD_BASES:
        for sq in range(SQUARE_NB):
            rotated = SQUARE_NB - 1 - sq
            table[base + sq] = base + SQUARE_NB + rotated
            table[base + SQUARE_NB + sq] = base + rotated
    return table


INV_BONA = _build_inverse()


def rotate_square(sq):
    return SQUARE_NB - 1 - sq


def halfkp_indices(bona, king_black, king_white):
    """Feature indices for both halves of a batch.

    `bona` is (N, 38) in black's numbering, the kings are (N,). Returns two
    (N, 38) int32 arrays: black's half and white's half. White's is where the
    whole design pays off, because it is the same two array lookups rather
    than a second walk over the board.
    """
    bona = np.asarray(bona, dtype=np.int32)
    black = np.asarray(king_black, dtype=np.int32)[:, None] * BONA_END + bona
    white = (SQUARE_NB - 1 - np.asarray(king_white, dtype=np.int32))[:, None] * BONA_END + INV_BONA[
        bona
    ]
    return black, white


# --- SFEN, for the cross-check against the engine --------------------------
#
# The trainer never needs this: the samples arrive as BonaPieces. verify.py
# does, because it has to hand the engine a position and work out the same
# position's features here.

_PIECE_LETTERS = "PLNSBRGK"
_PAWN, _LANCE, _KNIGHT, _SILVER, _BISHOP, _ROOK, _GOLD, _KING = range(1, 9)
_PROMOTE = 8

_BOARD_BASE_OF = {
    _PAWN: F_PAWN,
    _LANCE: F_LANCE,
    _KNIGHT: F_KNIGHT,
    _SILVER: F_SILVER,
    _BISHOP: F_BISHOP,
    _ROOK: F_ROOK,
    _GOLD: F_GOLD,
    _PAWN | _PROMOTE: F_GOLD,
    _LANCE | _PROMOTE: F_GOLD,
    _KNIGHT | _PROMOTE: F_GOLD,
    _SILVER | _PROMOTE: F_GOLD,
    _BISHOP | _PROMOTE: F_HORSE,
    _ROOK | _PROMOTE: F_DRAGON,
}

_HAND_BASE_OF = {
    _PAWN: (F_HAND_PAWN, E_HAND_PAWN),
    _LANCE: (F_HAND_LANCE, E_HAND_LANCE),
    _KNIGHT: (F_HAND_KNIGHT, E_HAND_KNIGHT),
    _SILVER: (F_HAND_SILVER, E_HAND_SILVER),
    _GOLD: (F_HAND_GOLD, E_HAND_GOLD),
    _BISHOP: (F_HAND_BISHOP, E_HAND_BISHOP),
    _ROOK: (F_HAND_ROOK, E_HAND_ROOK),
}

# The order the engine writes hands in, which is also the order it builds the
# BonaPiece list in. Only the order matters for the comparison, not the set.
_HAND_ORDER = [_ROOK, _BISHOP, _GOLD, _SILVER, _KNIGHT, _LANCE, _PAWN]


class SfenError(ValueError):
    pass


def parse_sfen(sfen: str):
    """(bona list, black king square, white king square, side to move).

    The BonaPiece list comes out in the same order src/nnue/features.cpp
    builds it: board squares in SFEN reading order, then hands.
    """
    fields = sfen.split()
    if len(fields) < 2:
        raise SfenError(f"not an SFEN: {sfen!r}")
    board_field, side_field = fields[0], fields[1]
    hand_field = fields[2] if len(fields) > 2 else "-"

    board = [0] * SQUARE_NB  # (colour << 4) | piece type
    square = 0
    promoted = False
    for ch in board_field:
        if ch == "/":
            continue
        if ch == "+":
            promoted = True
            continue
        if ch.isdigit():
            square += int(ch)
            continue
        upper = ch.upper()
        if upper not in _PIECE_LETTERS:
            raise SfenError(f"unknown piece {ch!r} in {sfen!r}")
        piece_type = _PIECE_LETTERS.index(upper) + 1
        if promoted:
            piece_type |= _PROMOTE
            promoted = False
        colour = 0 if ch.isupper() else 1
        if square >= SQUARE_NB:
            raise SfenError(f"too many squares in {sfen!r}")
        board[square] = (colour << 4) | piece_type
        square += 1
    if square != SQUARE_NB:
        raise SfenError(f"board has {square} squares, not {SQUARE_NB}: {sfen!r}")

    hands = {(0, pt): 0 for pt in _HAND_ORDER}
    hands.update({(1, pt): 0 for pt in _HAND_ORDER})
    if hand_field != "-":
        count = 0
        for ch in hand_field:
            if ch.isdigit():
                count = count * 10 + int(ch)
                continue
            upper = ch.upper()
            piece_type = _PIECE_LETTERS.index(upper) + 1
            colour = 0 if ch.isupper() else 1
            hands[(colour, piece_type)] += count if count else 1
            count = 0

    bona = []
    king = [None, None]
    for sq in range(SQUARE_NB):
        piece = board[sq]
        if piece == 0:
            continue
        colour, piece_type = piece >> 4, piece & 15
        if piece_type == _KING:
            king[colour] = sq
            continue
        base = _BOARD_BASE_OF[piece_type]
        bona.append(base + (0 if colour == 0 else SQUARE_NB) + sq)
    for colour in (0, 1):
        for piece_type in _HAND_ORDER:
            friendly, enemy = _HAND_BASE_OF[piece_type]
            base = friendly if colour == 0 else enemy
            for i in range(hands[(colour, piece_type)]):
                bona.append(base + i)

    if king[0] is None or king[1] is None:
        raise SfenError(f"HalfKP needs both kings: {sfen!r}")
    side = 0 if side_field == "b" else 1
    return np.array(bona, dtype=np.int32), king[0], king[1], side
