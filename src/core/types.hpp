#pragma once

#include <array>
#include <cstdint>

namespace luna {

enum Color : int {
  kBlack = 0,
  kWhite = 1,
  kColorNb = 2,
};

constexpr Color Opponent(Color c) {
  return static_cast<Color>(c ^ 1);
}

// Promotion is encoded as bit 3, so PromotedType(pt) == pt | 8 for every piece
// that can actually promote (pawn .. rook). Gold and king have no promoted
// form, which is why they sit at the top of the unpromoted range.
enum PieceType : int {
  kNoPieceType = 0,
  kPawn = 1,
  kLance = 2,
  kKnight = 3,
  kSilver = 4,
  kBishop = 5,
  kRook = 6,
  kGold = 7,
  kKing = 8,
  kProPawn = 9,
  kProLance = 10,
  kProKnight = 11,
  kProSilver = 12,
  kHorse = 13,
  kDragon = 14,
  kPieceTypeNb = 15,
};

constexpr int kPromoteBit = 8;

constexpr bool CanPromoteType(PieceType pt) {
  return pt >= kPawn && pt <= kRook;
}

constexpr bool IsPromotedType(PieceType pt) {
  return pt >= kProPawn;
}

constexpr PieceType PromotedType(PieceType pt) {
  return static_cast<PieceType>(pt | kPromoteBit);
}

// Piece type as it goes to the hand when captured: promoted pieces revert.
constexpr PieceType RawType(PieceType pt) {
  return IsPromotedType(pt) ? static_cast<PieceType>(pt & ~kPromoteBit) : pt;
}

// Hand pieces in the order SFEN conventionally writes them.
inline constexpr std::array<PieceType, 7> kHandTypes = {kRook,   kBishop, kGold, kSilver,
                                                        kKnight, kLance,  kPawn};

// A piece is (color << 4) | piece_type, so black pieces are 1..14 and white
// pieces are 17..30.
enum Piece : int {
  kNoPiece = 0,
  kPieceNb = 32,
};

constexpr int kColorShift = 4;

constexpr Piece MakePiece(Color c, PieceType pt) {
  return static_cast<Piece>((c << kColorShift) | pt);
}

constexpr PieceType TypeOf(Piece p) {
  return static_cast<PieceType>(p & 15);
}

// Only meaningful when p != kNoPiece.
constexpr Color ColorOf(Piece p) {
  return static_cast<Color>(p >> kColorShift);
}

// Squares are numbered in SFEN reading order: 0 is 9a, 8 is 1a, 80 is 1i.
// Files run 1..9 from right to left, ranks run 1..9 ('a'..'i') from top.
using Square = int;

constexpr int kSquareNb = 81;
constexpr Square kSquareNone = -1;

constexpr Square MakeSquare(int file, int rank) {
  return (rank - 1) * 9 + (9 - file);
}

constexpr int FileOf(Square sq) {
  return 9 - (sq % 9);
}

constexpr int RankOf(Square sq) {
  return sq / 9 + 1;
}

// Column index (0..8) matching the storage layout; column 0 is file 9.
constexpr int ColumnOf(Square sq) {
  return sq % 9;
}

constexpr bool IsSquareValid(Square sq) {
  return sq >= 0 && sq < kSquareNb;
}

// Rank seen from `c`: 1 is the opponent's back rank, 9 is own back rank.
constexpr int RelativeRank(Color c, int rank) {
  return c == kBlack ? rank : 10 - rank;
}

constexpr int RelativeRankOf(Color c, Square sq) {
  return RelativeRank(c, RankOf(sq));
}

constexpr bool InPromotionZone(Color c, Square sq) {
  return RelativeRankOf(c, sq) <= 3;
}

// Directions are listed clockwise starting from black's forward direction, so
// Opposite(d) == (d + 4) & 7 and a 180-degree rotation of a direction bitmask
// is a 4-bit rotate.
enum Direction : int {
  kNorth = 0,
  kNorthEast = 1,
  kEast = 2,
  kSouthEast = 3,
  kSouth = 4,
  kSouthWest = 5,
  kWest = 6,
  kNorthWest = 7,
  kDirectionNb = 8,
};

constexpr Direction Opposite(Direction d) {
  return static_cast<Direction>((d + 4) & 7);
}

}  // namespace luna
