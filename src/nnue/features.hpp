#pragma once

#include <array>
#include <cstdint>

#include "core/move.hpp"
#include "core/position.hpp"
#include "core/types.hpp"

namespace luna::nnue {

// HalfKP, the feature set every shogi NNUE has been built on since Bonanza's
// KPP: one feature per (own king square, other piece). "Half" because the
// network sees the position twice, once from each side's king.
//
// A piece is identified by a BonaPiece, the Bonanza numbering for "this piece,
// in this place". Both kings are left out of it — a king is never captured and
// never sits in hand, and its square is already the K half of the feature — so
// a shogi position always has exactly 38 of them: 40 pieces minus the kings,
// each either on a square or in somebody's hand.
//
// The numbering is split into "f" (friend: the piece belongs to the side the
// features are being built for) and "e" (enemy) halves so that swapping
// perspective is a table lookup. Board pieces get 81 slots each; hand pieces
// get one slot per piece held, so a hand of three pawns lights up three
// consecutive features rather than one feature with a count.
//
// The four small promoted pieces share the gold slots, because that is what
// they are: a promoted silver moves exactly like a gold and the network has
// nothing to gain from telling them apart.
enum : int {
  kFHandPawn = 0,
  kEHandPawn = 19,
  kFHandLance = 38,
  kEHandLance = 43,
  kFHandKnight = 48,
  kEHandKnight = 53,
  kFHandSilver = 58,
  kEHandSilver = 63,
  kFHandGold = 68,
  kEHandGold = 73,
  kFHandBishop = 78,
  kEHandBishop = 81,
  kFHandRook = 84,
  kEHandRook = 87,
  kBonaHandEnd = 90,

  kFPawn = kBonaHandEnd,
  kEPawn = kFPawn + 81,
  kFLance = kEPawn + 81,
  kELance = kFLance + 81,
  kFKnight = kELance + 81,
  kEKnight = kFKnight + 81,
  kFSilver = kEKnight + 81,
  kESilver = kFSilver + 81,
  kFGold = kESilver + 81,
  kEGold = kFGold + 81,
  kFBishop = kEGold + 81,
  kEBishop = kFBishop + 81,
  kFHorse = kEBishop + 81,
  kEHorse = kFHorse + 81,
  kFRook = kEHorse + 81,
  kERook = kFRook + 81,
  kFDragon = kERook + 81,
  kEDragon = kFDragon + 81,
  kBonaEnd = kEDragon + 81,  // 1548
};

using BonaPiece = uint16_t;

// 81 king squares times every BonaPiece.
constexpr int kFeatureDimensions = kSquareNb * kBonaEnd;  // 125388

// 40 pieces less the two kings. A position that has been played rather than
// invented always has exactly this many.
constexpr int kActiveFeatures = 38;

// One more than kActiveFeatures so a test position holding a spare piece still
// fits instead of overrunning.
constexpr int kMaxActiveFeatures = 40;

// The BonaPieces of a position as black sees them, in no particular order.
struct BonaList {
  std::array<BonaPiece, kMaxActiveFeatures> pieces{};
  int size = 0;

  void Add(BonaPiece p) {
    if (size < kMaxActiveFeatures) pieces[size++] = p;
  }
};

// 180-degree rotation of the board, which is how white's perspective is
// built: square 0 (9a) and square 80 (1i) trade places.
constexpr Square RotateSquare(Square sq) {
  return kSquareNb - 1 - sq;
}

// BonaPiece of a piece of `c` standing on `sq`, as black sees it. `pt` must
// not be kKing.
BonaPiece BoardBona(Color c, PieceType pt, Square sq);

// BonaPiece of the `index`-th (0-based) piece of type `pt` in `c`'s hand, as
// black sees it. `pt` must be one of kHandTypes.
BonaPiece HandBona(Color c, PieceType pt, int index);

// The same BonaPiece as white sees it: the board rotates and f/e swap.
// InvertBona(InvertBona(p)) == p.
BonaPiece InvertBona(BonaPiece p);

// Every BonaPiece in `pos`, as black sees them. Kings are skipped.
BonaList BuildBonaList(const Position& pos);

// The king square that indexes `perspective`'s half of the features. Returns
// kSquareNone when that side has no king, which only happens in contrived
// positions; callers that can meet one must check.
inline Square PerspectiveKing(const Position& pos, Color perspective) {
  const Square king = pos.KingSquare(perspective);
  if (king == kSquareNone) return kSquareNone;
  return perspective == kBlack ? king : RotateSquare(king);
}

// The feature index a BonaPiece occupies in `perspective`'s half. `bona` must
// already be in that perspective's numbering, i.e. run through InvertBona for
// white.
constexpr int FeatureIndex(Square perspective_king, BonaPiece bona) {
  return perspective_king * kBonaEnd + bona;
}

// Which BonaPieces a move adds and removes, in black's numbering. At most two
// of each: the piece that moved (its square, and its type when it promotes)
// and, on a capture, the captured piece leaving the board and reappearing in
// the capturer's hand.
struct FeatureDelta {
  std::array<BonaPiece, 2> added{};
  int added_size = 0;
  std::array<BonaPiece, 2> removed{};
  int removed_size = 0;

  // The side whose king moved, so whose half has to be rebuilt from scratch:
  // every one of its features is indexed by a square that just changed.
  // kColorNb when no king moved.
  int king_moved = kColorNb;

  void Add(BonaPiece p) { added[added_size++] = p; }
  void Remove(BonaPiece p) { removed[removed_size++] = p; }
};

// What one move did to the feature set. Takes only the undo record, so a run
// of moves can be replayed without reconstructing the positions in between.
FeatureDelta ComputeDelta(const Position::Undo& undo);

}  // namespace luna::nnue
