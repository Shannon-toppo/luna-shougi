#include "nnue/features.hpp"

namespace luna::nnue {
namespace {

// Where each piece type's 81 board slots start, for a piece owned by the side
// the features belong to. -1 for the two entries that have no BonaPiece: the
// empty square and the king.
constexpr std::array<int, kPieceTypeNb> kBoardBase = {
    -1,        // kNoPieceType
    kFPawn,    // kPawn
    kFLance,   // kLance
    kFKnight,  // kKnight
    kFSilver,  // kSilver
    kFBishop,  // kBishop
    kFRook,    // kRook
    kFGold,    // kGold
    -1,        // kKing
    kFGold,    // kProPawn    the four small promotions move like a gold and
    kFGold,    // kProLance   are numbered as one
    kFGold,    // kProKnight
    kFGold,    // kProSilver
    kFHorse,   // kHorse
    kFDragon,  // kDragon
};

// The enemy half of a board piece sits exactly 81 slots after the friendly one.
constexpr int kBoardSideStride = kSquareNb;

// Hand slots, which are not all the same width: 18 pawns can be held but only
// two rooks, and one slot is left spare in each range.
struct HandRange {
  int friendly;
  int enemy;
};

constexpr std::array<HandRange, kPieceTypeNb> kHandBase = {{
    {-1, -1},                      // kNoPieceType
    {kFHandPawn, kEHandPawn},      // kPawn
    {kFHandLance, kEHandLance},    // kLance
    {kFHandKnight, kEHandKnight},  // kKnight
    {kFHandSilver, kEHandSilver},  // kSilver
    {kFHandBishop, kEHandBishop},  // kBishop
    {kFHandRook, kEHandRook},      // kRook
    {kFHandGold, kEHandGold},      // kGold
    {-1, -1},                      // kKing
    {-1, -1},                      // kProPawn ... promoted pieces revert
    {-1, -1},                      // kProLance  before they reach a hand,
    {-1, -1},                      // kProKnight so these are never used
    {-1, -1},                      // kProSilver
    {-1, -1},                      // kHorse
    {-1, -1},                      // kDragon
}};

using InverseTable = std::array<BonaPiece, kBonaEnd>;

constexpr InverseTable BuildInverse() {
  InverseTable table{};

  // Hands only swap owner: a pawn held by black is, seen from white, a pawn
  // held by the enemy, and it is the same pawn in the same slot.
  constexpr std::array<HandRange, 7> hands = {{
      {kFHandPawn, kEHandPawn},
      {kFHandLance, kEHandLance},
      {kFHandKnight, kEHandKnight},
      {kFHandSilver, kEHandSilver},
      {kFHandGold, kEHandGold},
      {kFHandBishop, kEHandBishop},
      {kFHandRook, kEHandRook},
  }};
  for (const HandRange& range : hands) {
    const int width = range.enemy - range.friendly;
    for (int i = 0; i < width; ++i) {
      table[range.friendly + i] = static_cast<BonaPiece>(range.enemy + i);
      table[range.enemy + i] = static_cast<BonaPiece>(range.friendly + i);
    }
  }

  // Board pieces swap owner and rotate.
  constexpr std::array<int, 9> board = {
      kFPawn, kFLance, kFKnight, kFSilver, kFGold, kFBishop, kFHorse, kFRook, kFDragon};
  for (const int base : board) {
    for (Square sq = 0; sq < kSquareNb; ++sq) {
      const Square rotated = RotateSquare(sq);
      table[base + sq] = static_cast<BonaPiece>(base + kBoardSideStride + rotated);
      table[base + kBoardSideStride + sq] = static_cast<BonaPiece>(base + rotated);
    }
  }
  return table;
}

constexpr InverseTable kInverse = BuildInverse();

}  // namespace

BonaPiece BoardBona(Color c, PieceType pt, Square sq) {
  const int base = kBoardBase[pt];
  return static_cast<BonaPiece>(base + (c == kBlack ? 0 : kBoardSideStride) + sq);
}

BonaPiece HandBona(Color c, PieceType pt, int index) {
  const HandRange& range = kHandBase[pt];
  return static_cast<BonaPiece>((c == kBlack ? range.friendly : range.enemy) + index);
}

BonaPiece InvertBona(BonaPiece p) {
  return kInverse[p];
}

BonaList BuildBonaList(const Position& pos) {
  BonaList list;
  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = pos.PieceOn(sq);
    if (p == kNoPiece) continue;
    const PieceType pt = TypeOf(p);
    if (pt == kKing) continue;
    list.Add(BoardBona(ColorOf(p), pt, sq));
  }
  for (int c = 0; c < kColorNb; ++c) {
    const Color color = static_cast<Color>(c);
    for (const PieceType pt : kHandTypes) {
      const int count = pos.HandCount(color, pt);
      for (int i = 0; i < count; ++i) list.Add(HandBona(color, pt, i));
    }
  }
  return list;
}

FeatureDelta ComputeDelta(const Position::Undo& undo) {
  FeatureDelta delta;

  const Move m = undo.move;
  const Color us = ColorOf(undo.moved);
  const Square to = m.To();

  if (m.IsDrop()) {
    const PieceType pt = m.DroppedType();
    delta.Remove(HandBona(us, pt, undo.hand_index));
    delta.Add(BoardBona(us, pt, to));
    return delta;
  }

  const PieceType moved_type = TypeOf(undo.moved);
  const PieceType placed_type = m.IsPromotion() ? PromotedType(moved_type) : moved_type;

  if (moved_type == kKing) {
    // Every feature in this side's half is indexed by the king square, so
    // there is nothing incremental to do for it: the half has to be rebuilt.
    // The other half still needs the capture below.
    delta.king_moved = us;
  } else {
    delta.Remove(BoardBona(us, moved_type, m.From()));
    delta.Add(BoardBona(us, placed_type, to));
  }

  if (undo.captured != kNoPiece) {
    const PieceType captured = TypeOf(undo.captured);
    delta.Remove(BoardBona(Opponent(us), captured, to));
    delta.Add(HandBona(us, RawType(captured), undo.hand_index));
  }
  return delta;
}

}  // namespace luna::nnue
