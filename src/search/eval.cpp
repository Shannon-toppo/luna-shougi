#include "search/eval.hpp"

#include <array>

namespace luna::eval {
namespace {

// The values shogi engines have broadly converged on, scaled so an unpromoted
// pawn is 90. All four small promoted pieces are worth a gold, which is what
// makes 成り so attractive for a pawn and so marginal for a silver.
constexpr std::array<int, kPieceTypeNb> kValues = {
    0,      // kNoPieceType
    90,     // kPawn
    315,    // kLance
    405,    // kKnight
    495,    // kSilver
    855,    // kBishop
    990,    // kRook
    540,    // kGold
    15000,  // kKing
    540,    // kProPawn
    540,    // kProLance
    540,    // kProKnight
    540,    // kProSilver
    945,    // kHorse
    1395,   // kDragon
};

}  // namespace

int PieceValue(PieceType pt) {
  return kValues[pt];
}

int HandValue(PieceType pt) {
  return kValues[RawType(pt)];
}

int CaptureValue(PieceType pt) {
  return PieceValue(pt) + HandValue(RawType(pt));
}

int PromotionGain(PieceType pt) {
  return CanPromoteType(pt) ? kValues[PromotedType(pt)] - kValues[pt] : 0;
}

int Evaluate(const Position& pos) {
  int score = 0;

  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = pos.PieceOn(sq);
    if (p == kNoPiece) continue;
    const PieceType pt = TypeOf(p);
    if (pt == kKing) continue;
    score += ColorOf(p) == kBlack ? kValues[pt] : -kValues[pt];
  }

  for (const PieceType pt : kHandTypes) {
    const int diff = pos.HandCount(kBlack, pt) - pos.HandCount(kWhite, pt);
    score += diff * HandValue(pt);
  }

  return pos.SideToMove() == kBlack ? score : -score;
}

}  // namespace luna::eval
