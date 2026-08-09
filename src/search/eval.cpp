#include "search/eval.hpp"

#include <array>
#include <cstdint>

#include "search/pst.hpp"

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

// A piece in hand is worth a little more than the same piece on the board: it
// picks its square instead of being stuck on one, and nothing can attack it
// where it sits. Only a little, because it is also doing nothing until it is
// dropped, and because too large a gap makes the engine hoard pieces it
// should be using. Only the seven types that can be held are ever read.
constexpr std::array<int, kPieceTypeNb> kHandValues = {
    0,      // kNoPieceType
    100,    // kPawn
    335,    // kLance
    420,    // kKnight
    510,    // kSilver
    890,    // kBishop
    1030,   // kRook
    555,    // kGold
    15000,  // kKing
    100,    // kProPawn     reverts to a pawn when captured
    335,    // kProLance
    420,    // kProKnight
    510,    // kProSilver
    890,    // kHorse
    1030,   // kDragon
};

// What a piece is worth standing next to the enemy king, per point of
// proximity. This is the attacking half of king safety: a shogi attack is
// mostly a matter of getting enough pieces near the opposing castle, and the
// count of attackers matters more than which squares they are on.
constexpr std::array<int, kPieceTypeNb> kAttackWeight = {
    0,   // kNoPieceType
    2,   // kPawn
    3,   // kLance
    4,   // kKnight
    5,   // kSilver
    5,   // kBishop
    6,   // kRook
    6,   // kGold
    0,   // kKing        the two kings are handled by their own table
    6,   // kProPawn
    6,   // kProLance
    6,   // kProKnight
    6,   // kProSilver
    8,   // kHorse
    10,  // kDragon
};

// What a piece is worth standing next to its own king. Golds and silvers are
// what a castle is built out of; a rook defends badly because it wants to be
// on a file somewhere else, and a knight cannot cover any square next to it.
// The horse is the outlier at the top, which is the whole point of 馬は自陣に
// 引け: no other piece defends that many squares and survives that long.
constexpr std::array<int, kPieceTypeNb> kDefenceWeight = {
    0,  // kNoPieceType
    3,  // kPawn
    2,  // kLance
    1,  // kKnight
    6,  // kSilver
    2,  // kBishop
    1,  // kRook
    7,  // kGold
    0,  // kKing
    5,  // kProPawn
    5,  // kProLance
    5,  // kProKnight
    5,  // kProSilver
    8,  // kHorse
    4,  // kDragon
};

// Squares within this distance of a king count towards its safety.
constexpr int kProximityRange = 5;

constexpr int ChebyshevDistance(Square a, Square b) {
  const int file = FileOf(a) - FileOf(b);
  const int rank = RankOf(a) - RankOf(b);
  const int df = file < 0 ? -file : file;
  const int dr = rank < 0 ? -rank : rank;
  return df > dr ? df : dr;
}

using ProximityTable = std::array<std::array<uint8_t, kSquareNb>, kSquareNb>;

// 4 for a square adjacent to the king, falling to 0 five squares away.
// Tabulated by square pair so the evaluation loop pays one load instead of
// pulling both squares apart into files and ranks.
constexpr ProximityTable BuildProximity() {
  ProximityTable table{};
  for (Square a = 0; a < kSquareNb; ++a) {
    for (Square b = 0; b < kSquareNb; ++b) {
      const int distance = ChebyshevDistance(a, b);
      table[a][b] =
          static_cast<uint8_t>(distance < kProximityRange ? kProximityRange - distance : 0);
    }
  }
  return table;
}

constexpr ProximityTable kProximity = BuildProximity();

int Accumulate(const Position& pos, EvalTerms& terms) {
  const std::array<Square, kColorNb> king = {pos.KingSquare(kBlack), pos.KingSquare(kWhite)};

  int material = 0;
  int pst = 0;
  int king_safety = 0;

  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = pos.PieceOn(sq);
    if (p == kNoPiece) continue;

    const PieceType pt = TypeOf(p);
    const Color c = ColorOf(p);
    const int sign = c == kBlack ? 1 : -1;

    pst += sign * PstValue(c, pt, sq);
    // A king is priced by its table alone: it is never captured, and counting
    // it as both a defender and an attacker of itself would say nothing.
    if (pt == kKing) continue;

    material += sign * kValues[pt];

    int proximity = 0;
    const Square own = king[c];
    const Square foe = king[Opponent(c)];
    if (own != kSquareNone) proximity += kDefenceWeight[pt] * kProximity[sq][own];
    if (foe != kSquareNone) proximity += kAttackWeight[pt] * kProximity[sq][foe];
    king_safety += sign * proximity;
  }

  for (const PieceType pt : kHandTypes) {
    const int diff = pos.HandCount(kBlack, pt) - pos.HandCount(kWhite, pt);
    material += diff * kHandValues[pt];
  }

  const int tempo = pos.SideToMove() == kBlack ? kTempo : -kTempo;
  const int black = material + pst + king_safety + tempo;

  terms.material = material;
  terms.pst = pst;
  terms.king_safety = king_safety;
  terms.tempo = tempo;
  terms.total = pos.SideToMove() == kBlack ? black : -black;
  return terms.total;
}

}  // namespace

int PieceValue(PieceType pt) {
  return kValues[pt];
}

int HandValue(PieceType pt) {
  return kHandValues[RawType(pt)];
}

int CaptureValue(PieceType pt) {
  return PieceValue(pt) + HandValue(RawType(pt));
}

int PromotionGain(PieceType pt) {
  return CanPromoteType(pt) ? kValues[PromotedType(pt)] - kValues[pt] : 0;
}

int Evaluate(const Position& pos) {
  EvalTerms terms;
  return Accumulate(pos, terms);
}

EvalTerms Trace(const Position& pos) {
  EvalTerms terms;
  Accumulate(pos, terms);
  return terms;
}

}  // namespace luna::eval
