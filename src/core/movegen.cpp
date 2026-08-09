#include "core/movegen.hpp"

#include <bit>

#include "core/attacks.hpp"

namespace luna::movegen {
namespace {

Direction LowestDirection(uint8_t mask) {
  return static_cast<Direction>(std::countr_zero(mask));
}

// A move to `to` is worth generating without promotion only if the piece would
// still have somewhere to go afterwards (行きどころのない駒).
bool CanStayUnpromoted(Color us, PieceType pt, Square to) {
  const int rank = RelativeRankOf(us, to);
  switch (pt) {
    case kPawn:
    case kLance:
      return rank >= 2;
    case kKnight:
      return rank >= 3;
    default:
      return true;
  }
}

void AddBoardMove(Color us, PieceType pt, Square from, Square to, MoveList& out) {
  if (CanPromoteType(pt) && (InPromotionZone(us, from) || InPromotionZone(us, to))) {
    out.Add(Move::Normal(from, to, true));
  }
  if (CanStayUnpromoted(us, pt, to)) out.Add(Move::Normal(from, to, false));
}

void GeneratePieceMoves(const Position& pos, Color us, Square from, MoveList& out) {
  const PieceType pt = TypeOf(pos.PieceOn(from));

  for (uint8_t steps = StepDirections(us, pt); steps != 0; steps &= steps - 1) {
    const Direction d = LowestDirection(steps);
    const Square to = Neighbor(from, d);
    if (to == kSquareNone) continue;
    const Piece target = pos.PieceOn(to);
    if (target != kNoPiece && ColorOf(target) == us) continue;
    AddBoardMove(us, pt, from, to, out);
  }

  for (uint8_t slides = SlideDirections(us, pt); slides != 0; slides &= slides - 1) {
    const Direction d = LowestDirection(slides);
    const SquareList& line = Ray(from, d);
    for (int i = 0; i < line.size; ++i) {
      const Square to = line.squares[i];
      const Piece target = pos.PieceOn(to);
      if (target != kNoPiece) {
        if (ColorOf(target) != us) AddBoardMove(us, pt, from, to, out);
        break;
      }
      AddBoardMove(us, pt, from, to, out);
    }
  }

  if (pt == kKnight) {
    for (const uint8_t to : KnightAttacks(us, from)) {
      const Piece target = pos.PieceOn(to);
      if (target != kNoPiece && ColorOf(target) == us) continue;
      AddBoardMove(us, pt, from, to, out);
    }
  }
}

void GenerateDrops(const Position& pos, Color us, MoveList& out) {
  const Piece own_pawn = MakePiece(us, kPawn);
  const bool have_pawn = pos.HandCount(us, kPawn) > 0;

  // 二歩: at most one unpromoted pawn per file.
  bool file_has_pawn[9] = {};
  if (have_pawn) {
    for (Square sq = 0; sq < kSquareNb; ++sq) {
      if (pos.PieceOn(sq) == own_pawn) file_has_pawn[ColumnOf(sq)] = true;
    }
  }

  for (const PieceType pt : kHandTypes) {
    if (pos.HandCount(us, pt) == 0) continue;
    for (Square to = 0; to < kSquareNb; ++to) {
      if (pos.PieceOn(to) != kNoPiece) continue;
      const int rank = RelativeRankOf(us, to);
      if ((pt == kPawn || pt == kLance) && rank < 2) continue;
      if (pt == kKnight && rank < 3) continue;
      if (pt == kPawn && file_has_pawn[ColumnOf(to)]) continue;
      out.Add(Move::Drop(pt, to));
    }
  }
}

}  // namespace

void GeneratePseudoLegal(const Position& pos, MoveList& out) {
  const Color us = pos.SideToMove();
  for (Square from = 0; from < kSquareNb; ++from) {
    const Piece p = pos.PieceOn(from);
    if (p == kNoPiece || ColorOf(p) != us) continue;
    GeneratePieceMoves(pos, us, from, out);
  }
  GenerateDrops(pos, us, out);
}

void GenerateLegal(Position& pos, MoveList& out) {
  MoveList pseudo;
  GeneratePseudoLegal(pos, pseudo);

  const Color us = pos.SideToMove();
  for (const Move m : pseudo) {
    pos.DoMove(m);
    bool legal = !pos.IsKingAttacked(us);
    // 打ち歩詰め: a dropped pawn may give check, but not mate. Any other way of
    // delivering mate is fine, and a dropped pawn only ever gives check from
    // directly in front of the king, so InCheck() is the cheap pre-filter.
    if (legal && m.IsDrop() && m.DroppedType() == kPawn && pos.InCheck()) {
      legal = !IsCheckmate(pos);
    }
    pos.UndoMove();
    if (legal) out.Add(m);
  }
}

MoveList GenerateLegal(Position& pos) {
  MoveList moves;
  GenerateLegal(pos, moves);
  return moves;
}

bool IsCheckmate(Position& pos) {
  if (!pos.InCheck()) return false;
  MoveList replies;
  GenerateLegal(pos, replies);
  return replies.Empty();
}

bool IsLegal(Position& pos, Move m) {
  if (m.IsNone()) return false;
  MoveList moves;
  GenerateLegal(pos, moves);
  return moves.Contains(m);
}

}  // namespace luna::movegen
