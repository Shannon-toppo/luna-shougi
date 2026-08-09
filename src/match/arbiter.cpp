#include "match/arbiter.hpp"

#include "core/movegen.hpp"

namespace luna::match {
namespace {

// Rook and bishop are worth five points each in the declaration count,
// promoted or not; every other piece but the king is worth one.
int DeclarationPoints(PieceType pt) {
  switch (RawType(pt)) {
    case kRook:
    case kBishop:
      return 5;
    case kKing:
      return 0;
    default:
      return 1;
  }
}

}  // namespace

std::string ToString(Outcome outcome) {
  switch (outcome) {
    case Outcome::kBlackWin:
      return "black wins";
    case Outcome::kWhiteWin:
      return "white wins";
    case Outcome::kDraw:
      return "draw";
    case Outcome::kInProgress:
      break;
  }
  return "in progress";
}

std::string ToString(Reason reason) {
  switch (reason) {
    case Reason::kCheckmate:
      return "checkmate";
    case Reason::kRepetition:
      return "repetition";
    case Reason::kPerpetualCheck:
      return "perpetual check";
    case Reason::kMaxPly:
      return "move limit";
    case Reason::kDeclaration:
      return "declaration";
    case Reason::kResign:
      return "resignation";
    case Reason::kIllegalMove:
      return "illegal move";
    case Reason::kTimeout:
      return "flag";
    case Reason::kNoReply:
      return "no reply";
    case Reason::kNone:
      break;
  }
  return "none";
}

Arbiter::Arbiter(const Position& start, int max_ply)
    : pos_(start), start_sfen_(start.ToSfen()), start_side_(start.SideToMove()), max_ply_(max_ply) {
  Record();
  Decide();
}

Color Arbiter::SideAt(size_t index) const {
  // The arbiter watches the game from its first position, so who is to move
  // is just a matter of how many moves have been played since.
  return index % 2 == 0 ? start_side_ : Opponent(start_side_);
}

void Arbiter::Record() {
  keys_.push_back(pos_.Key());
  in_check_.push_back(pos_.InCheck());
}

void Arbiter::Play(Move m) {
  moves_.push_back(m);
  pos_.DoMove(m);
  Record();
  Decide();
}

void Arbiter::Decide() {
  legal_.Clear();
  movegen::GenerateLegal(pos_, legal_);

  // Shogi has no stalemate: no legal move means mated, in check or not.
  if (legal_.Empty()) {
    status_ = {LossFor(pos_.SideToMove()), Reason::kCheckmate};
    return;
  }

  const Verdict repetition = CheckRepetition();
  if (repetition.outcome != Outcome::kInProgress) {
    status_ = repetition;
    return;
  }

  if (static_cast<int>(moves_.size()) >= max_ply_) {
    status_ = {Outcome::kDraw, Reason::kMaxPly};
    return;
  }

  status_ = {};
}

Verdict Arbiter::CheckRepetition() const {
  const uint64_t key = keys_.back();
  int occurrences = 0;
  size_t first = keys_.size() - 1;
  for (size_t i = keys_.size(); i-- > 0;) {
    if (keys_[i] != key) continue;
    ++occurrences;
    first = i;
    if (occurrences == kRepetitionCount) break;
  }
  if (occurrences < kRepetitionCount) return {};

  // 連続王手の千日手. If one side was in check on every turn it had through
  // the cycle, the other side was the one repeating the checks, and it is the
  // checking side that loses rather than the game being drawn.
  for (const Color c : {kBlack, kWhite}) {
    bool always_in_check = true;
    bool had_a_turn = false;
    for (size_t i = first; i < keys_.size(); ++i) {
      if (SideAt(i) != c) continue;
      had_a_turn = true;
      if (!in_check_[i]) {
        always_in_check = false;
        break;
      }
    }
    if (had_a_turn && always_in_check) {
      return {LossFor(Opponent(c)), Reason::kPerpetualCheck};
    }
  }

  return {Outcome::kDraw, Reason::kRepetition};
}

bool Arbiter::CanDeclareWin() const {
  const Color us = pos_.SideToMove();
  const Square king = pos_.KingSquare(us);
  if (king == kSquareNone) return false;
  if (!InPromotionZone(us, king)) return false;
  if (pos_.InCheck()) return false;

  int pieces = 0;
  int points = 0;
  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = pos_.PieceOn(sq);
    if (p == kNoPiece || ColorOf(p) != us) continue;
    const PieceType pt = TypeOf(p);
    if (pt == kKing) continue;
    if (!InPromotionZone(us, sq)) continue;
    ++pieces;
    points += DeclarationPoints(pt);
  }
  if (pieces < kDeclarationPieces) return false;

  for (const PieceType pt : kHandTypes) {
    points += pos_.HandCount(us, pt) * DeclarationPoints(pt);
  }

  return points >= (us == kBlack ? kDeclarationPointsBlack : kDeclarationPointsWhite);
}

std::string Arbiter::PositionCommand() const {
  std::string command = "position sfen " + start_sfen_;
  if (!moves_.empty()) {
    command += " moves";
    for (const Move m : moves_) command += ' ' + ToUsi(m);
  }
  return command;
}

}  // namespace luna::match
