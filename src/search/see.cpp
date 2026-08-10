#include "search/see.hpp"

#include <algorithm>
#include <array>

#include "core/attacks.hpp"
#include "search/eval.hpp"

namespace luna {
namespace {

// An exchange cannot run longer than there are pieces able to reach the
// square: eight lines of at most eight pieces, plus four knights.
constexpr int kMaxSwaps = 72;

// The pieces standing along one of the eight lines out of the exchange square,
// walked outwards from it. Only the nearest one can attack; once it is spent
// the one behind it may take over, which is what makes a lance behind a lance
// count.
struct Line {
  const SquareList* ray = nullptr;
  int next = 0;  // index of the first square on the ray not yet looked at
  int distance = -1;  // ray index of the piece at the front
  Square square = kSquareNone;
  Piece piece = kNoPiece;
};

struct KnightAttacker {
  Square square = kSquareNone;
  Color color = kBlack;
  bool used = false;
};

// Moves the front to the next occupied square, which is how a spent attacker
// is taken off the line. Nothing on the board is modified: everything behind
// the front is still exactly where it was.
void Advance(Line& line, const Position& pos) {
  while (line.next < line.ray->size) {
    const Square sq = line.ray->squares[line.next];
    ++line.next;
    if (pos.PieceOn(sq) != kNoPiece) {
      line.distance = line.next - 1;
      line.square = sq;
      line.piece = pos.PieceOn(sq);
      return;
    }
  }
  line.distance = -1;
  line.square = kSquareNone;
  line.piece = kNoPiece;
}

// Whether the piece at the front of the line attacks the exchange square. Seen
// from the attacker, the square lies in the opposite direction; a single step
// only reaches it from the adjacent square.
bool AttacksSquare(const Line& line, Color by, Direction d) {
  if (line.piece == kNoPiece || ColorOf(line.piece) != by) return false;
  const Direction back = Opposite(d);
  const PieceType pt = TypeOf(line.piece);
  if (line.distance == 0 && ((StepDirections(by, pt) >> back) & 1)) return true;
  return ((SlideDirections(by, pt) >> back) & 1) != 0;
}

}  // namespace

int See(const Position& pos, Move m) {
  const Square to = m.To();

  std::array<Line, kDirectionNb> lines;
  for (int d = 0; d < kDirectionNb; ++d) {
    lines[d].ray = &Ray(to, static_cast<Direction>(d));
    Advance(lines[d], pos);
  }

  // Knights jump, so no exchange can block or reveal one: whichever knights
  // attack the square do so for the whole sequence.
  std::array<KnightAttacker, 4> knights{};
  int knight_count = 0;
  for (int c = 0; c < kColorNb; ++c) {
    const Color by = static_cast<Color>(c);
    for (const uint8_t from : KnightAttacks(Opponent(by), to)) {
      if (pos.PieceOn(from) == MakePiece(by, kKnight)) {
        knights[knight_count++] = {static_cast<Square>(from), by, false};
      }
    }
  }

  // The cheapest attacker is the one to use, because it is the one the other
  // side minds least losing. Reports the piece type and says where it came
  // from, so the caller can spend it.
  const auto least_valuable = [&](Color side, int& line_index, int& knight_index) {
    PieceType best = kNoPieceType;
    int best_value = 0;
    line_index = -1;
    knight_index = -1;
    for (int d = 0; d < kDirectionNb; ++d) {
      if (!AttacksSquare(lines[d], side, static_cast<Direction>(d))) continue;
      const PieceType pt = TypeOf(lines[d].piece);
      const int value = eval::PieceValue(pt);
      if (best == kNoPieceType || value < best_value) {
        best = pt;
        best_value = value;
        line_index = d;
        knight_index = -1;
      }
    }
    for (int i = 0; i < knight_count; ++i) {
      if (knights[i].used || knights[i].color != side) continue;
      if (best == kNoPieceType || eval::PieceValue(kKnight) < best_value) {
        best = kKnight;
        best_value = eval::PieceValue(kKnight);
        line_index = -1;
        knight_index = i;
      }
    }
    return best;
  };

  std::array<int, kMaxSwaps> gain{};
  const Piece victim = pos.PieceOn(to);
  gain[0] = victim == kNoPiece ? 0 : eval::CaptureValue(TypeOf(victim));

  // What stands on the square once `m` is played, and so what the first
  // recapture would win.
  PieceType occupant;
  if (m.IsDrop()) {
    occupant = m.DroppedType();
  } else {
    const Square from = m.From();
    const PieceType mover = TypeOf(pos.PieceOn(from));
    occupant = mover;
    if (m.IsPromotion()) {
      gain[0] += eval::PromotionGain(mover);
      occupant = PromotedType(mover);
    }
    // The mover has left `from`, which may open a line onto the square. A
    // legal move's origin is always the nearest piece on its line, or off the
    // lines entirely when a knight jumped.
    for (Line& line : lines) {
      if (line.square == from) {
        Advance(line, pos);
        break;
      }
    }
    for (int i = 0; i < knight_count; ++i) {
      if (knights[i].square == from) knights[i].used = true;
    }
  }

  int depth = 0;
  Color side = Opponent(pos.SideToMove());
  while (depth + 1 < kMaxSwaps) {
    // A king standing on the square ends the exchange: it is the one piece
    // that cannot be taken. Reachable only from an illegal king capture, which
    // the move generator never offers, but the arithmetic below would price a
    // king at 30000 and the answer would be nonsense.
    if (occupant == kKing) break;

    int line_index = 0;
    int knight_index = 0;
    const PieceType attacker = least_valuable(side, line_index, knight_index);
    if (attacker == kNoPieceType) break;

    // The king is the most expensive piece there is, so it only comes up as
    // the cheapest attacker when it is the only one left. Taking with it is
    // illegal while anything can take it back, which ends the exchange here.
    if (attacker == kKing) {
      int other_line = 0;
      int other_knight = 0;
      if (least_valuable(Opponent(side), other_line, other_knight) != kNoPieceType) break;
    }

    ++depth;
    gain[depth] = eval::CaptureValue(occupant) - gain[depth - 1];
    // Neither side has to keep trading. Once the running total is losing for
    // whoever is on move, and would still be losing if the other side stopped
    // first, the rest of the sequence cannot change the answer.
    if (std::max(-gain[depth - 1], gain[depth]) < 0) break;

    occupant = attacker;
    if (line_index >= 0) {
      Advance(lines[line_index], pos);
    } else {
      knights[knight_index].used = true;
    }
    side = Opponent(side);
  }

  // Fold the sequence back up. At every step the side to move takes the better
  // of capturing and standing pat, which is what makes this the value of the
  // exchange rather than of trading everything off.
  while (--depth >= 0) gain[depth] = -std::max(-gain[depth], gain[depth + 1]);
  return gain[0];
}

bool SeeGe(const Position& pos, Move m, int threshold) {
  return See(pos, m) >= threshold;
}

}  // namespace luna
