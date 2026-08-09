#include "core/move.hpp"

#include <cctype>

namespace luna {
namespace {

// Indexed by the raw (unpromoted) piece type.
constexpr char kPieceLetters[] = "?PLNSBRGK";

}  // namespace

bool MoveList::Contains(Move m) const {
  for (const Move candidate : *this) {
    if (candidate == m) return true;
  }
  return false;
}

char PieceTypeToChar(PieceType pt) {
  const PieceType raw = RawType(pt);
  if (raw < kPawn || raw > kKing) return 0;
  return kPieceLetters[raw];
}

PieceType PieceTypeFromChar(char c) {
  const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  for (int pt = kPawn; pt <= kKing; ++pt) {
    if (kPieceLetters[pt] == upper) return static_cast<PieceType>(pt);
  }
  return kNoPieceType;
}

std::string PieceTypeToSfen(PieceType pt) {
  const char letter = PieceTypeToChar(pt);
  if (letter == 0) return std::string();
  std::string s;
  if (IsPromotedType(pt)) s += '+';
  s += letter;
  return s;
}

std::string SquareToUsi(Square sq) {
  if (!IsSquareValid(sq)) return std::string();
  std::string s;
  s += static_cast<char>('0' + FileOf(sq));
  s += static_cast<char>('a' + RankOf(sq) - 1);
  return s;
}

Square SquareFromUsi(const std::string& s) {
  if (s.size() != 2) return kSquareNone;
  const int file = s[0] - '0';
  const int rank = s[1] - 'a' + 1;
  if (file < 1 || file > 9 || rank < 1 || rank > 9) return kSquareNone;
  return MakeSquare(file, rank);
}

std::string ToUsi(Move m) {
  if (m.IsNone()) return "resign";
  std::string s;
  if (m.IsDrop()) {
    s += PieceTypeToChar(m.DroppedType());
    s += '*';
    s += SquareToUsi(m.To());
    return s;
  }
  s += SquareToUsi(m.From());
  s += SquareToUsi(m.To());
  if (m.IsPromotion()) s += '+';
  return s;
}

Move MoveFromUsi(const std::string& s) {
  if (s.size() == 4 && s[1] == '*') {
    const PieceType pt = PieceTypeFromChar(s[0]);
    // Only unpromoted, non-king pieces can sit in hand.
    if (pt == kNoPieceType || pt == kKing) return Move::None();
    const Square to = SquareFromUsi(s.substr(2, 2));
    if (to == kSquareNone) return Move::None();
    return Move::Drop(pt, to);
  }
  if (s.size() != 4 && !(s.size() == 5 && s[4] == '+')) return Move::None();
  const Square from = SquareFromUsi(s.substr(0, 2));
  const Square to = SquareFromUsi(s.substr(2, 2));
  if (from == kSquareNone || to == kSquareNone || from == to) return Move::None();
  return Move::Normal(from, to, s.size() == 5);
}

}  // namespace luna
