#include "core/position.hpp"

#include <cctype>
#include <exception>
#include <sstream>

#include "core/attacks.hpp"
#include "core/zobrist.hpp"

namespace luna {
namespace {

bool IsDigit(char c) {
  return c >= '0' && c <= '9';
}

}  // namespace

Position::Position() {
  SetSfen(kStartSfen);
}

void Position::Clear() {
  board_.fill(kNoPiece);
  for (auto& per_color : hand_) per_color.fill(0);
  king_.fill(kSquareNone);
  side_ = kBlack;
  ply_ = 1;
  key_ = 0;
  history_.clear();
}

bool Position::SetSfen(const std::string& sfen) {
  std::istringstream iss(sfen);
  std::string board_field;
  std::string side_field;
  std::string hand_field;
  std::string ply_field;
  if (!(iss >> board_field >> side_field)) return false;
  if (!(iss >> hand_field)) hand_field = "-";
  if (!(iss >> ply_field)) ply_field = "1";

  // Parse into locals so a malformed SFEN leaves the position untouched.
  std::array<Piece, kSquareNb> board{};
  board.fill(kNoPiece);
  std::array<std::array<int, kPieceTypeNb>, kColorNb> hand{};

  int column = 0;
  int rank = 1;
  bool promoted = false;
  for (const char ch : board_field) {
    if (ch == '/') {
      if (column != 9 || promoted || rank >= 9) return false;
      column = 0;
      ++rank;
      continue;
    }
    if (ch == '+') {
      if (promoted) return false;
      promoted = true;
      continue;
    }
    if (IsDigit(ch)) {
      if (promoted || ch == '0') return false;
      column += ch - '0';
      if (column > 9) return false;
      continue;
    }
    PieceType pt = PieceTypeFromChar(ch);
    if (pt == kNoPieceType || column >= 9) return false;
    if (promoted) {
      if (!CanPromoteType(pt)) return false;
      pt = PromotedType(pt);
      promoted = false;
    }
    const Color color = std::isupper(static_cast<unsigned char>(ch)) ? kBlack : kWhite;
    board[(rank - 1) * 9 + column] = MakePiece(color, pt);
    ++column;
  }
  if (rank != 9 || column != 9 || promoted) return false;

  if (side_field != "b" && side_field != "w") return false;
  const Color side = side_field == "b" ? kBlack : kWhite;

  if (hand_field != "-") {
    int count = 0;
    for (const char ch : hand_field) {
      if (IsDigit(ch)) {
        count = count * 10 + (ch - '0');
        if (count > zobrist::kMaxHandCount) return false;
        continue;
      }
      const PieceType pt = PieceTypeFromChar(ch);
      if (pt == kNoPieceType || pt == kKing) return false;
      const Color color = std::isupper(static_cast<unsigned char>(ch)) ? kBlack : kWhite;
      hand[color][pt] += (count == 0 ? 1 : count);
      if (hand[color][pt] > zobrist::kMaxHandCount) return false;
      count = 0;
    }
    if (count != 0) return false;
  }

  int ply = 1;
  try {
    size_t consumed = 0;
    ply = std::stoi(ply_field, &consumed);
    if (consumed != ply_field.size() || ply < 1) return false;
  } catch (const std::exception&) {
    return false;
  }

  Clear();
  board_ = board;
  hand_ = hand;
  side_ = side;
  ply_ = ply;
  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = board_[sq];
    if (p != kNoPiece && TypeOf(p) == kKing) king_[ColorOf(p)] = sq;
  }
  key_ = ComputeKey();
  return true;
}

std::string Position::ToSfen() const {
  std::string sfen;
  for (int rank = 1; rank <= 9; ++rank) {
    int empty = 0;
    for (int file = 9; file >= 1; --file) {
      const Piece p = board_[MakeSquare(file, rank)];
      if (p == kNoPiece) {
        ++empty;
        continue;
      }
      if (empty > 0) {
        sfen += static_cast<char>('0' + empty);
        empty = 0;
      }
      std::string letters = PieceTypeToSfen(TypeOf(p));
      if (ColorOf(p) == kWhite) {
        letters.back() =
            static_cast<char>(std::tolower(static_cast<unsigned char>(letters.back())));
      }
      sfen += letters;
    }
    if (empty > 0) sfen += static_cast<char>('0' + empty);
    if (rank != 9) sfen += '/';
  }

  sfen += side_ == kBlack ? " b " : " w ";

  std::string hands;
  for (const Color color : {kBlack, kWhite}) {
    for (const PieceType pt : kHandTypes) {
      const int count = hand_[color][pt];
      if (count == 0) continue;
      if (count > 1) hands += std::to_string(count);
      const char letter = PieceTypeToChar(pt);
      hands += color == kBlack
                   ? letter
                   : static_cast<char>(std::tolower(static_cast<unsigned char>(letter)));
    }
  }
  sfen += hands.empty() ? "-" : hands;

  sfen += ' ';
  sfen += std::to_string(ply_);
  return sfen;
}

uint64_t Position::ComputeKey() const {
  const zobrist::Table& keys = zobrist::Keys();
  uint64_t key = 0;
  for (Square sq = 0; sq < kSquareNb; ++sq) {
    const Piece p = board_[sq];
    if (p != kNoPiece) key ^= keys.psq[p][sq];
  }
  for (int color = 0; color < kColorNb; ++color) {
    for (const PieceType pt : kHandTypes) key ^= keys.hand[color][pt][hand_[color][pt]];
  }
  if (side_ == kWhite) key ^= keys.side;
  return key;
}

bool Position::IsSquareAttacked(Square sq, Color by) const {
  const Color them = Opponent(by);

  // A knight of `by` attacks `sq` exactly when a knight of the other color on
  // `sq` would attack its square, because the two move sets are mirrors.
  for (const uint8_t from : KnightAttacks(them, sq)) {
    if (board_[from] == MakePiece(by, kKnight)) return true;
  }

  for (int d = 0; d < kDirectionNb; ++d) {
    const SquareList& line = Ray(sq, static_cast<Direction>(d));
    for (int i = 0; i < line.size; ++i) {
      const Square from = line.squares[i];
      const Piece p = board_[from];
      if (p == kNoPiece) continue;
      if (ColorOf(p) == by) {
        // Seen from `from`, our square lies in the opposite direction.
        const Direction back = Opposite(static_cast<Direction>(d));
        const PieceType pt = TypeOf(p);
        if (i == 0 && (StepDirections(by, pt) >> back) & 1) return true;
        if ((SlideDirections(by, pt) >> back) & 1) return true;
      }
      break;  // The first occupied square blocks everything behind it.
    }
  }
  return false;
}

bool Position::IsKingAttacked(Color c) const {
  const Square king = king_[c];
  if (king == kSquareNone) return false;
  return IsSquareAttacked(king, Opponent(c));
}

bool Position::IsRepetition() const {
  const int played = static_cast<int>(history_.size());
  for (int i = played - 1; i >= 0; --i) {
    const Undo& state = history_[i];
    // state.key is the position before state.move was played.
    if (state.captured != kNoPiece || state.move.IsDrop()) break;
    if ((played - i) % 2 == 0 && state.key == key_) return true;
  }
  return false;
}

void Position::DoMove(Move m) {
  const zobrist::Table& keys = zobrist::Keys();
  const Color us = side_;
  const Square to = m.To();
  Undo state{m, kNoPiece, kNoPiece, -1, key_};

  if (m.IsDrop()) {
    const PieceType pt = m.DroppedType();
    const Piece dropped = MakePiece(us, pt);
    state.moved = dropped;
    board_[to] = dropped;
    key_ ^= keys.psq[dropped][to];
    key_ ^= keys.hand[us][pt][hand_[us][pt]];
    --hand_[us][pt];
    state.hand_index = static_cast<int8_t>(hand_[us][pt]);
    key_ ^= keys.hand[us][pt][hand_[us][pt]];
  } else {
    const Square from = m.From();
    const Piece moved = board_[from];
    state.moved = moved;
    key_ ^= keys.psq[moved][from];
    board_[from] = kNoPiece;

    const Piece captured = board_[to];
    if (captured != kNoPiece) {
      state.captured = captured;
      const PieceType raw = RawType(TypeOf(captured));
      key_ ^= keys.psq[captured][to];
      key_ ^= keys.hand[us][raw][hand_[us][raw]];
      state.hand_index = static_cast<int8_t>(hand_[us][raw]);
      ++hand_[us][raw];
      key_ ^= keys.hand[us][raw][hand_[us][raw]];
    }

    const Piece placed =
        m.IsPromotion() ? MakePiece(us, PromotedType(TypeOf(moved))) : moved;
    board_[to] = placed;
    key_ ^= keys.psq[placed][to];
    if (TypeOf(placed) == kKing) king_[us] = to;
  }

  side_ = Opponent(us);
  key_ ^= keys.side;
  ++ply_;
  history_.push_back(state);
}

void Position::UndoMove() {
  const Undo state = history_.back();
  history_.pop_back();

  side_ = Opponent(side_);
  --ply_;
  key_ = state.key;

  const Color us = side_;
  const Move m = state.move;
  const Square to = m.To();

  if (m.IsDrop()) {
    board_[to] = kNoPiece;
    ++hand_[us][m.DroppedType()];
    return;
  }

  const Piece placed = board_[to];
  const Piece moved = m.IsPromotion() ? MakePiece(us, RawType(TypeOf(placed))) : placed;
  board_[m.From()] = moved;
  board_[to] = state.captured;
  if (state.captured != kNoPiece) --hand_[us][RawType(TypeOf(state.captured))];
  if (TypeOf(moved) == kKing) king_[us] = m.From();
}

void Position::DoNullMove() {
  side_ = Opponent(side_);
  key_ ^= zobrist::Keys().side;
  ++ply_;
}

void Position::UndoNullMove() {
  side_ = Opponent(side_);
  key_ ^= zobrist::Keys().side;
  --ply_;
}

}  // namespace luna
