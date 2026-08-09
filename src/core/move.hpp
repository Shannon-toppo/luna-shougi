#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "core/types.hpp"

namespace luna {

// A move packed into 16 bits:
//   bits  0-6  destination square
//   bits  7-13 origin square, or the dropped piece type for drops
//   bit  14    promotion flag
//   bit  15    drop flag
// The all-zero encoding would mean "board move from 9a to 9a", which is never
// legal, so it doubles as the "no move" value.
class Move {
 public:
  constexpr Move() = default;

  static constexpr Move None() { return Move(0); }

  static constexpr Move Normal(Square from, Square to, bool promote) {
    return Move(static_cast<uint16_t>(to | (from << 7) | (promote ? 1u << 14 : 0u)));
  }

  static constexpr Move Drop(PieceType pt, Square to) {
    return Move(static_cast<uint16_t>(to | (pt << 7) | (1u << 15)));
  }

  constexpr bool IsNone() const { return raw_ == 0; }
  constexpr bool IsDrop() const { return (raw_ & (1u << 15)) != 0; }
  constexpr bool IsPromotion() const { return (raw_ & (1u << 14)) != 0; }
  constexpr Square To() const { return raw_ & 0x7F; }
  constexpr Square From() const { return (raw_ >> 7) & 0x7F; }
  constexpr PieceType DroppedType() const { return static_cast<PieceType>((raw_ >> 7) & 0x7F); }
  constexpr uint16_t Raw() const { return raw_; }

  friend constexpr bool operator==(Move a, Move b) { return a.raw_ == b.raw_; }
  friend constexpr bool operator!=(Move a, Move b) { return a.raw_ != b.raw_; }

 private:
  explicit constexpr Move(uint16_t raw) : raw_(raw) {}

  uint16_t raw_ = 0;
};

// The most legal moves a shogi position is known to allow is 593; pseudo-legal
// generation can exceed that slightly, so leave generous headroom.
constexpr int kMaxMoves = 1024;

class MoveList {
 public:
  void Add(Move m) { moves_[size_++] = m; }
  void Clear() { size_ = 0; }
  int Size() const { return size_; }
  bool Empty() const { return size_ == 0; }
  Move operator[](int i) const { return moves_[i]; }
  bool Contains(Move m) const;

  const Move* begin() const { return moves_.data(); }
  const Move* end() const { return moves_.data() + size_; }

 private:
  std::array<Move, kMaxMoves> moves_{};
  int size_ = 0;
};

// USI notation: "7g7f", "7g7f+" for board moves, "P*5e" for drops.
std::string ToUsi(Move m);

// Returns Move::None() when the string is not valid USI move notation. The
// result is syntactic only; it is not checked against any position.
Move MoveFromUsi(const std::string& s);

// USI square notation, e.g. "7g".
std::string SquareToUsi(Square sq);

// Returns kSquareNone on failure.
Square SquareFromUsi(const std::string& s);

// Single-letter piece notation shared by USI and SFEN, always uppercase and
// always the unpromoted letter. Returns 0 / kNoPieceType on failure.
char PieceTypeToChar(PieceType pt);
PieceType PieceTypeFromChar(char c);

// SFEN piece notation including the '+' prefix, e.g. "+P" for a promoted pawn.
std::string PieceTypeToSfen(PieceType pt);

}  // namespace luna
