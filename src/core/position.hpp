#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "core/move.hpp"
#include "core/types.hpp"

namespace luna {

inline constexpr const char* kStartSfen =
    "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1";

class Position {
 public:
  // Starts at the initial position.
  Position();

  // Replaces the position. Returns false and leaves the object untouched when
  // the SFEN is malformed. Accepts the trailing move number as optional.
  bool SetSfen(const std::string& sfen);
  std::string ToSfen() const;

  Piece PieceOn(Square sq) const { return board_[sq]; }
  int HandCount(Color c, PieceType pt) const { return hand_[c][pt]; }
  Color SideToMove() const { return side_; }
  int GamePly() const { return ply_; }
  Square KingSquare(Color c) const { return king_[c]; }
  uint64_t Key() const { return key_; }

  // Number of moves played through DoMove that can still be taken back.
  int UndoableMoves() const { return static_cast<int>(history_.size()); }

  // True when any piece of `by` attacks `sq`, whatever occupies it.
  bool IsSquareAttacked(Square sq, Color by) const;

  // False when `c` has no king on the board, which only happens in contrived
  // test positions.
  bool IsKingAttacked(Color c) const;
  bool InCheck() const { return IsKingAttacked(side_); }

  // True when this exact position (board, hands and side to move alike) has
  // already occurred among the moves played through DoMove. Only the moves
  // since the last capture or drop are examined, because those two make every
  // earlier position unreachable: a captured piece changes hands for good.
  //
  // Shogi calls a draw on the fourth occurrence, but a search that treats the
  // first repetition as a draw is the usual and much cheaper approximation.
  // 連続王手の千日手 (perpetual check, a loss for the checking side) is not
  // distinguished here.
  bool IsRepetition() const;

  // `m` must be pseudo-legal for the current side to move.
  void DoMove(Move m);
  void UndoMove();

  // Recomputes the Zobrist key from scratch; used to verify incremental updates.
  uint64_t ComputeKey() const;

 private:
  struct StateInfo {
    Move move;
    Piece captured;
    uint64_t key;
  };

  void Clear();

  std::array<Piece, kSquareNb> board_{};
  std::array<std::array<int, kPieceTypeNb>, kColorNb> hand_{};
  std::array<Square, kColorNb> king_{};
  Color side_ = kBlack;
  int ply_ = 1;
  uint64_t key_ = 0;
  std::vector<StateInfo> history_;
};

}  // namespace luna
