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
  // What a DoMove has to be able to take back, plus enough to say what the
  // move did to the position without having that position at hand. That last
  // part is what lets incremental evaluation replay a run of moves from an
  // accumulator it computed several plies ago.
  struct Undo {
    Move move;
    // The piece as it stood before the move, or the dropped piece. Its colour
    // is the side that played.
    Piece moved;
    // kNoPiece when nothing was captured.
    Piece captured;
    // The hand slot the move filled or emptied, counting from 0: the slot the
    // captured piece landed in, or the one the dropped piece came out of. -1
    // when no hand changed.
    int8_t hand_index;
    // Zobrist key of the position the move was played in.
    uint64_t key;
  };

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

  // The `index`-th move played through DoMove, counting from 0. Incremental
  // evaluation walks this backwards to work out which features changed since
  // the last position it had computed. `index` must be in
  // [0, UndoableMoves()).
  const Undo& UndoAt(int index) const { return history_[static_cast<size_t>(index)]; }

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

  // Passes the turn. Nothing but the side to move changes, which is not a
  // legal thing to do in a game: the search uses it to ask what the opponent
  // would do with a free move. Illegal while in check, where the answer would
  // be "capture the king".
  //
  // Deliberately not recorded in the move history, so UndoNullMove rather than
  // UndoMove takes it back. A null move changes no piece and no hand, so
  // incremental evaluation, which replays the history to work out what moved,
  // stays correct without ever being told one happened.
  void DoNullMove();
  void UndoNullMove();

  // Recomputes the Zobrist key from scratch; used to verify incremental updates.
  uint64_t ComputeKey() const;

 private:
  void Clear();

  std::array<Piece, kSquareNb> board_{};
  std::array<std::array<int, kPieceTypeNb>, kColorNb> hand_{};
  std::array<Square, kColorNb> king_{};
  Color side_ = kBlack;
  int ply_ = 1;
  uint64_t key_ = 0;
  std::vector<Undo> history_;
};

}  // namespace luna
