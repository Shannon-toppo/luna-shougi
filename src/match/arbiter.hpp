#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/move.hpp"
#include "core/position.hpp"
#include "core/types.hpp"

namespace luna::match {

enum class Outcome {
  kInProgress,
  kBlackWin,
  kWhiteWin,
  kDraw,
};

// How a game finished. The first four the arbiter works out from the position
// on its own; the rest are things only whoever is driving the engines can see.
enum class Reason {
  kNone,
  kCheckmate,
  kRepetition,      // 千日手
  kPerpetualCheck,  // 連続王手の千日手, a loss for the side giving the checks
  kMaxPly,
  kDeclaration,  // 入玉宣言勝ち
  kResign,
  kIllegalMove,
  kTimeout,
  kNoReply,  // the engine stopped answering
};

struct Verdict {
  Outcome outcome = Outcome::kInProgress;
  Reason reason = Reason::kNone;
};

constexpr Outcome LossFor(Color loser) {
  return loser == kBlack ? Outcome::kWhiteWin : Outcome::kBlackWin;
}

std::string ToString(Outcome outcome);
std::string ToString(Reason reason);

// A game is a draw once the same position has been reached this many times.
constexpr int kRepetitionCount = 4;

// 入玉宣言勝ち, 27点法: rook and bishop count 5 and everything else 1, over
// the declaring side's pieces in the promotion zone and its whole hand. Black
// needs one point more than white because black moved first.
constexpr int kDeclarationPointsBlack = 28;
constexpr int kDeclarationPointsWhite = 27;
constexpr int kDeclarationPieces = 10;

// Long enough that a real game finishes inside it, short enough that two
// engines shuffling pieces do not hold up a match run.
constexpr int kDefaultMaxPly = 320;

// The rules half of a game between two engines: it holds the position, says
// which moves are legal, and decides when the game is over and why.
//
// It knows nothing about engines, clocks or processes, which is what lets it
// be tested by feeding it moves. The rules themselves come from luna_engine's
// move generator, so the arbiter can never disagree with the engine about
// what is legal — the two are the same code.
class Arbiter {
 public:
  explicit Arbiter(const Position& start, int max_ply = kDefaultMaxPly);

  const Position& Pos() const {
    return pos_;
  }
  const std::string& StartSfen() const {
    return start_sfen_;
  }
  const std::vector<Move>& Moves() const {
    return moves_;
  }
  Color SideToMove() const {
    return pos_.SideToMove();
  }

  // Where the game stands before the side to move has played. kInProgress
  // means it is their turn and the game is still on.
  const Verdict& Status() const {
    return status_;
  }
  bool Over() const {
    return status_.outcome != Outcome::kInProgress;
  }

  const MoveList& LegalMoves() const {
    return legal_;
  }
  bool IsLegal(Move m) const {
    return legal_.Contains(m);
  }

  // True when the side to move could claim 入玉宣言勝ち right now.
  bool CanDeclareWin() const;

  // `m` must satisfy IsLegal.
  void Play(Move m);

  // The "position sfen ... moves ..." line describing the game so far.
  std::string PositionCommand() const;

 private:
  void Record();
  void Decide();
  Verdict CheckRepetition() const;
  Color SideAt(size_t index) const;

  Position pos_;
  std::string start_sfen_;
  Color start_side_;
  int max_ply_;
  MoveList legal_;
  std::vector<Move> moves_;
  // One entry per position reached, the starting one included.
  std::vector<uint64_t> keys_;
  std::vector<bool> in_check_;
  Verdict status_;
};

}  // namespace luna::match
