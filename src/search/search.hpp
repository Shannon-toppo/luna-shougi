#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/move.hpp"
#include "core/position.hpp"
#include "search/eval.hpp"
#include "search/timeman.hpp"
#include "search/tt.hpp"

namespace luna {

// Ceiling on how far from the root any node can be, quiescence included.
constexpr int kMaxPly = 128;

// Ceiling on iterative deepening. Nothing this side of a mate search gets
// close, but a bare "go infinite" needs somewhere to stop.
constexpr int kMaxDepth = 64;

// Scores beyond this magnitude encode a forced mate rather than material.
constexpr int kMateInMaxPly = eval::kMate - kMaxPly;

constexpr bool IsMateScore(int score) {
  return score > kMateInMaxPly || score < -kMateInMaxPly;
}

// Plies until mate, positive when the side to move is the one mating. Only
// meaningful for a score IsMateScore accepts; this is what USI "score mate"
// reports.
constexpr int MateDistance(int score) {
  return score > 0 ? eval::kMate - score : -(eval::kMate + score);
}

struct SearchResult {
  Move best;  // None() only when the position has no legal move at all
  int score = 0;
  int depth = 0;
  int seldepth = 0;
  int64_t nodes = 0;
  int64_t time_ms = 0;
  std::vector<Move> pv;
};

// Alpha-beta with iterative deepening, a transposition table, quiescence
// search and MVV-LVA / killer / history move ordering.
//
// Single threaded and synchronous: Think runs to completion before returning,
// so "stop" cannot interrupt it. Time management is what ends a search.
class Search {
 public:
  // Called with one complete "info ..." line per finished iteration.
  using InfoSink = std::function<void(const std::string&)>;

  Search();

  void SetInfoSink(InfoSink sink) { info_sink_ = std::move(sink); }

  TranspositionTable& Tt() { return tt_; }

  // Drops everything carried over from the previous game.
  void NewGame();

  SearchResult Think(Position& pos, const SearchLimits& limits);

 private:
  // Per-ply scratch space. Kept off the call stack: a MoveList plus its scores
  // is over 6KB, and kMaxPly of those would not fit in a thread's stack.
  struct Node {
    MoveList moves;
    std::array<int, kMaxMoves> scores{};
    std::array<Move, 2> killers{};
  };

  // Drops use kSquareNb + piece type as their origin index.
  static constexpr int kHistoryOriginNb = kSquareNb + kPieceTypeNb;

  int AlphaBeta(Position& pos, int depth, int alpha, int beta, int ply);
  int Quiescence(Position& pos, int alpha, int beta, int ply);

  void ScoreMoves(const Position& pos, int ply, Move tt_move);

  // Moves the highest scoring of the moves from `index` onwards into `index`
  // and returns it. Ordering the tail lazily costs nothing when a cutoff comes
  // early, which is the common case.
  Move PickMove(int ply, int index);

  void UpdateKillers(int ply, Move m);
  void UpdateHistory(Color us, Move m, int depth);
  int& HistoryOf(Color us, Move m);

  void UpdatePv(int ply, Move m);

  bool ShouldStop();
  int64_t ElapsedMs() const;
  std::string BuildInfoLine(const SearchResult& result) const;

  TranspositionTable tt_;
  InfoSink info_sink_;
  std::vector<Node> stack_;
  std::vector<int> history_;
  // Triangular PV table. One extra ply so that a node at the deepest ply can
  // still read its (always empty) child entry.
  std::array<std::array<Move, kMaxPly>, kMaxPly + 1> pv_{};
  std::array<int, kMaxPly + 1> pv_length_{};

  SearchLimits limits_;
  TimeBudget budget_;
  std::chrono::steady_clock::time_point start_;
  int64_t node_count_ = 0;
  int seldepth_ = 0;
  bool stop_ = false;
};

}  // namespace luna
