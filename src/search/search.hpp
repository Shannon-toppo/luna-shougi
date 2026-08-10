#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
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

// More threads than this is not a machine anyone is playing shogi on, and the
// option has to have an upper bound to advertise.
constexpr int kMaxThreads = 128;

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
// The window is narrowed by principal variation search and the tree is cut
// down by null move pruning, reverse futility, futility and late move pruning
// of quiet moves, late move reductions, and SEE and delta pruning in
// quiescence. All of them can be wrong about a move; what keeps that
// affordable is that a mistake costs strength rather than correctness, and
// that the ones near the leaves are the ones that prune the most.
//
// Think searches with `Threads()` threads, sharing the transposition table and
// nothing else. This is Lazy SMP: the helper threads are not given part of the
// tree to search, they search the same tree from a different depth and at
// their own speed, and the work they save the main thread arrives through the
// table. It scales worse than a real parallel search and is a fraction of the
// code, which is the trade every engine of this size makes.
//
// Think blocks until the search is over. Stop is what ends it early, and it is
// the one member safe to call from another thread while a search runs.
class Search {
 public:
  // Called with one complete "info ..." line per finished iteration.
  using InfoSink = std::function<void(const std::string&)>;

  Search();
  ~Search();
  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;

  void SetInfoSink(InfoSink sink) { info_sink_ = std::move(sink); }

  TranspositionTable& Tt() { return tt_; }

  // Threads Think searches with, the calling one included. Clamped to
  // [1, kMaxThreads]. Not safe to call while a search is running.
  void SetThreads(int count);
  int Threads() const { return static_cast<int>(workers_.size()); }

  // Drops everything carried over from the previous game.
  void NewGame();

  // Searches `pos`, which is only read: each thread works on its own copy.
  SearchResult Think(Position& pos, const SearchLimits& limits);

  // Asks a running search to finish early. Safe to call from any thread. The
  // move that comes back is whatever the search had settled on, which is
  // always a legal one.
  //
  // A stop asked for before the search starts is honoured, not lost: Think
  // does not clear the flag on the way in. That matters because a GUI is
  // entitled to send "stop" the instant after "go", which can easily arrive
  // before the search thread has reached Think at all. Think clears the flag
  // on the way out instead, so the next search starts clean.
  void Stop() { stop_.store(true, std::memory_order_relaxed); }

  // Drops a stop request that no search consumed, which happens when one is
  // asked to stop after it had already finished on its own. Call it before
  // starting a search on another thread; calling it after that thread exists
  // would reintroduce the race it is here to avoid.
  void ClearStop() { stop_.store(false, std::memory_order_relaxed); }

 private:
  // One thread's worth of search: its own position, move stacks, killers,
  // history and node count. Defined in the source file, because nothing
  // outside the search has any business with it.
  class Worker;

  int64_t ElapsedMs() const;
  int64_t TotalNodes() const;
  std::string BuildInfoLine(const SearchResult& result) const;

  TranspositionTable tt_;
  InfoSink info_sink_;
  std::vector<std::unique_ptr<Worker>> workers_;

  // Shared by every worker for the length of one search, and written only
  // before the workers start.
  SearchLimits limits_;
  TimeBudget budget_;
  std::chrono::steady_clock::time_point start_;
  std::atomic<bool> stop_{false};
};

}  // namespace luna
