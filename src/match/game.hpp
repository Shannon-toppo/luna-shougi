#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/move.hpp"
#include "core/position.hpp"
#include "match/arbiter.hpp"
#include "match/usi_client.hpp"

namespace luna::match {

// What each side gets on the clock. Milliseconds throughout; 0 means the
// field is not in use. movetime, depth and nodes are alternatives to a real
// clock and are useful because they make a match reproducible.
struct TimeControl {
  int64_t main_ms = 0;
  int64_t inc_ms = 0;
  int64_t byoyomi_ms = 0;
  int64_t movetime_ms = 0;
  int depth = 0;
  int64_t nodes = 0;

  // Grace on top of what the clock allows before a move counts as late.
  // Starting a process, writing to a pipe and waking the other side up all
  // cost time that is not the engine's fault. Too small a margin turns a
  // measurement of strength into a measurement of scheduling luck.
  int64_t margin_ms = 500;

  bool Timed() const {
    return main_ms > 0 || byoyomi_ms > 0 || movetime_ms > 0;
  }
};

// How long to wait for a search with no clock on it before giving up on the
// engine entirely.
constexpr int64_t kUntimedMoveLimitMs = 300000;

struct GameRecord {
  Outcome outcome = Outcome::kInProgress;
  Reason reason = Reason::kNone;
  std::string start_sfen;
  std::vector<Move> moves;
  // What went wrong, when the game ended because something did rather than
  // because it was over.
  std::string message;

  std::string ToUsiLine() const;
};

// Plays one complete game. Returns when it has a result; an engine that
// answers too slowly, illegally or not at all loses, so this always
// terminates as long as the process layer honours its timeouts.
GameRecord PlayGame(
    UsiClient& black, UsiClient& white, const Position& start, const TimeControl& tc, int max_ply);

}  // namespace luna::match
