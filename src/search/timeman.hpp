#pragma once

#include <cstdint>

#include "core/types.hpp"

namespace luna {

// Everything "go" can ask for. Milliseconds throughout; 0 means "not given",
// which is also how USI omits a token.
struct SearchLimits {
  int btime = 0;
  int wtime = 0;
  int binc = 0;
  int winc = 0;
  int byoyomi = 0;
  int movetime = 0;
  int depth = 0;
  int64_t nodes = 0;
  bool infinite = false;
};

struct TimeBudget {
  // Iterative deepening stops starting new iterations once `soft_ms` is spent
  // and abandons the current one at `hard_ms`. Zero means no time limit.
  int64_t soft_ms = 0;
  int64_t hard_ms = 0;

  bool Unlimited() const { return hard_ms == 0; }
};

// Reserved for process wake-up and GUI round trip. Losing on time is worse
// than thinking 50ms less.
constexpr int64_t kTimeOverheadMs = 50;

// Main time is spread over roughly this many more moves. Byoyomi and
// increment are refilled every move and so get spent in full.
constexpr int kMovesToGo = 40;

// "go" with no limit at all still has to return a move.
constexpr int64_t kNoClockDefaultMs = 1000;

// There is no search thread yet, so "stop" cannot interrupt a search in
// flight and "go infinite" would never return. Capping it keeps the GUI
// usable; real infinite analysis waits for the phase 6 threaded search.
constexpr int64_t kInfiniteFallbackMs = 30000;

TimeBudget ComputeTimeBudget(const SearchLimits& limits, Color us);

}  // namespace luna
