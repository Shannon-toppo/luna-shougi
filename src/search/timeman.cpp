#include "search/timeman.hpp"

#include <algorithm>

namespace luna {

TimeBudget ComputeTimeBudget(const SearchLimits& limits, Color us) {
  // "go infinite" means what it says: the search runs until "stop" arrives.
  // That only became true once the search moved off the command thread, which
  // is what reads the "stop".
  if (limits.infinite) return {};

  if (limits.movetime > 0) {
    const int64_t t = std::max<int64_t>(1, limits.movetime - kTimeOverheadMs);
    return {t, t};
  }

  const int64_t remaining = us == kBlack ? limits.btime : limits.wtime;
  const int64_t increment = us == kBlack ? limits.binc : limits.winc;
  const int64_t byoyomi = limits.byoyomi;

  if (remaining <= 0 && increment <= 0 && byoyomi <= 0) {
    // No clock was given. A depth or node limit ends the search on its own;
    // a bare "go" needs some limit or it would run to the maximum depth.
    if (limits.depth > 0 || limits.nodes > 0) return {};
    return {kNoClockDefaultMs, kNoClockDefaultMs};
  }

  // Byoyomi and increment come back every move, so they are free to spend.
  // Only the main time has to last, and it is what the divisor applies to.
  int64_t soft = remaining / kMovesToGo + increment + byoyomi;

  // Overshooting the clock loses the game outright, so the hard limit is
  // capped by what is actually there to spend.
  const int64_t available = std::max<int64_t>(1, remaining + byoyomi - kTimeOverheadMs);
  const int64_t hard = std::clamp<int64_t>(soft * 2, 1, available);
  soft = std::clamp<int64_t>(soft, 1, hard);

  return {soft, hard};
}

}  // namespace luna
