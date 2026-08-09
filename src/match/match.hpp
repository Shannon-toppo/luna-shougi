#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/position.hpp"
#include "match/elo.hpp"
#include "match/game.hpp"
#include "match/usi_client.hpp"

namespace luna::match {

struct MatchConfig {
  EngineSpec first;
  EngineSpec second;

  // Rounded up to an even number: every opening is played twice with the
  // colours swapped, so that neither engine can be handed the better side of
  // an unbalanced start.
  int games = 100;

  // Random legal moves played before either engine is asked anything. Two
  // deterministic engines given the same position play the same game every
  // time, so without this a hundred-game match is one game counted a hundred
  // times.
  int opening_plies = 4;
  uint64_t seed = 1;

  int max_ply = kDefaultMaxPly;
  int concurrency = 1;
  TimeControl tc;
};

struct MatchReport {
  // From the first engine's point of view.
  MatchScore score;
  std::vector<GameRecord> games;
  // Set when the match could not be run at all, e.g. an engine that would
  // not start. Games already played are still in `games`.
  std::string error;
};

// The position `plies` random legal moves into a game. Stops early if the
// game somehow ends first, which a handful of random moves will not do.
Position RandomOpening(uint64_t seed, int plies);

// Plays the whole match. `log` receives one line per finished game plus the
// running tally, and is called from several threads but never concurrently.
MatchReport RunMatch(const MatchConfig& config, const std::function<void(const std::string&)>& log);

}  // namespace luna::match
