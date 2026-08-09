#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "match/arbiter.hpp"

namespace luna::datagen {

struct Config {
  std::string out_path;
  // Keep whatever the file already holds. A generation run that was stopped
  // and started again adds to its own output rather than replacing it.
  bool append = false;

  int games = 1000;
  // One of the two, depth unless nodes is set. Fixed depth is reproducible
  // and is what makes two runs on different machines comparable.
  int depth = 6;
  int64_t nodes = 0;

  // Random legal moves played before the search takes over. Without them
  // every game from a deterministic engine would be the same game.
  int opening_plies = 8;
  uint64_t seed = 1;
  int max_ply = match::kDefaultMaxPly;
  int concurrency = 1;

  // Per thread, so the default is well below the engine's: eight threads at
  // the engine's 64MB would be half a gigabyte of table for searches that are
  // only a few plies deep.
  int hash_mb = 32;

  // Positions the search already considers decided teach the network very
  // little and drag its output range around, so they are left out.
  int score_limit = 3000;

  // Network to generate with. Empty means the hand-written evaluation, which
  // is what the first generation of data has to use.
  std::string eval_file;

  // The SFEN of every position written, one per line, for training/verify.py
  // to hand back to the engine. Empty to skip it.
  std::string sfen_path;

  // Touching this file asks the run to finish the games in flight and stop.
  // Ctrl+C does the same. Either way the output stays usable and --append
  // picks up from there.
  std::string stop_file;
};

struct Report {
  int64_t games = 0;
  int64_t samples = 0;
  int64_t black_wins = 0;
  int64_t white_wins = 0;
  int64_t draws = 0;
  bool stopped_early = false;
  std::string error;
};

using Logger = std::function<void(const std::string&)>;

// Asks a run in progress to stop. Called from a signal handler, so it does
// nothing but set a flag.
void RequestStop();
bool StopRequested();

Report Run(const Config& config, const Logger& log);

}  // namespace luna::datagen
