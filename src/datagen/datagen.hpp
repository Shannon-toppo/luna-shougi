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

  // Fixed depth is reproducible, and reproducibility is what makes two
  // generation runs on different machines comparable.
  //
  // But a fixed depth does not bound the work. Quiescence has no pruning
  // beyond the stand-pat cutoff yet, so a middlegame with a full hand can
  // spend a hundred times what a quiet position spends at the same depth:
  // measured at depth 3, 2.5k nodes from the starting position against 355k
  // in a middlegame, reaching 26 plies past the horizon. Left alone, one
  // position like that stalls a run.
  //
  // So `nodes` is a cap that applies alongside `depth`, whichever comes
  // first, and it is on by default. It stays reproducible: the node count of
  // a given position at a given depth is the same everywhere. Set it to 0 to
  // take the brakes off, or set `depth` to 0 to search by node count alone.
  int depth = 6;
  int64_t nodes = 2'000'000;

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
  // Ctrl+C does the same. Either way the output stays usable.
  //
  // Carrying on afterwards is not automatic. `append` keeps the file, but the
  // games are numbered from 0 every time and game N is played with seed
  // `seed + N`, so the same command run twice plays the same games twice and
  // writes them both. The caller has to move `seed` past what is already done
  // and take the same amount off `games`.
  std::string stop_file;

  // Positions to label instead of self-playing, one SFEN per line. Set, and
  // `games`, `opening_plies`, `seed` and `max_ply` are all unused: there are
  // no games, only the positions in this file, each searched once at `depth`
  // and written out with the score that came back.
  //
  // This is what turns a search dump (src/search/evaldump.hpp) into something
  // training/baseline.py can read. Self-play cannot do it, because the whole
  // point of the dump is that it holds positions no self-play game reaches.
  std::string label_path;
};

struct Report {
  int64_t games = 0;
  int64_t samples = 0;
  int64_t black_wins = 0;
  int64_t white_wins = 0;
  int64_t draws = 0;
  bool stopped_early = false;
  std::string error;

  // Label mode only: lines that were read but produced no sample, because the
  // SFEN would not parse or the position was not the full 40 pieces MakeSample
  // needs. Worth reporting rather than swallowing -- a labelling run that
  // quietly dropped half its input still writes a plausible .bin.
  int64_t skipped = 0;
};

using Logger = std::function<void(const std::string&)>;

// Asks a run in progress to stop. Called from a signal handler, so it does
// nothing but set a flag.
void RequestStop();
bool StopRequested();

// Self-play, or -- when `config.label_path` is set -- one search per position
// in that file.
Report Run(const Config& config, const Logger& log);

}  // namespace luna::datagen
