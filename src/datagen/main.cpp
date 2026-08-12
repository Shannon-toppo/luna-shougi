#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "datagen/datagen.hpp"

namespace {

constexpr const char* kUsage = R"(luna-datagen: generate NNUE training data by self-play.

  luna-datagen --out FILE [options]
  luna-datagen --out FILE --label POSITIONS.sfen [options]

Output:
  --out FILE             where to write the samples (required)
  --append               add to FILE instead of replacing it
  --sfen-out FILE        also write the SFEN of every sample, one per line,
                         for training/verify.py to read back

Search (default: --depth 6 --nodes 2000000):
  --depth N              depth per move, 0 for no depth limit
  --nodes N              node cap per move, 0 for no cap. Applies alongside
                         --depth; whichever comes first ends the search. The
                         cap is on by default because quiescence is not yet
                         pruned, and one middlegame position can otherwise
                         cost a hundred times what a quiet one does.
  --hash MB              transposition table per thread (default 32)

Labelling instead of self-play:
  --label FILE           search every SFEN in FILE once, one per line, and
                         write the scores as samples in the same order.
                         --games, --openings, --seed and --maxply do nothing
                         in this mode; --depth, --nodes and --concurrency
                         still apply. Blank lines and lines starting with '#'
                         are skipped, so the engine's EvalDump file can be fed
                         in with its header and its other columns cut off.

                         Nothing is filtered: positions in check, mate scores
                         and scores past --score-limit are all labelled, which
                         is the point -- this exists to measure an evaluation
                         on the positions a search really visits, not on the
                         tidy subset self-play keeps.

                         The result field is 0 on every sample, because these
                         positions have no game around them. Fine to measure
                         against, wrong to train on: train.py would read every
                         one of them as a draw.

Games:
  --games N              default 1000
  --openings N           random opening plies before the search takes over
                         (default 8)
  --seed N               opening seed (default 1)
  --maxply N             adjudicate a draw after this many moves (default 320)
  --concurrency N        games in parallel (default 1)
  --score-limit N        skip positions the search scores past this (default
                         3000)
  --eval-file PATH       generate with an NNUE network instead of the
                         hand-written evaluation

Stopping:
  --stop-file PATH       create this file to end the run cleanly; Ctrl+C does
                         the same. Either way the output stays usable.

                         To carry on, --append keeps the existing file, but it
                         does NOT continue the game sequence: games are always
                         numbered from 0 and the seed of game N is --seed + N,
                         so rerunning the same command replays every game it
                         already played and writes all of it a second time.
                         Advance --seed past the games that are done and lower
                         --games by the same amount.
)";

void OnInterrupt(int) {
  luna::datagen::RequestStop();
}

bool ParseInt64(const std::string& text, int64_t& out) {
  try {
    size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size()) return false;
    out = value;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  luna::datagen::Config config;
  const std::vector<std::string> args(argv + 1, argv + argc);

  if (args.empty()) {
    std::cout << kUsage;
    return 1;
  }

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& flag = args[i];
    if (flag == "-h" || flag == "--help") {
      std::cout << kUsage;
      return 0;
    }
    if (flag == "--append") {
      config.append = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      std::cerr << "luna-datagen: " << flag << " needs a value\n\n" << kUsage;
      return 1;
    }
    const std::string& value = args[++i];

    if (flag == "--out") {
      config.out_path = value;
      continue;
    }
    if (flag == "--eval-file") {
      config.eval_file = value;
      continue;
    }
    if (flag == "--stop-file") {
      config.stop_file = value;
      continue;
    }
    if (flag == "--sfen-out") {
      config.sfen_path = value;
      continue;
    }
    if (flag == "--label") {
      config.label_path = value;
      continue;
    }

    int64_t number = 0;
    if (!ParseInt64(value, number)) {
      std::cerr << "luna-datagen: " << flag << ": " << value << " is not a number\n";
      return 1;
    }
    if (flag == "--games")
      config.games = static_cast<int>(number);
    else if (flag == "--depth")
      config.depth = static_cast<int>(number);
    else if (flag == "--nodes")
      config.nodes = number;
    else if (flag == "--hash")
      config.hash_mb = static_cast<int>(number);
    else if (flag == "--openings")
      config.opening_plies = static_cast<int>(number);
    else if (flag == "--seed")
      config.seed = static_cast<uint64_t>(number);
    else if (flag == "--maxply")
      config.max_ply = static_cast<int>(number);
    else if (flag == "--concurrency")
      config.concurrency = static_cast<int>(number);
    else if (flag == "--score-limit")
      config.score_limit = static_cast<int>(number);
    else {
      std::cerr << "luna-datagen: unknown option " << flag << "\n\n" << kUsage;
      return 1;
    }
  }

  if (config.out_path.empty()) {
    std::cerr << "luna-datagen: --out is required\n\n" << kUsage;
    return 1;
  }
  if (config.games < 1 || config.concurrency < 1 || config.max_ply < 1) {
    std::cerr << "luna-datagen: --games, --concurrency and --maxply have to be at least 1\n";
    return 1;
  }

  std::signal(SIGINT, OnInterrupt);
#ifdef SIGTERM
  std::signal(SIGTERM, OnInterrupt);
#endif

  const std::string per_move = (config.depth > 0 ? "depth " + std::to_string(config.depth) : "") +
                               (config.depth > 0 && config.nodes > 0 ? ", " : "") +
                               (config.nodes > 0 ? std::to_string(config.nodes) + " nodes" : "");

  if (config.label_path.empty()) {
    std::cout << config.games << " games, " << per_move << " per move, " << config.opening_plies
              << " opening plies, seed " << config.seed << ", concurrency " << config.concurrency
              << "\n";
  } else {
    std::cout << "labelling " << config.label_path << ", " << per_move << " per position"
              << ", concurrency " << config.concurrency << "\n";
  }
  std::cout << "writing " << config.out_path << (config.append ? " (appending)" : "") << "\n\n";
  std::cout.flush();

  const luna::datagen::Report report = luna::datagen::Run(config, [](const std::string& line) {
    std::cout << line << '\n';
    std::cout.flush();
  });

  if (!report.error.empty()) {
    std::cerr << "luna-datagen: " << report.error << '\n';
    return 1;
  }

  if (config.label_path.empty()) {
    std::cout << '\n'
              << report.games << " games, " << report.samples << " positions written\n"
              << "black " << report.black_wins << " - white " << report.white_wins << " - draw "
              << report.draws << '\n';
    if (report.stopped_early) {
      std::cout << "stopped early; rerun with --append to carry on\n";
    }
    return 0;
  }

  std::cout << '\n' << report.samples << " positions written\n";
  if (report.skipped > 0) {
    // Loud, because a labelling run that dropped part of its input still
    // produces a .bin that looks perfectly reasonable.
    std::cout << report.skipped << " lines produced nothing (unparsable, or not a full board)\n";
  }
  if (report.stopped_early) {
    std::cout << "stopped early; the positions that were reached are all in the output\n";
  }
  return 0;
}
