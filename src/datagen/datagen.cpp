#include "datagen/datagen.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "core/movegen.hpp"
#include "datagen/sample.hpp"
#include "nnue/evaluate.hpp"
#include "search/search.hpp"

namespace luna::datagen {
namespace {

std::atomic<bool> g_stop{false};

// One self-play game's worth of positions, still waiting for the result that
// will label them.
struct Game {
  std::vector<Sample> samples;
  // Parallel to `samples`, and empty unless SFENs were asked for.
  std::vector<std::string> sfens;
  match::Outcome outcome = match::Outcome::kInProgress;
};

Move PickRandomMove(const MoveList& moves, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> pick(0, moves.Size() - 1);
  return moves[pick(rng)];
}

Game PlayGame(const Config& config, Search& search, uint64_t seed) {
  std::mt19937_64 rng(seed);
  match::Arbiter arbiter(Position(), config.max_ply);
  Position pos;

  for (int i = 0; i < config.opening_plies && !arbiter.Over(); ++i) {
    const Move m = PickRandomMove(arbiter.LegalMoves(), rng);
    arbiter.Play(m);
    pos.DoMove(m);
  }

  SearchLimits limits;
  limits.depth = config.depth;
  // No clock is set, so ComputeTimeBudget leaves the search unlimited and the
  // depth and node count are the only things that end it. Both apply, and
  // whichever is reached first stops the search. That is what makes a
  // generation run reproducible: same seed, same data, whatever the machine.
  limits.nodes = config.nodes;

  Game game;
  search.NewGame();

  while (!arbiter.Over()) {
    const SearchResult result = search.Think(pos, limits);
    if (result.best.IsNone()) break;

    // The score comes back from the side to move; everything stored is from
    // black, so that the trainer never has to think about whose turn it was.
    const int black_score = pos.SideToMove() == kBlack ? result.score : -result.score;

    const bool usable = !pos.InCheck() && !IsMateScore(result.score) &&
                        (black_score < config.score_limit && black_score > -config.score_limit);
    Sample sample;
    if (usable && MakeSample(pos, black_score, sample)) {
      game.samples.push_back(sample);
      if (!config.sfen_path.empty()) game.sfens.push_back(pos.ToSfen());
    }

    if (!arbiter.IsLegal(result.best)) break;
    arbiter.Play(result.best);
    pos.DoMove(result.best);
  }

  game.outcome = arbiter.Status().outcome;
  return game;
}

int8_t ResultOf(match::Outcome outcome) {
  if (outcome == match::Outcome::kBlackWin) return 1;
  if (outcome == match::Outcome::kWhiteWin) return -1;
  return 0;
}

std::vector<std::string> ReadLines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    // Blank lines and the '#' header a dump file starts with are not
    // positions.
    if (line.empty() || line.front() == '#') continue;
    lines.push_back(line);
  }
  return lines;
}

// One search per position in `config.label_path`, written out in the order
// they were read.
//
// Nothing is filtered. Self-play drops positions that are in check, scored as
// a mate, or past `score_limit`, because those teach a network very little and
// drag its output range about. Here they are exactly what is being looked at:
// the question a labelled dump answers is where in the search's own
// distribution an evaluation goes wrong, and cutting the awkward part out of
// the distribution first would answer a different one. The bands in
// training/baseline.py are where that gets sorted out.
//
// `result` is 0 on every sample, because a position lifted out of a search
// tree has no game around it to have been won or lost. That makes this output
// fine to measure against and wrong to train on: train.py mixes the result
// into its target, and would read every one of these as a draw.
Report RunLabels(const Config& config, const Logger& log) {
  Report report;

  const std::vector<std::string> lines = ReadLines(config.label_path);
  if (lines.empty()) {
    report.error = "no positions in " + config.label_path;
    return report;
  }
  if (log) log(std::to_string(lines.size()) + " positions to label");

  SampleWriter writer;
  if (!writer.Open(config.out_path, config.sfen_path, config.append)) {
    report.error = "cannot write " + config.out_path;
    return report;
  }

  // Filled in place so that the output keeps the order of the input, whatever
  // order the threads happen to finish in.
  std::vector<Sample> samples(lines.size());
  std::vector<char> done(lines.size(), 0);

  std::atomic<size_t> next{0};
  std::atomic<int64_t> finished{0};
  std::mutex log_mutex;

  const auto worker = [&]() {
    Search search;
    search.Tt().Resize(static_cast<size_t>(config.hash_mb));

    SearchLimits limits;
    limits.depth = config.depth;
    limits.nodes = config.nodes;

    while (true) {
      if (StopRequested()) break;
      if (!config.stop_file.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(config.stop_file, ec)) {
          RequestStop();
          break;
        }
      }
      const size_t index = next.fetch_add(1);
      if (index >= lines.size()) break;

      Position pos;
      if (!pos.SetSfen(lines[index])) continue;

      // Every position is searched on its own, so the table must not carry
      // anything over from the last one: these are unrelated positions, not a
      // game, and a stale entry would make the score depend on what happened
      // to be labelled before it.
      search.NewGame();
      const SearchResult result = search.Think(pos, limits);

      const int black_score = pos.SideToMove() == kBlack ? result.score : -result.score;
      if (!MakeSample(pos, black_score, samples[index])) continue;
      done[index] = 1;

      const int64_t count = finished.fetch_add(1) + 1;
      if (log && count % 500 == 0) {
        std::lock_guard<std::mutex> lock(log_mutex);
        log(std::to_string(count) + "/" + std::to_string(lines.size()) + " labelled");
      }
    }
  };

  const int threads = config.concurrency < 1 ? 1 : config.concurrency;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (std::thread& thread : pool) thread.join();

  // Both files come out of this one filtered list, so they stay line for line
  // with each other even when a position in the middle was dropped.
  std::vector<Sample> keep;
  std::vector<std::string> sfens;
  for (size_t i = 0; i < lines.size(); ++i) {
    if (!done[i]) continue;
    keep.push_back(samples[i]);
    if (!config.sfen_path.empty()) sfens.push_back(lines[i]);
  }

  if (!writer.Write(keep, sfens)) {
    report.error = "writing " + config.out_path + " failed";
    return report;
  }

  report.samples = static_cast<int64_t>(keep.size());
  report.skipped = static_cast<int64_t>(lines.size()) - report.samples;
  report.stopped_early = StopRequested();
  return report;
}

}  // namespace

void RequestStop() {
  g_stop.store(true, std::memory_order_relaxed);
}

bool StopRequested() {
  return g_stop.load(std::memory_order_relaxed);
}

Report Run(const Config& config, const Logger& log) {
  Report report;
  g_stop.store(false, std::memory_order_relaxed);

  if (!config.eval_file.empty()) {
    std::string error;
    if (!nnue::Load(config.eval_file, error)) {
      report.error = error;
      return report;
    }
    if (log) log("network: " + config.eval_file + " (" + nnue::SimdName() + ")");
  }

  if (!config.label_path.empty()) return RunLabels(config, log);

  SampleWriter writer;
  if (!writer.Open(config.out_path, config.sfen_path, config.append)) {
    report.error = "cannot write " + config.out_path;
    return report;
  }

  std::atomic<int> next_game{0};
  std::mutex report_mutex;
  std::atomic<bool> write_failed{false};

  const auto worker = [&]() {
    Search search;
    search.Tt().Resize(static_cast<size_t>(config.hash_mb));
    while (true) {
      if (StopRequested() || write_failed.load(std::memory_order_relaxed)) break;
      if (!config.stop_file.empty()) {
        std::error_code ec;
        if (std::filesystem::exists(config.stop_file, ec)) {
          RequestStop();
          break;
        }
      }
      const int index = next_game.fetch_add(1);
      if (index >= config.games) break;

      Game game = PlayGame(config, search, config.seed + static_cast<uint64_t>(index));
      const int8_t result = ResultOf(game.outcome);
      for (Sample& sample : game.samples) sample.result = result;

      if (!writer.Write(game.samples, game.sfens)) {
        write_failed.store(true, std::memory_order_relaxed);
        break;
      }

      std::lock_guard<std::mutex> lock(report_mutex);
      ++report.games;
      report.samples += static_cast<int64_t>(game.samples.size());
      if (result > 0)
        ++report.black_wins;
      else if (result < 0)
        ++report.white_wins;
      else
        ++report.draws;
      if (log && report.games % 20 == 0) {
        log(std::to_string(report.games) + "/" + std::to_string(config.games) + " games, " +
            std::to_string(report.samples) + " positions");
      }
    }
  };

  const int threads = config.concurrency < 1 ? 1 : config.concurrency;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (std::thread& thread : pool) thread.join();

  if (write_failed.load(std::memory_order_relaxed)) {
    report.error = "writing " + config.out_path + " failed";
  }
  report.stopped_early = StopRequested() && report.games < config.games;
  return report;
}

}  // namespace luna::datagen
