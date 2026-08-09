#include "datagen/datagen.hpp"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <random>
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
  limits.nodes = config.nodes;
  // No clock is set, so ComputeTimeBudget leaves the search unlimited and the
  // depth or node count is the only thing that ends it. That is what makes a
  // generation run reproducible: same seed, same data, whatever the machine.
  if (config.nodes > 0) limits.depth = 0;

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
