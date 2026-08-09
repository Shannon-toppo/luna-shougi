#include "match/match.hpp"

#include <algorithm>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

#include "core/movegen.hpp"

namespace luna::match {
namespace {

// One opening played twice, once with each engine as black.
constexpr int kGamesPerRound = 2;

// A game as it comes back from a worker, kept with its index so the report
// reads in playing order however the threads happened to interleave.
struct FinishedGame {
  int index = 0;
  GameRecord record;
  bool first_was_black = false;
};

void Tally(MatchScore& score, const FinishedGame& game) {
  const Outcome first_wins = game.first_was_black ? Outcome::kBlackWin : Outcome::kWhiteWin;
  const Outcome first_loses = game.first_was_black ? Outcome::kWhiteWin : Outcome::kBlackWin;
  if (game.record.outcome == first_wins) {
    ++score.wins;
  } else if (game.record.outcome == first_loses) {
    ++score.losses;
  } else {
    ++score.draws;
  }
}

std::string Describe(const FinishedGame& game,
                     const std::string& first,
                     const std::string& second) {
  const std::string black = game.first_was_black ? first : second;
  const std::string white = game.first_was_black ? second : first;

  std::ostringstream line;
  line << "game " << game.index + 1 << ": " << black << " (b) vs " << white << " (w), "
       << game.record.moves.size() << " moves, " << ToString(game.record.outcome) << " by "
       << ToString(game.record.reason);
  if (!game.record.message.empty()) line << " [" << game.record.message << ']';
  return line.str();
}

}  // namespace

Position RandomOpening(uint64_t seed, int plies) {
  Position pos;
  std::mt19937_64 rng(seed);
  for (int i = 0; i < plies; ++i) {
    MoveList moves;
    movegen::GenerateLegal(pos, moves);
    if (moves.Empty()) break;
    std::uniform_int_distribution<int> pick(0, moves.Size() - 1);
    const Move m = moves[pick(rng)];
    pos.DoMove(m);
  }
  return pos;
}

MatchReport RunMatch(const MatchConfig& config,
                     const std::function<void(const std::string&)>& log) {
  MatchReport report;

  const int rounds = std::max(1, (config.games + kGamesPerRound - 1) / kGamesPerRound);
  const int total = rounds * kGamesPerRound;
  const int workers = std::clamp(config.concurrency, 1, rounds);

  std::vector<FinishedGame> finished(static_cast<size_t>(total));
  std::mutex mutex;
  MatchScore running;
  std::string first_error;

  const auto play_rounds = [&](int worker) {
    // Each worker owns its own pair of engine processes for the whole run.
    // Restarting them per game would measure process start-up as much as
    // playing strength, and would throw away the transposition table every
    // time for no reason.
    UsiClient first;
    UsiClient second;
    std::string error;
    if (!first.Start(config.first, error) || !second.Start(config.second, error)) {
      std::lock_guard<std::mutex> lock(mutex);
      if (first_error.empty()) first_error = error;
      return;
    }

    for (int round = worker; round < rounds; round += workers) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        if (!first_error.empty()) break;
      }

      const Position start =
          RandomOpening(config.seed + static_cast<uint64_t>(round), config.opening_plies);

      for (int leg = 0; leg < kGamesPerRound; ++leg) {
        const bool first_is_black = leg == 0;
        FinishedGame game;
        game.index = round * kGamesPerRound + leg;
        game.first_was_black = first_is_black;
        game.record = first_is_black ? PlayGame(first, second, start, config.tc, config.max_ply)
                                     : PlayGame(second, first, start, config.tc, config.max_ply);

        std::lock_guard<std::mutex> lock(mutex);
        finished[static_cast<size_t>(game.index)] = game;
        Tally(running, game);
        if (log) {
          log(Describe(game, first.Name(), second.Name()));
          log("  " + FormatScore(running) + ", " + FormatElo(running));
        }
      }
    }

    first.Quit();
    second.Quit();
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(workers - 1));
  for (int worker = 1; worker < workers; ++worker) {
    threads.emplace_back(play_rounds, worker);
  }
  play_rounds(0);
  for (std::thread& thread : threads) thread.join();

  report.error = first_error;
  report.score = running;
  for (FinishedGame& game : finished) {
    if (game.record.outcome == Outcome::kInProgress && game.record.moves.empty() &&
        game.record.start_sfen.empty()) {
      continue;  // never played, because the match was cut short
    }
    report.games.push_back(std::move(game.record));
  }
  return report;
}

}  // namespace luna::match
