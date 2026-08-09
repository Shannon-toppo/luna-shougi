#include "match/game.hpp"

#include <algorithm>
#include <array>
#include <sstream>

namespace luna::match {
namespace {

// The clocks, as USI wants them written.
struct Clocks {
  std::array<int64_t, kColorNb> remaining{};

  // Both clocks go out every move, so this does not depend on whose turn it
  // is; the engine reads its own off the command.
  std::string GoCommand(const TimeControl& tc) const {
    std::ostringstream go;
    go << "go";
    if (tc.movetime_ms > 0) {
      go << " movetime " << tc.movetime_ms;
    } else if (tc.depth > 0) {
      go << " depth " << tc.depth;
    } else if (tc.nodes > 0) {
      go << " nodes " << tc.nodes;
    } else {
      go << " btime " << std::max<int64_t>(remaining[kBlack], 0) << " wtime "
         << std::max<int64_t>(remaining[kWhite], 0);
      // byoyomi and increment are two ways of paying for the same thing and
      // no GUI sends both, so neither does this.
      if (tc.byoyomi_ms > 0) {
        go << " byoyomi " << tc.byoyomi_ms;
      } else if (tc.inc_ms > 0) {
        go << " binc " << tc.inc_ms << " winc " << tc.inc_ms;
      }
    }
    return go.str();
  }

  // Everything the engine could spend on this move without flagging.
  int64_t Allowance(Color us, const TimeControl& tc) const {
    if (tc.movetime_ms > 0) return tc.movetime_ms;
    if (!tc.Timed()) return kUntimedMoveLimitMs;
    return std::max<int64_t>(remaining[us], 0) + tc.byoyomi_ms;
  }

  // False when `us` has just overstepped.
  bool Charge(Color us, const TimeControl& tc, int64_t elapsed_ms) {
    if (tc.movetime_ms > 0) return elapsed_ms <= tc.movetime_ms + tc.margin_ms;
    if (!tc.Timed()) return true;

    remaining[us] -= elapsed_ms;
    if (remaining[us] < 0) {
      // What is left over came out of byoyomi, which refills every move. With
      // no byoyomi to fall back on, any overshoot at all is a lost game.
      if (-remaining[us] > tc.byoyomi_ms + tc.margin_ms) return false;
      remaining[us] = 0;
    }
    remaining[us] += tc.inc_ms;
    return true;
  }
};

// How much longer than its own allowance an engine is given before we stop
// waiting at all. Past this it is not late, it is hung, and there is nothing
// to gain from holding the whole match up for it.
constexpr int64_t kHangGraceMs = 5000;

GameRecord Finish(const Arbiter& arbiter, Outcome outcome, Reason reason, std::string message) {
  GameRecord record;
  record.outcome = outcome;
  record.reason = reason;
  record.start_sfen = arbiter.StartSfen();
  record.moves = arbiter.Moves();
  record.message = std::move(message);
  return record;
}

}  // namespace

std::string GameRecord::ToUsiLine() const {
  std::ostringstream line;
  line << "position sfen " << start_sfen;
  if (!moves.empty()) {
    line << " moves";
    for (const Move m : moves) line << ' ' << ToUsi(m);
  }
  line << " # " << ToString(outcome) << " by " << ToString(reason);
  if (!message.empty()) line << " (" << message << ')';
  return line.str();
}

GameRecord PlayGame(
    UsiClient& black, UsiClient& white, const Position& start, const TimeControl& tc, int max_ply) {
  Arbiter arbiter(start, max_ply);

  Clocks clocks;
  clocks.remaining[kBlack] = tc.main_ms;
  clocks.remaining[kWhite] = tc.main_ms;

  std::string error;
  if (!black.NewGame(error)) {
    return Finish(arbiter, LossFor(kBlack), Reason::kNoReply, error);
  }
  if (!white.NewGame(error)) {
    return Finish(arbiter, LossFor(kWhite), Reason::kNoReply, error);
  }

  while (!arbiter.Over()) {
    const Color us = arbiter.SideToMove();
    UsiClient& engine = us == kBlack ? black : white;

    const int64_t allowance = clocks.Allowance(us, tc);
    const std::chrono::milliseconds timeout(allowance + tc.margin_ms + kHangGraceMs);

    std::string token;
    int64_t elapsed_ms = 0;
    if (!engine.Go(
            arbiter.PositionCommand(), clocks.GoCommand(tc), timeout, token, elapsed_ms, error)) {
      return Finish(arbiter, LossFor(us), Reason::kNoReply, error);
    }

    if (!clocks.Charge(us, tc, elapsed_ms)) {
      return Finish(arbiter,
                    LossFor(us),
                    Reason::kTimeout,
                    engine.Name() + " took " + std::to_string(elapsed_ms) + "ms with " +
                        std::to_string(allowance) + "ms to spend");
    }

    if (token == "resign") {
      return Finish(arbiter, LossFor(us), Reason::kResign, {});
    }
    if (token == "win") {
      // 入玉宣言勝ち. A declaration that does not hold up loses the game,
      // which is what the rule says and also what keeps a buggy engine from
      // claiming its way to a good score.
      if (arbiter.CanDeclareWin()) {
        return Finish(arbiter, LossFor(Opponent(us)), Reason::kDeclaration, {});
      }
      return Finish(arbiter,
                    LossFor(us),
                    Reason::kIllegalMove,
                    engine.Name() + " declared a win it had not earned");
    }

    const Move move = MoveFromUsi(token);
    if (move.IsNone() || !arbiter.IsLegal(move)) {
      return Finish(arbiter, LossFor(us), Reason::kIllegalMove, engine.Name() + " played " + token);
    }
    arbiter.Play(move);
  }

  const Verdict verdict = arbiter.Status();
  return Finish(arbiter, verdict.outcome, verdict.reason, {});
}

}  // namespace luna::match
