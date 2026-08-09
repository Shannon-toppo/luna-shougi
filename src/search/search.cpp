#include "search/search.hpp"

#include <algorithm>
#include <sstream>

#include "core/movegen.hpp"

namespace luna {
namespace {

// Ordering bands, from best to worst. Captures and promotions sit above the
// killers, killers above every quiet move, and the transposition table move
// above all of them: it is the one move already known to be good here.
constexpr int kTtMoveScore = 1 << 24;
constexpr int kTacticalBase = 1 << 20;
constexpr int kKillerScore[2] = {kTacticalBase - 1, kTacticalBase - 2};

// History has to stay below the killers so it can only reorder quiet moves
// among themselves.
constexpr int kMaxHistory = kTacticalBase - 3;

// Time is only read every so many nodes; the clock call is not free.
constexpr int64_t kTimeCheckMask = 1023;

// A new iteration that cannot finish is wasted, so deepening stops well
// before the soft limit rather than at it.
constexpr int64_t kIterationStartNumerator = 6;
constexpr int64_t kIterationStartDenominator = 10;

// A move is quiet when it neither captures nor promotes. Must be called
// before the move is played.
bool IsQuiet(const Position& pos, Move m) {
  if (m.IsDrop()) return true;
  return !m.IsPromotion() && pos.PieceOn(m.To()) == kNoPiece;
}

}  // namespace

Search::Search() {
  stack_.resize(kMaxPly + 1);
  history_.assign(static_cast<size_t>(kColorNb) * kHistoryOriginNb * kSquareNb, 0);
}

void Search::NewGame() {
  tt_.Clear();
  std::fill(history_.begin(), history_.end(), 0);
  for (Node& node : stack_) node.killers.fill(Move::None());
}

int& Search::HistoryOf(Color us, Move m) {
  const int origin = m.IsDrop() ? kSquareNb + m.DroppedType() : m.From();
  return history_[(static_cast<size_t>(us) * kHistoryOriginNb + origin) * kSquareNb + m.To()];
}

void Search::UpdateKillers(int ply, Move m) {
  std::array<Move, 2>& killers = stack_[ply].killers;
  if (killers[0] == m) return;
  killers[1] = killers[0];
  killers[0] = m;
}

void Search::UpdateHistory(Color us, Move m, int depth) {
  int& entry = HistoryOf(us, m);
  entry = std::min(entry + depth * depth, kMaxHistory);
}

void Search::ScoreMoves(const Position& pos, int ply, Move tt_move) {
  Node& node = stack_[ply];
  const Color us = pos.SideToMove();

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = node.moves[i];
    if (m == tt_move) {
      node.scores[i] = kTtMoveScore;
      continue;
    }

    if (!IsQuiet(pos, m)) {
      const Piece captured = pos.PieceOn(m.To());
      const PieceType mover = TypeOf(pos.PieceOn(m.From()));
      int tactical = 0;
      if (captured != kNoPiece) {
        // MVV-LVA: the fatter the victim and the cheaper the attacker, the
        // more likely the capture is to be worth searching first.
        tactical += eval::CaptureValue(TypeOf(captured)) * 16 - eval::PieceValue(mover);
      }
      if (m.IsPromotion()) tactical += eval::PromotionGain(mover);
      // The king is priced high enough to swamp MVV-LVA, so a capture by the
      // king can come out negative. It is still a capture and belongs in this
      // band, just at the bottom of it.
      node.scores[i] = kTacticalBase + std::max(tactical, 1);
    } else if (m == node.killers[0]) {
      node.scores[i] = kKillerScore[0];
    } else if (m == node.killers[1]) {
      node.scores[i] = kKillerScore[1];
    } else {
      node.scores[i] = HistoryOf(us, m);
    }
  }
}

Move Search::PickMove(int ply, int index) {
  Node& node = stack_[ply];
  int best = index;
  for (int i = index + 1; i < node.moves.Size(); ++i) {
    if (node.scores[i] > node.scores[best]) best = i;
  }
  if (best != index) {
    std::swap(node.moves[index], node.moves[best]);
    std::swap(node.scores[index], node.scores[best]);
  }
  return node.moves[index];
}

void Search::UpdatePv(int ply, Move m) {
  pv_[ply][0] = m;
  const int tail = std::min(pv_length_[ply + 1], kMaxPly - ply - 1);
  for (int i = 0; i < tail; ++i) pv_[ply][i + 1] = pv_[ply + 1][i];
  pv_length_[ply] = tail + 1;
}

int64_t Search::ElapsedMs() const {
  const auto elapsed = std::chrono::steady_clock::now() - start_;
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

bool Search::ShouldStop() {
  if (stop_) return true;
  if (limits_.nodes > 0 && node_count_ >= limits_.nodes) {
    stop_ = true;
  } else if (!budget_.Unlimited() && (node_count_ & kTimeCheckMask) == 0 &&
             ElapsedMs() >= budget_.hard_ms) {
    stop_ = true;
  }
  return stop_;
}

int Search::Quiescence(Position& pos, int alpha, int beta, int ply) {
  ++node_count_;
  if (ShouldStop()) return eval::kDraw;

  seldepth_ = std::max(seldepth_, ply);
  if (ply >= kMaxPly) return eval::Evaluate(pos);

  const bool in_check = pos.InCheck();

  // Standing pat: the side to move is not obliged to capture, so the static
  // evaluation is a lower bound on what it can get. That reasoning does not
  // hold in check, where every move is forced to be an evasion.
  int best = -eval::kInfinite;
  if (!in_check) {
    best = eval::Evaluate(pos);
    if (best >= beta) return best;
    alpha = std::max(alpha, best);
  }

  Node& node = stack_[ply];
  node.moves.Clear();
  if (in_check) {
    movegen::GenerateLegal(pos, node.moves);
    if (node.moves.Empty()) return -eval::kMate + ply;
  } else {
    movegen::GenerateLegalCaptures(pos, node.moves);
  }

  ScoreMoves(pos, ply, Move::None());

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = PickMove(ply, i);
    pos.DoMove(m);
    const int score = -Quiescence(pos, -beta, -alpha, ply + 1);
    pos.UndoMove();
    if (stop_) return eval::kDraw;

    if (score > best) {
      best = score;
      if (score > alpha) {
        alpha = score;
        if (score >= beta) break;
      }
    }
  }

  return best;
}

int Search::AlphaBeta(Position& pos, int depth, int alpha, int beta, int ply) {
  ++node_count_;
  pv_length_[ply] = 0;
  if (ShouldStop()) return eval::kDraw;

  if (ply > 0) {
    if (pos.IsRepetition()) return eval::kDraw;

    // Mate distance pruning. A mate found here can be no shorter than `ply`
    // plies, so a window already promising something better is unreachable.
    alpha = std::max(alpha, -eval::kMate + ply);
    beta = std::min(beta, eval::kMate - ply - 1);
    if (alpha >= beta) return alpha;
  }

  if (ply >= kMaxPly) return eval::Evaluate(pos);

  // Being in check means the position is unsettled and the reply is nearly
  // forced, so it is cheap to look one ply further. This also keeps the
  // quiescence search from ever being entered while in check at depth 0.
  const bool in_check = pos.InCheck();
  if (in_check && depth < kMaxDepth) ++depth;

  if (depth <= 0) return Quiescence(pos, alpha, beta, ply);

  TtProbe probe;
  Move tt_move;
  if (tt_.Probe(pos.Key(), ply, probe)) {
    tt_move = probe.move;
    // The stored bound only answers this window if it was proven at least as
    // deeply. The root is excluded so that a PV is always produced.
    if (ply > 0 && probe.depth >= depth) {
      if (probe.bound == Bound::kExact) return probe.value;
      if (probe.bound == Bound::kLower && probe.value >= beta) return probe.value;
      if (probe.bound == Bound::kUpper && probe.value <= alpha) return probe.value;
    }
  }

  Node& node = stack_[ply];
  node.moves.Clear();
  movegen::GenerateLegal(pos, node.moves);
  // Shogi has no stalemate: no legal move means mated, whether in check or not.
  if (node.moves.Empty()) return -eval::kMate + ply;

  ScoreMoves(pos, ply, tt_move);

  const Color us = pos.SideToMove();
  int best = -eval::kInfinite;
  Move best_move;
  Bound bound = Bound::kUpper;

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = PickMove(ply, i);
    const bool quiet = IsQuiet(pos, m);

    pos.DoMove(m);
    const int score = -AlphaBeta(pos, depth - 1, -beta, -alpha, ply + 1);
    pos.UndoMove();
    if (stop_) return eval::kDraw;

    if (score <= best) continue;
    best = score;
    best_move = m;
    if (score <= alpha) continue;

    alpha = score;
    bound = Bound::kExact;
    UpdatePv(ply, m);

    if (score >= beta) {
      bound = Bound::kLower;
      // Quiet moves that cause a cutoff are worth trying early elsewhere;
      // captures are already ordered by what they win.
      if (quiet) {
        UpdateKillers(ply, m);
        UpdateHistory(us, m, depth);
      }
      break;
    }
  }

  tt_.Store(pos.Key(), ply, depth, best, bound, best_move);
  return best;
}

std::string Search::BuildInfoLine(const SearchResult& result) const {
  std::ostringstream info;
  info << "info depth " << result.depth << " seldepth " << result.seldepth << " time "
       << result.time_ms << " nodes " << result.nodes;
  if (result.time_ms > 0) info << " nps " << result.nodes * 1000 / result.time_ms;
  info << " hashfull " << tt_.HashFull();

  if (IsMateScore(result.score)) {
    info << " score mate " << MateDistance(result.score);
  } else {
    info << " score cp " << result.score;
  }

  if (!result.pv.empty()) {
    info << " pv";
    for (const Move m : result.pv) info << ' ' << ToUsi(m);
  }
  return info.str();
}

SearchResult Search::Think(Position& pos, const SearchLimits& limits) {
  limits_ = limits;
  budget_ = ComputeTimeBudget(limits, pos.SideToMove());
  start_ = std::chrono::steady_clock::now();
  node_count_ = 0;
  seldepth_ = 0;
  stop_ = false;

  tt_.NewSearch();
  // Killers belong to a position, which the opponent's reply has changed.
  // History is about which moves tend to work and only needs to fade.
  for (Node& node : stack_) node.killers.fill(Move::None());
  for (int& entry : history_) entry /= 2;

  SearchResult result;
  MoveList root_moves;
  movegen::GenerateLegal(pos, root_moves);
  if (root_moves.Empty()) return result;

  // Whatever happens from here on, there is a move to return.
  result.best = root_moves[0];

  const int max_depth = limits.depth > 0 ? std::min(limits.depth, kMaxDepth) : kMaxDepth;
  for (int depth = 1; depth <= max_depth; ++depth) {
    const int score = AlphaBeta(pos, depth, -eval::kInfinite, eval::kInfinite, 0);

    // An aborted iteration searched only part of the root moves, so its score
    // is not comparable with the completed one before it. The exception is the
    // very first iteration, whose partial result is all there is.
    const bool usable = !stop_ || (depth == 1 && pv_length_[0] > 0);
    if (usable) {
      result.score = score;
      result.depth = depth;
      result.seldepth = seldepth_;
      result.nodes = node_count_;
      result.time_ms = ElapsedMs();
      result.pv.assign(pv_[0].begin(), pv_[0].begin() + pv_length_[0]);
      if (!result.pv.empty()) result.best = result.pv.front();
      if (info_sink_) info_sink_(BuildInfoLine(result));
    }

    if (stop_) break;
    // A forced mate is the end of the line; searching deeper cannot improve it.
    if (score >= kMateInMaxPly) break;
    if (!budget_.Unlimited() && ElapsedMs() * kIterationStartDenominator >=
                                    budget_.soft_ms * kIterationStartNumerator) {
      break;
    }
  }

  result.nodes = node_count_;
  result.time_ms = ElapsedMs();
  return result;
}

}  // namespace luna
