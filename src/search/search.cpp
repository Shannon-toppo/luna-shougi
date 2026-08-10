#include "search/search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <thread>

#include "core/movegen.hpp"
#include "nnue/evaluate.hpp"
#include "search/see.hpp"

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

// How far short of alpha a capture may fall and still be searched. Material is
// not the only thing a move changes, and one move's worth of positional swing
// is about a pawn.
constexpr int kDeltaMargin = 200;

// Reverse futility: a position this far above beta is not going to fall back
// under it within `depth` moves, so there is nothing to search for. The margin
// is per ply of remaining depth and is deliberately wider than a pawn, because
// a shogi position can swing hard on a single drop.
constexpr int kReverseFutilityMaxDepth = 6;
constexpr int kReverseFutilityMargin = 190;

// Null move pruning: give the opponent a free move and see whether the
// position still beats beta. Shogi barely has zugzwang — a captured piece
// comes back as a drop, so a side with anything in hand always has a move
// worth making — which is why passing is a fair test of "I am winning here
// even a tempo down". It is still wrong in check, where the free move would be
// taking the king.
constexpr int kNullMoveMinDepth = 2;
constexpr int kNullMoveBaseReduction = 2;
constexpr int kNullMoveDepthDivisor = 6;

// Futility: a quiet move at low depth that leaves the static evaluation this
// far below alpha is not going to make up the difference.
constexpr int kFutilityMaxDepth = 6;
constexpr int kFutilityMargin = 180;

// Late move pruning: how many quiet moves are searched at each depth before
// the rest are taken on trust from the move ordering. Indexed by depth.
constexpr int kLateMoveMaxDepth = 6;
constexpr int kLateMoveCount[kLateMoveMaxDepth + 1] = {0, 6, 9, 14, 21, 30, 41};

// Captures that lose this much material are not worth a full search at low
// depth. Scaled by depth so a shallow node prunes harder.
constexpr int kSeePruningMaxDepth = 6;
constexpr int kSeePruningMargin = 100;

// Late move reductions start here: deep enough for a reduction to save
// anything, and late enough in the move list that the ordering has already
// been given its chance.
constexpr int kLmrMinDepth = 3;
constexpr int kLmrMinMoveIndex = 3;

// Reduction table, in plies, indexed by depth and move number. The shape is
// the one every engine has converged on: logarithmic in both, so a deep node
// reduces more than a shallow one and the tail of a long move list is cut
// harder than its head.
constexpr int kReductionDepths = kMaxDepth + 1;
constexpr int kReductionMoves = 64;

const std::array<std::array<uint8_t, kReductionMoves>, kReductionDepths>& Reductions() {
  static const auto table = [] {
    std::array<std::array<uint8_t, kReductionMoves>, kReductionDepths> t{};
    for (int depth = 1; depth < kReductionDepths; ++depth) {
      for (int move = 1; move < kReductionMoves; ++move) {
        const double r = 0.75 + std::log(depth) * std::log(move) / 2.25;
        t[depth][move] = static_cast<uint8_t>(r);
      }
    }
    return t;
  }();
  return table;
}

int Reduction(int depth, int move_index) {
  const int d = std::clamp(depth, 0, kReductionDepths - 1);
  const int m = std::clamp(move_index, 0, kReductionMoves - 1);
  return Reductions()[d][m];
}

// A move is quiet when it neither captures nor promotes. Must be called
// before the move is played.
bool IsQuiet(const Position& pos, Move m) {
  if (m.IsDrop()) return true;
  return !m.IsPromotion() && pos.PieceOn(m.To()) == kNoPiece;
}

// The most `m` can add to the material count: what it captures plus what it
// gains by promoting. Must be called before the move is played.
int MaterialGain(const Position& pos, Move m) {
  if (m.IsDrop()) return 0;
  int gain = 0;
  const Piece captured = pos.PieceOn(m.To());
  if (captured != kNoPiece) gain += eval::CaptureValue(TypeOf(captured));
  if (m.IsPromotion()) gain += eval::PromotionGain(TypeOf(pos.PieceOn(m.From())));
  return gain;
}

}  // namespace

// One thread's search. Everything a search touches that is not the
// transposition table lives here, one copy per thread, so the only memory two
// threads write to in common is the table.
class Search::Worker {
 public:
  Worker(Search& search, int id) : search_(search), id_(id) {
    stack_.resize(kMaxPly + 1);
    history_.assign(static_cast<size_t>(kColorNb) * kHistoryOriginNb * kSquareNb, 0);
  }

  void NewGame() {
    std::fill(history_.begin(), history_.end(), 0);
    for (Node& node : stack_) node.killers.fill(Move::None());
  }

  // Sets up for one search of `root`, which is copied.
  void Prepare(const Position& root, Move fallback) {
    pos_ = root;
    nodes_.store(0, std::memory_order_relaxed);
    local_nodes_ = 0;
    seldepth_ = 0;
    result_ = SearchResult{};
    result_.best = fallback;

    // Killers belong to a position, which the opponent's reply has changed.
    // History is about which moves tend to work and only needs to fade.
    for (Node& node : stack_) node.killers.fill(Move::None());
    for (int& entry : history_) entry /= 2;
  }

  // Iterative deepening, up to and including `max_depth`.
  void Iterate(int max_depth);

  int64_t Nodes() const { return nodes_.load(std::memory_order_relaxed); }
  const SearchResult& Result() const { return result_; }

 private:
  // Per-ply scratch space. Kept off the call stack: a MoveList plus its scores
  // is over 6KB, and kMaxPly of those would not fit in a thread's stack.
  struct Node {
    MoveList moves;
    std::array<int, kMaxMoves> scores{};
    std::array<Move, 2> killers{};
  };

  // Drops use kSquareNb + piece type as their origin index.
  static constexpr int kHistoryOriginNb = kSquareNb + kPieceTypeNb;

  int AlphaBeta(int depth, int alpha, int beta, int ply, bool null_ok);
  int Quiescence(int alpha, int beta, int ply);

  void ScoreMoves(int ply, Move tt_move);
  Move PickMove(int ply, int index);
  void UpdateKillers(int ply, Move m);
  void UpdateHistory(Color us, Move m, int depth);
  int& HistoryOf(Color us, Move m);
  void UpdatePv(int ply, Move m);

  bool ShouldStop();
  bool Stopped() const { return search_.stop_.load(std::memory_order_relaxed); }

  Search& search_;
  int id_;
  Position pos_;

  std::vector<Node> stack_;
  std::vector<int> history_;
  // Triangular PV table. One extra ply so that a node at the deepest ply can
  // still read its (always empty) child entry.
  std::array<std::array<Move, kMaxPly>, kMaxPly + 1> pv_{};
  std::array<int, kMaxPly + 1> pv_length_{};

  // Published so that the main thread can total up the search's node count
  // while the others are still running; `local_nodes_` is the copy the hot
  // path actually counts with.
  std::atomic<int64_t> nodes_{0};
  int64_t local_nodes_ = 0;
  int seldepth_ = 0;
  SearchResult result_;
};

int& Search::Worker::HistoryOf(Color us, Move m) {
  const int origin = m.IsDrop() ? kSquareNb + m.DroppedType() : m.From();
  return history_[(static_cast<size_t>(us) * kHistoryOriginNb + origin) * kSquareNb + m.To()];
}

void Search::Worker::UpdateKillers(int ply, Move m) {
  std::array<Move, 2>& killers = stack_[ply].killers;
  if (killers[0] == m) return;
  killers[1] = killers[0];
  killers[0] = m;
}

void Search::Worker::UpdateHistory(Color us, Move m, int depth) {
  int& entry = HistoryOf(us, m);
  entry = std::min(entry + depth * depth, kMaxHistory);
}

void Search::Worker::ScoreMoves(int ply, Move tt_move) {
  Node& node = stack_[ply];
  const Color us = pos_.SideToMove();

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = node.moves[i];
    if (m == tt_move) {
      node.scores[i] = kTtMoveScore;
      continue;
    }

    if (!IsQuiet(pos_, m)) {
      const Piece captured = pos_.PieceOn(m.To());
      const PieceType mover = TypeOf(pos_.PieceOn(m.From()));
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

Move Search::Worker::PickMove(int ply, int index) {
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

void Search::Worker::UpdatePv(int ply, Move m) {
  pv_[ply][0] = m;
  const int tail = std::min(pv_length_[ply + 1], kMaxPly - ply - 1);
  for (int i = 0; i < tail; ++i) pv_[ply][i + 1] = pv_[ply + 1][i];
  pv_length_[ply] = tail + 1;
}

bool Search::Worker::ShouldStop() {
  if (Stopped()) return true;

  // Reading the clock, and summing every thread's node count, are both too
  // expensive to do per node. The exception is a node limit with a single
  // thread: that is what makes a datagen run reproducible, so it is checked
  // exactly rather than every 1024 nodes.
  const bool solo = search_.workers_.size() == 1;
  const bool sample = (local_nodes_ & kTimeCheckMask) == 0;

  if (search_.limits_.nodes > 0 && (solo || sample)) {
    const int64_t counted = solo ? local_nodes_ : search_.TotalNodes();
    if (counted >= search_.limits_.nodes) {
      search_.Stop();
      return true;
    }
  }
  if (sample && !search_.budget_.Unlimited() &&
      search_.ElapsedMs() >= search_.budget_.hard_ms) {
    search_.Stop();
    return true;
  }
  return false;
}

int Search::Worker::Quiescence(int alpha, int beta, int ply) {
  ++local_nodes_;
  nodes_.store(local_nodes_, std::memory_order_relaxed);
  if (ShouldStop()) return eval::kDraw;

  seldepth_ = std::max(seldepth_, ply);
  if (ply >= kMaxPly) return eval::Evaluate(pos_);

  const bool in_check = pos_.InCheck();

  // Standing pat: the side to move is not obliged to capture, so the static
  // evaluation is a lower bound on what it can get. That reasoning does not
  // hold in check, where every move is forced to be an evasion.
  int best = -eval::kInfinite;
  int stand_pat = 0;
  if (!in_check) {
    stand_pat = eval::Evaluate(pos_);
    best = stand_pat;
    if (best >= beta) return best;
    alpha = std::max(alpha, best);
  }

  Node& node = stack_[ply];
  node.moves.Clear();
  if (in_check) {
    movegen::GenerateLegal(pos_, node.moves);
    if (node.moves.Empty()) return -eval::kMate + ply;
  } else {
    movegen::GenerateLegalCaptures(pos_, node.moves);
  }

  ScoreMoves(ply, Move::None());

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = PickMove(ply, i);

    // Neither of these applies in check, where the move is not being played
    // for what it wins but because the king has to get out of the way.
    if (!in_check) {
      // Delta pruning. Winning the piece outright, promotion included, still
      // leaves the side to move short of alpha by more than one move's worth
      // of positional swing, so the move cannot rescue the position.
      if (stand_pat + MaterialGain(pos_, m) + kDeltaMargin <= alpha) continue;

      // Captures that lose the exchange are the whole reason an unpruned
      // quiescence search explodes: every piece in hand offers another way to
      // start a losing trade, and each one drags the search a ply deeper.
      if (!SeeGe(pos_, m, 0)) continue;
    }

    pos_.DoMove(m);
    const int score = -Quiescence(-beta, -alpha, ply + 1);
    pos_.UndoMove();
    if (Stopped()) return eval::kDraw;

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

int Search::Worker::AlphaBeta(int depth, int alpha, int beta, int ply, bool null_ok) {
  // A window wider than one point means this node can still improve the
  // principal variation, and everything below is more careful there: the
  // score of a PV node is used as a score, while the score of any other node
  // only has to be on the right side of a bound.
  const bool pv_node = beta - alpha > 1;

  ++local_nodes_;
  nodes_.store(local_nodes_, std::memory_order_relaxed);
  pv_length_[ply] = 0;
  if (ShouldStop()) return eval::kDraw;

  if (ply > 0) {
    if (pos_.IsRepetition()) return eval::kDraw;

    // Mate distance pruning. A mate found here can be no shorter than `ply`
    // plies, so a window already promising something better is unreachable.
    alpha = std::max(alpha, -eval::kMate + ply);
    beta = std::min(beta, eval::kMate - ply - 1);
    if (alpha >= beta) return alpha;
  }

  if (ply >= kMaxPly) return eval::Evaluate(pos_);

  // Being in check means the position is unsettled and the reply is nearly
  // forced, so it is cheap to look one ply further. This also keeps the
  // quiescence search from ever being entered while in check at depth 0.
  const bool in_check = pos_.InCheck();
  if (in_check && depth < kMaxDepth) ++depth;

  if (depth <= 0) return Quiescence(alpha, beta, ply);

  TtProbe probe;
  Move tt_move;
  if (search_.tt_.Probe(pos_.Key(), ply, probe)) {
    tt_move = probe.move;
    // The stored bound only answers this window if it was proven at least as
    // deeply. The root is excluded so that a PV is always produced.
    if (ply > 0 && probe.depth >= depth) {
      if (probe.bound == Bound::kExact) return probe.value;
      if (probe.bound == Bound::kLower && probe.value >= beta) return probe.value;
      if (probe.bound == Bound::kUpper && probe.value <= alpha) return probe.value;
    }
  }

  // Pruning that replaces a search with a guess about the static evaluation
  // only belongs where a guess is enough: outside the principal variation,
  // outside check, and outside a window that is already about a mate, where
  // material margins say nothing.
  const bool prune_ok = !pv_node && !in_check && !IsMateScore(beta) && !IsMateScore(alpha);
  int static_eval = 0;

  if (prune_ok) {
    static_eval = eval::Evaluate(pos_);

    // Reverse futility. Being this far above beta with this little depth left
    // means the opponent cannot claw it back, so there is nothing to look at.
    if (depth <= kReverseFutilityMaxDepth &&
        static_eval - kReverseFutilityMargin * depth >= beta) {
      return static_eval;
    }

    // Null move pruning.
    if (null_ok && depth >= kNullMoveMinDepth && static_eval >= beta) {
      const int reduction = kNullMoveBaseReduction + depth / kNullMoveDepthDivisor;
      pos_.DoNullMove();
      const int score = -AlphaBeta(depth - reduction - 1, -beta, -beta + 1, ply + 1, false);
      pos_.UndoNullMove();
      if (Stopped()) return eval::kDraw;
      if (score >= beta) {
        // A mate score proved by giving the opponent a free move is not a
        // mate; only the bound it establishes is worth anything.
        return IsMateScore(score) ? beta : score;
      }
    }
  }

  Node& node = stack_[ply];
  node.moves.Clear();
  movegen::GenerateLegal(pos_, node.moves);
  // Shogi has no stalemate: no legal move means mated, whether in check or not.
  if (node.moves.Empty()) return -eval::kMate + ply;

  ScoreMoves(ply, tt_move);

  const Color us = pos_.SideToMove();
  int best = -eval::kInfinite;
  Move best_move;
  Bound bound = Bound::kUpper;
  int quiets_tried = 0;

  for (int i = 0; i < node.moves.Size(); ++i) {
    const Move m = PickMove(ply, i);
    const bool quiet = IsQuiet(pos_, m);

    // Skipping moves is only safe once there is a score to fall back on. With
    // none, a node that pruned everything would report -kInfinite and be read
    // as a mate.
    if (prune_ok && best > -eval::kInfinite) {
      if (quiet) {
        // Late move pruning: this far down a well ordered list, at this
        // little depth, the remaining quiet moves are not the answer.
        if (depth <= kLateMoveMaxDepth && quiets_tried >= kLateMoveCount[depth]) continue;

        // Futility: nothing this move does to the material count can lift a
        // static evaluation that far below alpha.
        if (depth <= kFutilityMaxDepth && static_eval + kFutilityMargin * depth <= alpha) {
          continue;
        }
      } else if (depth <= kSeePruningMaxDepth && !SeeGe(pos_, m, -kSeePruningMargin * depth)) {
        // A capture that loses this much material needs a reason the search
        // has not found in the first few moves.
        continue;
      }
    }
    if (quiet) ++quiets_tried;

    pos_.DoMove(m);

    int score;
    const int child_depth = depth - 1;
    if (i == 0) {
      // The first move gets the full window. In a non-PV node that window is
      // already one point wide, so this is the same zero-window search every
      // other move gets.
      score = -AlphaBeta(child_depth, -beta, -alpha, ply + 1, true);
    } else {
      // Late move reductions. A quiet move this far down the list is being
      // searched to confirm it is not better than the ones before it, and a
      // shallower search answers that most of the time. When it comes back
      // above alpha the answer was not good enough and the move is searched
      // again at full depth.
      int reduction = 0;
      if (quiet && depth >= kLmrMinDepth && i >= kLmrMinMoveIndex) {
        reduction = Reduction(depth, i);
        if (pv_node) --reduction;
        reduction = std::clamp(reduction, 0, std::max(child_depth - 1, 0));
      }

      score = -AlphaBeta(child_depth - reduction, -alpha - 1, -alpha, ply + 1, true);
      if (reduction > 0 && score > alpha) {
        score = -AlphaBeta(child_depth, -alpha - 1, -alpha, ply + 1, true);
      }
      // A zero-window search only says which side of alpha the move is on. A
      // PV node needs the number itself, and only from a move that landed
      // inside the window.
      if (pv_node && score > alpha && score < beta) {
        score = -AlphaBeta(child_depth, -beta, -alpha, ply + 1, true);
      }
    }

    pos_.UndoMove();
    if (Stopped()) return eval::kDraw;

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

  search_.tt_.Store(pos_.Key(), ply, depth, best, bound, best_move);
  return best;
}

void Search::Worker::Iterate(int max_depth) {
  // Helper threads start deeper than the main one and so reach a given depth
  // at a different time and in a different order. That is the whole mechanism
  // of Lazy SMP: what one thread proves early lands in the shared table, and
  // the others walk into it and skip the work.
  const int first_depth = 1 + (id_ % 3);

  // Only the main thread is held to the depth that was asked for. A helper
  // that stopped there would go idle while the main thread was still finishing
  // that same iteration — the longest one — and take a core's worth of table
  // filling with it. Nothing it finds past the limit is ever reported; it is
  // there to keep feeding the main thread until the main thread is done.
  const int last_depth = id_ == 0 ? max_depth : kMaxDepth;

  for (int depth = std::min(first_depth, last_depth); depth <= last_depth; ++depth) {
    const int score = AlphaBeta(depth, -eval::kInfinite, eval::kInfinite, 0, true);

    // Only the main thread reports. A helper's numbers are not wrong, they
    // just describe a search nobody asked for, and interleaving them would
    // make the depth counter in a GUI jump about.
    if (id_ == 0) {
      // An aborted iteration searched only part of the root moves, so its
      // score is not comparable with the completed one before it. The
      // exception is the very first iteration, whose partial result is all
      // there is.
      const bool usable = !Stopped() || (result_.depth == 0 && pv_length_[0] > 0);
      if (usable) {
        result_.score = score;
        result_.depth = depth;
        result_.seldepth = seldepth_;
        result_.nodes = search_.TotalNodes();
        result_.time_ms = search_.ElapsedMs();
        result_.pv.assign(pv_[0].begin(), pv_[0].begin() + pv_length_[0]);
        if (!result_.pv.empty()) result_.best = result_.pv.front();
        if (search_.info_sink_) search_.info_sink_(search_.BuildInfoLine(result_));
      }
    }

    if (Stopped()) break;
    // A forced mate is the end of the line; searching deeper cannot improve it.
    if (score >= kMateInMaxPly) break;
    // Only the main thread's clock decides when to stop deepening. A helper
    // that gave up here would leave the machine idle while the main thread is
    // still working.
    if (id_ == 0 && !search_.budget_.Unlimited() &&
        search_.ElapsedMs() * kIterationStartDenominator >=
            search_.budget_.soft_ms * kIterationStartNumerator) {
      break;
    }
  }

  // The main thread finishing is what ends the search: the helpers exist to
  // feed its table, and there is nothing to do with what they find afterwards.
  if (id_ == 0) search_.Stop();
}

Search::Search() {
  SetThreads(1);
}

Search::~Search() = default;

void Search::SetThreads(int count) {
  count = std::clamp(count, 1, kMaxThreads);
  if (static_cast<int>(workers_.size()) == count) return;

  workers_.clear();
  for (int i = 0; i < count; ++i) {
    workers_.push_back(std::make_unique<Worker>(*this, i));
  }
}

void Search::NewGame() {
  tt_.Clear();
  for (std::unique_ptr<Worker>& worker : workers_) worker->NewGame();
  nnue::ClearCache();
}

int64_t Search::ElapsedMs() const {
  const auto elapsed = std::chrono::steady_clock::now() - start_;
  return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

int64_t Search::TotalNodes() const {
  int64_t total = 0;
  for (const std::unique_ptr<Worker>& worker : workers_) total += worker->Nodes();
  return total;
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

  tt_.NewSearch();

  SearchResult result;
  MoveList root_moves;
  movegen::GenerateLegal(pos, root_moves);
  if (root_moves.Empty()) {
    stop_.store(false, std::memory_order_relaxed);
    return result;
  }

  // Whatever happens from here on, there is a move to return.
  for (std::unique_ptr<Worker>& worker : workers_) worker->Prepare(pos, root_moves[0]);

  const int max_depth = limits.depth > 0 ? std::min(limits.depth, kMaxDepth) : kMaxDepth;

  std::vector<std::thread> helpers;
  helpers.reserve(workers_.size() - 1);
  for (size_t i = 1; i < workers_.size(); ++i) {
    helpers.emplace_back([this, i, max_depth] { workers_[i]->Iterate(max_depth); });
  }

  workers_[0]->Iterate(max_depth);
  for (std::thread& helper : helpers) helper.join();

  result = workers_[0]->Result();
  result.nodes = TotalNodes();
  result.time_ms = ElapsedMs();

  // Cleared here rather than on the way in. A stop asked for between the
  // moment this search was started on another thread and the moment that
  // thread got here has to survive, or the search it was meant to end runs
  // forever.
  stop_.store(false, std::memory_order_relaxed);
  return result;
}

}  // namespace luna
