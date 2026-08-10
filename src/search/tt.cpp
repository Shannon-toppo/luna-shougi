#include "search/tt.hpp"

#include <algorithm>

#include "search/eval.hpp"

namespace luna {
namespace {

// Anything this large is a mate score rather than a material count: the most
// material one side can ever hold is well under 25000.
constexpr int kMateScoreFloor = eval::kMate - 1000;

// Mate scores are stored as "distance from this node" so the entry stays
// correct when the same position turns up at a different ply.
int16_t ToStorage(int value, int ply) {
  if (value >= kMateScoreFloor) return static_cast<int16_t>(value + ply);
  if (value <= -kMateScoreFloor) return static_cast<int16_t>(value - ply);
  return static_cast<int16_t>(value);
}

int FromStorage(int value, int ply) {
  if (value >= kMateScoreFloor) return value - ply;
  if (value <= -kMateScoreFloor) return value + ply;
  return value;
}

uint64_t Pack(uint16_t tag, uint16_t move, int16_t value, uint8_t depth, uint8_t gen_bound) {
  return (static_cast<uint64_t>(tag) << 48) | (static_cast<uint64_t>(move) << 32) |
         (static_cast<uint64_t>(static_cast<uint16_t>(value)) << 16) |
         (static_cast<uint64_t>(depth) << 8) | gen_bound;
}

// Every access is relaxed. The table is a cache of results that can each be
// recomputed, so a thread reading a slightly stale word costs at most the
// search it saved; there is nothing here that another thread's write has to
// happen-before.
constexpr std::memory_order kOrder = std::memory_order_relaxed;

}  // namespace

void TranspositionTable::Resize(size_t mb) {
  mb = std::clamp<size_t>(mb, 1, kMaxSizeMb);

  size_t count = (mb * 1024 * 1024) / sizeof(uint64_t);
  size_t power_of_two = 1;
  while (power_of_two * 2 <= count) power_of_two *= 2;
  count = power_of_two;

  entries_ = std::vector<std::atomic<uint64_t>>(count);
  mask_ = count - 1;
  generation_ = 0;
  Clear();
}

void TranspositionTable::Clear() {
  for (std::atomic<uint64_t>& entry : entries_) entry.store(0, kOrder);
  generation_ = 0;
}

void TranspositionTable::NewSearch() {
  generation_ = static_cast<uint8_t>(generation_ + kGenerationStep);
}

bool TranspositionTable::Probe(uint64_t key, int ply, TtProbe& out) const {
  const uint64_t entry = entries_[key & mask_].load(kOrder);
  if (BoundOf(entry) == Bound::kNone || TagOf(entry) != TagFor(key)) return false;

  out.value = FromStorage(ValueOf(entry), ply);
  out.depth = DepthOf(entry);
  out.move = Move::FromRaw(MoveOf(entry));
  out.bound = BoundOf(entry);
  return true;
}

void TranspositionTable::Store(uint64_t key, int ply, int depth, int value, Bound bound,
                               Move move) {
  std::atomic<uint64_t>& slot = entries_[key & mask_];
  const uint64_t entry = slot.load(kOrder);
  const uint16_t tag = TagFor(key);
  const bool same = TagOf(entry) == tag && BoundOf(entry) != Bound::kNone;

  // Refresh our own entry unconditionally, take over slots left by an earlier
  // search, and otherwise only displace a shallower result from this search.
  const bool replace = same || BoundOf(entry) == Bound::kNone ||
                       GenerationOf(entry) != generation_ ||
                       depth >= static_cast<int>(DepthOf(entry));
  if (!replace) return;

  // A cutoff can come from a node whose best move was never established; keep
  // the previous move for this position rather than storing nothing.
  if (move.IsNone() && same) move = Move::FromRaw(MoveOf(entry));

  slot.store(Pack(tag, move.Raw(), ToStorage(value, ply),
                  static_cast<uint8_t>(std::clamp(depth, 0, 255)),
                  static_cast<uint8_t>(generation_ | static_cast<uint8_t>(bound))),
             kOrder);
}

int TranspositionTable::HashFull() const {
  const size_t sample = std::min<size_t>(1000, entries_.size());
  if (sample == 0) return 0;

  size_t used = 0;
  for (size_t i = 0; i < sample; ++i) {
    const uint64_t entry = entries_[i].load(kOrder);
    if (BoundOf(entry) != Bound::kNone && GenerationOf(entry) == generation_) ++used;
  }
  return static_cast<int>(used * 1000 / sample);
}

}  // namespace luna
