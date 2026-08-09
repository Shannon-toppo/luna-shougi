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

}  // namespace

void TranspositionTable::Resize(size_t mb) {
  mb = std::clamp<size_t>(mb, 1, kMaxSizeMb);

  size_t count = (mb * 1024 * 1024) / sizeof(Entry);
  size_t power_of_two = 1;
  while (power_of_two * 2 <= count) power_of_two *= 2;
  count = power_of_two;

  entries_.assign(count, Entry{});
  mask_ = count - 1;
  generation_ = 0;
}

void TranspositionTable::Clear() {
  std::fill(entries_.begin(), entries_.end(), Entry{});
  generation_ = 0;
}

void TranspositionTable::NewSearch() {
  generation_ = static_cast<uint8_t>(generation_ + kGenerationStep);
}

bool TranspositionTable::Probe(uint64_t key, int ply, TtProbe& out) const {
  const Entry& e = entries_[key & mask_];
  if (BoundOf(e) == Bound::kNone || e.key32 != static_cast<uint32_t>(key >> 32)) return false;

  out.value = FromStorage(e.value, ply);
  out.depth = e.depth;
  out.move = Move::FromRaw(e.move);
  out.bound = BoundOf(e);
  return true;
}

void TranspositionTable::Store(uint64_t key, int ply, int depth, int value, Bound bound,
                               Move move) {
  Entry& e = entries_[key & mask_];
  const uint32_t key32 = static_cast<uint32_t>(key >> 32);

  // Refresh our own entry unconditionally, take over slots left by an earlier
  // search, and otherwise only displace a shallower result from this search.
  const bool replace = e.key32 == key32 || BoundOf(e) == Bound::kNone ||
                       GenerationOf(e) != generation_ || depth >= e.depth;
  if (!replace) return;

  // A cutoff can come from a node whose best move was never established; keep
  // the previous move for this position rather than storing nothing.
  if (move.IsNone() && e.key32 == key32) move = Move::FromRaw(e.move);

  e.key32 = key32;
  e.move = move.Raw();
  e.value = ToStorage(value, ply);
  e.depth = static_cast<uint8_t>(std::clamp(depth, 0, 255));
  e.gen_bound = static_cast<uint8_t>(generation_ | static_cast<uint8_t>(bound));
}

int TranspositionTable::HashFull() const {
  const size_t sample = std::min<size_t>(1000, entries_.size());
  if (sample == 0) return 0;

  size_t used = 0;
  for (size_t i = 0; i < sample; ++i) {
    const Entry& e = entries_[i];
    if (BoundOf(e) != Bound::kNone && GenerationOf(e) == generation_) ++used;
  }
  return static_cast<int>(used * 1000 / sample);
}

}  // namespace luna
