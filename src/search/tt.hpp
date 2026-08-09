#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/move.hpp"

namespace luna {

// What a stored score says about the true value of the position.
enum class Bound : uint8_t {
  kNone = 0,
  kUpper = 1,  // the search failed low: the true value is at most `value`
  kLower = 2,  // the search failed high: the true value is at least `value`
  kExact = 3,
};

struct TtProbe {
  int value = 0;
  int depth = 0;
  Move move;
  Bound bound = Bound::kNone;
};

// One entry per slot, indexed by the low bits of the Zobrist key. Entries are
// verified against the high 32 bits of the key, which is not a proof of
// identity: callers must treat a probed move as a hint and check it against
// generated moves before playing it.
class TranspositionTable {
 public:
  static constexpr size_t kDefaultSizeMb = 64;
  static constexpr size_t kMaxSizeMb = 4096;

  TranspositionTable() { Resize(kDefaultSizeMb); }

  // Rounds the entry count down to a power of two and clears the table. Sizes
  // below one entry's worth are rounded up to a single entry.
  void Resize(size_t mb);
  void Clear();

  // Ages the table so entries from earlier searches become replaceable.
  void NewSearch();

  // `ply` undoes the mate-score encoding applied by Store.
  bool Probe(uint64_t key, int ply, TtProbe& out) const;

  // `value` is relative to `ply`; mate scores are rewritten to be relative to
  // the root before being stored.
  void Store(uint64_t key, int ply, int depth, int value, Bound bound, Move move);

  // Permille of slots holding an entry from the current search, sampled over
  // the first 1000 slots. This is what USI "hashfull" wants.
  int HashFull() const;

  size_t EntryCount() const { return entries_.size(); }

 private:
  struct Entry {
    uint32_t key32 = 0;
    uint16_t move = 0;
    int16_t value = 0;
    uint8_t depth = 0;
    uint8_t gen_bound = 0;  // generation in the high 6 bits, Bound in the low 2
  };

  static constexpr uint8_t kBoundMask = 3;
  static constexpr uint8_t kGenerationStep = 4;

  static Bound BoundOf(const Entry& e) { return static_cast<Bound>(e.gen_bound & kBoundMask); }
  static uint8_t GenerationOf(const Entry& e) {
    return static_cast<uint8_t>(e.gen_bound & ~kBoundMask);
  }

  std::vector<Entry> entries_;
  size_t mask_ = 0;
  uint8_t generation_ = 0;
};

}  // namespace luna
