#pragma once

#include <atomic>
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

// One entry per slot, indexed by the low bits of the Zobrist key and verified
// against its high 16 bits.
//
// Sixteen bits of tag is not a proof of identity, and neither was the 32 the
// table used to carry: callers must treat a probed move as a hint and check it
// against generated moves before playing it. What sixteen buys is that the
// whole entry fits in one 64-bit word, which is what makes the table safe to
// share between search threads without a lock. A reader always sees a whole
// entry, never a key from one position with a score from another. Two
// positions landing on the same slot with the same tag stay possible, at
// roughly one probe in 65536 of the ones that collide at all, and the search
// carries that risk the way every engine does.
//
// Stores race with each other and the loser is simply overwritten. Nothing is
// lost that another search will not find again.
class TranspositionTable {
 public:
  static constexpr size_t kDefaultSizeMb = 64;
  static constexpr size_t kMaxSizeMb = 4096;

  TranspositionTable() { Resize(kDefaultSizeMb); }

  // Rounds the entry count down to a power of two and clears the table. Sizes
  // below one entry's worth are rounded up to a single entry.
  //
  // Not safe while a search is running.
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
  // The packed entry, low bits first:
  //
  //   bits  0-7   generation in the high 6 bits, Bound in the low 2
  //   bits  8-15  depth
  //   bits 16-31  value
  //   bits 32-47  move
  //   bits 48-63  tag, the high 16 bits of the Zobrist key
  //
  // An all-zero word has Bound::kNone and so reads as empty.
  static constexpr uint8_t kBoundMask = 3;
  static constexpr uint8_t kGenerationStep = 4;

  static uint16_t TagOf(uint64_t e) { return static_cast<uint16_t>(e >> 48); }
  static uint16_t MoveOf(uint64_t e) { return static_cast<uint16_t>(e >> 32); }
  static int16_t ValueOf(uint64_t e) { return static_cast<int16_t>(e >> 16); }
  static uint8_t DepthOf(uint64_t e) { return static_cast<uint8_t>(e >> 8); }
  static Bound BoundOf(uint64_t e) { return static_cast<Bound>(e & kBoundMask); }
  static uint8_t GenerationOf(uint64_t e) {
    return static_cast<uint8_t>(static_cast<uint8_t>(e) & ~kBoundMask);
  }

  static uint16_t TagFor(uint64_t key) { return static_cast<uint16_t>(key >> 48); }

  std::vector<std::atomic<uint64_t>> entries_;
  size_t mask_ = 0;
  uint8_t generation_ = 0;
};

}  // namespace luna
