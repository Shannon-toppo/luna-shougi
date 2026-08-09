#include "core/zobrist.hpp"

namespace luna::zobrist {
namespace {

// splitmix64: good enough for hashing and reproducible without <random>, whose
// engines are specified but whose distributions are not.
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t Next() {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

 private:
  uint64_t state_;
};

}  // namespace

Table::Table() {
  SplitMix64 rng(0x6C756E61UL);  // "luna"
  for (auto& per_piece : psq) {
    for (uint64_t& key : per_piece) key = rng.Next();
  }
  for (auto& per_color : hand) {
    for (auto& per_type : per_color) {
      // Count 0 contributes nothing, which keeps an empty hand out of the key.
      per_type[0] = 0;
      for (size_t n = 1; n < per_type.size(); ++n) per_type[n] = rng.Next();
    }
  }
  side = rng.Next();
}

const Table& Keys() {
  static const Table table;
  return table;
}

}  // namespace luna::zobrist
