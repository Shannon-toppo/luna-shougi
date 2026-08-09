#pragma once

#include <array>
#include <cstdint>

#include "core/types.hpp"

namespace luna::zobrist {

// A hand can hold at most 18 pawns, so counts 0..18 need 19 slots.
constexpr int kMaxHandCount = 18;

struct Table {
  std::array<std::array<uint64_t, kSquareNb>, kPieceNb> psq{};
  std::array<std::array<std::array<uint64_t, kMaxHandCount + 1>, kPieceTypeNb>, kColorNb> hand{};
  uint64_t side = 0;

  Table();
};

// Deterministic across runs and platforms: the keys come from a fixed seed.
const Table& Keys();

}  // namespace luna::zobrist
