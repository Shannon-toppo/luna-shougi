#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "core/move.hpp"
#include "core/position.hpp"

namespace luna {

// Counts leaf nodes of the legal move tree `depth` plies deep. Depth 0 is one
// node (the position itself).
uint64_t Perft(Position& pos, int depth);

// Perft split by first move, which is what you diff against a reference engine
// to find where the two generators disagree.
std::vector<std::pair<Move, uint64_t>> PerftDivide(Position& pos, int depth);

}  // namespace luna
