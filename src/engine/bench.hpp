#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "search/search.hpp"

namespace luna {

// A fixed set of positions searched to a fixed depth. The same binary produces
// the same node count every run, which makes it the cheapest signal there is
// that a change to the search did what it was meant to: a pruning change that
// leaves the node count untouched is not running, and one that changes the
// best moves everywhere is doing more than pruning.
//
// Reproducibility is the whole point, so the search is reset between positions
// and the numbers only hold for a single thread.
constexpr int kDefaultBenchDepth = 8;

struct BenchResult {
  int64_t nodes = 0;
  int64_t time_ms = 0;
  // One line per position, then a total. Written for a human to read.
  std::vector<std::string> lines;
};

BenchResult RunBench(Search& search, int depth);

}  // namespace luna
