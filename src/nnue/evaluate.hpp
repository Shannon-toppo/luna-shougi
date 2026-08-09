#pragma once

#include <string>

#include "core/position.hpp"
#include "nnue/network.hpp"

namespace luna::nnue {

// The network the engine plays with, and the accumulator caches that go with
// it.
//
// One network, shared; one accumulator cache per thread, held in thread-local
// storage. Loading is only safe while no search is running, which for now is
// always true because the search runs inside the USI command handler. When
// phase 6 puts the search on its own thread, loading has to happen between
// searches, the same way the transposition table is resized.

// Replaces the network. On failure the previous one is dropped and `error`
// says why.
bool Load(const std::string& path, std::string& error);

// Goes back to the hand-written evaluation.
void Unload();

// Hands back the shared network, empty and marked loaded, for a caller that
// builds one in memory instead of reading a file. The tests use it; so would
// anything that generates a network programmatically.
Network& InstallNetwork();

bool IsLoaded();

// The path the current network came from, empty when none is loaded.
const std::string& LoadedPath();

// False when the position has no king for one of the sides, which HalfKP
// cannot describe. Only contrived positions manage this, but the evaluation
// has to answer for them too.
inline bool CanEvaluate(const Position& pos) {
  return pos.KingSquare(kBlack) != kSquareNone && pos.KingSquare(kWhite) != kSquareNone;
}

// Static evaluation in engine units, from the side to move's point of view.
// Requires IsLoaded() and CanEvaluate(pos).
int Evaluate(const Position& pos);

// Drops this thread's cached accumulators. Not needed for correctness — every
// cached accumulator is checked against the Zobrist key of the position it
// was computed for — but it keeps a long game from holding on to the memory.
void ClearCache();

}  // namespace luna::nnue
