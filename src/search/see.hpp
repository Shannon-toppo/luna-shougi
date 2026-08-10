#pragma once

#include "core/move.hpp"
#include "core/position.hpp"

namespace luna {

// Static exchange evaluation: what the side to move ends up with, in engine
// units, after playing `m` and letting both sides trade off on its destination
// square as long as trading pays. A negative result means the move loses
// material to a recapture the search has not looked at yet.
//
// This is an approximation, deliberately:
//
// - Only the destination square is considered. A recapture that hangs a piece
//   somewhere else is invisible.
// - Recaptures never promote. Modelling that would need move generation, and
//   the sequences where it changes the sign are rare.
// - Pinned pieces still count as defenders. Finding out otherwise costs a
//   legality check per attacker, which is more than the answer is worth here.
//
// It is exact about one thing that matters: a piece behind a piece on the same
// line takes over when the one in front is taken, so a lance behind a lance
// and a rook behind a rook both count.
//
// `m` must be pseudo-legal in `pos`, which is left untouched.
int See(const Position& pos, Move m);

// See(pos, m) >= threshold.
bool SeeGe(const Position& pos, Move m, int threshold);

}  // namespace luna
