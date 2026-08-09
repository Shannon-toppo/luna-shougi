#pragma once

#include "core/types.hpp"

namespace luna::eval {

// Positional value of a piece of type `pt` belonging to `c` standing on `sq`,
// on the same scale as the material values in eval.hpp. Kings are included:
// where the king stands is the single most important positional fact in a
// shogi middlegame.
//
// The tables say nothing about the other pieces on the board, so everything
// they express has to be true on average rather than in any given position.
// That is what keeps the numbers small next to material.
int PstValue(Color c, PieceType pt, Square sq);

}  // namespace luna::eval
