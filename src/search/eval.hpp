#pragma once

#include "core/position.hpp"
#include "core/types.hpp"

namespace luna::eval {

// Scores are in the engine's own unit, with an unpromoted pawn worth 90. USI
// "score cp" reports them unchanged; GUIs only ever display the number.
constexpr int kDraw = 0;

// A side mated at `ply` scores -(kMate - ply), so a mate found sooner always
// beats one found later. kInfinite is the widest usable alpha-beta window.
constexpr int kMate = 32000;
constexpr int kInfinite = 32001;

// Value of a piece standing on the board. The king is worth far more than any
// combination of the rest so that it dominates move ordering, but Evaluate
// leaves it out: both sides always have exactly one.
int PieceValue(PieceType pt);

// Value of a piece sitting in hand. A piece in hand can be dropped anywhere,
// which is worth at least as much as the same piece on the board; phase 3
// keeps the two equal and lets phase 4 separate them.
int HandValue(PieceType pt);

// Material swing from capturing a piece of type `pt`: the opponent loses it
// from the board and we gain its unpromoted form in hand.
int CaptureValue(PieceType pt);

// Material gained by promoting a piece of type `pt`, or 0 when it cannot
// promote.
int PromotionGain(PieceType pt);

// Material balance from the side to move's point of view.
int Evaluate(const Position& pos);

}  // namespace luna::eval
