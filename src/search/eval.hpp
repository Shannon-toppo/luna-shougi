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

// Having the move is worth something on its own. Without this the evaluation
// is symmetric between the two sides of an exchange, and the search happily
// walks into positions it only likes because it is about to lose the tempo.
constexpr int kTempo = 20;

// Value of a piece standing on the board. The king is worth far more than any
// combination of the rest so that it dominates move ordering, but Evaluate
// leaves it out: both sides always have exactly one.
int PieceValue(PieceType pt);

// Value of a piece sitting in hand, which is a little more than the same
// piece on the board: it can be dropped on any square, and until it is
// dropped nothing can attack it. Promoted types revert, the way they do when
// captured.
int HandValue(PieceType pt);

// Material swing from capturing a piece of type `pt`: the opponent loses it
// from the board and we gain its unpromoted form in hand.
int CaptureValue(PieceType pt);

// Material gained by promoting a piece of type `pt`, or 0 when it cannot
// promote.
int PromotionGain(PieceType pt);

// The evaluation split into its parts. Everything except `total` is from
// black's point of view, so the parts are comparable between positions;
// `total` is from the side to move's, like Evaluate. Meant for the "eval"
// debug command and for tests, not for the search.
struct EvalTerms {
  int material = 0;
  int pst = 0;
  int king_safety = 0;
  int tempo = 0;
  int total = 0;
};

// Static evaluation from the side to move's point of view: material, piece
// placement, how close each side's pieces are to the two kings, and the
// tempo bonus.
int Evaluate(const Position& pos);

// Evaluate, keeping the parts. Trace(pos).total == Evaluate(pos) always.
EvalTerms Trace(const Position& pos);

}  // namespace luna::eval
