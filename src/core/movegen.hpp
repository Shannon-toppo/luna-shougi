#pragma once

#include "core/move.hpp"
#include "core/position.hpp"

namespace luna::movegen {

// Moves that respect piece movement, the drop restrictions that can be checked
// without searching (二歩, 行きどころのない駒), and nothing else. In particular
// these may leave the moving side's king in check.
void GeneratePseudoLegal(const Position& pos, MoveList& out);

// Fully legal moves: pseudo-legal, minus moves that leave the king in check,
// minus pawn drops that deliver mate (打ち歩詰め).
//
// Takes the position by reference because legality is decided by playing each
// move; `pos` is always restored before returning.
void GenerateLegal(Position& pos, MoveList& out);

MoveList GenerateLegal(Position& pos);

// True when `pos` is check and the side to move has no legal reply.
bool IsCheckmate(Position& pos);

// True when `m` is one of the legal moves in `pos`.
bool IsLegal(Position& pos, Move m);

}  // namespace luna::movegen
