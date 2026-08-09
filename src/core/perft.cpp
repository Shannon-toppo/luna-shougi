#include "core/perft.hpp"

#include "core/movegen.hpp"

namespace luna {

uint64_t Perft(Position& pos, int depth) {
  if (depth <= 0) return 1;

  MoveList moves;
  movegen::GenerateLegal(pos, moves);
  if (depth == 1) return static_cast<uint64_t>(moves.Size());

  uint64_t nodes = 0;
  for (const Move m : moves) {
    pos.DoMove(m);
    nodes += Perft(pos, depth - 1);
    pos.UndoMove();
  }
  return nodes;
}

std::vector<std::pair<Move, uint64_t>> PerftDivide(Position& pos, int depth) {
  std::vector<std::pair<Move, uint64_t>> result;
  if (depth <= 0) return result;

  MoveList moves;
  movegen::GenerateLegal(pos, moves);
  result.reserve(moves.Size());
  for (const Move m : moves) {
    pos.DoMove(m);
    result.emplace_back(m, Perft(pos, depth - 1));
    pos.UndoMove();
  }
  return result;
}

}  // namespace luna
