#pragma once

#include <array>
#include <cstdint>

#include "core/types.hpp"

namespace luna {

// A short list of squares; nothing on a shogi board needs more than 8.
struct SquareList {
  uint8_t size = 0;
  std::array<uint8_t, 8> squares{};

  const uint8_t* begin() const { return squares.data(); }
  const uint8_t* end() const { return squares.data() + size; }
};

// Bitmask of directions the piece can step one square along. Horse and dragon
// include their single-step directions here, not in SlideDirections.
uint8_t StepDirections(Color c, PieceType pt);

// Bitmask of directions the piece slides along until blocked.
uint8_t SlideDirections(Color c, PieceType pt);

// Adjacent square in direction `d`, or kSquareNone when off the board.
Square Neighbor(Square sq, Direction d);

// Every square from `sq` along `d`, ordered by increasing distance.
const SquareList& Ray(Square sq, Direction d);

// The one or two squares a knight of color `c` on `sq` attacks.
const SquareList& KnightAttacks(Color c, Square sq);

}  // namespace luna
