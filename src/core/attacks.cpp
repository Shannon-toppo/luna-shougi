#include "core/attacks.hpp"

namespace luna {
namespace {

// Column delta, then rank delta, for each Direction. Column 0 is file 9, so a
// positive column delta moves toward file 1.
constexpr int kDeltaColumn[kDirectionNb] = {0, +1, +1, +1, 0, -1, -1, -1};
constexpr int kDeltaRank[kDirectionNb] = {-1, -1, 0, +1, +1, +1, 0, -1};

constexpr uint8_t Bit(Direction d) {
  return static_cast<uint8_t>(1u << d);
}

// Rotating a direction mask by half a turn maps black's move set onto white's.
constexpr uint8_t RotateHalfTurn(uint8_t mask) {
  return static_cast<uint8_t>((mask << 4) | (mask >> 4));
}

constexpr uint8_t kOrthogonal = Bit(kNorth) | Bit(kEast) | Bit(kSouth) | Bit(kWest);
constexpr uint8_t kDiagonal = Bit(kNorthEast) | Bit(kSouthEast) | Bit(kSouthWest) | Bit(kNorthWest);

// Knight moves are generated from a dedicated table, not from these masks.
constexpr uint8_t BlackStepDirections(PieceType pt) {
  switch (pt) {
    case kPawn:
      return Bit(kNorth);
    case kSilver:
      return Bit(kNorth) | Bit(kNorthEast) | Bit(kNorthWest) | Bit(kSouthEast) | Bit(kSouthWest);
    case kGold:
    case kProPawn:
    case kProLance:
    case kProKnight:
    case kProSilver:
      return Bit(kNorth) | Bit(kNorthEast) | Bit(kEast) | Bit(kWest) | Bit(kNorthWest) |
             Bit(kSouth);
    case kKing:
      return kOrthogonal | kDiagonal;
    case kHorse:
      return kOrthogonal;
    case kDragon:
      return kDiagonal;
    default:
      return 0;
  }
}

constexpr uint8_t BlackSlideDirections(PieceType pt) {
  switch (pt) {
    case kLance:
      return Bit(kNorth);
    case kBishop:
    case kHorse:
      return kDiagonal;
    case kRook:
    case kDragon:
      return kOrthogonal;
    default:
      return 0;
  }
}

constexpr bool OnBoard(int column, int rank) {
  return column >= 0 && column < 9 && rank >= 1 && rank <= 9;
}

constexpr Square FromColumnRank(int column, int rank) {
  return (rank - 1) * 9 + column;
}

struct Tables {
  std::array<std::array<Square, kDirectionNb>, kSquareNb> neighbor{};
  std::array<std::array<SquareList, kDirectionNb>, kSquareNb> ray{};
  std::array<std::array<SquareList, kSquareNb>, kColorNb> knight{};
  std::array<std::array<uint8_t, kPieceTypeNb>, kColorNb> step_dirs{};
  std::array<std::array<uint8_t, kPieceTypeNb>, kColorNb> slide_dirs{};

  Tables() {
    for (Square sq = 0; sq < kSquareNb; ++sq) {
      for (int d = 0; d < kDirectionNb; ++d) {
        int column = ColumnOf(sq) + kDeltaColumn[d];
        int rank = RankOf(sq) + kDeltaRank[d];
        neighbor[sq][d] = OnBoard(column, rank) ? FromColumnRank(column, rank) : kSquareNone;

        SquareList& line = ray[sq][d];
        while (OnBoard(column, rank)) {
          line.squares[line.size++] = static_cast<uint8_t>(FromColumnRank(column, rank));
          column += kDeltaColumn[d];
          rank += kDeltaRank[d];
        }
      }
    }

    for (int c = 0; c < kColorNb; ++c) {
      const int forward = (c == kBlack) ? -1 : +1;
      for (Square sq = 0; sq < kSquareNb; ++sq) {
        SquareList& list = knight[c][sq];
        for (const int side : {-1, +1}) {
          const int column = ColumnOf(sq) + side;
          const int rank = RankOf(sq) + 2 * forward;
          if (OnBoard(column, rank)) {
            list.squares[list.size++] = static_cast<uint8_t>(FromColumnRank(column, rank));
          }
        }
      }
      for (int pt = 0; pt < kPieceTypeNb; ++pt) {
        const uint8_t steps = BlackStepDirections(static_cast<PieceType>(pt));
        const uint8_t slides = BlackSlideDirections(static_cast<PieceType>(pt));
        step_dirs[c][pt] = (c == kBlack) ? steps : RotateHalfTurn(steps);
        slide_dirs[c][pt] = (c == kBlack) ? slides : RotateHalfTurn(slides);
      }
    }
  }
};

const Tables& Get() {
  static const Tables tables;
  return tables;
}

}  // namespace

uint8_t StepDirections(Color c, PieceType pt) {
  return Get().step_dirs[c][pt];
}

uint8_t SlideDirections(Color c, PieceType pt) {
  return Get().slide_dirs[c][pt];
}

Square Neighbor(Square sq, Direction d) {
  return Get().neighbor[sq][d];
}

const SquareList& Ray(Square sq, Direction d) {
  return Get().ray[sq][d];
}

const SquareList& KnightAttacks(Color c, Square sq) {
  return Get().knight[c][sq];
}

}  // namespace luna
