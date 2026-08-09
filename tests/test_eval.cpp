#include "search/eval.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/position.hpp"
#include "core/types.hpp"

namespace {

luna::Position PositionOf(const std::string& sfen) {
  luna::Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

}  // namespace

TEST_CASE("the start position is materially even", "[eval]") {
  const luna::Position pos;

  REQUIRE(luna::eval::Evaluate(pos) == 0);
}

TEST_CASE("evaluation is from the side to move's point of view", "[eval]") {
  // Black is a rook up. The same board scores the opposite way for white.
  const std::string board = "4k4/9/9/9/9/9/9/7R1/4K4";
  const luna::Position black_to_move = PositionOf(board + " b - 1");
  const luna::Position white_to_move = PositionOf(board + " w - 1");

  REQUIRE(luna::eval::Evaluate(black_to_move) > 0);
  REQUIRE(luna::eval::Evaluate(white_to_move) == -luna::eval::Evaluate(black_to_move));
}

TEST_CASE("pieces in hand count the same as pieces on the board", "[eval]") {
  const luna::Position on_board = PositionOf("4k4/9/9/9/9/9/9/7R1/4K4 b - 1");
  const luna::Position in_hand = PositionOf("4k4/9/9/9/9/9/9/9/4K4 b R 1");

  REQUIRE(luna::eval::Evaluate(in_hand) == luna::eval::Evaluate(on_board));
  REQUIRE(luna::eval::Evaluate(in_hand) == luna::eval::PieceValue(luna::kRook));
}

TEST_CASE("kings are left out of the material count", "[eval]") {
  // Only black has a king, and the balance is still even.
  const luna::Position pos = PositionOf("9/9/9/9/9/9/9/9/4K4 b - 1");

  REQUIRE(luna::eval::Evaluate(pos) == 0);
}

TEST_CASE("piece values are ordered the way shogi pieces rank", "[eval]") {
  using namespace luna;

  REQUIRE(eval::PieceValue(kPawn) < eval::PieceValue(kLance));
  REQUIRE(eval::PieceValue(kLance) < eval::PieceValue(kKnight));
  REQUIRE(eval::PieceValue(kKnight) < eval::PieceValue(kSilver));
  REQUIRE(eval::PieceValue(kSilver) < eval::PieceValue(kGold));
  REQUIRE(eval::PieceValue(kGold) < eval::PieceValue(kBishop));
  REQUIRE(eval::PieceValue(kBishop) < eval::PieceValue(kRook));

  // Promoting a major piece adds to it, but a horse still comes out below a
  // rook: the bishop it grew from was that much cheaper to begin with.
  REQUIRE(eval::PieceValue(kBishop) < eval::PieceValue(kHorse));
  REQUIRE(eval::PieceValue(kHorse) < eval::PieceValue(kRook));
  REQUIRE(eval::PieceValue(kRook) < eval::PieceValue(kDragon));

  // The four small promoted pieces are all worth exactly a gold.
  for (const PieceType pt : {kProPawn, kProLance, kProKnight, kProSilver}) {
    REQUIRE(eval::PieceValue(pt) == eval::PieceValue(kGold));
  }
}

TEST_CASE("capturing counts the piece twice: off their board and into our hand",
          "[eval]") {
  using namespace luna;

  // A promoted pawn is worth a gold on the board but only a pawn in hand.
  REQUIRE(eval::CaptureValue(kProPawn) ==
          eval::PieceValue(kProPawn) + eval::PieceValue(kPawn));
  REQUIRE(eval::CaptureValue(kRook) == 2 * eval::PieceValue(kRook));
}

TEST_CASE("promotion gain is positive for every promotable piece", "[eval]") {
  using namespace luna;

  for (const PieceType pt : {kPawn, kLance, kKnight, kSilver, kBishop, kRook}) {
    REQUIRE(eval::PromotionGain(pt) > 0);
  }
  REQUIRE(eval::PromotionGain(kGold) == 0);
  REQUIRE(eval::PromotionGain(kKing) == 0);
  // Promoting a pawn gains the most, which is why と金 is worth chasing.
  REQUIRE(eval::PromotionGain(kPawn) > eval::PromotionGain(kSilver));
}
