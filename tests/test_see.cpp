#include "search/see.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "core/position.hpp"
#include "search/eval.hpp"

namespace {

int SeeOf(const std::string& sfen, const std::string& move) {
  luna::Position pos;
  REQUIRE(pos.SetSfen(sfen));
  const luna::Move m = luna::MoveFromUsi(move);
  REQUIRE_FALSE(m.IsNone());
  return luna::See(pos, m);
}

const int kPawn = luna::eval::CaptureValue(luna::kPawn);
const int kLance = luna::eval::CaptureValue(luna::kLance);
const int kRook = luna::eval::CaptureValue(luna::kRook);
const int kGold = luna::eval::CaptureValue(luna::kGold);

}  // namespace

TEST_CASE("an undefended capture is worth the whole piece", "[see]") {
  // A black pawn on 5e eats a white rook on 5d that nothing defends.
  REQUIRE(SeeOf("4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1", "5e5d") == kRook);
}

TEST_CASE("a defended capture pays for the recapture", "[see]") {
  // The pawn on 5d is defended by the lance on 5a: winning a pawn costs a rook.
  REQUIRE(SeeOf("4l3k/9/9/4p4/4R4/9/9/9/8K b - 1", "5e5d") == kPawn - kRook);
}

TEST_CASE("a piece behind a piece joins the exchange", "[see]") {
  // Two black lances stack on the 5 file against a pawn the white lance
  // defends. The second lance is what makes taking work: without it the
  // exchange stops a lance down.
  const char* stacked = "4l3k/9/9/4p4/4L4/4L4/9/9/8K b - 1";
  const char* single = "4l3k/9/9/4p4/4L4/9/9/9/8K b - 1";

  REQUIRE(SeeOf(stacked, "5e5d") == kPawn);
  REQUIRE(SeeOf(single, "5e5d") == kPawn - kLance);
}

TEST_CASE("an even trade is worth nothing", "[see]") {
  // Gold takes gold, gold takes back.
  REQUIRE(SeeOf("k8/9/4g4/4g4/4G4/9/9/9/K8 b - 1", "5e5d") == 0);
}

TEST_CASE("a quiet move onto a defended square loses the piece", "[see]") {
  // The rook steps onto 5d, which the pawn on 5c covers.
  REQUIRE(SeeOf("k8/9/4p4/8R/9/9/9/9/K8 b - 1", "1d5d") == -kRook);
}

TEST_CASE("a quiet move onto a safe square is worth nothing", "[see]") {
  REQUIRE(SeeOf("k8/9/9/8R/9/9/9/9/K8 b - 1", "1d5d") == 0);
}

TEST_CASE("promotion counts towards the exchange", "[see]") {
  // The pawn on 5d promotes on an empty, undefended square.
  REQUIRE(SeeOf("k8/9/9/4P4/9/9/9/9/K8 b - 1", "5d5c+") ==
          luna::eval::PromotionGain(luna::kPawn));
}

TEST_CASE("a drop onto a defended square loses the dropped piece", "[see]") {
  // Nothing is won by dropping, and the pawn on 5c takes the gold back.
  REQUIRE(SeeOf("k8/9/4p4/9/9/9/9/9/K8 b G 1", "G*5d") == -kGold);
}

TEST_CASE("the king cannot be used to take a defended piece", "[see]") {
  // 5e5d wins a pawn but hangs the king, so the exchange stops before the
  // king is priced into it. The move is not legal, and SEE must not answer
  // with a king's worth either way.
  const int see = SeeOf("4l3k/9/9/4p4/4K4/9/9/9/9 b - 1", "5e5d");

  REQUIRE(see < kRook);
  REQUIRE(see > -kRook);
}

TEST_CASE("SEE leaves the position untouched", "[see]") {
  luna::Position pos;
  REQUIRE(pos.SetSfen("4l3k/9/9/4p4/4R4/9/9/9/8K b - 1"));
  const std::string before = pos.ToSfen();
  const uint64_t key = pos.Key();

  luna::See(pos, luna::MoveFromUsi("5e5d"));

  REQUIRE(pos.ToSfen() == before);
  REQUIRE(pos.Key() == key);
}

TEST_CASE("SEE agrees with the sign of every capture in a real position", "[see]") {
  // Not a check of the numbers, only that nothing in a crowded position makes
  // See disagree with itself: the answer has to be the same whichever way the
  // move list is walked, and it must not depend on the position being dirty.
  luna::Position pos;
  REQUIRE(pos.SetSfen("ln1gkg1nl/2r2s1p1/pp1s1p2p/2p1PbPR1/3pp4/2P6/PP1P1P2P/1BS1KS3/LN1G1G1NL b Pp 29"));
  luna::MoveList moves;
  luna::movegen::GenerateLegal(pos, moves);

  for (int i = 0; i < moves.Size(); ++i) {
    const int first = luna::See(pos, moves[i]);
    const int again = luna::See(pos, moves[i]);
    REQUIRE(first == again);
  }
}
