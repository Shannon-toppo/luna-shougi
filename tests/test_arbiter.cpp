#include "match/arbiter.hpp"

#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/position.hpp"
#include "match/match.hpp"

namespace {

using luna::match::Arbiter;
using luna::match::Outcome;
using luna::match::Reason;

luna::Position PositionOf(const std::string& sfen) {
  luna::Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

// Plays a space-separated list of USI moves, checking each one is legal.
void PlayUsi(Arbiter& arbiter, const std::string& moves) {
  std::istringstream iss(moves);
  std::string token;
  while (iss >> token) {
    const luna::Move m = luna::MoveFromUsi(token);
    INFO("playing " << token);
    REQUIRE_FALSE(m.IsNone());
    REQUIRE(arbiter.IsLegal(m));
    arbiter.Play(m);
  }
}

// A pair of kings alone on the board, free to shuffle for as long as they
// like. Black is on 5i and white on 5a.
constexpr const char* kBareKings = "4k4/9/9/9/9/9/9/9/4K4 b - 1";

// White is to move and in check from a black rook down the file. Neither
// king can be reached by anything else, so the checks can go on forever.
constexpr const char* kPerpetual = "4k4/9/9/9/9/9/9/9/4R3K w - 1";

}  // namespace

TEST_CASE("a fresh game is in progress with every legal move available", "[arbiter]") {
  Arbiter arbiter{luna::Position()};

  REQUIRE(arbiter.Status().outcome == Outcome::kInProgress);
  REQUIRE_FALSE(arbiter.Over());
  REQUIRE(arbiter.SideToMove() == luna::kBlack);
  REQUIRE(arbiter.LegalMoves().Size() == 30);
  REQUIRE(arbiter.IsLegal(luna::MoveFromUsi("7g7f")));
  REQUIRE_FALSE(arbiter.IsLegal(luna::MoveFromUsi("7g7e")));
}

TEST_CASE("the position command names the start and every move since", "[arbiter]") {
  Arbiter arbiter{luna::Position()};
  REQUIRE(arbiter.PositionCommand() == "position sfen " + std::string(luna::kStartSfen));

  PlayUsi(arbiter, "7g7f 3c3d");
  REQUIRE(arbiter.PositionCommand().find(" moves 7g7f 3c3d") != std::string::npos);
  REQUIRE(arbiter.Moves().size() == 2);
}

TEST_CASE("a mated side loses", "[arbiter]") {
  // White's king on 5a has nowhere to go: the gold on 5b covers every square
  // around it and the gold on 5c defends the gold.
  Arbiter arbiter{PositionOf("4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1")};

  REQUIRE(arbiter.Over());
  REQUIRE(arbiter.Status().outcome == Outcome::kBlackWin);
  REQUIRE(arbiter.Status().reason == Reason::kCheckmate);
}

TEST_CASE("the same position four times is a draw", "[arbiter]") {
  Arbiter arbiter{PositionOf(kBareKings)};

  // Each round trip brings the position back once more. The third time it
  // appears the game is still on; the fourth ends it.
  PlayUsi(arbiter, "5i4i 5a4a 4i5i 4a5a");
  REQUIRE_FALSE(arbiter.Over());
  PlayUsi(arbiter, "5i4i 5a4a 4i5i 4a5a");
  REQUIRE_FALSE(arbiter.Over());
  PlayUsi(arbiter, "5i4i 5a4a 4i5i 4a5a");

  REQUIRE(arbiter.Over());
  REQUIRE(arbiter.Status().outcome == Outcome::kDraw);
  REQUIRE(arbiter.Status().reason == Reason::kRepetition);
}

TEST_CASE("repeating a position while checking loses the game", "[arbiter]") {
  // 連続王手の千日手. Black chases white's king with the rook and never lets
  // it out of check, so black loses instead of the game being drawn.
  Arbiter arbiter{PositionOf(kPerpetual)};

  PlayUsi(arbiter, "5a4a 5i4i 4a5a 4i5i");
  PlayUsi(arbiter, "5a4a 5i4i 4a5a 4i5i");
  PlayUsi(arbiter, "5a4a 5i4i 4a5a 4i5i");

  REQUIRE(arbiter.Over());
  REQUIRE(arbiter.Status().outcome == Outcome::kWhiteWin);
  REQUIRE(arbiter.Status().reason == Reason::kPerpetualCheck);
}

TEST_CASE("a game that will not end is adjudicated a draw", "[arbiter]") {
  Arbiter arbiter{PositionOf(kBareKings), 4};

  PlayUsi(arbiter, "5i4i 5a4a 4i5i");
  REQUIRE_FALSE(arbiter.Over());
  PlayUsi(arbiter, "4a5a");

  REQUIRE(arbiter.Status().outcome == Outcome::kDraw);
  REQUIRE(arbiter.Status().reason == Reason::kMaxPly);
}

TEST_CASE("a king that has entered with enough behind it may declare", "[arbiter]") {
  // Black's king is on 5a with twelve pieces in the promotion zone, worth
  // 28 points with the rooks and bishops counting five each.
  const Arbiter can{PositionOf("RBGSKSGBR/3PPPP2/9/9/9/9/9/9/4k4 b - 1")};
  REQUIRE(can.CanDeclareWin());

  // One pawn fewer is 27 points, which is enough for white but not for
  // black: black moved first and needs one more.
  const Arbiter one_short{PositionOf("RBGSKSGBR/3PPP3/9/9/9/9/9/9/4k4 b - 1")};
  REQUIRE_FALSE(one_short.CanDeclareWin());
  const Arbiter as_white{PositionOf("4K4/9/9/9/9/9/9/3pppp2/rbgsksgbr w - 1")};
  REQUIRE(as_white.CanDeclareWin());
}

TEST_CASE("a declaration needs the king in the zone, out of check and supported", "[arbiter]") {
  // The same material, but the king never entered.
  const Arbiter outside{PositionOf("RBGS1SGBR/3PPPP2/9/9/9/9/9/9/4K1k2 b - 1")};
  REQUIRE_FALSE(outside.CanDeclareWin());

  // Twenty-eight points, but only eight pieces up there with the king. The
  // rule wants a real entering formation, not two big pieces on their own.
  const Arbiter thin{PositionOf("RBRBK4/3PPPP2/9/9/9/9/9/9/4k4 b 4G 1")};
  REQUIRE_FALSE(thin.CanDeclareWin());

  // In check, which no amount of material makes up for.
  const Arbiter checked{PositionOf("RBGSKSGBR/2PP1PP2/9/9/4r4/9/9/9/4k4 b - 1")};
  REQUIRE_FALSE(checked.CanDeclareWin());
}

TEST_CASE("pieces in hand count towards a declaration", "[arbiter]") {
  // Ten pieces in the zone worth ten points between them, so everything else
  // has to come out of the hand. A rook, two bishops and two golds is 27 and
  // one short; a third gold makes it 28.
  const Arbiter short_of_it{PositionOf("SSSGKGSSS/3P1P3/9/9/9/9/9/9/4k4 b RBB2G 1")};
  REQUIRE_FALSE(short_of_it.CanDeclareWin());

  const Arbiter enough{PositionOf("SSSGKGSSS/3P1P3/9/9/9/9/9/9/4k4 b RBB3G 1")};
  REQUIRE(enough.CanDeclareWin());
}

TEST_CASE("random openings vary with the seed and stop where told", "[match]") {
  const luna::Position four = luna::match::RandomOpening(1, 4);
  const luna::Position same = luna::match::RandomOpening(1, 4);
  const luna::Position other = luna::match::RandomOpening(2, 4);

  REQUIRE(four.ToSfen() == same.ToSfen());
  REQUIRE(four.ToSfen() != other.ToSfen());
  REQUIRE(four.SideToMove() == luna::kBlack);
  REQUIRE(luna::match::RandomOpening(1, 0).ToSfen() == std::string(luna::kStartSfen));

  // An opening is a legal position with a game still to play in it.
  Arbiter arbiter{four};
  REQUIRE_FALSE(arbiter.Over());
}
