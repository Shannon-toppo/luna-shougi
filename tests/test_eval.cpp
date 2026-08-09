#include "search/eval.hpp"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/position.hpp"
#include "core/types.hpp"
#include "mirror.hpp"
#include "search/pst.hpp"

namespace {

luna::Position PositionOf(const std::string& sfen) {
  luna::Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

using test::MirrorSfen;

}  // namespace

TEST_CASE("the start position is even apart from the tempo bonus", "[eval]") {
  const luna::Position pos;
  const luna::eval::EvalTerms terms = luna::eval::Trace(pos);

  REQUIRE(terms.material == 0);
  REQUIRE(terms.pst == 0);
  REQUIRE(terms.king_safety == 0);
  REQUIRE(terms.total == luna::eval::kTempo);
}

TEST_CASE("a position and its mirror score the same", "[eval]") {
  const std::vector<std::string> sfens = {
      luna::kStartSfen,
      // A castled black king against a white king still in the centre.
      "ln1g3nl/1r2k1gs1/p1ppppbpp/9/1p5P1/2P6/PPSPPPP1P/1BG2K1R1/LN2G2NL b Ss 1",
      // Both sides holding pieces, several of them promoted.
      "l6nl/4+P1gk1/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
      // A lone white king with black pieces closing in and the rest in hand.
      "4k4/9/4G4/3+P5/9/9/9/9/4K4 b 2R2B3G4S4N4L17P 1",
  };

  for (const std::string& sfen : sfens) {
    const luna::Position pos = PositionOf(sfen);
    const luna::Position mirror = PositionOf(MirrorSfen(sfen));

    INFO(sfen << "  ->  " << MirrorSfen(sfen));
    REQUIRE(luna::eval::Evaluate(mirror) == luna::eval::Evaluate(pos));
  }
}

TEST_CASE("the traced terms add up to the score", "[eval]") {
  const luna::Position pos =
      PositionOf("ln1g3nl/1r2k1gs1/p1ppppbpp/9/1p5P1/2P6/PPSPPPP1P/1BG2K1R1/LN2G2NL b Ss 1");
  const luna::eval::EvalTerms terms = luna::eval::Trace(pos);

  REQUIRE(terms.material + terms.pst + terms.king_safety + terms.tempo == terms.total);
  REQUIRE(terms.total == luna::eval::Evaluate(pos));
}

TEST_CASE("the traced terms are from black's point of view", "[eval]") {
  // Black is a rook up on the board. Only the sign of `total` should follow
  // whose turn it is; the parts describe the position itself.
  const std::string board = "4k4/9/9/9/9/9/9/7R1/4K4";
  const luna::eval::EvalTerms black = luna::eval::Trace(PositionOf(board + " b - 1"));
  const luna::eval::EvalTerms white = luna::eval::Trace(PositionOf(board + " w - 1"));

  REQUIRE(black.material == white.material);
  REQUIRE(black.material > 0);
  REQUIRE(black.pst == white.pst);
  REQUIRE(black.king_safety == white.king_safety);
  REQUIRE(black.tempo == -white.tempo);
  REQUIRE(white.total == -(white.material + white.pst + white.king_safety) + luna::eval::kTempo);
}

TEST_CASE("a piece in hand is worth a little more than the same piece on the board", "[eval]") {
  using namespace luna;

  for (const PieceType pt : kHandTypes) {
    REQUIRE(eval::HandValue(pt) > eval::PieceValue(pt));
    // Not so much more that giving up the piece to make a drop looks like a
    // material loss worth avoiding.
    REQUIRE(eval::HandValue(pt) - eval::PieceValue(pt) < eval::PieceValue(kPawn) / 2);
  }

  // A promoted piece goes to the hand unpromoted, so that is what it is worth
  // there.
  REQUIRE(eval::HandValue(kProPawn) == eval::HandValue(kPawn));
  REQUIRE(eval::HandValue(kDragon) == eval::HandValue(kRook));
}

TEST_CASE("kings are left out of the material count", "[eval]") {
  // Only black has a king, and the material balance is still even.
  const luna::Position pos = PositionOf("9/9/9/9/9/9/9/9/4K4 b - 1");

  REQUIRE(luna::eval::Trace(pos).material == 0);
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

TEST_CASE("capturing counts the piece twice: off their board and into our hand", "[eval]") {
  using namespace luna;

  // A promoted pawn is worth a gold on the board but only a pawn in hand.
  REQUIRE(eval::CaptureValue(kProPawn) == eval::PieceValue(kProPawn) + eval::HandValue(kPawn));
  REQUIRE(eval::CaptureValue(kRook) == eval::PieceValue(kRook) + eval::HandValue(kRook));
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

TEST_CASE("the piece-square tables mirror between the two sides", "[eval][pst]") {
  using namespace luna;

  for (int pt = kPawn; pt < kPieceTypeNb; ++pt) {
    for (Square sq = 0; sq < kSquareNb; ++sq) {
      const Square mirrored = kSquareNb - 1 - sq;
      INFO("piece " << pt << " on square " << sq);
      REQUIRE(eval::PstValue(kWhite, static_cast<PieceType>(pt), mirrored) ==
              eval::PstValue(kBlack, static_cast<PieceType>(pt), sq));
    }
  }
}

TEST_CASE("the piece-square tables do not favour one wing", "[eval][pst]") {
  using namespace luna;

  // Which side of the board an opening attacks is decided by the opening, not
  // by the piece, so no table may lean left or right.
  for (int pt = kPawn; pt < kPieceTypeNb; ++pt) {
    for (int rank = 1; rank <= 9; ++rank) {
      for (int file = 1; file <= 9; ++file) {
        INFO("piece " << pt << " on file " << file << " rank " << rank);
        REQUIRE(eval::PstValue(kBlack, static_cast<PieceType>(pt), MakeSquare(file, rank)) ==
                eval::PstValue(kBlack, static_cast<PieceType>(pt), MakeSquare(10 - file, rank)));
      }
    }
  }
}

TEST_CASE("the tables want the king castled and the pieces advanced", "[eval][pst]") {
  using namespace luna;

  // A king on 8i, where 矢倉 and 穴熊 both put it, against one still on its
  // starting square in the middle of the back rank.
  REQUIRE(eval::PstValue(kBlack, kKing, MakeSquare(8, 9)) >
          eval::PstValue(kBlack, kKing, MakeSquare(5, 9)));
  // A king that has walked out into the open is far worse than either.
  REQUIRE(eval::PstValue(kBlack, kKing, MakeSquare(5, 5)) <
          eval::PstValue(kBlack, kKing, MakeSquare(5, 9)) - 20);

  // Pawns, knights and silvers are all worth more the further they get.
  for (const PieceType pt : {kPawn, kKnight, kSilver}) {
    REQUIRE(eval::PstValue(kBlack, pt, MakeSquare(5, 4)) >
            eval::PstValue(kBlack, pt, MakeSquare(5, 7)));
  }

  // 馬は自陣に引け: unlike the dragon, the horse is worth more back home.
  REQUIRE(eval::PstValue(kBlack, kHorse, MakeSquare(5, 8)) >
          eval::PstValue(kBlack, kHorse, MakeSquare(5, 2)));
  REQUIRE(eval::PstValue(kBlack, kDragon, MakeSquare(5, 2)) >
          eval::PstValue(kBlack, kDragon, MakeSquare(5, 8)));
}

TEST_CASE("a castled king is safer than a bare one", "[eval]") {
  // The same material, with black's king behind a gold and a silver and
  // white's king out on its own.
  const luna::Position castled = PositionOf("4k4/9/9/9/9/9/9/6GS1/6K2 b - 1");
  const luna::Position bare = PositionOf("4k4/9/9/6GS1/9/9/9/9/6K2 b - 1");

  REQUIRE(luna::eval::Trace(castled).king_safety > luna::eval::Trace(bare).king_safety);
}

TEST_CASE("pieces near the enemy king are worth more than pieces far from it", "[eval]") {
  // A black gold two squares from white's king, and the same gold three ranks
  // further back. Black's own king sits in the far corner in both, out of
  // range of the gold, so only the attacking half of the term moves.
  const luna::Position close = PositionOf("4k4/9/4G4/9/9/9/9/9/8K b - 1");
  const luna::Position far = PositionOf("4k4/9/9/9/9/4G4/9/9/8K b - 1");

  REQUIRE(luna::eval::Trace(close).king_safety > luna::eval::Trace(far).king_safety);
}
