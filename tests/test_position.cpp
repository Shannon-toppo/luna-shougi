#include "core/position.hpp"

#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/movegen.hpp"

namespace {

using luna::Color;
using luna::Move;
using luna::MoveFromUsi;
using luna::Position;
using luna::Square;

Position FromSfen(const std::string& sfen) {
  Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

}  // namespace

TEST_CASE("square numbering matches USI notation", "[position]") {
  REQUIRE(luna::MakeSquare(9, 1) == 0);
  REQUIRE(luna::MakeSquare(1, 1) == 8);
  REQUIRE(luna::MakeSquare(1, 9) == 80);
  REQUIRE(luna::SquareToUsi(luna::MakeSquare(7, 7)) == "7g");
  REQUIRE(luna::SquareFromUsi("7g") == luna::MakeSquare(7, 7));
  REQUIRE(luna::SquareFromUsi("0a") == luna::kSquareNone);
  REQUIRE(luna::SquareFromUsi("7j") == luna::kSquareNone);

  for (Square sq = 0; sq < luna::kSquareNb; ++sq) {
    REQUIRE(luna::MakeSquare(luna::FileOf(sq), luna::RankOf(sq)) == sq);
  }
}

TEST_CASE("a default position is the start position", "[position]") {
  const Position pos;
  REQUIRE(pos.ToSfen() == luna::kStartSfen);
  REQUIRE(pos.SideToMove() == luna::kBlack);
  REQUIRE(pos.GamePly() == 1);
  REQUIRE(pos.KingSquare(luna::kBlack) == luna::MakeSquare(5, 9));
  REQUIRE(pos.KingSquare(luna::kWhite) == luna::MakeSquare(5, 1));
  REQUIRE(pos.PieceOn(luna::MakeSquare(7, 7)) == luna::MakePiece(luna::kBlack, luna::kPawn));
}

TEST_CASE("sfen survives a round trip", "[position][sfen]") {
  const std::vector<std::string> sfens = {
      luna::kStartSfen,
      "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
      "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1",
      "9/9/9/9/9/9/9/9/9 w - 42",
  };
  for (const std::string& sfen : sfens) {
    const Position pos = FromSfen(sfen);
    REQUIRE(pos.ToSfen() == sfen);
  }
}

TEST_CASE("hand pieces parse regardless of the order they are written", "[position][sfen]") {
  const Position canonical =
      FromSfen("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1");
  const Position shuffled =
      FromSfen("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1");

  REQUIRE(shuffled.ToSfen() == canonical.ToSfen());
  REQUIRE(shuffled.Key() == canonical.Key());
  REQUIRE(shuffled.HandCount(luna::kWhite, luna::kPawn) == 5);
  REQUIRE(shuffled.HandCount(luna::kBlack, luna::kRook) == 1);
  REQUIRE(shuffled.HandCount(luna::kBlack, luna::kGold) == 1);
  REQUIRE(shuffled.HandCount(luna::kWhite, luna::kGold) == 1);
}

TEST_CASE("malformed sfen is rejected and leaves the position untouched", "[position][sfen]") {
  const std::vector<std::string> bad = {
      "",
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1 b - 1",              // eight ranks
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNLL b - 1",   // ten files
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL x - 1",    // bad side
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b K 1",    // king in hand
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - x",    // bad move number
      "+Gnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",   // golds never promote
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL+ b - 1",   // dangling '+'
  };
  Position pos;
  for (const std::string& sfen : bad) {
    INFO("sfen: " << sfen);
    REQUIRE_FALSE(pos.SetSfen(sfen));
    REQUIRE(pos.ToSfen() == luna::kStartSfen);
  }
}

TEST_CASE("do and undo restore the position exactly", "[position]") {
  Position pos;
  const std::string before = pos.ToSfen();
  const uint64_t key_before = pos.Key();

  pos.DoMove(MoveFromUsi("7g7f"));
  REQUIRE(pos.SideToMove() == luna::kWhite);
  REQUIRE(pos.GamePly() == 2);
  REQUIRE(pos.PieceOn(luna::MakeSquare(7, 6)) == luna::MakePiece(luna::kBlack, luna::kPawn));
  REQUIRE(pos.PieceOn(luna::MakeSquare(7, 7)) == luna::kNoPiece);
  REQUIRE(pos.Key() != key_before);

  pos.UndoMove();
  REQUIRE(pos.ToSfen() == before);
  REQUIRE(pos.Key() == key_before);
  REQUIRE(pos.UndoableMoves() == 0);
}

TEST_CASE("capturing puts the unpromoted piece in hand", "[position]") {
  // A promoted black pawn on 5b captures white's gold on 4a and comes back as a
  // plain pawn when the move is taken back.
  Position pos = FromSfen("4kg3/4+P4/9/9/9/9/9/9/4K4 b - 1");
  pos.DoMove(MoveFromUsi("5b4a"));

  REQUIRE(pos.HandCount(luna::kBlack, luna::kGold) == 1);
  REQUIRE(pos.PieceOn(luna::MakeSquare(4, 1)) == luna::MakePiece(luna::kBlack, luna::kProPawn));
  REQUIRE(pos.Key() == pos.ComputeKey());

  pos.UndoMove();
  REQUIRE(pos.HandCount(luna::kBlack, luna::kGold) == 0);
  REQUIRE(pos.ToSfen() == "4kg3/4+P4/9/9/9/9/9/9/4K4 b - 1");
}

TEST_CASE("promotion is undone back to the unpromoted piece", "[position]") {
  Position pos = FromSfen("4k4/9/4P4/9/9/9/9/9/4K4 b - 1");
  pos.DoMove(MoveFromUsi("5c5b+"));
  REQUIRE(pos.PieceOn(luna::MakeSquare(5, 2)) == luna::MakePiece(luna::kBlack, luna::kProPawn));

  pos.UndoMove();
  REQUIRE(pos.PieceOn(luna::MakeSquare(5, 3)) == luna::MakePiece(luna::kBlack, luna::kPawn));
  REQUIRE(pos.ToSfen() == "4k4/9/4P4/9/9/9/9/9/4K4 b - 1");
}

TEST_CASE("dropping and taking back restores the hand", "[position]") {
  Position pos = FromSfen("4k4/9/9/9/9/9/9/9/4K4 b P 1");
  pos.DoMove(MoveFromUsi("P*5e"));
  REQUIRE(pos.HandCount(luna::kBlack, luna::kPawn) == 0);
  REQUIRE(pos.PieceOn(luna::MakeSquare(5, 5)) == luna::MakePiece(luna::kBlack, luna::kPawn));
  REQUIRE(pos.Key() == pos.ComputeKey());

  pos.UndoMove();
  REQUIRE(pos.HandCount(luna::kBlack, luna::kPawn) == 1);
  REQUIRE(pos.ToSfen() == "4k4/9/9/9/9/9/9/9/4K4 b P 1");
}

TEST_CASE("the incremental key tracks the recomputed key through a game", "[position][zobrist]") {
  Position pos;
  // A deterministic walk through the move list: always take the same index so
  // the test covers captures, promotions and drops without a random source.
  const uint64_t start_key = pos.Key();
  std::vector<uint64_t> keys;
  for (int i = 0; i < 40; ++i) {
    const luna::MoveList moves = luna::movegen::GenerateLegal(pos);
    if (moves.Empty()) break;
    keys.push_back(pos.Key());
    pos.DoMove(moves[(i * 7 + 3) % moves.Size()]);
    INFO("after move " << i);
    REQUIRE(pos.Key() == pos.ComputeKey());
  }
  REQUIRE(pos.UndoableMoves() == static_cast<int>(keys.size()));

  while (pos.UndoableMoves() > 0) {
    pos.UndoMove();
    REQUIRE(pos.Key() == keys[pos.UndoableMoves()]);
    REQUIRE(pos.Key() == pos.ComputeKey());
  }
  REQUIRE(pos.Key() == start_key);
  REQUIRE(pos.ToSfen() == luna::kStartSfen);
}

TEST_CASE("attack queries see sliders, steppers and jumps", "[position][attacks]") {
  // Black rook 5e, black knight 3g, black horse 9i, white king 5a.
  const Position pos = FromSfen("4k4/9/9/9/4R4/9/6N2/9/+B8 b - 1");
  const Color black = luna::kBlack;

  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(5, 1), black));   // rook up the file
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(1, 5), black));   // rook across the rank
  REQUIRE_FALSE(pos.IsSquareAttacked(luna::MakeSquare(4, 4), black));
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(4, 5), black));   // knight jump to 4e
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(2, 5), black));   // knight jump to 2e
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(8, 8), black));   // horse diagonal
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(9, 8), black));   // horse single step
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(8, 9), black));   // horse single step
  REQUIRE_FALSE(pos.IsSquareAttacked(luna::MakeSquare(7, 9), black));
}

TEST_CASE("a rook is blocked by the piece in front of it", "[position][attacks]") {
  // Black rook 5i, black pawn 5e. The rook still "attacks" the square its own
  // pawn stands on, but nothing beyond it. 5d is reached by the pawn, not the
  // rook, so look one square further for the shielded squares.
  const Position pos = FromSfen("4k4/9/9/9/4P4/9/9/9/4RK3 b - 1");
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(5, 5), luna::kBlack));
  REQUIRE(pos.IsSquareAttacked(luna::MakeSquare(5, 4), luna::kBlack));  // the pawn
  REQUIRE_FALSE(pos.IsSquareAttacked(luna::MakeSquare(5, 3), luna::kBlack));
  REQUIRE_FALSE(pos.IsSquareAttacked(luna::MakeSquare(5, 1), luna::kBlack));
}
