#include "core/movegen.hpp"

#include <algorithm>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/position.hpp"

namespace {

using luna::Move;
using luna::MoveFromUsi;
using luna::MoveList;
using luna::Position;

Position FromSfen(const std::string& sfen) {
  Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

bool HasMove(Position& pos, const std::string& usi) {
  const Move m = MoveFromUsi(usi);
  REQUIRE_FALSE(m.IsNone());
  return luna::movegen::IsLegal(pos, m);
}

// Every legal move for the piece standing on `square`, sorted for comparison.
std::vector<std::string> MovesFrom(Position& pos, const std::string& square) {
  const luna::Square from = luna::SquareFromUsi(square);
  std::vector<std::string> result;
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    if (!m.IsDrop() && m.From() == from) result.push_back(luna::ToUsi(m));
  }
  std::sort(result.begin(), result.end());
  return result;
}

}  // namespace

TEST_CASE("the start position has the 30 known opening moves", "[movegen]") {
  Position pos;
  REQUIRE(luna::movegen::GenerateLegal(pos).Size() == 30);
  REQUIRE(HasMove(pos, "7g7f"));
  REQUIRE(HasMove(pos, "1g1f"));
  REQUIRE(HasMove(pos, "2h7h"));  // rook slides along the empty part of rank h
  REQUIRE(HasMove(pos, "6i7h"));
  REQUIRE(HasMove(pos, "5i5h"));
  REQUIRE_FALSE(HasMove(pos, "2h2d"));  // blocked by its own pawn on 2g
  REQUIRE_FALSE(HasMove(pos, "7g7e"));  // pawns move one square
}

TEST_CASE("white generates the mirror image of black's opening moves", "[movegen]") {
  Position pos = FromSfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1");
  REQUIRE(luna::movegen::GenerateLegal(pos).Size() == 30);
  REQUIRE(HasMove(pos, "3c3d"));
  REQUIRE(HasMove(pos, "8b3b"));
  REQUIRE(HasMove(pos, "5a5b"));
  REQUIRE_FALSE(HasMove(pos, "3c3b"));  // pawns do not move backwards
}

TEST_CASE("pawns may not be dropped onto a file that already holds one", "[movegen][nifu]") {
  Position pos = FromSfen("8k/9/9/9/9/9/4P4/9/4K4 b P 1");
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    if (m.IsDrop() && m.DroppedType() == luna::kPawn) {
      INFO("illegal 二歩 drop: " << luna::ToUsi(m));
      REQUIRE(luna::FileOf(m.To()) != 5);
    }
  }
  REQUIRE_FALSE(HasMove(pos, "P*5e"));
  REQUIRE(HasMove(pos, "P*4e"));
}

TEST_CASE("a promoted pawn does not block a pawn drop on its file", "[movegen][nifu]") {
  // 二歩 counts unpromoted pawns only, so the と金 on 5c leaves file 5 open.
  Position pos = FromSfen("8k/9/4+P4/9/9/9/9/9/4K4 b P 1");
  REQUIRE(HasMove(pos, "P*5e"));
}

TEST_CASE("a pawn may not be dropped to give mate", "[movegen][uchifuzume]") {
  // White's king on 1a is boxed in by its own lance on 2a and black's silver on
  // 1c, which also defends 1b. P*1b would be mate, so it is illegal. L*1b is
  // exactly as fatal and perfectly legal: the rule restricts pawn drops only.
  Position pos = FromSfen("7lk/9/8S/9/9/9/9/7L1/8K b LP 1");

  REQUIRE_FALSE(HasMove(pos, "P*1b"));
  REQUIRE(HasMove(pos, "L*1b"));
  REQUIRE(HasMove(pos, "P*2b"));  // an ordinary drop on the same rank
}

TEST_CASE("a pawn drop that gives check but not mate is legal", "[movegen][uchifuzume]") {
  // Same shape without the lance on 2a, so the king simply steps to 2a.
  Position pos = FromSfen("8k/9/8S/9/9/9/9/9/8K b P 1");
  REQUIRE(HasMove(pos, "P*1b"));
}

TEST_CASE("a pawn may be moved to give mate", "[movegen][uchifuzume]") {
  // Pushing a pawn into mate is legal; only dropping one is not. Black's pawn
  // on 1c walks into the same mating square as the drop above.
  Position pos = FromSfen("7lk/9/7SP/9/9/9/9/7L1/8K b - 1");
  REQUIRE(HasMove(pos, "1c1b"));
}

TEST_CASE("pawns lances and knights must promote when they would be stuck",
          "[movegen][promotion]") {
  SECTION("pawn reaching the last rank") {
    Position pos = FromSfen("8k/4P4/9/9/9/9/9/9/4K4 b - 1");
    REQUIRE(MovesFrom(pos, "5b") == std::vector<std::string>{"5b5a+"});
  }
  SECTION("lance reaching the last rank") {
    Position pos = FromSfen("8k/4L4/9/9/9/9/9/9/4K4 b - 1");
    REQUIRE(MovesFrom(pos, "5b") == std::vector<std::string>{"5b5a+"});
  }
  SECTION("knight reaching either of the last two ranks") {
    Position pos = FromSfen("8k/9/4N4/9/9/9/9/9/4K4 b - 1");
    REQUIRE(MovesFrom(pos, "5c") == std::vector<std::string>({"5c4a+", "5c6a+"}));
  }
  SECTION("a lance stopping on the second rank may decline promotion") {
    Position pos = FromSfen("8k/9/4L4/9/9/9/9/9/4K4 b - 1");
    REQUIRE(MovesFrom(pos, "5c") == std::vector<std::string>({"5c5a+", "5c5b", "5c5b+"}));
  }
}

TEST_CASE("pieces may not be dropped where they could never move", "[movegen][drops]") {
  Position pos = FromSfen("8k/9/9/9/9/9/9/9/4K4 b PLN 1");
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    if (!m.IsDrop()) continue;
    const int rank = luna::RankOf(m.To());
    INFO("drop: " << luna::ToUsi(m));
    switch (m.DroppedType()) {
      case luna::kPawn:
      case luna::kLance:
        REQUIRE(rank >= 2);
        break;
      case luna::kKnight:
        REQUIRE(rank >= 3);
        break;
      default:
        break;
    }
  }
  REQUIRE_FALSE(HasMove(pos, "N*5b"));
  REQUIRE(HasMove(pos, "N*5c"));
  REQUIRE_FALSE(HasMove(pos, "L*5a"));
  REQUIRE(HasMove(pos, "L*5b"));
  REQUIRE_FALSE(HasMove(pos, "P*5a"));
}

TEST_CASE("promotion is optional when it is not forced", "[movegen][promotion]") {
  SECTION("entering the promotion zone") {
    Position pos = FromSfen("8k/9/9/4S4/9/9/9/9/4K4 b - 1");
    REQUIRE(HasMove(pos, "5d5c"));
    REQUIRE(HasMove(pos, "5d5c+"));
  }
  SECTION("leaving the promotion zone still allows promotion") {
    // A silver steps diagonally backwards, never straight back.
    Position pos = FromSfen("8k/9/4S4/9/9/9/9/9/4K4 b - 1");
    REQUIRE(HasMove(pos, "5c4d"));
    REQUIRE(HasMove(pos, "5c4d+"));
    REQUIRE_FALSE(HasMove(pos, "5c5d"));
  }
  SECTION("a move that never touches the zone cannot promote") {
    Position pos = FromSfen("8k/9/9/9/9/4S4/9/9/4K4 b - 1");
    REQUIRE(HasMove(pos, "5f5e"));
    REQUIRE_FALSE(HasMove(pos, "5f5e+"));
  }
}

TEST_CASE("gold and king never promote", "[movegen][promotion]") {
  Position pos = FromSfen("8k/9/3GK4/9/9/9/9/9/9 b - 1");
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    INFO("move: " << luna::ToUsi(m));
    REQUIRE_FALSE(m.IsPromotion());
  }
}

TEST_CASE("an already promoted piece does not promote again", "[movegen][promotion]") {
  Position pos = FromSfen("8k/9/4+P4/9/9/9/9/9/4K4 b - 1");
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    if (m.IsDrop() || m.From() != luna::SquareFromUsi("5c")) continue;
    INFO("move: " << luna::ToUsi(m));
    REQUIRE_FALSE(m.IsPromotion());
  }
}

TEST_CASE("a pinned piece may only move along the pin", "[movegen][legality]") {
  // Black's gold on 5e is pinned to its king on 5i by white's rook on 5a.
  Position pos = FromSfen("4r3k/9/9/9/4G4/9/9/9/4K4 b - 1");
  REQUIRE(MovesFrom(pos, "5e") == std::vector<std::string>({"5e5d", "5e5f"}));
}

TEST_CASE("the king may not step into an attacked square", "[movegen][legality]") {
  // White's rook on 5a rakes the whole of file 5.
  Position pos = FromSfen("4r3k/9/9/9/9/9/9/9/4K4 b - 1");
  REQUIRE(MovesFrom(pos, "5i") == std::vector<std::string>({"5i4h", "5i4i", "5i6h", "5i6i"}));
}

TEST_CASE("every move must answer a check", "[movegen][legality]") {
  Position blocked = FromSfen("4r3k/9/9/9/9/9/9/4G4/4K4 b - 1");
  REQUIRE_FALSE(blocked.InCheck());  // the gold on 5h shields the king

  Position pos = FromSfen("4r3k/9/9/9/9/9/9/9/4K4 b G 1");
  REQUIRE(pos.InCheck());
  for (const Move m : luna::movegen::GenerateLegal(pos)) {
    pos.DoMove(m);
    INFO("move leaves the king in check: " << luna::ToUsi(m));
    REQUIRE_FALSE(pos.IsKingAttacked(luna::kBlack));
    pos.UndoMove();
  }
  REQUIRE(HasMove(pos, "G*5e"));        // interposing on the file
  REQUIRE_FALSE(HasMove(pos, "G*4e"));  // anywhere else ignores the check
}

TEST_CASE("checkmate leaves no legal move", "[movegen][legality]") {
  // Head gold mate: black's gold on 5b is backed up by the gold on 5c.
  Position pos = FromSfen("4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1");
  REQUIRE(pos.InCheck());
  REQUIRE(luna::movegen::IsCheckmate(pos));
  REQUIRE(luna::movegen::GenerateLegal(pos).Empty());
}

TEST_CASE("the known maximum of 593 legal moves is reproduced", "[movegen]") {
  Position pos = FromSfen("R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1");
  REQUIRE(luna::movegen::GenerateLegal(pos).Size() == 593);
}

TEST_CASE("every generated move survives DoMove and UndoMove", "[movegen]") {
  // 指し手生成祭り, the position the Japanese engine community uses to stress
  // move generation.
  Position pos = FromSfen("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1");
  const std::string before = pos.ToSfen();
  const uint64_t key = pos.Key();
  const MoveList moves = luna::movegen::GenerateLegal(pos);
  REQUIRE(moves.Size() == 207);

  for (const Move m : moves) {
    pos.DoMove(m);
    INFO("move: " << luna::ToUsi(m));
    REQUIRE(pos.Key() == pos.ComputeKey());
    pos.UndoMove();
    REQUIRE(pos.ToSfen() == before);
    REQUIRE(pos.Key() == key);
  }
}

TEST_CASE("pseudo-legal generation is a superset of legal generation", "[movegen]") {
  const std::vector<std::string> sfens = {
      luna::kStartSfen,
      "4r3k/9/9/9/4G4/9/9/9/4K4 b - 1",
      "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
  };
  for (const std::string& sfen : sfens) {
    Position pos = FromSfen(sfen);
    MoveList pseudo;
    luna::movegen::GeneratePseudoLegal(pos, pseudo);
    const MoveList legal = luna::movegen::GenerateLegal(pos);
    INFO("sfen: " << sfen);
    REQUIRE(legal.Size() <= pseudo.Size());
    for (const Move m : legal) REQUIRE(pseudo.Contains(m));
  }
}
