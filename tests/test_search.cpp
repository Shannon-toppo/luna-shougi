#include "search/search.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "core/position.hpp"

namespace {

luna::SearchLimits DepthLimit(int depth) {
  luna::SearchLimits limits;
  limits.depth = depth;
  return limits;
}

// Runs a fixed-depth search on `sfen` and returns the result.
luna::SearchResult SearchSfen(const std::string& sfen, int depth) {
  luna::Position pos;
  REQUIRE(pos.SetSfen(sfen));
  luna::Search search;
  return search.Think(pos, DepthLimit(depth));
}

// White king on 5a, black golds on 5c and 4c. 5c5b is mate: the gold covers
// every escape square and the one on 4c stops the king from taking it.
constexpr const char* kMateInOne = "4k4/9/4GG3/9/9/9/9/9/4K4 b - 1";

// Same idea, but the mating gold comes from hand.
constexpr const char* kMateInOneByDrop = "4k4/9/4G4/9/9/9/9/9/4K4 b G 1";

// 9c9b first, taking the whole b rank away from the white king, and whichever
// way it steps a gold drop beside it mates.
constexpr const char* kMateInThree = "4k4/9/R8/9/9/9/9/9/4K4 b G 1";

}  // namespace

TEST_CASE("search returns a legal move and counts nodes", "[search]") {
  luna::Position pos;
  luna::Search search;

  const luna::SearchResult result = search.Think(pos, DepthLimit(3));

  REQUIRE(result.depth == 3);
  REQUIRE(result.nodes > 0);
  REQUIRE(luna::movegen::IsLegal(pos, result.best));
  REQUIRE_FALSE(result.pv.empty());
  REQUIRE(result.pv.front() == result.best);
}

TEST_CASE("search leaves the position exactly as it found it", "[search]") {
  luna::Position pos;
  const std::string before = pos.ToSfen();
  const uint64_t key = pos.Key();
  luna::Search search;

  search.Think(pos, DepthLimit(4));

  REQUIRE(pos.ToSfen() == before);
  REQUIRE(pos.Key() == key);
  REQUIRE(pos.UndoableMoves() == 0);
}

TEST_CASE("search finds a mate in one", "[search][mate]") {
  const luna::SearchResult result = SearchSfen(kMateInOne, 3);

  REQUIRE(luna::IsMateScore(result.score));
  REQUIRE(luna::MateDistance(result.score) == 1);
  REQUIRE(luna::ToUsi(result.best) == "5c5b");
}

TEST_CASE("search finds a mate in one delivered by a drop", "[search][mate]") {
  const luna::SearchResult result = SearchSfen(kMateInOneByDrop, 3);

  REQUIRE(luna::IsMateScore(result.score));
  REQUIRE(luna::MateDistance(result.score) == 1);
  REQUIRE(luna::ToUsi(result.best) == "G*5b");
}

TEST_CASE("search finds a mate in three", "[search][mate]") {
  const luna::SearchResult result = SearchSfen(kMateInThree, 5);

  REQUIRE(luna::IsMateScore(result.score));
  REQUIRE(luna::MateDistance(result.score) == 3);
}

TEST_CASE("the mated side has no move to return", "[search][mate]") {
  const luna::SearchResult result = SearchSfen("4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1", 3);

  REQUIRE(result.best.IsNone());
}

TEST_CASE("search reports being mated as a negative mate score", "[search][mate]") {
  // Black's king on 1i has one square to run to, 2i, because the white rook
  // on 9h owns the h rank. G*2h then mates, and the rook defends the gold.
  const luna::SearchResult result = SearchSfen("4k4/9/9/9/9/9/9/r8/8K b g 1", 4);

  REQUIRE(luna::IsMateScore(result.score));
  REQUIRE(luna::MateDistance(result.score) == -2);
  REQUIRE(luna::ToUsi(result.best) == "1i2i");
}

TEST_CASE("search takes free material", "[search]") {
  // Black's pawn on 5e can eat an undefended rook on 5d.
  const luna::SearchResult result = SearchSfen("4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1", 3);

  REQUIRE(luna::ToUsi(result.best) == "5e5d");
  REQUIRE(result.score > 0);
}

TEST_CASE("quiescence keeps the search out of a losing capture", "[search][qsearch]") {
  // The pawn on 5d is defended by the lance on 5a. Taking it wins a pawn and
  // loses a rook, which only a search that follows the recapture can see.
  const luna::SearchResult result = SearchSfen("4l3k/9/9/4p4/4R4/9/9/9/8K b - 1", 1);

  REQUIRE(luna::ToUsi(result.best) != "5e5d");
}

TEST_CASE("a deeper search is at least as good at spotting the mate", "[search][mate]") {
  // The mate is one ply deep, so every depth from there on must find it.
  for (const int depth : {2, 3, 4, 5}) {
    const luna::SearchResult result = SearchSfen(kMateInOne, depth);
    REQUIRE(luna::MateDistance(result.score) == 1);
  }
}

TEST_CASE("a node limit bounds the search", "[search][limits]") {
  luna::Position pos;
  luna::Search search;
  luna::SearchLimits limits;
  limits.nodes = 5000;

  const luna::SearchResult result = search.Think(pos, limits);

  REQUIRE(result.nodes >= 5000);
  // The limit is only checked between nodes, so allow a small overshoot.
  REQUIRE(result.nodes < 5000 + 1000);
  REQUIRE(luna::movegen::IsLegal(pos, result.best));
}

TEST_CASE("a movetime limit bounds the search", "[search][limits]") {
  luna::Position pos;
  luna::Search search;
  luna::SearchLimits limits;
  limits.movetime = 300;

  const luna::SearchResult result = search.Think(pos, limits);

  REQUIRE(result.time_ms < 2000);
  REQUIRE(luna::movegen::IsLegal(pos, result.best));
}

TEST_CASE("one info line is emitted per completed iteration", "[search][info]") {
  luna::Position pos;
  luna::Search search;
  std::vector<std::string> lines;
  search.SetInfoSink([&lines](const std::string& line) { lines.push_back(line); });

  search.Think(pos, DepthLimit(4));

  REQUIRE(lines.size() == 4);
  for (const std::string& line : lines) {
    REQUIRE(line.rfind("info ", 0) == 0);
    REQUIRE(line.find(" score ") != std::string::npos);
    REQUIRE(line.find(" pv ") != std::string::npos);
  }
  REQUIRE(lines.back().find("depth 4") != std::string::npos);
}

TEST_CASE("a mate score is reported as score mate, not score cp", "[search][info]") {
  luna::Position pos;
  REQUIRE(pos.SetSfen(kMateInOne));
  luna::Search search;
  std::vector<std::string> lines;
  search.SetInfoSink([&lines](const std::string& line) { lines.push_back(line); });

  search.Think(pos, DepthLimit(3));

  REQUIRE_FALSE(lines.empty());
  REQUIRE(lines.back().find("score mate 1") != std::string::npos);
  REQUIRE(lines.back().find("score cp") == std::string::npos);
}

TEST_CASE("a repeated position is taken as a draw when losing", "[search]") {
  // White holds a rook, so black has nothing to play for. Both kings have
  // been shuffling; stepping back to a position already seen is worth 0,
  // which beats every move that keeps the game going a rook down.
  luna::Position pos;
  REQUIRE(pos.SetSfen("4k4/9/9/9/9/9/9/9/4K4 b r 1"));
  for (const char* move : {"5i5h", "5a5b", "5h5i", "5b5a", "5i5h", "5a5b"}) {
    pos.DoMove(luna::MoveFromUsi(move));
  }

  luna::Search search;
  const luna::SearchResult result = search.Think(pos, DepthLimit(4));

  REQUIRE(luna::ToUsi(result.best) == "5h5i");
  REQUIRE(result.score == 0);
}

TEST_CASE("NewGame clears what the previous game taught the search", "[search]") {
  luna::Position pos;
  luna::Search search;
  // A small table so that the first 1000 slots hashfull samples get used.
  search.Tt().Resize(1);
  search.Think(pos, DepthLimit(4));
  REQUIRE(search.Tt().HashFull() > 0);

  search.NewGame();

  REQUIRE(search.Tt().HashFull() == 0);
}
