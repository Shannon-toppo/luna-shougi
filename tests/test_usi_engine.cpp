#include "engine/usi_engine.hpp"

#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "core/position.hpp"

namespace {

bool Contains(const std::vector<std::string>& lines, const std::string& prefix) {
  for (const auto& line : lines) {
    if (line.rfind(prefix, 0) == 0) return true;
  }
  return false;
}

// Extracts the move token following "bestmove " from a HandleCommand result.
std::string BestmoveOf(const std::vector<std::string>& response) {
  REQUIRE(response.size() == 1);
  REQUIRE(response.front().rfind("bestmove ", 0) == 0);
  return response.front().substr(std::string("bestmove ").size());
}

}  // namespace

TEST_CASE("usi command performs the id/usiok handshake", "[usi]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("usi");

  REQUIRE(Contains(response, "id name"));
  REQUIRE(Contains(response, "id author"));
  REQUIRE(response.back() == "usiok");
}

TEST_CASE("isready responds with readyok", "[usi]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("isready");

  REQUIRE(response == std::vector<std::string>{"readyok"});
}

TEST_CASE("quit sets the quit flag and produces no output", "[usi]") {
  luna::UsiEngine engine;
  REQUIRE_FALSE(engine.ShouldQuit());

  const auto response = engine.HandleCommand("quit");

  REQUIRE(response.empty());
  REQUIRE(engine.ShouldQuit());
}

TEST_CASE("unimplemented commands are ignored without error", "[usi]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("position startpos moves 7g7f");

  REQUIRE(response.empty());
  REQUIRE_FALSE(engine.ShouldQuit());
}

TEST_CASE("Run processes a full handshake and stops at quit", "[usi]") {
  luna::UsiEngine engine;
  std::istringstream in("usi\nisready\nquit\n");
  std::ostringstream out;

  engine.Run(in, out);
  const std::string output = out.str();

  REQUIRE(output.find("usiok") != std::string::npos);
  REQUIRE(output.find("readyok") != std::string::npos);
}

TEST_CASE("go from the start position returns a legal move", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos");
  const std::string bestmove = BestmoveOf(engine.HandleCommand("go btime 10000 wtime 10000"));

  luna::Position pos;
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(bestmove)));
}

TEST_CASE("position replays moves before go picks from the resulting position",
          "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos moves 7g7f 3c3d");
  const std::string bestmove = BestmoveOf(engine.HandleCommand("go byoyomi 5000"));

  luna::Position pos;
  pos.SetSfen(luna::kStartSfen);
  pos.DoMove(luna::MoveFromUsi("7g7f"));
  pos.DoMove(luna::MoveFromUsi("3c3d"));
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(bestmove)));
}

TEST_CASE("position sfen sets an arbitrary position for go", "[usi][go]") {
  luna::UsiEngine engine;
  const std::string sfen = "4k4/4G4/9/9/9/9/9/9/4K4 b - 1";
  engine.HandleCommand("position sfen " + sfen);
  const std::string bestmove = BestmoveOf(engine.HandleCommand("go"));

  luna::Position pos;
  pos.SetSfen(sfen);
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(bestmove)));
}

TEST_CASE("go resigns when the side to move is checkmated", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position sfen 4k4/4G4/4G4/9/9/9/9/9/4K4 w - 1");

  REQUIRE(engine.HandleCommand("go") == std::vector<std::string>{"bestmove resign"});
}

TEST_CASE("ParseGoParams reads time control tokens", "[usi][go]") {
  const auto params = luna::ParseGoParams("go btime 10000 wtime 9000 byoyomi 5000");

  REQUIRE(params.btime == 10000);
  REQUIRE(params.wtime == 9000);
  REQUIRE(params.byoyomi == 5000);
  REQUIRE_FALSE(params.infinite);
  REQUIRE_FALSE(params.ponder);
}

TEST_CASE("ParseGoParams reads flags and increments", "[usi][go]") {
  const auto params = luna::ParseGoParams("go infinite ponder binc 1000 winc 2000");

  REQUIRE(params.infinite);
  REQUIRE(params.ponder);
  REQUIRE(params.binc == 1000);
  REQUIRE(params.winc == 2000);
  REQUIRE_FALSE(params.btime.has_value());
}
