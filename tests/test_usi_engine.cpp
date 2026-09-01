#include "engine/usi_engine.hpp"

#include <chrono>
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
// The search's info lines come first; the bestmove is always the last line.
std::string BestmoveOf(const std::vector<std::string>& response) {
  REQUIRE_FALSE(response.empty());
  REQUIRE(response.back().rfind("bestmove ", 0) == 0);
  return response.back().substr(std::string("bestmove ").size());
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

  // readyok has to be last, because that is the word the GUI waits for. What
  // comes before it is the EvalFile default being applied, which is where the
  // one slow thing at startup belongs; info lines are legal anywhere.
  REQUIRE(response.back() == "readyok");
  for (size_t i = 0; i + 1 < response.size(); ++i) {
    REQUIRE(response[i].rfind("info string ", 0) == 0);
  }

  // A second one does no work and says nothing: reloading the network on
  // every liveness check would be 64MB of nothing.
  REQUIRE(engine.HandleCommand("isready") == std::vector<std::string>{"readyok"});
}

TEST_CASE("the EvalFile default is applied at isready, and only until told otherwise", "[usi]") {
  // No executable path and no eval.nnue in the working directory, so the
  // default finds nothing. That it says so is the point: the engine is about
  // to play with the weaker evaluation and must not do it quietly.
  {
    luna::UsiEngine engine;
    const auto response = engine.HandleCommand("isready");
    REQUIRE(Contains(response, "info string eval hand-written"));
    REQUIRE(Contains(response, "info string no eval.nnue"));
  }

  // An empty EvalFile is how a GUI says hand-written, and it has to survive
  // isready. Before this default existed, empty was simply the default and
  // nothing could overwrite it; now something could, so it is worth a test.
  {
    luna::UsiEngine engine;
    const auto set = engine.HandleCommand("setoption name EvalFile value ");
    REQUIRE(set == std::vector<std::string>{"info string eval hand-written"});
    REQUIRE(engine.HandleCommand("isready") == std::vector<std::string>{"readyok"});
  }

  // And the advertised default is the filename the search looks for, not a
  // blank. A GUI that echoes defaults back has to send something that works.
  {
    luna::UsiEngine engine;
    REQUIRE(Contains(engine.HandleCommand("usi"),
                     "option name EvalFile type string default eval.nnue"));
  }
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
  const std::string bestmove = BestmoveOf(engine.HandleCommand("go byoyomi 200"));

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

TEST_CASE("usi advertises the hash option", "[usi]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("usi");

  REQUIRE(Contains(response, "option name USI_Hash type spin"));
  // Pondering is not implemented, so it must not be offered.
  REQUIRE_FALSE(Contains(response, "option name USI_Ponder"));
}

TEST_CASE("setoption resizes the hash without disturbing the search", "[usi]") {
  luna::UsiEngine engine;
  engine.HandleCommand("setoption name USI_Hash value 1");
  engine.HandleCommand("position startpos");

  const auto response = engine.HandleCommand("go depth 3");

  luna::Position pos;
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(BestmoveOf(response))));
}

TEST_CASE("setoption keeps a value's spaces but not what surrounds it", "[usi]") {
  luna::UsiEngine engine;

  // A network lives wherever the user keeps it, and that path can have spaces
  // in it, so the value is everything after "value" rather than one token.
  // What it must not keep is the carriage return a GUI on the other end of a
  // pipe leaves behind: the path would then name a file that does not exist.
  //
  // Compared whole rather than by prefix: a trailing carriage return still
  // matches a prefix, which is exactly the mistake this test is here to catch.
  const auto response =
      engine.HandleCommand("setoption name EvalFile value C:\\nets\\no such net.nnue\r");
  REQUIRE_FALSE(response.empty());
  REQUIRE(response[0] == "info string EvalFile failed: cannot open C:\\nets\\no such net.nnue");

  // Whatever the path was, a failed load leaves the engine playing.
  engine.HandleCommand("position startpos");
  REQUIRE(BestmoveOf(engine.HandleCommand("go depth 2")).size() >= 4);
}

TEST_CASE("setoption with a malformed value is ignored", "[usi]") {
  luna::UsiEngine engine;
  engine.HandleCommand("setoption name USI_Hash value huge");
  engine.HandleCommand("position startpos");

  REQUIRE(BestmoveOf(engine.HandleCommand("go depth 2")).size() >= 4);
}

TEST_CASE("go reports info lines before the bestmove", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos");

  const auto response = engine.HandleCommand("go depth 3");

  REQUIRE(response.size() > 1);
  REQUIRE(Contains(response, "info depth 1"));
  REQUIRE(Contains(response, "info depth 3"));
  REQUIRE(response.back().rfind("bestmove ", 0) == 0);
  for (size_t i = 0; i + 1 < response.size(); ++i) {
    REQUIRE(response[i].rfind("info ", 0) == 0);
  }
}

TEST_CASE("go plays the mate instead of a random legal move", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position sfen 4k4/9/4GG3/9/9/9/9/9/4K4 b - 1");

  const auto response = engine.HandleCommand("go depth 3");

  REQUIRE(BestmoveOf(response) == "5c5b");
}

TEST_CASE("byoyomi bounds how long go takes", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos");

  const auto start = std::chrono::steady_clock::now();
  engine.HandleCommand("go btime 0 wtime 0 byoyomi 200");
  const auto elapsed = std::chrono::steady_clock::now() - start;

  REQUIRE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() < 3000);
}

TEST_CASE("quit during a search ends it and still answers the go", "[usi][go]") {
  // "go" no longer holds the command loop, so the "quit" behind it arrives
  // while the search is still running. Quitting has to cut the search short
  // rather than wait for it, and the search still owes the GUI a bestmove on
  // its way out. A depth this large would take far longer than the bound.
  luna::UsiEngine engine;
  std::istringstream in("position startpos\ngo depth 30\nquit\n");
  std::ostringstream out;

  const auto start = std::chrono::steady_clock::now();
  engine.Run(in, out);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  REQUIRE(out.str().find("bestmove ") != std::string::npos);
  REQUIRE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() < 3000);
  REQUIRE(engine.ShouldQuit());
}

TEST_CASE("usinewgame resets the position and the search", "[usi]") {
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos moves 7g7f");
  engine.HandleCommand("usinewgame");

  const std::string bestmove = BestmoveOf(engine.HandleCommand("go depth 2"));

  luna::Position pos;
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(bestmove)));
}

TEST_CASE("usi advertises the thread count option", "[usi][options]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("usi");

  bool found = false;
  for (const std::string& line : response) {
    if (line.rfind("option name Threads type spin", 0) == 0) found = true;
  }
  REQUIRE(found);
}

TEST_CASE("a search on several threads still returns a legal move", "[usi][go]") {
  luna::UsiEngine engine;
  engine.HandleCommand("setoption name Threads value 4");
  engine.HandleCommand("position startpos");

  const std::string bestmove = BestmoveOf(engine.HandleCommand("go depth 6"));

  luna::Position pos;
  REQUIRE(luna::movegen::IsLegal(pos, luna::MoveFromUsi(bestmove)));
}

TEST_CASE("an unreadable thread count leaves the engine working", "[usi][options]") {
  luna::UsiEngine engine;
  engine.HandleCommand("setoption name Threads value plenty");
  engine.HandleCommand("position startpos");

  REQUIRE(BestmoveOf(engine.HandleCommand("go depth 2")).size() >= 4);
}

TEST_CASE("stop ends an infinite search", "[usi][go]") {
  // Without a search thread this would never return: "go infinite" has no
  // limit of its own and the "stop" behind it would never be read.
  luna::UsiEngine engine;
  std::istringstream in("position startpos\ngo infinite\nstop\nquit\n");
  std::ostringstream out;

  engine.Run(in, out);

  REQUIRE(out.str().find("bestmove ") != std::string::npos);
}

TEST_CASE("commands are answered while a search is running", "[usi][go]") {
  // A GUI uses "isready" to check the engine is alive, including mid-search.
  // The reply has to come out before the search is over, not after it.
  luna::UsiEngine engine;
  std::istringstream in("position startpos\ngo infinite\nisready\nstop\nquit\n");
  std::ostringstream out;

  engine.Run(in, out);

  const std::string text = out.str();
  const size_t ready = text.find("readyok");
  const size_t bestmove = text.find("bestmove ");
  REQUIRE(ready != std::string::npos);
  REQUIRE(bestmove != std::string::npos);
  REQUIRE(ready < bestmove);
}
