#include "engine/usi_engine.hpp"

#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace {

bool Contains(const std::vector<std::string>& lines, const std::string& prefix) {
  for (const auto& line : lines) {
    if (line.rfind(prefix, 0) == 0) return true;
  }
  return false;
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
