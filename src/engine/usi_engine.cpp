#include "engine/usi_engine.hpp"

#include <istream>
#include <ostream>
#include <sstream>

namespace luna {

namespace {
constexpr const char* kEngineName = "luna-shougi";
constexpr const char* kEngineAuthor = "Toppo";
}  // namespace

std::vector<std::string> UsiEngine::HandleCommand(const std::string& line) {
  std::istringstream iss(line);
  std::string cmd;
  iss >> cmd;

  if (cmd == "usi") {
    return {
        std::string("id name ") + kEngineName,
        std::string("id author ") + kEngineAuthor,
        "usiok",
    };
  }
  if (cmd == "isready") {
    return {"readyok"};
  }
  if (cmd == "quit") {
    quit_requested_ = true;
    return {};
  }
  // Everything else (setoption, usinewgame, position, go, stop, ...) is
  // silently ignored in phase 0.
  return {};
}

void UsiEngine::Run(std::istream& in, std::ostream& out) {
  std::string line;
  while (!quit_requested_ && std::getline(in, line)) {
    for (const auto& response : HandleCommand(line)) {
      out << response << "\n";
    }
    out.flush();
  }
}

}  // namespace luna
