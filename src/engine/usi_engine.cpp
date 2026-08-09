#include "engine/usi_engine.hpp"

#include <istream>
#include <ostream>
#include <sstream>

#include "core/move.hpp"
#include "core/movegen.hpp"

namespace luna {

namespace {
constexpr const char* kEngineName = "luna-shougi";
constexpr const char* kEngineAuthor = "Toppo";
}  // namespace

GoParams ParseGoParams(const std::string& line) {
  GoParams params;
  std::istringstream iss(line);
  std::string token;
  while (iss >> token) {
    if (token == "go") continue;
    if (token == "infinite") {
      params.infinite = true;
      continue;
    }
    if (token == "ponder") {
      params.ponder = true;
      continue;
    }
    int value = 0;
    if (!(iss >> value)) break;
    if (token == "btime") params.btime = value;
    else if (token == "wtime") params.wtime = value;
    else if (token == "binc") params.binc = value;
    else if (token == "winc") params.winc = value;
    else if (token == "byoyomi") params.byoyomi = value;
    else if (token == "movetime") params.movetime = value;
    else if (token == "depth") params.depth = value;
    else if (token == "nodes") params.nodes = value;
    // Unknown keys with a following integer (e.g. mate) are consumed and
    // ignored rather than tripping up the tokens after them.
  }
  return params;
}

void UsiEngine::HandlePosition(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  iss >> token;  // "position"
  iss >> token;  // "startpos" or "sfen"

  std::string next;
  if (token == "startpos") {
    position_ = Position();
    iss >> next;
  } else if (token == "sfen") {
    std::string sfen;
    while (iss >> next) {
      if (next == "moves") break;
      if (!sfen.empty()) sfen += ' ';
      sfen += next;
    }
    if (!position_.SetSfen(sfen)) return;
  } else {
    return;
  }

  if (next != "moves") return;

  std::string move_str;
  while (iss >> move_str) {
    const Move m = MoveFromUsi(move_str);
    if (m.IsNone() || !movegen::IsLegal(position_, m)) break;
    position_.DoMove(m);
  }
}

std::vector<std::string> UsiEngine::HandleGo(const std::string& line) {
  const GoParams params = ParseGoParams(line);
  (void)params;  // Timing and depth limits arrive with phase 3's search.

  const MoveList legal = movegen::GenerateLegal(position_);
  if (legal.Empty()) return {"bestmove resign"};

  std::uniform_int_distribution<int> dist(0, legal.Size() - 1);
  return {std::string("bestmove ") + ToUsi(legal[dist(rng_)])};
}

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
  if (cmd == "usinewgame") {
    position_ = Position();
    return {};
  }
  if (cmd == "position") {
    HandlePosition(line);
    return {};
  }
  if (cmd == "go") {
    return HandleGo(line);
  }
  if (cmd == "quit") {
    quit_requested_ = true;
    return {};
  }
  // Everything else (setoption, stop, ponderhit, gameover, ...) is silently
  // ignored: phase 2's move choice is instant, so there is never a search in
  // flight for "stop" to cancel.
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
