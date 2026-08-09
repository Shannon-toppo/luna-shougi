#include "engine/usi_engine.hpp"

#include <istream>
#include <ostream>
#include <sstream>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "search/eval.hpp"
#include "search/timeman.hpp"
#include "search/tt.hpp"

namespace luna {

namespace {
constexpr const char* kEngineName = "luna-shougi";
constexpr const char* kEngineAuthor = "Toppo";

// USI_Ponder is deliberately not offered: without a search thread there is no
// way to hold a "go ponder" open until "ponderhit", so a GUI must not be told
// pondering works.
constexpr const char* kHashOption = "USI_Hash";
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

void UsiEngine::HandleSetOption(const std::string& line) {
  // setoption name <id> value <x>
  std::istringstream iss(line);
  std::string token;
  std::string name;
  std::string value;
  iss >> token;  // "setoption"
  if (!(iss >> token) || token != "name") return;
  if (!(iss >> name)) return;
  if (!(iss >> token) || token != "value") return;
  if (!(iss >> value)) return;

  if (name == kHashOption) {
    try {
      search_.Tt().Resize(static_cast<size_t>(std::stoul(value)));
    } catch (const std::exception&) {
      // A GUI sending a non-numeric size gets the current one kept.
    }
  }
}

std::vector<std::string> UsiEngine::HandleGo(const std::string& line) {
  const GoParams params = ParseGoParams(line);

  SearchLimits limits;
  limits.btime = params.btime.value_or(0);
  limits.wtime = params.wtime.value_or(0);
  limits.binc = params.binc.value_or(0);
  limits.winc = params.winc.value_or(0);
  limits.byoyomi = params.byoyomi.value_or(0);
  limits.movetime = params.movetime.value_or(0);
  limits.depth = params.depth.value_or(0);
  limits.nodes = params.nodes.value_or(0);
  limits.infinite = params.infinite;

  // With no sink installed by Run, the info lines travel back with the
  // bestmove instead of being streamed. Tests read them that way.
  std::vector<std::string> response;
  if (info_sink_) {
    search_.SetInfoSink(info_sink_);
  } else {
    search_.SetInfoSink([&response](const std::string& info) { response.push_back(info); });
  }

  const SearchResult result = search_.Think(position_, limits);
  search_.SetInfoSink(nullptr);

  response.push_back(result.best.IsNone() ? "bestmove resign"
                                          : "bestmove " + ToUsi(result.best));
  return response;
}

std::vector<std::string> UsiEngine::HandleCommand(const std::string& line) {
  std::istringstream iss(line);
  std::string cmd;
  iss >> cmd;

  if (cmd == "usi") {
    std::ostringstream hash_option;
    hash_option << "option name " << kHashOption << " type spin default "
                << TranspositionTable::kDefaultSizeMb << " min 1 max "
                << TranspositionTable::kMaxSizeMb;
    return {
        std::string("id name ") + kEngineName,
        std::string("id author ") + kEngineAuthor,
        hash_option.str(),
        "usiok",
    };
  }
  if (cmd == "isready") {
    return {"readyok"};
  }
  if (cmd == "usinewgame") {
    position_ = Position();
    search_.NewGame();
    return {};
  }
  if (cmd == "setoption") {
    HandleSetOption(line);
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
  if (cmd == "eval") {
    // Not part of USI. The evaluation is hand-written and its terms pull
    // against each other, so being able to ask what a position scores and why
    // is the only practical way to tell a tuning change from a bug. Sent as
    // "info string" so a GUI that somehow receives it just ignores it.
    const eval::EvalTerms terms = eval::Trace(position_);
    std::ostringstream trace;
    trace << "info string eval material " << terms.material << " pst " << terms.pst << " king "
          << terms.king_safety << " tempo " << terms.tempo << " total " << terms.total;
    return {trace.str()};
  }
  // Everything else (stop, ponderhit, gameover, ...) is silently ignored.
  // "stop" in particular cannot be honoured: the search runs inside this same
  // call, so nothing reads input while it is in flight.
  return {};
}

void UsiEngine::Run(std::istream& in, std::ostream& out) {
  // Streaming the search's info lines is the whole reason a GUI shows a depth
  // counter ticking up instead of freezing until the move appears.
  info_sink_ = [&out](const std::string& info) {
    out << info << "\n";
    out.flush();
  };

  std::string line;
  while (!quit_requested_ && std::getline(in, line)) {
    for (const auto& response : HandleCommand(line)) {
      out << response << "\n";
    }
    out.flush();
  }

  info_sink_ = nullptr;
}

}  // namespace luna
