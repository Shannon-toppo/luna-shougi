#include "engine/usi_engine.hpp"

#include <istream>
#include <ostream>
#include <sstream>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "engine/bench.hpp"
#include "nnue/evaluate.hpp"
#include "search/eval.hpp"
#include "search/timeman.hpp"
#include "search/tt.hpp"

namespace luna {

namespace {
constexpr const char* kEngineName = "luna-shougi";
constexpr const char* kEngineAuthor = "Toppo";

// USI_Ponder is deliberately not offered. There is a search thread now, so
// holding a "go ponder" open until "ponderhit" has become possible, but the
// engine does not do it yet and a GUI must not be told otherwise.
constexpr const char* kHashOption = "USI_Hash";

// Path to the NNUE network. Empty, the default, means the hand-written
// evaluation. A GUI sets it through its engine settings dialog like any other
// string option.
constexpr const char* kEvalFileOption = "EvalFile";

// How many threads the search runs on, the one it is started from included.
constexpr const char* kThreadsOption = "Threads";
}  // namespace

UsiEngine::UsiEngine() {
  // Installed once. Where the lines end up depends on whether Run has given
  // us somewhere to write, which is decided in Emit rather than here.
  search_.SetInfoSink([this](const std::string& info) { Emit(info); });
}

UsiEngine::~UsiEngine() {
  StopSearch();
}

void UsiEngine::Emit(const std::string& line) {
  std::lock_guard<std::mutex> lock(output_mutex_);
  if (output_) {
    output_(line);
  } else {
    pending_.push_back(line);
  }
}

void UsiEngine::StartSearch(const SearchLimits& limits) {
  StopSearch();

  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    pending_.clear();
  }

  // The search gets its own copy of the board. A GUI is not supposed to send
  // "position" while a search runs, but if one does, the search finishes on
  // the position it was started with rather than on a half-changed one.
  search_position_ = position_;

  // Done here, on this thread, and never after the search thread exists. A
  // "stop" can arrive before that thread has even reached Think, and from this
  // point on every one of them counts.
  search_.ClearStop();

  search_thread_ = std::thread([this, limits] {
    const SearchResult result = search_.Think(search_position_, limits);
    Emit(result.best.IsNone() ? "bestmove resign" : "bestmove " + ToUsi(result.best));
  });
}

void UsiEngine::StopSearch() {
  if (!search_thread_.joinable()) return;
  search_.Stop();
  search_thread_.join();
}

std::vector<std::string> UsiEngine::WaitForSearch() {
  if (search_thread_.joinable()) search_thread_.join();

  std::lock_guard<std::mutex> lock(output_mutex_);
  std::vector<std::string> lines;
  lines.swap(pending_);
  return lines;
}

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

std::vector<std::string> UsiEngine::HandleSetOption(const std::string& line) {
  // setoption name <id> value <x>
  std::istringstream iss(line);
  std::string token;
  std::string name;
  iss >> token;  // "setoption"
  if (!(iss >> token) || token != "name") return {};
  if (!(iss >> name)) return {};
  if (!(iss >> token) || token != "value") return {};

  // Everything after "value" is the value, spaces included: a network sits
  // wherever the user keeps it, and that path can have spaces in it.
  std::string value;
  std::getline(iss, value);
  const size_t first = value.find_first_not_of(" \t");
  const size_t last = value.find_last_not_of(" \t\r");
  value = first == std::string::npos ? "" : value.substr(first, last - first + 1);

  // Every option below reaches into something a running search is using, so
  // none of them may be applied while one is in flight.
  StopSearch();

  if (name == kHashOption) {
    try {
      search_.Tt().Resize(static_cast<size_t>(std::stoul(value)));
    } catch (const std::exception&) {
      // A GUI sending a non-numeric size gets the current one kept.
    }
    return {};
  }

  if (name == kThreadsOption) {
    try {
      search_.SetThreads(std::stoi(value));
    } catch (const std::exception&) {
      // Same as above: an unreadable count leaves the current one alone.
    }
    return {};
  }

  if (name == kEvalFileOption) {
    if (value.empty()) {
      nnue::Unload();
      return {"info string eval hand-written"};
    }
    std::string error;
    if (!nnue::Load(value, error)) {
      // Falling back rather than refusing to start: a wrong path in a GUI
      // config should cost strength, not a game.
      return {"info string EvalFile failed: " + error, "info string eval hand-written"};
    }
    return {"info string eval nnue " + value + " (" + nnue::SimdName() + ")"};
  }
  return {};
}

std::vector<std::string> UsiEngine::HandleEval() const {
  std::ostringstream trace;
  if (nnue::IsLoaded() && nnue::CanEvaluate(position_)) {
    trace << "info string eval nnue " << nnue::Evaluate(position_) << " net " << nnue::LoadedPath()
          << " (" << nnue::SimdName() << ")";
    return {trace.str()};
  }
  const eval::EvalTerms terms = eval::Trace(position_);
  trace << "info string eval material " << terms.material << " pst " << terms.pst << " king "
        << terms.king_safety << " tempo " << terms.tempo << " total " << terms.total;
  return {trace.str()};
}

std::vector<std::string> UsiEngine::HandleBench(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  iss >> token;  // "bench"

  int depth = kDefaultBenchDepth;
  if (iss >> depth) {
    if (depth < 1) depth = 1;
  } else {
    depth = kDefaultBenchDepth;
  }

  // A bench is six searches deep enough to produce hundreds of info lines,
  // and none of them are what was asked for: the summary per position is.
  search_.SetInfoSink(nullptr);
  const BenchResult bench = RunBench(search_, depth);
  search_.SetInfoSink([this](const std::string& info) { Emit(info); });

  // The table the bench ran with is full of positions from six unrelated
  // games; leaving it in place would poison the next real search.
  search_.NewGame();

  std::vector<std::string> response;
  response.reserve(bench.lines.size());
  for (const std::string& text : bench.lines) response.push_back("info string " + text);
  return response;
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

  StartSearch(limits);

  // Run installs an output sink and reads the next command while the search
  // runs, which is what lets a "stop" get through. Without one there is nobody
  // to receive a bestmove that arrives later, so this call is the whole search.
  bool streaming;
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    streaming = static_cast<bool>(output_);
  }
  return streaming ? std::vector<std::string>{} : WaitForSearch();
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
    std::ostringstream threads_option;
    threads_option << "option name " << kThreadsOption << " type spin default 1 min 1 max "
                   << kMaxThreads;
    return {
        std::string("id name ") + kEngineName,
        std::string("id author ") + kEngineAuthor,
        hash_option.str(),
        threads_option.str(),
        std::string("option name ") + kEvalFileOption + " type string default ",
        "usiok",
    };
  }
  if (cmd == "isready") {
    // Answered straight away, search running or not. A GUI uses this as a
    // liveness check and will give up on an engine that goes quiet.
    return {"readyok"};
  }
  if (cmd == "usinewgame") {
    StopSearch();
    position_ = Position();
    search_.NewGame();
    return {};
  }
  if (cmd == "setoption") {
    return HandleSetOption(line);
  }
  if (cmd == "position") {
    StopSearch();
    HandlePosition(line);
    return {};
  }
  if (cmd == "go") {
    return HandleGo(line);
  }
  if (cmd == "stop") {
    // The search returns the best move it has so far, and the bestmove line
    // comes from the search thread on its way out, not from here.
    StopSearch();
    return {};
  }
  if (cmd == "gameover") {
    StopSearch();
    return {};
  }
  if (cmd == "quit") {
    StopSearch();
    quit_requested_ = true;
    return {};
  }
  if (cmd == "eval") {
    // Not part of USI. Being able to ask what a position scores is the only
    // practical way to tell a tuning change from a bug, and with a network
    // loaded it is also how the C++ side gets compared against the trainer.
    // Sent as "info string" so a GUI that somehow receives it just ignores it.
    return HandleEval();
  }
  if (cmd == "bench") {
    // Not part of USI either. A fixed depth over a fixed set of positions is
    // the one measurement that is exactly reproducible, so it is what says
    // whether a search change is doing anything at all.
    StopSearch();
    return HandleBench(line);
  }
  // Everything else (ponderhit, ...) is silently ignored.
  return {};
}

void UsiEngine::Run(std::istream& in, std::ostream& out) {
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    // Installing this is what puts the engine in streaming mode: "go" stops
    // waiting for its own search, and the info lines and the bestmove are
    // written by the search thread as they happen. The lock is held for every
    // write because that thread and this one both use `out`.
    output_ = [this, &out](const std::string& text) {
      out << text << "\n";
      out.flush();
    };
  }

  std::string line;
  while (!quit_requested_ && std::getline(in, line)) {
    for (const std::string& response : HandleCommand(line)) {
      Emit(response);
    }
  }

  // The stream is about to go out of scope; nothing may write to it after
  // this, so the search has to be over before the sink is taken away.
  StopSearch();
  {
    std::lock_guard<std::mutex> lock(output_mutex_);
    output_ = nullptr;
  }
}

}  // namespace luna
