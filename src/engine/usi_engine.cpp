#include "engine/usi_engine.hpp"

#include <istream>
#include <ostream>
#include <sstream>

#include "core/move.hpp"
#include "core/movegen.hpp"
#include "engine/bench.hpp"
#include "nnue/evaluate.hpp"
#include "search/eval.hpp"
#include "search/evaldump.hpp"
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

// Path to the NNUE network. Empty means the hand-written evaluation; a GUI
// sets it through its engine settings dialog like any other string option.
//
// The advertised default is a bare filename rather than nothing, and that is
// the whole mechanism behind loading a network without being told to. A
// relative path is resolved against the engine's own directory before the
// working directory, so an engine and its `eval.nnue` sitting in one folder
// work whatever directory the GUI happens to launch from.
//
// Advertising it beats treating an empty value as "look for one anyway". Some
// GUIs send every option back at startup: with an empty default they would
// send empty and the network would never load, and with this default they
// send `eval.nnue` and it does. It also leaves empty meaning exactly what it
// has always meant -- the hand-written evaluation -- so a GUI that clears the
// field still gets what clearing it looks like.
constexpr const char* kEvalFileOption = "EvalFile";
constexpr const char* kEvalFileDefault = "eval.nnue";

// The directory part of a path, without the separator, or empty if there is
// none. Both separators, because a Windows GUI may hand over either.
std::string DirectoryOf(const std::string& path) {
  const size_t cut = path.find_last_of("/\\");
  return cut == std::string::npos ? std::string() : path.substr(0, cut);
}

// Whether a path already says where it starts from, and so must not be
// resolved against anything. "C:\..." and "C:/..." as well as the two
// separators, because this reads paths a Windows GUI wrote.
bool IsAbsolutePath(const std::string& path) {
  if (path.empty()) return false;
  if (path[0] == '/' || path[0] == '\\') return true;
  return path.size() >= 2 && path[1] == ':';
}

// How many threads the search runs on, the one it is started from included.
constexpr const char* kThreadsOption = "Threads";

// Debug: where to write every static evaluation the search takes, and how
// many to skip between the ones written. Empty path, the default, records
// nothing and costs nothing. See src/search/evaldump.hpp for why this exists.
constexpr const char* kEvalDumpOption = "EvalDump";
constexpr const char* kEvalDumpEveryOption = "EvalDumpEvery";
constexpr int64_t kMaxEvalDumpEvery = 100'000'000;
}  // namespace

UsiEngine::UsiEngine(std::string executable_path)
    : executable_dir_(DirectoryOf(executable_path)) {
  // Installed once. Where the lines end up depends on whether Run has given
  // us somewhere to write, which is decided in Emit rather than here.
  search_.SetInfoSink([this](const std::string& info) { Emit(info); });
}

UsiEngine::~UsiEngine() {
  StopSearch();
  // The search is what writes to the dump, so it has to be over first. Closing
  // here rather than leaving it to the stream's own destructor keeps a dump
  // from outliving the engine that opened it, which is what the tests need.
  eval::dump::Close();
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
    // Even an empty value counts as the GUI having chosen, so that "isready"
    // does not then load a network over the top of a deliberate blank.
    eval_file_chosen_ = true;
    return LoadEvalFile(value, false);
  }

  if (name == kEvalDumpEveryOption) {
    try {
      eval::dump::SetStride(std::stoll(value));
    } catch (const std::exception&) {
      // Same as the numeric options above: an unreadable count is ignored.
    }
    return {"info string eval dump every " + std::to_string(eval::dump::Stride())};
  }

  if (name == kEvalDumpOption) {
    if (value.empty()) {
      // Reported on the way out because the counts are the only evidence the
      // dump ran at all, and a file with nothing in it looks the same as one
      // that was never opened.
      const std::string written = std::to_string(eval::dump::Count());
      const std::string seen = std::to_string(eval::dump::Seen());
      eval::dump::Close();
      return {"info string eval dump off, " + written + " of " + seen + " evaluations written"};
    }
    std::string error;
    if (!eval::dump::Open(value, error)) {
      // Unlike EvalFile there is no falling back to be done: nothing was
      // recorded, and a measurement run has to hear about that.
      return {"info string EvalDump failed: " + error};
    }
    std::vector<std::string> response{"info string eval dump " + value + " every " +
                                      std::to_string(eval::dump::Stride())};
    if (search_.Threads() > 1) {
      // Not an error. The records are still complete and none of them are
      // torn, but the helper threads pick their own depths, so two runs of the
      // same position do not produce the same file.
      response.push_back("info string eval dump: Threads " + std::to_string(search_.Threads()) +
                         " makes the dump unreproducible; set Threads 1 to compare two runs");
    }
    return response;
  }
  return {};
}

std::vector<std::string> UsiEngine::LoadEvalFile(const std::string& value, bool from_default) {
  if (value.empty()) {
    nnue::Unload();
    return {"info string eval hand-written"};
  }

  // The engine's own directory first, so that a folder holding the engine and
  // its network works from any working directory, and the value as given
  // second, which is what an absolute path from a GUI needs.
  std::vector<std::string> candidates;
  if (!executable_dir_.empty() && !IsAbsolutePath(value)) {
    candidates.push_back(executable_dir_ + "/" + value);
  }
  candidates.push_back(value);

  std::string error;
  for (const std::string& path : candidates) {
    std::string one;
    if (nnue::Load(path, one)) {
      return {"info string eval nnue " + path + " (" + nnue::SimdName() + ")"};
    }
    if (error.empty()) error = one;
  }

  if (from_default) {
    // Nobody asked for this network, so its absence is ordinary and must not
    // read like a fault. Said out loud all the same: the engine is about to
    // play with the weaker evaluation, and silence there is the failure this
    // default exists to prevent.
    return {"info string no " + std::string(kEvalFileDefault) +
            " beside the engine; set EvalFile to use a network",
            "info string eval hand-written"};
  }
  // Falling back rather than refusing to start: a wrong path in a GUI
  // config should cost strength, not a game.
  return {"info string EvalFile failed: " + error, "info string eval hand-written"};
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
    std::ostringstream dump_every_option;
    dump_every_option << "option name " << kEvalDumpEveryOption
                      << " type spin default 1 min 1 max " << kMaxEvalDumpEvery;
    return {
        std::string("id name ") + kEngineName,
        std::string("id author ") + kEngineAuthor,
        hash_option.str(),
        threads_option.str(),
        std::string("option name ") + kEvalFileOption + " type string default " + kEvalFileDefault,
        std::string("option name ") + kEvalDumpOption + " type string default ",
        dump_every_option.str(),
        "usiok",
    };
  }
  if (cmd == "isready") {
    // Answered straight away, search running or not. A GUI uses this as a
    // liveness check and will give up on an engine that goes quiet.
    //
    // The one thing done first is the EvalFile default, and it is done here
    // rather than in the constructor because this is where USI puts the work
    // that takes a moment, and because a GUI that does send the option has to
    // get its word in before the default is applied. Once only: reloading a
    // 64MB network on every readiness check would be a lot of nothing.
    if (!eval_file_chosen_) {
      eval_file_chosen_ = true;
      std::vector<std::string> response = LoadEvalFile(kEvalFileDefault, true);
      response.push_back("readyok");
      return response;
    }
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
