#pragma once

#include <functional>
#include <iosfwd>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "core/position.hpp"
#include "search/search.hpp"

namespace luna {

// Parsed "go" parameters. A field is nullopt when its token was absent from
// the command line.
struct GoParams {
  std::optional<int> btime;
  std::optional<int> wtime;
  std::optional<int> binc;
  std::optional<int> winc;
  std::optional<int> byoyomi;
  std::optional<int> movetime;
  std::optional<int> depth;
  std::optional<int> nodes;
  bool infinite = false;
  bool ponder = false;
};

// Accepts either a full "go ..." line or just its arguments.
GoParams ParseGoParams(const std::string& line);

// The USI handshake, position/go/bestmove, and an alpha-beta search behind
// "go". Two commands outside USI are understood: "eval" reports the static
// evaluation of the current position broken into its terms, and "bench"
// searches a fixed set of positions to a fixed depth.
//
// "setoption name EvalFile value <path>" loads an NNUE network, after which
// it, rather than the hand-written terms, is what "go" searches with.
// "setoption name Threads value <n>" is how many threads the search runs on.
//
// The search runs on its own thread, so commands keep being read while it
// thinks. That is what makes "stop" work and what makes "go infinite" mean
// what it says. Anything that would pull the ground out from under a running
// search — a new position, an option, a new game, quitting — stops it first.
class UsiEngine {
 public:
  UsiEngine();
  ~UsiEngine();
  UsiEngine(const UsiEngine&) = delete;
  UsiEngine& operator=(const UsiEngine&) = delete;

  // Processes a single line of USI input and returns the response lines to
  // write, in order. May return an empty vector for commands with no
  // response (including unrecognized commands, which are ignored).
  //
  // Used on its own, without Run, this stays synchronous: "go" waits for the
  // search and hands back its info lines and bestmove together. Run installs
  // an output sink, and with one installed "go" returns at once and the search
  // writes for itself.
  std::vector<std::string> HandleCommand(const std::string& line);

  // Reads commands from `in` and writes responses to `out` until a "quit"
  // command is processed or `in` reaches EOF. Unlike bare HandleCommand, the
  // search does not block the reading of the next command.
  void Run(std::istream& in, std::ostream& out);

  bool ShouldQuit() const { return quit_requested_; }

 private:
  void HandlePosition(const std::string& line);
  std::vector<std::string> HandleSetOption(const std::string& line);
  std::vector<std::string> HandleGo(const std::string& line);
  std::vector<std::string> HandleEval() const;
  std::vector<std::string> HandleBench(const std::string& line);

  void StartSearch(const SearchLimits& limits);
  // Asks the search to finish and waits for it. Safe to call when none is
  // running.
  void StopSearch();
  // Everything the search produced, once it is over.
  std::vector<std::string> WaitForSearch();

  // One line out, from whichever thread produced it.
  void Emit(const std::string& line);

  Position position_;
  Search search_;

  // The position the search thread owns, so that a "position" command arriving
  // mid-search cannot move the board out from under it.
  Position search_position_;
  std::thread search_thread_;

  // Installed by Run only. Without it, output is collected in `pending_` and
  // returned from HandleCommand instead.
  std::function<void(const std::string&)> output_;
  std::mutex output_mutex_;
  std::vector<std::string> pending_;

  bool quit_requested_ = false;
};

}  // namespace luna
