#pragma once

#include <iosfwd>
#include <optional>
#include <string>
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

// Phase 3: the USI handshake, position/go/bestmove, and an alpha-beta search
// behind "go".
//
// The search runs inside HandleCommand, so nothing is read from the GUI while
// it is thinking and "stop" cannot arrive in time to be acted on. Time
// management is what ends a search; a search thread arrives in phase 6.
class UsiEngine {
 public:
  // Processes a single line of USI input and returns the response lines to
  // write, in order. May return an empty vector for commands with no
  // response (including unrecognized commands, which are ignored).
  std::vector<std::string> HandleCommand(const std::string& line);

  // Reads commands from `in` and writes responses to `out` until a "quit"
  // command is processed or `in` reaches EOF. Unlike HandleCommand, this
  // streams "info" lines out as the search produces them.
  void Run(std::istream& in, std::ostream& out);

  bool ShouldQuit() const { return quit_requested_; }

 private:
  void HandlePosition(const std::string& line);
  void HandleSetOption(const std::string& line);
  std::vector<std::string> HandleGo(const std::string& line);

  Position position_;
  Search search_;
  Search::InfoSink info_sink_;
  bool quit_requested_ = false;
};

}  // namespace luna
