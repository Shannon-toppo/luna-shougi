#pragma once

#include <iosfwd>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "core/position.hpp"

namespace luna {

// Parsed "go" parameters. A field is nullopt when its token was absent from
// the command line. Phase 2's move choice ignores all of these; they are
// parsed now so phase 3's time management has somewhere to read from.
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

// Phase 2: handles the USI handshake plus position/go/bestmove. The move
// chosen by "go" is a uniform-random legal move; search arrives in phase 3.
class UsiEngine {
 public:
  // Processes a single line of USI input and returns the response lines to
  // write, in order. May return an empty vector for commands with no
  // response (including unrecognized commands, which are ignored).
  std::vector<std::string> HandleCommand(const std::string& line);

  // Reads commands from `in` and writes responses to `out` until a "quit"
  // command is processed or `in` reaches EOF.
  void Run(std::istream& in, std::ostream& out);

  bool ShouldQuit() const { return quit_requested_; }

 private:
  void HandlePosition(const std::string& line);
  std::vector<std::string> HandleGo(const std::string& line);

  Position position_;
  std::mt19937 rng_{std::random_device{}()};
  bool quit_requested_ = false;
};

}  // namespace luna
