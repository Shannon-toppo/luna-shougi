#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace luna {

// Phase 0: handles only the minimum USI handshake (usi / isready / quit).
// Move generation and search are added in later phases.
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
  bool quit_requested_ = false;
};

}  // namespace luna
