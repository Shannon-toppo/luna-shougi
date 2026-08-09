#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "match/process.hpp"

namespace luna::match {

struct EngineOption {
  std::string name;
  std::string value;
};

// How an engine is launched and configured.
struct EngineSpec {
  std::string path;
  std::vector<std::string> args;  // extra arguments after the path
  std::vector<EngineOption> options;
  // Shown in the match report. Defaults to whatever the engine calls itself.
  std::string label;
};

// How long to wait for an engine that is not thinking. An engine which needs
// longer than this to answer "isready" is loading something enormous, and one
// that needs longer to quit is stuck.
constexpr std::chrono::milliseconds kHandshakeTimeout{30000};

// The other side of a USI conversation: one engine, running as a child
// process, driven one command at a time.
//
// Every call is synchronous. That suits a match runner, where there is
// nothing to do between sending "go" and receiving "bestmove", and it means
// no part of this has to think about a search running in the background.
class UsiClient {
 public:
  bool Start(const EngineSpec& spec, std::string& error);
  void Quit();

  // What the engine answered to "id name", or the label from the spec.
  const std::string& Name() const {
    return name_;
  }

  bool NewGame(std::string& error);

  // Sends `position` and then `go`, and waits for the bestmove. `move` comes
  // back as the raw USI token, which may be "resign" or "win". `elapsed_ms`
  // is what the engine actually took, measured here rather than trusted from
  // the engine, because that is what a clock has to be based on.
  bool Go(const std::string& position,
          const std::string& go,
          std::chrono::milliseconds timeout,
          std::string& move,
          int64_t& elapsed_ms,
          std::string& error);

 private:
  // Reads lines until one starts with `token`. Returns false on timeout or if
  // the engine stops talking first.
  bool WaitFor(const std::string& token, std::chrono::milliseconds timeout, std::string& line);

  Process process_;
  std::string name_;
};

}  // namespace luna::match
