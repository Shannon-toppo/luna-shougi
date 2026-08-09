#include "match/usi_client.hpp"

#include <sstream>

namespace luna::match {
namespace {

// The first whitespace-separated word of `line`.
std::string FirstToken(const std::string& line) {
  std::istringstream iss(line);
  std::string token;
  iss >> token;
  return token;
}

}  // namespace

bool UsiClient::WaitFor(const std::string& token,
                        std::chrono::milliseconds timeout,
                        std::string& line) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    const auto left = deadline - std::chrono::steady_clock::now();
    if (left <= std::chrono::steady_clock::duration::zero()) return false;
    if (!process_.ReadLine(line, std::chrono::duration_cast<std::chrono::milliseconds>(left))) {
      return false;
    }
    if (FirstToken(line) == token) return true;
    // Anything else is an info line, an id line, an option line or noise the
    // engine wrote to standard error. None of it is our business here.
    if (FirstToken(line) == "id") {
      std::istringstream iss(line);
      std::string word;
      iss >> word >> word;
      if (word == "name" && std::getline(iss, word)) {
        const size_t start = word.find_first_not_of(' ');
        if (start != std::string::npos) name_ = word.substr(start);
      }
    }
  }
}

bool UsiClient::Start(const EngineSpec& spec, std::string& error) {
  name_ = spec.label.empty() ? spec.path : spec.label;

  std::vector<std::string> args = {spec.path};
  args.insert(args.end(), spec.args.begin(), spec.args.end());
  if (!process_.Start(args, error)) return false;

  std::string line;
  if (!process_.WriteLine("usi")) {
    error = spec.path + ": the engine closed its input";
    return false;
  }
  if (!WaitFor("usiok", kHandshakeTimeout, line)) {
    error = spec.path + ": no usiok";
    return false;
  }
  // A label given on the command line wins over what the engine calls itself:
  // the whole point of a match is that the two sides can share a name.
  if (!spec.label.empty()) name_ = spec.label;

  for (const EngineOption& option : spec.options) {
    if (!process_.WriteLine("setoption name " + option.name + " value " + option.value)) {
      error = spec.path + ": the engine closed its input";
      return false;
    }
  }

  if (!process_.WriteLine("isready") || !WaitFor("readyok", kHandshakeTimeout, line)) {
    error = spec.path + ": no readyok";
    return false;
  }
  return true;
}

bool UsiClient::NewGame(std::string& error) {
  std::string line;
  if (!process_.WriteLine("usinewgame") || !process_.WriteLine("isready") ||
      !WaitFor("readyok", kHandshakeTimeout, line)) {
    error = name_ + ": did not answer isready before the game";
    return false;
  }
  return true;
}

bool UsiClient::Go(const std::string& position,
                   const std::string& go,
                   std::chrono::milliseconds timeout,
                   std::string& move,
                   int64_t& elapsed_ms,
                   std::string& error) {
  const auto start = std::chrono::steady_clock::now();
  if (!process_.WriteLine(position) || !process_.WriteLine(go)) {
    error = name_ + ": the engine closed its input";
    return false;
  }

  std::string line;
  if (!WaitFor("bestmove", timeout, line)) {
    error = name_ + ": no bestmove within " + std::to_string(timeout.count()) + "ms";
    return false;
  }
  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - start)
                   .count();

  std::istringstream iss(line);
  std::string token;
  iss >> token >> move;
  if (move.empty()) {
    error = name_ + ": bestmove with no move";
    return false;
  }
  return true;
}

void UsiClient::Quit() {
  process_.WriteLine("quit");
  process_.Stop();
}

}  // namespace luna::match
