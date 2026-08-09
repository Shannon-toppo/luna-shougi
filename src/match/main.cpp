#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

#include "match/match.hpp"

namespace {

using luna::match::EngineOption;
using luna::match::EngineSpec;
using luna::match::MatchConfig;

constexpr const char* kUsage = R"(luna-match: play two USI engines against each other.

  luna-match --engine A [engine options] --engine B [engine options] [match options]

Engine options apply to the engine named by the most recent --engine:
  --name NAME            what to call it in the report (default: its id name)
  --arg ARG              an extra command line argument for it
  --option NAME=VALUE    a USI setoption, repeatable

Time control, pick one (default: --byoyomi 1000):
  --time MS              main time per side
  --inc MS               increment per move, with --time
  --byoyomi MS           byoyomi per move
  --movetime MS          fixed time per move
  --depth N              fixed depth per move, no clock
  --nodes N              fixed node count per move, no clock
  --margin MS            grace before a move counts as late (default 500)

Match options:
  --games N              default 100, rounded up to an even number
  --openings N           random opening plies before play (default 4)
  --seed N               opening seed (default 1)
  --maxply N             adjudicate a draw after this many moves (default 320)
  --concurrency N        games in parallel (default 1)
  --record FILE          write every finished game to FILE
  --quiet                only print the final result
)";

bool ParseInt64(const std::string& text, int64_t& out) {
  try {
    size_t consumed = 0;
    const long long value = std::stoll(text, &consumed);
    if (consumed != text.size()) return false;
    out = value;
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

class CommandLine {
 public:
  CommandLine(int argc, char** argv) : args_(argv + 1, argv + argc) {}

  // False when the arguments do not make sense; `error` says why.
  bool Parse(MatchConfig& config, std::string& record_path, bool& quiet, std::string& error);

 private:
  // The engine the engine-scoped options are currently talking about.
  EngineSpec* current_ = nullptr;
  std::vector<std::string> args_;
  size_t at_ = 0;

  bool Value(const std::string& flag, std::string& out, std::string& error) {
    if (at_ + 1 >= args_.size()) {
      error = flag + " needs a value";
      return false;
    }
    out = args_[++at_];
    return true;
  }

  bool Number(const std::string& flag, int64_t& out, std::string& error) {
    std::string text;
    if (!Value(flag, text, error)) return false;
    if (!ParseInt64(text, out)) {
      error = flag + ": " + text + " is not a number";
      return false;
    }
    return true;
  }
};

bool CommandLine::Parse(MatchConfig& config,
                        std::string& record_path,
                        bool& quiet,
                        std::string& error) {
  int engines = 0;
  bool time_control_given = false;

  for (at_ = 0; at_ < args_.size(); ++at_) {
    const std::string flag = args_[at_];
    std::string text;
    int64_t number = 0;

    if (flag == "--engine") {
      if (!Value(flag, text, error)) return false;
      if (engines >= 2) {
        error = "only two engines can play";
        return false;
      }
      current_ = engines == 0 ? &config.first : &config.second;
      current_->path = text;
      ++engines;
      continue;
    }

    if (flag == "--name" || flag == "--arg" || flag == "--option") {
      if (current_ == nullptr) {
        error = flag + " has to come after an --engine";
        return false;
      }
      if (!Value(flag, text, error)) return false;
      if (flag == "--name") {
        current_->label = text;
      } else if (flag == "--arg") {
        current_->args.push_back(text);
      } else {
        const size_t equals = text.find('=');
        if (equals == std::string::npos) {
          error = "--option wants NAME=VALUE, got " + text;
          return false;
        }
        current_->options.push_back(EngineOption{text.substr(0, equals), text.substr(equals + 1)});
      }
      continue;
    }

    if (flag == "--time" || flag == "--inc" || flag == "--byoyomi" || flag == "--movetime" ||
        flag == "--depth" || flag == "--nodes" || flag == "--margin") {
      if (!Number(flag, number, error)) return false;
      if (flag == "--time") config.tc.main_ms = number;
      if (flag == "--inc") config.tc.inc_ms = number;
      if (flag == "--byoyomi") config.tc.byoyomi_ms = number;
      if (flag == "--movetime") config.tc.movetime_ms = number;
      if (flag == "--depth") config.tc.depth = static_cast<int>(number);
      if (flag == "--nodes") config.tc.nodes = number;
      if (flag == "--margin") config.tc.margin_ms = number;
      if (flag != "--margin" && flag != "--inc") time_control_given = true;
      continue;
    }

    if (flag == "--games" || flag == "--openings" || flag == "--seed" || flag == "--maxply" ||
        flag == "--concurrency") {
      if (!Number(flag, number, error)) return false;
      if (flag == "--games") config.games = static_cast<int>(number);
      if (flag == "--openings") config.opening_plies = static_cast<int>(number);
      if (flag == "--seed") config.seed = static_cast<uint64_t>(number);
      if (flag == "--maxply") config.max_ply = static_cast<int>(number);
      if (flag == "--concurrency") config.concurrency = static_cast<int>(number);
      continue;
    }

    if (flag == "--record") {
      if (!Value(flag, record_path, error)) return false;
      continue;
    }
    if (flag == "--quiet") {
      quiet = true;
      continue;
    }

    error = "unknown option " + flag;
    return false;
  }

  if (engines != 2) {
    error = "two engines are needed; give --engine twice";
    return false;
  }
  if (config.games < 1) {
    error = "--games has to be at least 1";
    return false;
  }
  if (config.opening_plies < 0) {
    error = "--openings cannot be negative";
    return false;
  }
  if (config.max_ply < 1) {
    error = "--maxply has to be at least 1";
    return false;
  }
  if (config.concurrency < 1) {
    error = "--concurrency has to be at least 1";
    return false;
  }
  // Every opening is played twice, once with each engine as black, so an odd
  // request rounds up. Doing it here rather than inside the match keeps the
  // number printed at the start honest.
  config.games += config.games % 2;
  if (!time_control_given) config.tc.byoyomi_ms = 1000;
  return true;
}

std::string DescribeTimeControl(const luna::match::TimeControl& tc) {
  if (tc.movetime_ms > 0) return std::to_string(tc.movetime_ms) + "ms per move";
  if (tc.depth > 0) return "depth " + std::to_string(tc.depth);
  if (tc.nodes > 0) return std::to_string(tc.nodes) + " nodes per move";

  std::string text;
  if (tc.main_ms > 0) text += std::to_string(tc.main_ms) + "ms main time";
  if (tc.byoyomi_ms > 0) {
    if (!text.empty()) text += " + ";
    text += std::to_string(tc.byoyomi_ms) + "ms byoyomi";
  } else if (tc.inc_ms > 0) {
    if (!text.empty()) text += " + ";
    text += std::to_string(tc.inc_ms) + "ms increment";
  }
  return text;
}

}  // namespace

int main(int argc, char** argv) {
  MatchConfig config;
  std::string record_path;
  bool quiet = false;

  if (argc < 2) {
    std::cout << kUsage;
    return 1;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << kUsage;
      return 0;
    }
  }

  std::string error;
  if (!CommandLine(argc, argv).Parse(config, record_path, quiet, error)) {
    std::cerr << "luna-match: " << error << "\n\n" << kUsage;
    return 1;
  }

  const std::string first = config.first.label.empty() ? config.first.path : config.first.label;
  const std::string second = config.second.label.empty() ? config.second.path : config.second.label;

  std::cout << first << " vs " << second << "\n"
            << "  " << DescribeTimeControl(config.tc) << ", " << config.games << " games, "
            << config.opening_plies << " opening plies, seed " << config.seed << ", concurrency "
            << config.concurrency << "\n\n";
  std::cout.flush();

  std::mutex printing;
  const auto log = [&](const std::string& line) {
    if (quiet) return;
    std::lock_guard<std::mutex> lock(printing);
    std::cout << line << '\n';
    std::cout.flush();
  };

  const luna::match::MatchReport report = luna::match::RunMatch(config, log);

  if (!record_path.empty()) {
    std::ofstream out(record_path);
    if (!out) {
      std::cerr << "luna-match: could not write " << record_path << '\n';
    } else {
      for (const luna::match::GameRecord& game : report.games) out << game.ToUsiLine() << '\n';
    }
  }

  std::cout << '\n'
            << first << " vs " << second << ": " << FormatScore(report.score) << '\n'
            << FormatElo(report.score) << '\n';

  if (!report.error.empty()) {
    std::cerr << "luna-match: " << report.error << '\n';
    return 1;
  }
  return 0;
}
