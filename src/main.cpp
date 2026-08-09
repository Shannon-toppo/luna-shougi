#include <chrono>
#include <iostream>
#include <string>

#include "core/perft.hpp"
#include "core/position.hpp"
#include "engine/usi_engine.hpp"

namespace {

int RunPerft(int argc, char** argv) {
  // luna-shougi perft <depth> [sfen...]   (no sfen means the start position)
  if (argc < 3) {
    std::cerr << "usage: luna-shougi perft <depth> [sfen]\n";
    return 1;
  }

  int depth = 0;
  try {
    depth = std::stoi(argv[2]);
  } catch (const std::exception&) {
    std::cerr << "perft: depth must be an integer\n";
    return 1;
  }

  luna::Position pos;
  if (argc > 3) {
    std::string sfen = argv[3];
    for (int i = 4; i < argc; ++i) sfen += " " + std::string(argv[i]);
    if (!pos.SetSfen(sfen)) {
      std::cerr << "perft: could not parse sfen: " << sfen << "\n";
      return 1;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  const auto divided = luna::PerftDivide(pos, depth);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  uint64_t total = 0;
  for (const auto& [move, nodes] : divided) {
    std::cout << luna::ToUsi(move) << ": " << nodes << "\n";
    total += nodes;
  }
  if (depth <= 0) total = 1;

  const double seconds = std::chrono::duration<double>(elapsed).count();
  std::cout << "\nsfen  " << pos.ToSfen() << "\ndepth " << depth << "\nnodes " << total
            << "\ntime  " << seconds << "s\n";
  if (seconds > 0.0) {
    std::cout << "nps   " << static_cast<uint64_t>(total / seconds) << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "perft") return RunPerft(argc, argv);

  luna::UsiEngine engine;
  engine.Run(std::cin, std::cout);
  return 0;
}
