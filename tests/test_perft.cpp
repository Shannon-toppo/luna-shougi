#include "core/perft.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "core/position.hpp"

namespace {

using luna::Perft;
using luna::Position;

Position FromSfen(const std::string& sfen) {
  Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

void CheckPerft(const std::string& sfen, const std::vector<uint64_t>& expected) {
  Position pos = FromSfen(sfen);
  for (size_t i = 0; i < expected.size(); ++i) {
    const int depth = static_cast<int>(i) + 1;
    INFO("sfen: " << sfen << "  depth: " << depth);
    REQUIRE(Perft(pos, depth) == expected[i]);
    // Perft must leave the position exactly as it found it.
    REQUIRE(pos.ToSfen() == sfen);
  }
}

}  // namespace

TEST_CASE("perft of depth zero counts the position itself", "[perft]") {
  Position pos;
  REQUIRE(Perft(pos, 0) == 1);
}

// The reference counts everyone quotes for the initial position.
// https://groups.google.com/g/shogi-l/c/U7hmtThbk1k
TEST_CASE("perft matches the published counts for the start position", "[perft]") {
  CheckPerft(luna::kStartSfen, {30, 900, 25470, 719731, 19861490});
}

TEST_CASE("perft matches the published count at depth six", "[perft][.slow]") {
  Position pos;
  REQUIRE(Perft(pos, 6) == 547581517);
}

// Forced knight promotion near the far edge, from the TalkChess shogi perft
// thread. https://talkchess.com/viewtopic.php?t=71550
TEST_CASE("perft matches the reference counts for the knight promotion position", "[perft]") {
  CheckPerft("7k1/9/9/9/l2s5/6snb/9/9/8K b 2G 1", {76, 2606, 201658, 6768869});
}

// Same thread. Here P*1b is mate, so it is not a legal move; an engine that
// misses 打ち歩詰め counts 86 at depth one instead of 85.
TEST_CASE("perft matches the reference counts for the pawn drop mate position", "[perft]") {
  CheckPerft("7lk/9/8S/9/9/9/9/7L1/8K b P 1", {85, 639, 10786, 167089});
}

// 指し手生成祭り. No published counts to compare against, so these are this
// engine's own numbers, kept as a regression baseline over a position that is
// dense in captures, promotions and drops.
TEST_CASE("perft is stable for the move generation stress position", "[perft]") {
  CheckPerft("l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1",
             {207, 28684, 4809015});
}

TEST_CASE("perft divide sums to the plain perft count", "[perft]") {
  Position pos;
  const auto divided = luna::PerftDivide(pos, 4);
  REQUIRE(divided.size() == 30);

  uint64_t total = 0;
  for (const auto& [move, nodes] : divided) total += nodes;
  REQUIRE(total == 719731);
  REQUIRE(pos.ToSfen() == luna::kStartSfen);
}
