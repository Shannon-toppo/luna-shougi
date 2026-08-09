#include "search/tt.hpp"

#include <catch2/catch_test_macros.hpp>

#include "core/move.hpp"
#include "search/eval.hpp"

namespace {

// A key whose low bits pick a slot and whose high bits are the stored tag.
constexpr uint64_t kKey = 0x0123456789abcdefULL;

}  // namespace

TEST_CASE("a stored entry comes back unchanged", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  const luna::Move move = luna::MoveFromUsi("7g7f");

  tt.Store(kKey, 0, 7, 123, luna::Bound::kExact, move);

  luna::TtProbe probe;
  REQUIRE(tt.Probe(kKey, 0, probe));
  REQUIRE(probe.value == 123);
  REQUIRE(probe.depth == 7);
  REQUIRE(probe.bound == luna::Bound::kExact);
  REQUIRE(probe.move == move);
}

TEST_CASE("probing an unknown key misses", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  luna::TtProbe probe;

  REQUIRE_FALSE(tt.Probe(kKey, 0, probe));

  // Same slot, different tag: the entry must not be handed out.
  tt.Store(kKey, 0, 3, 50, luna::Bound::kExact, luna::MoveFromUsi("7g7f"));
  REQUIRE_FALSE(tt.Probe(kKey ^ (1ULL << 63), 0, probe));
}

TEST_CASE("mate scores keep their distance when the ply changes", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  luna::TtProbe probe;

  // Scores count plies from the root, so the same position reached at a
  // different ply must come back with the mate the same distance away.
  const int mated_two_plies_on = -(luna::eval::kMate - 7);
  tt.Store(kKey, 5, 4, mated_two_plies_on, luna::Bound::kExact, luna::Move::None());
  REQUIRE(tt.Probe(kKey, 9, probe));
  REQUIRE(probe.value == -(luna::eval::kMate - 11));

  const int mate_three_plies_on = luna::eval::kMate - 8;
  tt.Store(kKey, 5, 4, mate_three_plies_on, luna::Bound::kExact, luna::Move::None());
  REQUIRE(tt.Probe(kKey, 9, probe));
  REQUIRE(probe.value == luna::eval::kMate - 12);
}

TEST_CASE("a plain score is not touched by the ply adjustment", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  tt.Store(kKey, 5, 4, -42, luna::Bound::kUpper, luna::Move::None());

  luna::TtProbe probe;
  REQUIRE(tt.Probe(kKey, 9, probe));
  REQUIRE(probe.value == -42);
}

TEST_CASE("a shallower result does not evict a deeper one from the same search",
          "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  tt.NewSearch();

  const uint64_t other_key = kKey ^ (1ULL << 63);  // same slot, different tag
  tt.Store(kKey, 0, 10, 100, luna::Bound::kExact, luna::Move::None());
  tt.Store(other_key, 0, 2, 200, luna::Bound::kExact, luna::Move::None());

  luna::TtProbe probe;
  REQUIRE(tt.Probe(kKey, 0, probe));
  REQUIRE(probe.value == 100);
}

TEST_CASE("entries from an earlier search are replaceable", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  tt.NewSearch();
  tt.Store(kKey, 0, 10, 100, luna::Bound::kExact, luna::Move::None());

  tt.NewSearch();
  const uint64_t other_key = kKey ^ (1ULL << 63);
  tt.Store(other_key, 0, 2, 200, luna::Bound::kExact, luna::Move::None());

  luna::TtProbe probe;
  REQUIRE_FALSE(tt.Probe(kKey, 0, probe));
  REQUIRE(tt.Probe(other_key, 0, probe));
  REQUIRE(probe.value == 200);
}

TEST_CASE("Resize gives a power-of-two entry count and empties the table", "[tt]") {
  luna::TranspositionTable tt;

  tt.Resize(1);
  const size_t count = tt.EntryCount();
  REQUIRE(count > 0);
  REQUIRE((count & (count - 1)) == 0);

  tt.Store(kKey, 0, 5, 1, luna::Bound::kExact, luna::Move::None());
  tt.Resize(2);

  luna::TtProbe probe;
  REQUIRE_FALSE(tt.Probe(kKey, 0, probe));
  REQUIRE(tt.EntryCount() == 2 * count);
}

TEST_CASE("hashfull is zero on an empty table and rises as it fills", "[tt]") {
  luna::TranspositionTable tt;
  tt.Resize(1);
  tt.NewSearch();
  REQUIRE(tt.HashFull() == 0);

  // Slots 0..499 of the 1000 sampled, so half of the sample.
  for (uint64_t i = 0; i < 500; ++i) {
    tt.Store(i, 0, 1, 0, luna::Bound::kExact, luna::Move::None());
  }
  REQUIRE(tt.HashFull() == 500);
}
