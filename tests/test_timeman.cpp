#include "search/timeman.hpp"

#include <catch2/catch_test_macros.hpp>

#include "core/types.hpp"

TEST_CASE("movetime is used almost in full", "[timeman]") {
  luna::SearchLimits limits;
  limits.movetime = 5000;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  REQUIRE(budget.hard_ms == 5000 - luna::kTimeOverheadMs);
  REQUIRE(budget.soft_ms == budget.hard_ms);
}

TEST_CASE("a depth or node limit needs no clock", "[timeman]") {
  luna::SearchLimits limits;
  limits.depth = 8;

  REQUIRE(luna::ComputeTimeBudget(limits, luna::kBlack).Unlimited());

  limits = {};
  limits.nodes = 100000;
  REQUIRE(luna::ComputeTimeBudget(limits, luna::kBlack).Unlimited());
}

TEST_CASE("a bare go still gets a limit", "[timeman]") {
  const luna::TimeBudget budget = luna::ComputeTimeBudget({}, luna::kBlack);

  REQUIRE_FALSE(budget.Unlimited());
  REQUIRE(budget.hard_ms == luna::kNoClockDefaultMs);
}

TEST_CASE("byoyomi alone is spent nearly in full every move", "[timeman]") {
  luna::SearchLimits limits;
  limits.byoyomi = 5000;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  REQUIRE(budget.hard_ms == 5000 - luna::kTimeOverheadMs);
  REQUIRE(budget.soft_ms == budget.hard_ms);
}

TEST_CASE("main time is spread over the moves still to come", "[timeman]") {
  luna::SearchLimits limits;
  limits.btime = 600000;
  limits.wtime = 600000;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  REQUIRE(budget.soft_ms == 600000 / luna::kMovesToGo);
  REQUIRE(budget.hard_ms == 2 * budget.soft_ms);
}

TEST_CASE("each side reads its own clock", "[timeman]") {
  luna::SearchLimits limits;
  limits.btime = 600000;
  limits.wtime = 60000;
  limits.binc = 5000;
  limits.winc = 1000;

  const luna::TimeBudget black = luna::ComputeTimeBudget(limits, luna::kBlack);
  const luna::TimeBudget white = luna::ComputeTimeBudget(limits, luna::kWhite);

  REQUIRE(black.soft_ms == 600000 / luna::kMovesToGo + 5000);
  REQUIRE(white.soft_ms == 60000 / luna::kMovesToGo + 1000);
}

TEST_CASE("the budget never exceeds what is on the clock", "[timeman]") {
  luna::SearchLimits limits;
  limits.btime = 900;
  limits.byoyomi = 0;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  REQUIRE(budget.hard_ms <= 900 - luna::kTimeOverheadMs);
  REQUIRE(budget.soft_ms <= budget.hard_ms);
}

TEST_CASE("a nearly flat clock still leaves a positive budget", "[timeman]") {
  luna::SearchLimits limits;
  limits.btime = 1;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  REQUIRE(budget.soft_ms >= 1);
  REQUIRE(budget.hard_ms >= 1);
}

TEST_CASE("go infinite sets no time limit at all", "[timeman]") {
  luna::SearchLimits limits;
  limits.infinite = true;

  const luna::TimeBudget budget = luna::ComputeTimeBudget(limits, luna::kBlack);

  // Nothing but "stop" ends an infinite search, which is only honest now that
  // the search runs on its own thread and "stop" can arrive while it does.
  REQUIRE(budget.Unlimited());
}
