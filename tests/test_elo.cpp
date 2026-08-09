#include "match/elo.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using Catch::Approx;
using luna::match::EloDiff;
using luna::match::EloError;
using luna::match::Los;
using luna::match::MatchScore;

TEST_CASE("the score rate counts a draw as half a win", "[elo]") {
  REQUIRE(MatchScore{}.Rate() == 0.5);
  REQUIRE(MatchScore{10, 0, 0}.Rate() == 1.0);
  REQUIRE(MatchScore{0, 10, 0}.Rate() == 0.0);
  REQUIRE(MatchScore{0, 0, 10}.Rate() == 0.5);
  REQUIRE(MatchScore{5, 3, 2}.Rate() == Approx(0.6));
}

TEST_CASE("Elo follows the logistic curve it is defined on", "[elo]") {
  REQUIRE(EloDiff(0.5) == Approx(0.0));
  REQUIRE(EloDiff(0.75) == Approx(190.848).margin(0.01));
  REQUIRE(EloDiff(0.25) == Approx(-EloDiff(0.75)));

  // A perfect score means an unbounded difference, which no finite match has
  // actually shown, so both ends are reported at the limit instead.
  REQUIRE(EloDiff(1.0) == luna::match::kEloLimit);
  REQUIRE(EloDiff(0.0) == -luna::match::kEloLimit);
}

TEST_CASE("the error bar shrinks as more games are played", "[elo]") {
  const double few = EloError(MatchScore{6, 4, 0});
  const double more = EloError(MatchScore{60, 40, 0});
  const double many = EloError(MatchScore{600, 400, 0});

  REQUIRE(few > more);
  REQUIRE(more > many);
  REQUIRE(many > 0.0);
}

TEST_CASE("a drawish match measures the difference more tightly", "[elo]") {
  // Both score 50%, but the games that were all draws pin the rating
  // difference far better than the ones that swung either way.
  REQUIRE(EloError(MatchScore{0, 0, 100}) < EloError(MatchScore{50, 50, 0}));
}

TEST_CASE("a match too short to say anything reports no error bar", "[elo]") {
  REQUIRE(EloError(MatchScore{}) == 0.0);
  REQUIRE(EloError(MatchScore{1, 0, 0}) == 0.0);
}

TEST_CASE("likelihood of superiority ignores draws", "[elo]") {
  REQUIRE(Los(MatchScore{5, 5, 0}) == Approx(0.5));
  REQUIRE(Los(MatchScore{5, 5, 90}) == Approx(0.5));
  REQUIRE(Los(MatchScore{}) == Approx(0.5));

  // Winning more than losing makes it likely, and more games at the same
  // margin makes it likelier still.
  REQUIRE(Los(MatchScore{60, 40, 0}) > 0.9);
  REQUIRE(Los(MatchScore{600, 400, 0}) > Los(MatchScore{60, 40, 0}));
  REQUIRE(Los(MatchScore{40, 60, 0}) == Approx(1.0 - Los(MatchScore{60, 40, 0})));
}

TEST_CASE("the formatted report says what happened", "[elo]") {
  const std::string score = luna::match::FormatScore(MatchScore{52, 41, 7});
  REQUIRE(score.find("52 - 41 - 7") != std::string::npos);
  REQUIRE(score.find("100 games") != std::string::npos);
  REQUIRE(score.find("55.5%") != std::string::npos);

  const std::string elo = luna::match::FormatElo(MatchScore{52, 41, 7});
  REQUIRE(elo.find("Elo +") != std::string::npos);
  REQUIRE(elo.find("+/-") != std::string::npos);
  REQUIRE(elo.find("LOS") != std::string::npos);
}
