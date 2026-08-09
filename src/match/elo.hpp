#pragma once

#include <string>

namespace luna::match {

// The result of a match from one engine's point of view.
struct MatchScore {
  int wins = 0;
  int losses = 0;
  int draws = 0;

  int Games() const {
    return wins + losses + draws;
  }

  // Points per game, counting a draw as half. 0.5 for an empty match, which
  // is the only honest answer to "how did it do" before it has played.
  double Rate() const;
};

// A rate of 0 or 1 implies an infinite rating difference, which is never what
// a finite match has actually shown. Both ends are reported as this instead.
constexpr double kEloLimit = 800.0;

// Rating difference implied by a score rate, by the logistic curve Elo is
// defined on: 0.5 is level, 0.75 is about 191 points.
double EloDiff(double rate);

// Half-width of the 95% confidence interval on EloDiff(score.Rate()),
// worked out from how the individual games actually fell. This is the number
// that says whether a change is worth keeping: an improvement smaller than
// its own error bar has not been measured, only guessed at.
double EloError(const MatchScore& score);

// Likelihood of superiority — the probability that the engine really is the
// better of the two. Only decisive games carry information about which side
// is stronger, so draws do not enter.
double Los(const MatchScore& score);

// "52 - 41 - 7 (W-L-D), 100 games, 55.5%"
std::string FormatScore(const MatchScore& score);

// "Elo +38.4 +/- 66.2 (95%), LOS 87.3%"
std::string FormatElo(const MatchScore& score);

}  // namespace luna::match
