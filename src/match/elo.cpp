#include "match/elo.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace luna::match {
namespace {

double Squared(double x) {
  return x * x;
}

// Two-sided 95%.
constexpr double kConfidenceZ = 1.959963985;

std::string Fixed(double value, int places, bool signed_) {
  // A difference that rounds away to nothing still carries the sign of
  // whatever tiny number it came from, and "-0.0 Elo" reads as a result.
  const double scale = std::pow(10.0, places);
  double rounded = std::round(value * scale) / scale;
  if (rounded == 0.0) rounded = std::abs(rounded);

  std::ostringstream out;
  out.setf(std::ios::fixed);
  out.precision(places);
  if (signed_ && rounded >= 0.0) out << '+';
  out << rounded;
  return out.str();
}

}  // namespace

double MatchScore::Rate() const {
  const int games = Games();
  if (games == 0) return 0.5;
  return (wins + 0.5 * draws) / games;
}

double EloDiff(double rate) {
  if (rate <= 0.0) return -kEloLimit;
  if (rate >= 1.0) return kEloLimit;
  return std::clamp(-400.0 * std::log10(1.0 / rate - 1.0), -kEloLimit, kEloLimit);
}

double EloError(const MatchScore& score) {
  const int games = score.Games();
  if (games < 2) return 0.0;

  const double rate = score.Rate();
  // Spread of the per-game result, which is 1, 0.5 or 0. Taking it from the
  // games played rather than assuming a fixed draw rate is what makes the
  // interval tighten as draws pile up: a drawish match says more per game
  // about the rating difference than a wild one does.
  const double variance = (score.wins * Squared(1.0 - rate) + score.draws * Squared(0.5 - rate) +
                           score.losses * Squared(0.0 - rate)) /
                          games;
  const double error = std::sqrt(variance / games);

  const double low = EloDiff(rate - kConfidenceZ * error);
  const double high = EloDiff(rate + kConfidenceZ * error);
  return (high - low) / 2.0;
}

double Los(const MatchScore& score) {
  const int decisive = score.wins + score.losses;
  if (decisive == 0) return 0.5;
  return 0.5 * (1.0 + std::erf((score.wins - score.losses) / std::sqrt(2.0 * decisive)));
}

std::string FormatScore(const MatchScore& score) {
  std::ostringstream out;
  out << score.wins << " - " << score.losses << " - " << score.draws << " (W-L-D), "
      << score.Games() << " games, " << Fixed(100.0 * score.Rate(), 1, false) << '%';
  return out.str();
}

std::string FormatElo(const MatchScore& score) {
  std::ostringstream out;
  out << "Elo " << Fixed(EloDiff(score.Rate()), 1, true) << " +/- "
      << Fixed(EloError(score), 1, false) << " (95%), LOS " << Fixed(100.0 * Los(score), 1, false)
      << '%';
  return out.str();
}

}  // namespace luna::match
