#include "engine/bench.hpp"

#include <array>
#include <sstream>

#include "core/position.hpp"

namespace luna {
namespace {

// Positions taken from self-play games, chosen for the shape of their search
// trees rather than for being interesting to a player. The two with large
// hands are the ones that matter: a quiescence search with nothing but
// stand-pat explodes there and nowhere else.
constexpr std::array<const char*, 6> kBenchPositions = {
    kStartSfen,
    // Opening, both sides developed, one pawn in hand each.
    "ln1gkg1nl/2r2s1p1/pp1s1p2p/2p1PbPR1/3pp4/2P6/PP1P1P2P/1BS1KS3/LN1G1G1NL b Pp 29",
    // Middlegame, closed, almost nothing in hand.
    "l3sksnl/r3g1gb1/ppn3p1p/2p1p2p1/3P1pP2/2PBP4/PPN2PNPP/L1GSKS1R1/5G2L w p 32",
    // Middlegame with a big hand on both sides: bishop, knight, lance and four
    // pawns against a silver.
    "ln1g3nl/1r1k1+B1g1/3p4p/1p3pP2/p3pr3/1PPP2S2/P2S1P2P/4KS3/LN1G1G2+p w Sbnl4p 70",
    // Both kings exposed, pieces in hand on both sides.
    "+P2gk2n1/3sr1s1l/p1ppp1p2/1b5l1/1N1L1pN1p/2G4P1/P2S1PP1P/4R1S2/L2GKG2+p b Pbn3p 61",
    // Endgame, seven pawns and two minors in hand.
    "lr1k1+Ns2/2sg2g2/pp2lp3/2+BP2N2/P4P1pP/1P2P4/7+b1/2SGKGR2/LN5+s1 b NL7Pp 77",
};

}  // namespace

BenchResult RunBench(Search& search, int depth) {
  BenchResult result;

  SearchLimits limits;
  limits.depth = depth;

  for (size_t i = 0; i < kBenchPositions.size(); ++i) {
    Position pos;
    if (!pos.SetSfen(kBenchPositions[i])) continue;

    // Every position starts from an empty table and empty history, so one
    // position's numbers do not depend on the ones searched before it.
    search.NewGame();
    const SearchResult one = search.Think(pos, limits);

    result.nodes += one.nodes;
    result.time_ms += one.time_ms;

    std::ostringstream line;
    line << "position " << (i + 1) << '/' << kBenchPositions.size() << " depth " << one.depth
         << " nodes " << one.nodes << " time " << one.time_ms << " score " << one.score
         << " bestmove " << (one.best.IsNone() ? "none" : ToUsi(one.best));
    result.lines.push_back(line.str());
  }

  std::ostringstream total;
  total << "bench depth " << depth << " nodes " << result.nodes << " time " << result.time_ms
        << " nps " << (result.time_ms > 0 ? result.nodes * 1000 / result.time_ms : 0);
  result.lines.push_back(total.str());
  return result;
}

}  // namespace luna
