#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "core/movegen.hpp"
#include "core/position.hpp"
#include "datagen/sample.hpp"
#include "mirror.hpp"
#include "nnue/evaluate.hpp"
#include "nnue/features.hpp"
#include "nnue/network.hpp"
#include "nnue/simd.hpp"
#include "search/eval.hpp"

namespace {

using luna::Color;
using luna::kBlack;
using luna::kWhite;
using luna::Move;
using luna::MoveList;
using luna::Position;
using luna::Square;
namespace nnue = luna::nnue;

// xorshift64*, so the tests are the same run to run and on every machine. A
// SIMD kernel that only fails on one input in a million is exactly the kind
// of bug this file exists to catch, and that needs the inputs to be
// reproducible when it does.
class Rng {
 public:
  explicit Rng(uint64_t seed) : state_(seed) {}

  uint64_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_;
  }

  // Uniform on [low, high].
  int Range(int low, int high) {
    return low + static_cast<int>(Next() % static_cast<uint64_t>(high - low + 1));
  }

 private:
  uint64_t state_;
};

// Weights that exercise the arithmetic without overflowing it: 38 features
// plus a bias have to stay inside an int16 accumulator, and the int8 layers
// have to stay inside what maddubs can hold.
void FillNetwork(nnue::Network& network, uint64_t seed) {
  nnue::Weights& w = network.Parameters();
  Rng rng(seed);

  for (size_t i = 0; i < w.ft_weight.size(); ++i) {
    w.ft_weight.data()[i] = static_cast<int16_t>(rng.Range(-40, 40));
  }
  for (size_t i = 0; i < w.ft_bias.size(); ++i) {
    w.ft_bias.data()[i] = static_cast<int16_t>(rng.Range(-127, 127));
  }
  for (size_t i = 0; i < w.l1_weight.size(); ++i) {
    w.l1_weight.data()[i] = static_cast<int8_t>(rng.Range(-127, 127));
  }
  for (size_t i = 0; i < w.l1_bias.size(); ++i) {
    w.l1_bias.data()[i] = rng.Range(-4000, 4000);
  }
  for (size_t i = 0; i < w.l2_weight.size(); ++i) {
    w.l2_weight.data()[i] = static_cast<int8_t>(rng.Range(-127, 127));
  }
  for (size_t i = 0; i < w.l2_bias.size(); ++i) {
    w.l2_bias.data()[i] = rng.Range(-4000, 4000);
  }
  for (size_t i = 0; i < w.l3_weight.size(); ++i) {
    w.l3_weight.data()[i] = static_cast<int8_t>(rng.Range(-127, 127));
  }
  w.l3_bias.data()[0] = rng.Range(-nnue::kOutputBiasScale, nnue::kOutputBiasScale);
}

// The evaluation with nothing cached and nothing incremental: what
// nnue::Evaluate has to agree with, however it got there.
int ReferenceEval(const nnue::Network& network, const Position& pos) {
  nnue::Accumulator acc;
  const nnue::BonaList list = nnue::BuildBonaList(pos);
  network.Refresh(acc, kBlack, list, nnue::PerspectiveKing(pos, kBlack));
  network.Refresh(acc, kWhite, list, nnue::PerspectiveKing(pos, kWhite));
  return network.Propagate(acc, pos.SideToMove());
}

std::vector<int> FeatureSet(const Position& pos, Color perspective) {
  const nnue::BonaList list = nnue::BuildBonaList(pos);
  const Square king = nnue::PerspectiveKing(pos, perspective);
  std::vector<int> indices;
  indices.reserve(static_cast<size_t>(list.size));
  for (int i = 0; i < list.size; ++i) {
    const nnue::BonaPiece bona =
        perspective == kBlack ? list.pieces[i] : nnue::InvertBona(list.pieces[i]);
    indices.push_back(nnue::FeatureIndex(king, bona));
  }
  std::sort(indices.begin(), indices.end());
  return indices;
}

Position PositionOf(const std::string& sfen) {
  Position pos;
  REQUIRE(pos.SetSfen(sfen));
  return pos;
}

// A middlegame with pieces in hand, promoted pieces on the board and both
// kings out of their starting squares. Hand-written positions are the only
// way to reach one of these without playing fifty moves first.
constexpr const char* kMiddlegameSfen =
    "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w GR5pnsg 1";

}  // namespace

TEST_CASE("inverting a BonaPiece twice is the identity", "[nnue]") {
  for (int p = 0; p < nnue::kBonaEnd; ++p) {
    const nnue::BonaPiece inverted = nnue::InvertBona(static_cast<nnue::BonaPiece>(p));
    REQUIRE(inverted < nnue::kBonaEnd);
    REQUIRE(nnue::InvertBona(inverted) == p);
  }
}

TEST_CASE("inverting a board piece swaps its colour and rotates its square", "[nnue]") {
  for (Square sq = 0; sq < luna::kSquareNb; ++sq) {
    for (const luna::PieceType pt : {luna::kPawn,
                                     luna::kLance,
                                     luna::kKnight,
                                     luna::kSilver,
                                     luna::kGold,
                                     luna::kBishop,
                                     luna::kRook,
                                     luna::kHorse,
                                     luna::kDragon,
                                     luna::kProPawn,
                                     luna::kProSilver}) {
      const nnue::BonaPiece black = nnue::BoardBona(kBlack, pt, sq);
      const nnue::BonaPiece white = nnue::BoardBona(kWhite, pt, nnue::RotateSquare(sq));
      REQUIRE(nnue::InvertBona(black) == white);
    }
  }
}

TEST_CASE("every position has one feature per piece that is not a king", "[nnue]") {
  const Position start;
  REQUIRE(nnue::BuildBonaList(start).size == nnue::kActiveFeatures);
  REQUIRE(nnue::BuildBonaList(PositionOf(kMiddlegameSfen)).size == nnue::kActiveFeatures);

  // Captured pieces move from the board to a hand, and both are features, so
  // the count cannot change however the game goes.
  Position pos = PositionOf("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1");
  const Move capture[] = {Move::Normal(luna::MakeSquare(7, 7), luna::MakeSquare(7, 6), false),
                          Move::Normal(luna::MakeSquare(3, 3), luna::MakeSquare(3, 4), false),
                          Move::Normal(luna::MakeSquare(8, 8), luna::MakeSquare(2, 2), true)};
  for (const Move m : capture) {
    REQUIRE(luna::movegen::IsLegal(pos, m));
    pos.DoMove(m);
    REQUIRE(nnue::BuildBonaList(pos).size == nnue::kActiveFeatures);
  }
}

TEST_CASE("a position and its mirror look the same from opposite sides", "[nnue]") {
  for (const char* sfen :
       {luna::kStartSfen, kMiddlegameSfen, "9/9/9/4k4/9/4K4/9/9/9 b 2R2B4G4S4N4L18P 1"}) {
    const Position pos = PositionOf(sfen);
    const Position mirrored = PositionOf(test::MirrorSfen(sfen));
    REQUIRE(FeatureSet(pos, kBlack) == FeatureSet(mirrored, kWhite));
    REQUIRE(FeatureSet(pos, kWhite) == FeatureSet(mirrored, kBlack));
  }
}

TEST_CASE("the vector kernels compute what the scalar ones do", "[nnue]") {
  Rng rng(0xC0FFEE);

  SECTION("accumulator updates") {
    constexpr int kDim = nnue::kTransformedDim;
    std::vector<int16_t> row(kDim);
    std::vector<int16_t> vector_acc(kDim);
    for (int i = 0; i < kDim; ++i) {
      row[static_cast<size_t>(i)] = static_cast<int16_t>(rng.Range(-2000, 2000));
      vector_acc[static_cast<size_t>(i)] = static_cast<int16_t>(rng.Range(-2000, 2000));
    }
    std::vector<int16_t> scalar_acc = vector_acc;

    nnue::AddRow(vector_acc.data(), row.data(), kDim);
    nnue::reference::AddRow(scalar_acc.data(), row.data(), kDim);
    REQUIRE(vector_acc == scalar_acc);

    nnue::SubRow(vector_acc.data(), row.data(), kDim);
    nnue::reference::SubRow(scalar_acc.data(), row.data(), kDim);
    REQUIRE(vector_acc == scalar_acc);
  }

  SECTION("clipped relu, including both ends of the clamp") {
    constexpr int kDim = nnue::kTransformedDim;
    std::vector<int16_t> in(kDim);
    for (int i = 0; i < kDim; ++i)
      in[static_cast<size_t>(i)] = static_cast<int16_t>(rng.Range(-300, 300));
    std::vector<uint8_t> vector_out(kDim);
    std::vector<uint8_t> scalar_out(kDim);
    nnue::ClippedRelu(in.data(), vector_out.data(), kDim);
    nnue::reference::ClippedRelu(in.data(), scalar_out.data(), kDim);
    REQUIRE(vector_out == scalar_out);
  }

  SECTION("affine layers at every shape the network uses") {
    const int shapes[][2] = {
        {nnue::kL1In, nnue::kL1Out}, {nnue::kL2In, nnue::kL2Out}, {nnue::kL3In, 1}};
    for (const auto& shape : shapes) {
      const int in_dim = shape[0];
      const int out_dim = shape[1];
      std::vector<int8_t> weights(static_cast<size_t>(in_dim) * out_dim);
      std::vector<int32_t> biases(static_cast<size_t>(out_dim));
      std::vector<uint8_t> input(static_cast<size_t>(in_dim));
      for (auto& w : weights) w = static_cast<int8_t>(rng.Range(-128, 127));
      for (auto& b : biases) b = rng.Range(-100000, 100000);
      // Clipped ReLU never produces more than 127, and the vector kernels
      // rely on that, so the test has to respect it too.
      for (auto& v : input) v = static_cast<uint8_t>(rng.Range(0, 127));

      std::vector<int32_t> vector_out(static_cast<size_t>(out_dim));
      std::vector<int32_t> scalar_out(static_cast<size_t>(out_dim));
      nnue::Affine(weights.data(), biases.data(), input.data(), vector_out.data(), in_dim, out_dim);
      nnue::reference::Affine(
          weights.data(), biases.data(), input.data(), scalar_out.data(), in_dim, out_dim);
      REQUIRE(vector_out == scalar_out);
    }
  }
}

TEST_CASE("a network survives a trip through the file format", "[nnue]") {
  nnue::Network written;
  FillNetwork(written, 12345);

  std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
  REQUIRE(written.WriteTo(stream));

  nnue::Network read;
  std::string error;
  REQUIRE(read.ReadFrom(stream, error));
  REQUIRE(error.empty());

  const nnue::Weights& a = written.Parameters();
  const nnue::Weights& b = read.Parameters();
  REQUIRE(a.ft_weight.size() == b.ft_weight.size());
  // Checking 32 million weights one at a time is not worth the seconds; a
  // stride that is coprime with the row length walks every part of the array.
  for (size_t i = 0; i < a.ft_weight.size(); i += 9973) {
    REQUIRE(a.ft_weight.data()[i] == b.ft_weight.data()[i]);
  }
  for (size_t i = 0; i < a.ft_bias.size(); ++i) {
    REQUIRE(a.ft_bias.data()[i] == b.ft_bias.data()[i]);
  }
  for (size_t i = 0; i < a.l1_weight.size(); ++i) {
    REQUIRE(a.l1_weight.data()[i] == b.l1_weight.data()[i]);
  }
  for (size_t i = 0; i < a.l3_weight.size(); ++i) {
    REQUIRE(a.l3_weight.data()[i] == b.l3_weight.data()[i]);
  }
  REQUIRE(a.l3_bias.data()[0] == b.l3_bias.data()[0]);

  const Position pos = PositionOf(kMiddlegameSfen);
  REQUIRE(ReferenceEval(written, pos) == ReferenceEval(read, pos));
}

TEST_CASE("a file that is not a network is refused rather than believed", "[nnue]") {
  nnue::Network network;
  std::string error;

  std::stringstream empty(std::ios::in | std::ios::out | std::ios::binary);
  REQUIRE_FALSE(network.ReadFrom(empty, error));
  REQUIRE_FALSE(error.empty());

  std::stringstream wrong(std::ios::in | std::ios::out | std::ios::binary);
  wrong.write("NOTANNUEnotanetworkatallreally...", 33);
  REQUIRE_FALSE(network.ReadFrom(wrong, error));

  // A header that agrees with the build but a body that stops early is the
  // one that would otherwise be read as a network of zeros.
  nnue::Network good;
  FillNetwork(good, 7);
  std::stringstream truncated(std::ios::in | std::ios::out | std::ios::binary);
  REQUIRE(good.WriteTo(truncated));
  std::string bytes = truncated.str();
  bytes.resize(bytes.size() / 2);
  std::stringstream half(bytes, std::ios::in | std::ios::binary);
  REQUIRE_FALSE(network.ReadFrom(half, error));
  REQUIRE_FALSE(network.Loaded());
}

TEST_CASE("updating an accumulator gives the same score as rebuilding it", "[nnue]") {
  nnue::Network& network = nnue::InstallNetwork();
  FillNetwork(network, 99);
  nnue::ClearCache();

  Rng rng(4242);
  Position pos;
  MoveList moves;

  // A random game rather than a scripted one: 240 plies from the start
  // position runs through captures, promotions, drops and king moves in
  // whatever order they turn up, which is the point.
  for (int ply = 0; ply < 240; ++ply) {
    // GenerateLegal appends, so the list has to be emptied first.
    moves.Clear();
    luna::movegen::GenerateLegal(pos, moves);
    if (moves.Empty()) break;
    pos.DoMove(moves[rng.Range(0, moves.Size() - 1)]);

    // Not every ply, so that the walk back has to cross more than one move
    // sometimes, the way it does when a search skips evaluating a node.
    if (rng.Range(0, 2) == 0) continue;
    REQUIRE(nnue::Evaluate(pos) == ReferenceEval(network, pos));
  }

  // Taking moves back and playing different ones leaves stale accumulators
  // behind at exactly the plies about to be reused.
  for (int i = 0; i < 40 && pos.UndoableMoves() > 0; ++i) pos.UndoMove();
  for (int ply = 0; ply < 60; ++ply) {
    moves.Clear();
    luna::movegen::GenerateLegal(pos, moves);
    if (moves.Empty()) break;
    pos.DoMove(moves[rng.Range(0, moves.Size() - 1)]);
    REQUIRE(nnue::Evaluate(pos) == ReferenceEval(network, pos));
  }

  nnue::Unload();
}

TEST_CASE("a king move rebuilds only the half that needs it", "[nnue]") {
  nnue::Network& network = nnue::InstallNetwork();
  FillNetwork(network, 5150);
  nnue::ClearCache();

  // 5i5h and 5a5b are king moves; the ones between them are not, so the
  // rebuild has to happen in the middle of a chain of incremental updates.
  Position pos;
  const Move script[] = {
      Move::Normal(luna::MakeSquare(5, 9), luna::MakeSquare(5, 8), false),
      Move::Normal(luna::MakeSquare(5, 1), luna::MakeSquare(5, 2), false),
      Move::Normal(luna::MakeSquare(7, 7), luna::MakeSquare(7, 6), false),
      Move::Normal(luna::MakeSquare(3, 3), luna::MakeSquare(3, 4), false),
      Move::Normal(luna::MakeSquare(5, 8), luna::MakeSquare(4, 8), false),
      Move::Normal(luna::MakeSquare(5, 2), luna::MakeSquare(6, 2), false),
  };
  for (const Move m : script) {
    REQUIRE(luna::movegen::IsLegal(pos, m));
    pos.DoMove(m);
    REQUIRE(nnue::Evaluate(pos) == ReferenceEval(network, pos));
  }

  nnue::Unload();
}

TEST_CASE("a training sample survives a trip through its record", "[datagen]") {
  const Position pos = PositionOf(kMiddlegameSfen);
  luna::datagen::Sample sample;
  REQUIRE(luna::datagen::MakeSample(pos, -437, sample));
  sample.result = -1;

  REQUIRE(sample.king_black == pos.KingSquare(kBlack));
  REQUIRE(sample.king_white == pos.KingSquare(kWhite));
  REQUIRE(sample.side_to_move == static_cast<uint8_t>(kWhite));
  REQUIRE(sample.score == -437);

  unsigned char bytes[luna::datagen::kSampleBytes];
  sample.Encode(bytes);
  luna::datagen::Sample back;
  luna::datagen::Sample::Decode(bytes, back);

  REQUIRE(back.bona == sample.bona);
  REQUIRE(back.king_black == sample.king_black);
  REQUIRE(back.king_white == sample.king_white);
  REQUIRE(back.side_to_move == sample.side_to_move);
  REQUIRE(back.result == sample.result);
  REQUIRE(back.score == sample.score);
  REQUIRE(back.ply == sample.ply);
}

TEST_CASE("a position with no king falls back to the hand-written terms", "[nnue]") {
  nnue::Network& network = nnue::InstallNetwork();
  FillNetwork(network, 1);

  const Position kingless = PositionOf("9/9/9/9/4p4/9/9/9/9 b - 1");
  REQUIRE_FALSE(nnue::CanEvaluate(kingless));
  // eval::Evaluate has to answer for it anyway rather than index a feature
  // table with a square that does not exist.
  REQUIRE(luna::eval::Evaluate(kingless) == luna::eval::Trace(kingless).total);

  nnue::Unload();
}
