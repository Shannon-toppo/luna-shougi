#include "nnue/network.hpp"

#include <array>
#include <cstring>
#include <istream>
#include <ostream>

namespace luna::nnue {
namespace {

constexpr size_t kFeatureWeightCount = static_cast<size_t>(kFeatureDimensions) * kTransformedDim;

uint32_t ReadU32(const unsigned char* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void WriteU32(unsigned char* p, uint32_t value) {
  p[0] = static_cast<unsigned char>(value & 0xFF);
  p[1] = static_cast<unsigned char>((value >> 8) & 0xFF);
  p[2] = static_cast<unsigned char>((value >> 16) & 0xFF);
  p[3] = static_cast<unsigned char>((value >> 24) & 0xFF);
}

template <typename T>
bool ReadBuffer(std::istream& in, AlignedBuffer<T>& buffer) {
  in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.Bytes()));
  return static_cast<size_t>(in.gcount()) == buffer.Bytes();
}

template <typename T>
bool WriteBuffer(std::ostream& out, const AlignedBuffer<T>& buffer) {
  out.write(reinterpret_cast<const char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.Bytes()));
  return static_cast<bool>(out);
}

uint8_t Clamp8(int32_t value) {
  const int32_t shifted = value >> kWeightScaleBits;
  if (shifted < 0) return 0;
  if (shifted > kActivationScale) return static_cast<uint8_t>(kActivationScale);
  return static_cast<uint8_t>(shifted);
}

}  // namespace

void Weights::Allocate() {
  ft_weight.Resize(kFeatureWeightCount);
  ft_bias.Resize(kTransformedDim);
  l1_weight.Resize(static_cast<size_t>(kL1Out) * kL1In);
  l1_bias.Resize(kL1Out);
  l2_weight.Resize(static_cast<size_t>(kL2Out) * kL2In);
  l2_bias.Resize(kL2Out);
  l3_weight.Resize(kL3In);
  l3_bias.Resize(1);

  std::memset(ft_weight.data(), 0, ft_weight.Bytes());
  std::memset(ft_bias.data(), 0, ft_bias.Bytes());
  std::memset(l1_weight.data(), 0, l1_weight.Bytes());
  std::memset(l1_bias.data(), 0, l1_bias.Bytes());
  std::memset(l2_weight.data(), 0, l2_weight.Bytes());
  std::memset(l2_bias.data(), 0, l2_bias.Bytes());
  std::memset(l3_weight.data(), 0, l3_weight.Bytes());
  std::memset(l3_bias.data(), 0, l3_bias.Bytes());
}

Weights& Network::Parameters() {
  if (!weights_.Allocated()) weights_.Allocate();
  loaded_ = true;
  return weights_;
}

bool Network::ReadFrom(std::istream& in, std::string& error) {
  loaded_ = false;

  unsigned char header[kHeaderBytes];
  in.read(reinterpret_cast<char*>(header), kHeaderBytes);
  if (static_cast<size_t>(in.gcount()) != kHeaderBytes) {
    error = "file is too short to hold a header";
    return false;
  }
  if (std::memcmp(header, kFileMagic, sizeof(kFileMagic)) != 0) {
    error = "not a luna nnue file";
    return false;
  }
  if (ReadU32(header + 8) != kFileVersion) {
    error = "unsupported file version";
    return false;
  }

  // Every dimension is fixed at compile time, so a file for a different shape
  // has to be rejected rather than reinterpreted. Saying which number is
  // wrong saves a lot of guessing when the trainer and the engine drift.
  const std::array<uint32_t, 4> want = {static_cast<uint32_t>(kFeatureDimensions),
                                        static_cast<uint32_t>(kTransformedDim),
                                        static_cast<uint32_t>(kL1Out),
                                        static_cast<uint32_t>(kL2Out)};
  const std::array<const char*, 4> names = {
      "feature dimensions", "transformed dimensions", "L1 outputs", "L2 outputs"};
  for (size_t i = 0; i < want.size(); ++i) {
    const uint32_t got = ReadU32(header + 12 + 4 * i);
    if (got != want[i]) {
      error = std::string("wrong ") + names[i] + ": file has " + std::to_string(got) +
              ", engine was built for " + std::to_string(want[i]);
      return false;
    }
  }

  if (!weights_.Allocated()) weights_.Allocate();

  const bool ok = ReadBuffer(in, weights_.ft_bias) && ReadBuffer(in, weights_.ft_weight) &&
                  ReadBuffer(in, weights_.l1_bias) && ReadBuffer(in, weights_.l1_weight) &&
                  ReadBuffer(in, weights_.l2_bias) && ReadBuffer(in, weights_.l2_weight) &&
                  ReadBuffer(in, weights_.l3_bias) && ReadBuffer(in, weights_.l3_weight);
  if (!ok) {
    error = "file ended in the middle of the weights";
    return false;
  }

  loaded_ = true;
  return true;
}

bool Network::WriteTo(std::ostream& out) const {
  if (!weights_.Allocated()) return false;

  unsigned char header[kHeaderBytes] = {};
  std::memcpy(header, kFileMagic, sizeof(kFileMagic));
  WriteU32(header + 8, kFileVersion);
  WriteU32(header + 12, static_cast<uint32_t>(kFeatureDimensions));
  WriteU32(header + 16, static_cast<uint32_t>(kTransformedDim));
  WriteU32(header + 20, static_cast<uint32_t>(kL1Out));
  WriteU32(header + 24, static_cast<uint32_t>(kL2Out));
  out.write(reinterpret_cast<const char*>(header), kHeaderBytes);

  return WriteBuffer(out, weights_.ft_bias) && WriteBuffer(out, weights_.ft_weight) &&
         WriteBuffer(out, weights_.l1_bias) && WriteBuffer(out, weights_.l1_weight) &&
         WriteBuffer(out, weights_.l2_bias) && WriteBuffer(out, weights_.l2_weight) &&
         WriteBuffer(out, weights_.l3_bias) && WriteBuffer(out, weights_.l3_weight);
}

void Network::Refresh(Accumulator& acc,
                      Color perspective,
                      const BonaList& list,
                      Square king) const {
  int16_t* half = acc.half[perspective].data();
  std::memcpy(half, weights_.ft_bias.data(), kTransformedDim * sizeof(int16_t));
  for (int i = 0; i < list.size; ++i) {
    const BonaPiece bona = perspective == kBlack ? list.pieces[i] : InvertBona(list.pieces[i]);
    AddRow(half, FeatureRow(FeatureIndex(king, bona)), kTransformedDim);
  }
}

int Network::Propagate(const Accumulator& acc, Color side_to_move) const {
  // The side to move goes first, always. The network is trained that way, and
  // it is the only thing that tells it whose turn it is.
  alignas(kSimdAlignment) std::array<uint8_t, kL1In> input{};
  ClippedRelu(acc.half[side_to_move].data(), input.data(), kTransformedDim);
  ClippedRelu(
      acc.half[Opponent(side_to_move)].data(), input.data() + kTransformedDim, kTransformedDim);

  alignas(kSimdAlignment) std::array<int32_t, kL1Out> l1{};
  Affine(
      weights_.l1_weight.data(), weights_.l1_bias.data(), input.data(), l1.data(), kL1In, kL1Out);
  alignas(kSimdAlignment) std::array<uint8_t, kL1Out> a1{};
  for (int i = 0; i < kL1Out; ++i) a1[i] = Clamp8(l1[i]);

  alignas(kSimdAlignment) std::array<int32_t, kL2Out> l2{};
  Affine(weights_.l2_weight.data(), weights_.l2_bias.data(), a1.data(), l2.data(), kL2In, kL2Out);
  alignas(kSimdAlignment) std::array<uint8_t, kL2Out> a2{};
  for (int i = 0; i < kL2Out; ++i) a2[i] = Clamp8(l2[i]);

  int32_t out = 0;
  Affine(weights_.l3_weight.data(), weights_.l3_bias.data(), a2.data(), &out, kL3In, 1);

  const int score = out / kFvScale;
  if (score > kMaxEvalScore) return kMaxEvalScore;
  if (score < -kMaxEvalScore) return -kMaxEvalScore;
  return score;
}

}  // namespace luna::nnue
