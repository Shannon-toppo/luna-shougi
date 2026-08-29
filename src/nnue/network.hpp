#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <new>
#include <string>

#include "core/types.hpp"
#include "nnue/features.hpp"
#include "nnue/simd.hpp"

namespace luna::nnue {

// The network: HalfKP -> 256 per perspective -> 512 -> 32 -> 32 -> 1.
//
// Small on purpose, and 256 is enough to be clearly stronger than a
// hand-written evaluation while keeping the feature transformer at 64MB.
//
// The width is a build option rather than a plain constant so that two
// widths can be compiled from one tree and played against each other. It has
// to be a compile-time constant either way: ReadFrom checks the file against
// it and refuses a network of a different shape, so a 512 binary cannot read
// a 256 network and the comparison has to be run as two executables.
//
// What a wider first layer costs was measured rather than assumed (bench 9,
// identical node counts, a 256 net zero-padded to 512 so that both binaries
// evaluate the same numbers):
//
//   256 -> 512 transformed dims    0.86x nps, 0.84x with eight games running
//   32  -> 64  L1 outputs          0.90x nps
//
// Which says the older note here -- that the first layer is where all the
// memory traffic goes -- was wrong at this size. Doubling the L1 outputs adds
// the same 16k multiply-accumulates as doubling the transformer does and
// costs almost as much, so the bill is the dense L1 arithmetic (~12%) and not
// the transformer's memory traffic (~5%). The transformer is updated
// incrementally, which is what makes it cheap.
#ifndef LUNA_TRANSFORMED_DIM
#define LUNA_TRANSFORMED_DIM 256
#endif
constexpr int kTransformedDim = LUNA_TRANSFORMED_DIM;
constexpr int kL1In = kTransformedDim * 2;
constexpr int kL1Out = 32;
constexpr int kL2In = kL1Out;
constexpr int kL2Out = 32;
constexpr int kL3In = kL2Out;

// --- Quantization ---------------------------------------------------------
//
// The trainer works in floats; the engine works in integers, and the two have
// to agree exactly or the net that was measured is not the net that plays.
// The scheme, which follows the one Stockfish and YaneuraOu settled on:
//
//   feature transformer  weights and biases int16, one float unit = 127
//   activation           clipped ReLU to [0, 127], stored as uint8
//   hidden layers        weights int8 at 64, biases int32 at 127*64 = 8128,
//                        so the int32 output is the float times 8128 and a
//                        shift of 6 brings it back into activation range
//   output layer         biases int32 at C*kFvScale, weights int8 at
//                        C*kFvScale/127, so the int32 output is the float
//                        times C*kFvScale
//
// One scale for both hidden layers turned out to be one too few. Measured
// across 43 checkpoints, L1 and L2 want opposite things:
//
//   max|L1 weight|  0.12 .. 0.82    a third of the range, 1.7 bits wasted
//   max|L2 weight|  0.25 .. 1.9844  pinned at the ceiling in 28 of the 43
//
// So L1 is quantized more coarsely than it needs to be while L2 is quantized
// as finely as it can be and still wants more. Giving L1 a scale of 128 halves
// its rounding error and costs nothing, because no checkpoint has ever come
// near 127/128 there. L2 is left alone: halving its limit would clip most of
// the networks that exist.
//
// Which scale L1 is on is a property of the file, not of this build, and it
// is written in the header. Generation 0 is every network built before this
// -- the header's spare word was zero in all of them, which is what makes
// them still readable here rather than merely still parseable. Nets whose
// recorded results are in docs/ can go on playing.
//
// C is the trainer's Ponanza constant: the network learns a win rate whose
// argument is a score divided by C, and the quantizer folds C into the output
// layer's numbers on the way out. So dividing by kFvScale = 16 lands in this
// engine's units, where an unpromoted pawn is 90, and it does so whatever C
// was.
//
// Nothing here needs to know C, and nothing here does. That is worth stating
// plainly, because it is not obvious and it was got wrong once: raising C from
// 600 to 1800 changes the trainer, the loss and the numbers written into the
// file, and changes nothing about this file's arithmetic. A network built at
// either constant scores correctly through the same code, which is what lets
// two of them play each other.
//
// One caveat for anyone comparing a file against the model it came from: the
// biases in it are not exactly the float bias times the scale above. Rounding
// the weights of a dense layer to int8 is unbiased per weight but not per
// layer -- every input is used by every position, so what is left is one fixed
// offset rather than noise, worth tens of points at the output and of either
// sign. training/quantize.py measures each layer's offset and folds it into
// that layer's bias, which is int32 and has the room. Nothing on this side
// changes; the engine reads the numbers it is given.
constexpr int kActivationScale = 127;
constexpr int kWeightScaleBits = 6;
constexpr int kWeightScale = 1 << kWeightScaleBits;
constexpr int kHiddenBiasScale = kActivationScale * kWeightScale;  // 8128
constexpr int kFvScale = 16;

// L1's weight scale, indexed by the file's scale generation. L2 and the
// output layer are the same in every generation, so this array is the whole
// difference between them.
constexpr int kScaleGenerationNb = 2;
constexpr int kL1WeightScaleBits[kScaleGenerationNb] = {6, 7};
constexpr uint32_t kCurrentScaleGeneration = 1;

// The evaluation is clamped here so that a broken or wildly extrapolating net
// can never produce something the search would read as a forced mate.
constexpr int kMaxEvalScore = 16000;

// --- File format ----------------------------------------------------------
//
// Little-endian throughout; every machine this runs on is.
//
//   0   8   "LUNANNUE"
//   8   4   version
//   12  4   feature dimensions
//   16  4   transformed dimensions
//   20  4   L1 outputs
//   24  4   L2 outputs
//   28  4   scale generation
//
// then, in order: ft bias, ft weights, L1 bias, L1 weights, L2 bias, L2
// weights, L3 bias, L3 weights. Weight matrices are row-major by output.
inline constexpr char kFileMagic[8] = {'L', 'U', 'N', 'A', 'N', 'N', 'U', 'E'};

// Still 1, and it has never been anything else. Moving the trainer's Ponanza
// constant briefly looked like a format change and is not one: the constant
// lives in the quantized output layer, so a file built at any value reads and
// scores correctly here. Bumping the version only broke the older networks for
// no reason. The version is for changes to the layout above.
//
// Splitting L1 onto its own scale did not bump it either, for the same reason
// and by design: the word at 28 was reserved and written as zero, so an old
// file already says "generation 0" and reads correctly. Bumping the version
// would have made this engine reject every network whose result is written
// down in docs/, which is a comparison lost for nothing.
constexpr uint32_t kFileVersion = 1;
constexpr size_t kHeaderBytes = 32;

// Memory the SIMD kernels can load from with vector instructions.
template <typename T>
class AlignedBuffer {
 public:
  AlignedBuffer() = default;
  ~AlignedBuffer() { Free(); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  void Resize(size_t count) {
    Free();
    if (count == 0) return;
    data_ = static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{kSimdAlignment}));
    size_ = count;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool Empty() const { return size_ == 0; }
  size_t Bytes() const { return size_ * sizeof(T); }

 private:
  void Free() {
    if (data_ != nullptr) ::operator delete(data_, std::align_val_t{kSimdAlignment});
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  size_t size_ = 0;
};

// The accumulated first layer, one half per perspective. Keeping both halves
// of every position on the search stack is what makes the incremental update
// possible: a move changes two or three features, so the next accumulator is
// a handful of vector adds away from this one instead of 38 of them.
struct alignas(kSimdAlignment) Accumulator {
  std::array<std::array<int16_t, kTransformedDim>, kColorNb> half{};
};

// The quantized parameters, laid out the way the file stores them.
struct Weights {
  AlignedBuffer<int16_t> ft_weight;  // [kFeatureDimensions][kTransformedDim]
  AlignedBuffer<int16_t> ft_bias;    // [kTransformedDim]
  AlignedBuffer<int8_t> l1_weight;   // [kL1Out][kL1In]
  AlignedBuffer<int32_t> l1_bias;    // [kL1Out]
  AlignedBuffer<int8_t> l2_weight;   // [kL2Out][kL2In]
  AlignedBuffer<int32_t> l2_bias;    // [kL2Out]
  AlignedBuffer<int8_t> l3_weight;   // [kL3In]
  AlignedBuffer<int32_t> l3_bias;    // [1]

  // Which scale l1_weight and l1_bias are on; an index into
  // kL1WeightScaleBits, taken from the file. Zero is the right default and
  // not just a convenient one: it is what every network written before the
  // split says, and what a hand-built one gets.
  uint32_t scale_generation = 0;

  // All zeros, which evaluates every position as a draw. Tests fill it in;
  // ReadFrom calls it before reading.
  void Allocate();
  bool Allocated() const { return !ft_bias.Empty(); }
};

class Network {
 public:
  // False leaves the network unusable and puts the reason in `error`. The
  // file is around 64MB, almost all of it the feature transformer.
  bool ReadFrom(std::istream& in, std::string& error);
  bool WriteTo(std::ostream& out) const;

  bool Loaded() const { return loaded_; }
  void Clear() { loaded_ = false; }

  // For tests and for tooling that builds a network rather than reading one.
  // Marks the network usable.
  Weights& Parameters();
  const Weights& Parameters() const { return weights_; }

  const int16_t* FeatureRow(int index) const {
    return weights_.ft_weight.data() + static_cast<size_t>(index) * kTransformedDim;
  }

  // Rebuilds one half of `acc` from nothing. `king` is the king square as
  // `perspective` sees it and `list` is in black's numbering.
  void Refresh(Accumulator& acc, Color perspective, const BonaList& list, Square king) const;

  // Runs the layers above the accumulator. The result is in engine units and
  // from `side_to_move`'s point of view.
  int Propagate(const Accumulator& acc, Color side_to_move) const;

 private:
  Weights weights_;
  bool loaded_ = false;
};

}  // namespace luna::nnue
