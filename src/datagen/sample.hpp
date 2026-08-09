#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "core/position.hpp"
#include "nnue/features.hpp"

namespace luna::datagen {

// One training position, 84 bytes, little-endian.
//
// What is stored is the BonaPiece list rather than the position, because that
// is what the trainer actually needs and it turns feature extraction on the
// Python side into two array lookups instead of a board walk. Both halves of
// HalfKP follow from these fields: black's half is (king_black, bona), and
// white's is (rotated king_white, inverted bona), with the inversion a fixed
// 1548-entry table.
//
//   0   76  uint16 bona[38]     BonaPieces in black's numbering
//   76  1   uint8  king_black
//   77  1   uint8  king_white
//   78  1   uint8  side_to_move  0 black, 1 white
//   79  1   int8   result        +1 black won, 0 draw, -1 white won
//   80  2   int16  score         search score, black's point of view
//   82  2   uint16 ply
//
// Fixed size on purpose: the dataset reader seeks to a sample by multiplying,
// which is what makes a training run resumable at an exact position in the
// data without reading everything before it.
constexpr int kSampleBytes = 84;
constexpr int kSampleBona = nnue::kActiveFeatures;

struct Sample {
  std::array<uint16_t, kSampleBona> bona{};
  uint8_t king_black = 0;
  uint8_t king_white = 0;
  uint8_t side_to_move = 0;
  int8_t result = 0;
  int16_t score = 0;
  uint16_t ply = 0;

  void Encode(unsigned char* out) const;
  static void Decode(const unsigned char* in, Sample& out);
};

// False when `pos` is not a full 40-piece position, which a real game never
// is but a hand-made SFEN can be. `score` is from black's point of view;
// `result` is filled in later, once the game has one.
bool MakeSample(const Position& pos, int score, Sample& out);

// Writes samples in blocks under a lock, so several self-play threads sharing
// one file cannot interleave inside a record.
class SampleWriter {
 public:
  // `append` keeps an existing file, which is how a run that was stopped and
  // restarted adds to the data it already made.
  // `sfen_path` may be empty; when it is not, the SFEN of every sample is
  // written alongside, line for line, which is what training/verify.py feeds
  // back to the engine.
  bool Open(const std::string& path, const std::string& sfen_path, bool append);
  bool Write(const std::vector<Sample>& samples, const std::vector<std::string>& sfens);
  int64_t Count();

 private:
  std::mutex mutex_;
  std::ofstream out_;
  std::ofstream sfen_out_;
  int64_t count_ = 0;
};

}  // namespace luna::datagen
