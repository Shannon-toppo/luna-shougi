#include "datagen/sample.hpp"

namespace luna::datagen {
namespace {

void PutU16(unsigned char* p, uint16_t value) {
  p[0] = static_cast<unsigned char>(value & 0xFF);
  p[1] = static_cast<unsigned char>(value >> 8);
}

uint16_t GetU16(const unsigned char* p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

}  // namespace

void Sample::Encode(unsigned char* out) const {
  for (int i = 0; i < kSampleBona; ++i) PutU16(out + 2 * i, bona[i]);
  out[76] = king_black;
  out[77] = king_white;
  out[78] = side_to_move;
  out[79] = static_cast<unsigned char>(result);
  PutU16(out + 80, static_cast<uint16_t>(score));
  PutU16(out + 82, ply);
}

void Sample::Decode(const unsigned char* in, Sample& out) {
  for (int i = 0; i < kSampleBona; ++i) out.bona[i] = GetU16(in + 2 * i);
  out.king_black = in[76];
  out.king_white = in[77];
  out.side_to_move = in[78];
  out.result = static_cast<int8_t>(in[79]);
  out.score = static_cast<int16_t>(GetU16(in + 80));
  out.ply = GetU16(in + 82);
}

bool MakeSample(const Position& pos, int score, Sample& out) {
  if (pos.KingSquare(kBlack) == kSquareNone || pos.KingSquare(kWhite) == kSquareNone) return false;

  const nnue::BonaList list = nnue::BuildBonaList(pos);
  if (list.size != kSampleBona) return false;

  for (int i = 0; i < kSampleBona; ++i) out.bona[i] = list.pieces[i];
  out.king_black = static_cast<uint8_t>(pos.KingSquare(kBlack));
  out.king_white = static_cast<uint8_t>(pos.KingSquare(kWhite));
  out.side_to_move = static_cast<uint8_t>(pos.SideToMove());
  out.result = 0;
  out.score = static_cast<int16_t>(score);
  out.ply = static_cast<uint16_t>(pos.GamePly());
  return true;
}

bool SampleWriter::Open(const std::string& path, const std::string& sfen_path, bool append) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto mode = std::ios::binary | (append ? std::ios::app : std::ios::trunc);
  out_.open(path, mode);
  if (!out_) return false;
  if (!sfen_path.empty()) {
    sfen_out_.open(sfen_path, append ? std::ios::app : std::ios::trunc);
    if (!sfen_out_) return false;
  }
  return true;
}

bool SampleWriter::Write(const std::vector<Sample>& samples,
                         const std::vector<std::string>& sfens) {
  if (samples.empty()) return true;

  std::vector<unsigned char> block(samples.size() * kSampleBytes);
  for (size_t i = 0; i < samples.size(); ++i) samples[i].Encode(block.data() + i * kSampleBytes);

  std::string text;
  for (const std::string& sfen : sfens) {
    text += sfen;
    text += '\n';
  }

  std::lock_guard<std::mutex> lock(mutex_);
  out_.write(reinterpret_cast<const char*>(block.data()),
             static_cast<std::streamsize>(block.size()));
  if (!out_) return false;
  if (sfen_out_.is_open()) {
    sfen_out_ << text;
    if (!sfen_out_) return false;
  }
  count_ += static_cast<int64_t>(samples.size());
  return true;
}

int64_t SampleWriter::Count() {
  std::lock_guard<std::mutex> lock(mutex_);
  return count_;
}

}  // namespace luna::datagen
