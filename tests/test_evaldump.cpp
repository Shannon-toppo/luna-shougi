#include "search/evaldump.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "datagen/datagen.hpp"
#include "datagen/sample.hpp"
#include "engine/usi_engine.hpp"
#include "search/eval.hpp"

namespace {

// A file that deletes itself. The dump and the labeller both exist to write
// files, so there is no testing either of them without one, and a test that
// leaves rubbish behind in the temp directory is its own small bug.
class ScratchFile {
 public:
  explicit ScratchFile(const std::string& name) {
    path_ = (std::filesystem::temp_directory_path() / ("luna-test-" + name)).string();
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  ~ScratchFile() {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }
  ScratchFile(const ScratchFile&) = delete;
  ScratchFile& operator=(const ScratchFile&) = delete;

  const std::string& Path() const { return path_; }

  bool Exists() const {
    std::error_code ec;
    return std::filesystem::exists(path_, ec);
  }

  void Write(const std::string& text) const {
    std::ofstream out(path_, std::ios::trunc);
    out << text;
  }

  std::vector<std::string> Lines() const {
    std::vector<std::string> lines;
    std::ifstream in(path_);
    std::string line;
    while (std::getline(in, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
    }
    return lines;
  }

 private:
  std::string path_;
};

struct DumpRecord {
  std::string site;
  int score = 0;
  int ply = 0;
  int worker = 0;
  std::string sfen;
};

// Reads back a dump, header and all. The header is checked here rather than
// skipped silently: it is what tells a reader by hand what the columns are.
std::vector<DumpRecord> ReadDump(const ScratchFile& file) {
  std::vector<DumpRecord> records;
  const std::vector<std::string> lines = file.Lines();
  REQUIRE_FALSE(lines.empty());
  REQUIRE(lines.front().rfind("# site", 0) == 0);

  for (size_t i = 1; i < lines.size(); ++i) {
    std::istringstream fields(lines[i]);
    DumpRecord record;
    std::string score;
    std::string ply;
    std::string worker;
    REQUIRE(std::getline(fields, record.site, '\t'));
    REQUIRE(std::getline(fields, score, '\t'));
    REQUIRE(std::getline(fields, ply, '\t'));
    REQUIRE(std::getline(fields, worker, '\t'));
    REQUIRE(std::getline(fields, record.sfen));
    record.score = std::stoi(score);
    record.ply = std::stoi(ply);
    record.worker = std::stoi(worker);
    records.push_back(record);
  }
  return records;
}

// Runs one search with the dump pointed at `file`, and closes it again.
void SearchWithDump(const ScratchFile& file, const std::string& go, const std::string& every = "") {
  luna::UsiEngine engine;
  if (!every.empty()) {
    engine.HandleCommand("setoption name EvalDumpEvery value " + every);
  }
  const auto opened = engine.HandleCommand("setoption name EvalDump value " + file.Path());
  REQUIRE_FALSE(opened.empty());
  REQUIRE(opened.front().rfind("info string eval dump ", 0) == 0);

  engine.HandleCommand("position startpos");
  engine.HandleCommand(go);
  engine.HandleCommand("setoption name EvalDump value ");
}

}  // namespace

TEST_CASE("the evaluation dump is off until it is asked for", "[evaldump]") {
  REQUIRE_FALSE(luna::eval::dump::Enabled());

  // A whole search with no dump open leaves no file and turns nothing on.
  luna::UsiEngine engine;
  engine.HandleCommand("position startpos");
  engine.HandleCommand("go depth 4");

  REQUIRE_FALSE(luna::eval::dump::Enabled());
  REQUIRE(luna::eval::dump::Count() == 0);
}

TEST_CASE("the dump records the score the search actually used", "[evaldump]") {
  // The reason this file exists at all: a record has to be the number the
  // search worked with, on the position it worked on. Every line is replayed
  // through the same evaluation to check that it is.
  ScratchFile file("dump-scores.tsv");
  SearchWithDump(file, "go depth 5");

  const std::vector<DumpRecord> records = ReadDump(file);
  REQUIRE(records.size() > 100);

  for (const DumpRecord& record : records) {
    luna::Position pos;
    REQUIRE(pos.SetSfen(record.sfen));
    REQUIRE(luna::eval::Evaluate(pos) == record.score);
  }
}

TEST_CASE("the dump says which of the search's evaluations each record is",
          "[evaldump]") {
  ScratchFile file("dump-sites.tsv");
  SearchWithDump(file, "go depth 6");

  const std::vector<DumpRecord> records = ReadDump(file);
  int stand_pat = 0;
  int prune = 0;
  for (const DumpRecord& record : records) {
    if (record.site == "standpat") ++stand_pat;
    if (record.site == "prune") ++prune;
    REQUIRE(record.ply >= 0);
    REQUIRE(record.worker == 0);

    // Neither of the two common sites is ever reached in check, which is what
    // lets the analysis treat the site column as the answer to "was this
    // position settled" without having to recompute it.
    if (record.site == "standpat" || record.site == "prune") {
      luna::Position pos;
      REQUIRE(pos.SetSfen(record.sfen));
      REQUIRE_FALSE(pos.InCheck());
    }
  }

  // A depth-6 search prunes and it quiesces, so both have to show up. The
  // horizon site is not required: reaching ply 128 takes a pathological
  // position.
  REQUIRE(stand_pat > 0);
  REQUIRE(prune > 0);
}

TEST_CASE("the dump stride keeps one evaluation in every N", "[evaldump]") {
  ScratchFile all("dump-stride-1.tsv");
  SearchWithDump(all, "go depth 5", "1");
  const size_t total = ReadDump(all).size();

  ScratchFile sampled("dump-stride-16.tsv");
  SearchWithDump(sampled, "go depth 5", "16");
  const size_t kept = ReadDump(sampled).size();

  REQUIRE(total > 100);
  // The same search either way, so the count is exact rather than close.
  REQUIRE(kept == (total + 15) / 16);
}

TEST_CASE("closing the dump reports what it wrote and stops recording",
          "[evaldump]") {
  ScratchFile file("dump-close.tsv");

  luna::UsiEngine engine;
  engine.HandleCommand("setoption name EvalDump value " + file.Path());
  engine.HandleCommand("position startpos");
  engine.HandleCommand("go depth 4");

  const auto closed = engine.HandleCommand("setoption name EvalDump value ");
  REQUIRE(closed.size() == 1);
  REQUIRE(closed.front().rfind("info string eval dump off, ", 0) == 0);
  REQUIRE(closed.front().find(" of ") != std::string::npos);
  REQUIRE_FALSE(luna::eval::dump::Enabled());

  const size_t written = ReadDump(file).size();
  REQUIRE(written > 0);

  // A second search after the close must not add to the file.
  engine.HandleCommand("go depth 4");
  REQUIRE(ReadDump(file).size() == written);
}

TEST_CASE("a dump that cannot be opened says so instead of going quiet",
          "[evaldump]") {
  // The failure this guards against is the one from docs/nnue-gen1.md: an
  // option that silently does nothing produces a measurement session with no
  // data and no complaint.
  luna::UsiEngine engine;
  const auto response =
      engine.HandleCommand("setoption name EvalDump value no-such-directory/dump.tsv");

  REQUIRE(response.size() == 1);
  REQUIRE(response.front().rfind("info string EvalDump failed:", 0) == 0);
  REQUIRE_FALSE(luna::eval::dump::Enabled());
}

TEST_CASE("the usi handshake offers the dump options", "[evaldump][usi]") {
  luna::UsiEngine engine;
  const auto response = engine.HandleCommand("usi");

  bool dump = false;
  bool every = false;
  for (const std::string& line : response) {
    if (line.rfind("option name EvalDump type string", 0) == 0) dump = true;
    if (line.rfind("option name EvalDumpEvery type spin", 0) == 0) every = true;
  }
  REQUIRE(dump);
  REQUIRE(every);
}

TEST_CASE("labelling searches every position and keeps the input order",
          "[datagen][label]") {
  // Three positions with obviously different material, so that a mix-up
  // between a line and its label cannot pass.
  const std::vector<std::string> sfens = {
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
      "lnsgkgsnl/1r5b1/pppppp1pp/6p2/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL b - 3",
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
  };

  ScratchFile input("label-in.sfen");
  ScratchFile bin("label-out.bin");
  ScratchFile sfen_out("label-out.sfen");

  // A header line and a blank one, because a dump file is fed straight in.
  std::string text = "# site\tscore\tply\tworker\tsfen\n\n";
  for (const std::string& sfen : sfens) text += sfen + "\n";
  input.Write(text);

  luna::datagen::Config config;
  config.out_path = bin.Path();
  config.sfen_path = sfen_out.Path();
  config.label_path = input.Path();
  config.depth = 3;
  config.concurrency = 3;

  const luna::datagen::Report report = luna::datagen::Run(config, nullptr);

  REQUIRE(report.error.empty());
  REQUIRE(report.samples == 3);
  REQUIRE(report.skipped == 0);

  // The .sfen has to come back in the order it went in, or every label in the
  // .bin belongs to a different position than the one beside it.
  REQUIRE(sfen_out.Lines() == sfens);

  std::ifstream in(bin.Path(), std::ios::binary);
  std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
  REQUIRE(bytes.size() == sfens.size() * luna::datagen::kSampleBytes);

  for (size_t i = 0; i < sfens.size(); ++i) {
    luna::datagen::Sample sample;
    luna::datagen::Sample::Decode(bytes.data() + i * luna::datagen::kSampleBytes, sample);

    luna::Position pos;
    REQUIRE(pos.SetSfen(sfens[i]));
    REQUIRE(sample.side_to_move == static_cast<uint8_t>(pos.SideToMove()));
    REQUIRE(sample.ply == static_cast<uint16_t>(pos.GamePly()));
    // No game around these positions, so nothing won or lost. Recorded here
    // because it is the reason this output must not be trained on.
    REQUIRE(sample.result == 0);
  }
}

TEST_CASE("labelling reports the lines it could not use", "[datagen][label]") {
  ScratchFile input("label-bad-in.sfen");
  ScratchFile bin("label-bad-out.bin");

  input.Write(
      "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1\n"
      "this is not a position\n"
      // Legal to write down, but not the full forty pieces HalfKP needs.
      "9/9/9/4k4/9/4K4/9/9/9 b - 1\n");

  luna::datagen::Config config;
  config.out_path = bin.Path();
  config.label_path = input.Path();
  config.depth = 2;

  const luna::datagen::Report report = luna::datagen::Run(config, nullptr);

  REQUIRE(report.error.empty());
  REQUIRE(report.samples == 1);
  REQUIRE(report.skipped == 2);
}

TEST_CASE("labelling an empty list is an error rather than an empty file",
          "[datagen][label]") {
  ScratchFile input("label-empty-in.sfen");
  ScratchFile bin("label-empty-out.bin");
  input.Write("# nothing but a header\n");

  luna::datagen::Config config;
  config.out_path = bin.Path();
  config.label_path = input.Path();

  const luna::datagen::Report report = luna::datagen::Run(config, nullptr);

  REQUIRE_FALSE(report.error.empty());
  REQUIRE_FALSE(bin.Exists());
}
