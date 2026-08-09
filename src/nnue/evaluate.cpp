#include "nnue/evaluate.hpp"

#include <fstream>
#include <vector>

#include "nnue/features.hpp"
#include "nnue/simd.hpp"

namespace luna::nnue {
namespace {

// How far back Evaluate will look for an accumulator it can update instead of
// rebuilding. Every extra ply costs a copy of the accumulator plus a few row
// updates, and a rebuild costs 38 row updates, so there is a point past which
// walking back stops paying. In practice the parent is nearly always there:
// quiescence evaluates every node it visits.
constexpr int kMaxChain = 4;

struct Entry {
  Accumulator acc;
  uint64_t key = 0;
  bool valid = false;
};

// Indexed by how many moves have been played into the position, so a search
// that walks down and back up reuses the same slots. The key check is what
// makes that safe: taking a move back and playing a different one leaves a
// stale entry behind, and it is rejected the moment its key does not match.
struct Cache {
  std::vector<Entry> entries;
  uint32_t generation = 0;
};

Network& TheNetwork() {
  static Network network;
  return network;
}

std::string& ThePath() {
  static std::string path;
  return path;
}

// Bumped every time the network changes, so caches built from the old weights
// are discarded rather than silently mixed with the new ones.
uint32_t& TheGeneration() {
  static uint32_t generation = 1;
  return generation;
}

Cache& ThreadCache() {
  static thread_local Cache cache;
  if (cache.generation != TheGeneration()) {
    cache.entries.clear();
    cache.generation = TheGeneration();
  }
  return cache;
}

BonaPiece Orient(Color perspective, BonaPiece bona) {
  return perspective == kBlack ? bona : InvertBona(bona);
}

void RefreshBoth(const Network& net, const Position& pos, Entry& entry) {
  const BonaList list = BuildBonaList(pos);
  net.Refresh(entry.acc, kBlack, list, PerspectiveKing(pos, kBlack));
  net.Refresh(entry.acc, kWhite, list, PerspectiveKing(pos, kWhite));
}

// Applies one move's delta to `acc`, which must hold the position before it.
// `pos` is the position after, and is only read when the move was a king move
// and a half has to be rebuilt — which the caller only allows for the last
// move of a chain, where "after" really is `pos`.
void ApplyDelta(const Network& net,
                Accumulator& acc,
                const Position::Undo& undo,
                const Position& pos) {
  const FeatureDelta delta = ComputeDelta(undo);
  for (int c = 0; c < kColorNb; ++c) {
    const Color perspective = static_cast<Color>(c);
    const Square king = PerspectiveKing(pos, perspective);
    if (delta.king_moved == c) {
      net.Refresh(acc, perspective, BuildBonaList(pos), king);
      continue;
    }
    int16_t* half = acc.half[perspective].data();
    for (int i = 0; i < delta.removed_size; ++i) {
      SubRow(half,
             net.FeatureRow(FeatureIndex(king, Orient(perspective, delta.removed[i]))),
             kTransformedDim);
    }
    for (int i = 0; i < delta.added_size; ++i) {
      AddRow(half,
             net.FeatureRow(FeatureIndex(king, Orient(perspective, delta.added[i]))),
             kTransformedDim);
    }
  }
}

// The Zobrist key of the position `index` moves into the game, which is the
// key stored against the move played from it.
uint64_t KeyAt(const Position& pos, int index, int depth) {
  return index == depth ? pos.Key() : pos.UndoAt(index).key;
}

// Where to start replaying from, or -1 for "rebuild".
int FindChainStart(const Position& pos, const Cache& cache, int depth) {
  int start = -1;
  for (int j = depth - 1; j >= 0 && depth - j <= kMaxChain; --j) {
    if (cache.entries[j].valid && cache.entries[j].key == pos.UndoAt(j).key) {
      start = j;
      break;
    }
  }
  if (start < 0) return -1;

  // A king move rebuilds a whole half, and rebuilding needs the position the
  // king ended up in. Only the last move of a chain ends up in a position we
  // still have, and only when it is the sole link, because otherwise the
  // earlier links would have been computed against the wrong king square.
  for (int i = start; i < depth; ++i) {
    if (TypeOf(pos.UndoAt(i).moved) == kKing) return start == depth - 1 ? start : -1;
  }
  return start;
}

}  // namespace

bool Load(const std::string& path, std::string& error) {
  Unload();

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot open " + path;
    return false;
  }
  if (!TheNetwork().ReadFrom(file, error)) {
    error = path + ": " + error;
    return false;
  }
  ThePath() = path;
  return true;
}

void Unload() {
  TheNetwork().Clear();
  ThePath().clear();
  ++TheGeneration();
}

Network& InstallNetwork() {
  Unload();
  Network& network = TheNetwork();
  network.Parameters();  // allocates if needed and marks it loaded
  return network;
}

bool IsLoaded() {
  return TheNetwork().Loaded();
}

const std::string& LoadedPath() {
  return ThePath();
}

void ClearCache() {
  ThreadCache().entries.clear();
}

int Evaluate(const Position& pos) {
  const Network& net = TheNetwork();
  Cache& cache = ThreadCache();
  const int depth = pos.UndoableMoves();
  if (static_cast<int>(cache.entries.size()) <= depth) {
    cache.entries.resize(static_cast<size_t>(depth) + 1);
  }
  if (!(cache.entries[depth].valid && cache.entries[depth].key == pos.Key())) {
    const int start = FindChainStart(pos, cache, depth);
    if (start < 0) {
      RefreshBoth(net, pos, cache.entries[depth]);
      cache.entries[depth].key = pos.Key();
      cache.entries[depth].valid = true;
    } else {
      for (int i = start; i < depth; ++i) {
        cache.entries[i + 1].acc = cache.entries[i].acc;
        ApplyDelta(net, cache.entries[i + 1].acc, pos.UndoAt(i), pos);
        cache.entries[i + 1].key = KeyAt(pos, i + 1, depth);
        cache.entries[i + 1].valid = true;
      }
    }
  }

  return net.Propagate(cache.entries[depth].acc, pos.SideToMove());
}

}  // namespace luna::nnue
