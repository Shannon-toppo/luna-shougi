#include "search/evaldump.hpp"

#include <fstream>
#include <mutex>

namespace luna::eval::dump {
namespace {

std::mutex g_mutex;
std::ofstream g_out;
int64_t g_count = 0;
int64_t g_seen = 0;
int64_t g_stride = 1;

}  // namespace

namespace detail {
std::atomic<bool> g_enabled{false};
}  // namespace detail

const char* SiteName(Site site) {
  switch (site) {
    case Site::kStandPat:
      return "standpat";
    case Site::kPrune:
      return "prune";
    case Site::kHorizon:
      return "horizon";
  }
  return "?";
}

bool Open(const std::string& path, std::string& error) {
  Close();

  std::lock_guard<std::mutex> lock(g_mutex);
  g_out.open(path, std::ios::trunc);
  if (!g_out) {
    error = "cannot write " + path;
    return false;
  }
  // A header, because this file is read by hand as often as by a script and
  // the columns are not guessable. Readers skip lines starting with '#'.
  g_out << "# site\tscore\tply\tworker\tsfen\n";
  g_count = 0;
  g_seen = 0;
  detail::g_enabled.store(true, std::memory_order_relaxed);
  return true;
}

void Close() {
  // Cleared before the lock is taken, so a search still running cannot enter
  // Record and block behind a close.
  detail::g_enabled.store(false, std::memory_order_relaxed);

  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_out.is_open()) g_out.close();
  g_out.clear();
}

void SetStride(int64_t stride) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_stride = stride < 1 ? 1 : stride;
}

int64_t Stride() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_stride;
}

void Record(const Position& pos, int score, Site site, int ply, int worker) {
  std::lock_guard<std::mutex> lock(g_mutex);
  // Enabled() was read outside the lock, so the file may have been closed
  // between that read and here.
  if (!g_out.is_open()) return;

  // Counted before the stride is applied: `g_seen` is the size of the
  // population this file is a sample of, and the analysis needs it to know
  // how much of the tree one line stands for.
  const bool keep = g_seen % g_stride == 0;
  ++g_seen;
  if (!keep) return;

  g_out << SiteName(site) << '\t' << score << '\t' << ply << '\t' << worker << '\t' << pos.ToSfen()
        << '\n';
  ++g_count;
}

int64_t Count() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_count;
}

int64_t Seen() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_seen;
}

}  // namespace luna::eval::dump
