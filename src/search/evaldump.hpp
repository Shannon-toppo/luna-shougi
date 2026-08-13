#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "core/position.hpp"

namespace luna::eval::dump {

// A debug tap on every static evaluation the search takes.
//
// The point of it is that a self-play game record is not the distribution an
// evaluation is judged on. A game record holds the positions that were
// reached; the search evaluates the ones it considered and threw away, which
// is a hundred thousand times as many and includes the middle of exchanges,
// positions one ply into a losing capture, and whatever quiescence walks
// through on its way to a quiet score. Two generations of network were
// measured against reconstructed self-play holdouts, and both measurements
// missed the match result by hundreds of Elo. This writes down what the search
// actually looks at instead of guessing at it. docs/nnue-gen2.md is the
// record of the guessing.
//
// Off unless a path is set, and off is one relaxed load of a bool. The check
// is left to the caller so that it inlines: a build that never turns this on
// pays a predictable branch next to an evaluation that already costs hundreds
// of cycles.

// Which of the search's evaluation call sites a record came from. Different
// sites see different parts of the tree, and a network can be wrong in one
// without being wrong in another, so they are written down separately rather
// than pooled.
enum class Site {
  // Quiescence stand-pat: the score a side gets for declining to capture.
  // Never in check. This is the great majority of every dump.
  kStandPat,
  // The static evaluation that the reverse-futility, null-move and futility
  // margins are compared against. Non-PV nodes only, never in check.
  kPrune,
  // A node that ran out of ply. Rare, and the only site that can be in check.
  kHorizon,
};

// The name written in the file's first column, and what the analysis groups
// by. Total order matches the enum.
const char* SiteName(Site site);

namespace detail {
// The flag Enabled() reads. Defined in the .cpp; declared here so the check
// can inline into the search rather than costing a call.
extern std::atomic<bool> g_enabled;
}  // namespace detail

// Whether anything is being recorded. Cheap enough to call per evaluation.
inline bool Enabled() {
  return detail::g_enabled.load(std::memory_order_relaxed);
}

// Opens `path` for writing and starts recording, replacing whatever the file
// held. False, with `error` set, when it cannot be opened -- a debug option
// that silently does nothing is how a measurement session gets thrown away, so
// the caller is expected to pass the reason on.
bool Open(const std::string& path, std::string& error);

// Stops recording and closes the file. Safe when nothing is open.
void Close();

// Records one evaluation in every `stride`. 1 records all of them. A depth-6
// search evaluates six figures' worth of positions per move, so the stride is
// what keeps a game's worth of dump down to something a labelling run can get
// through. Values below 1 are read as 1.
void SetStride(int64_t stride);
int64_t Stride();

// Writes one record: the site, the score the search is about to use (from the
// side to move), the ply it was taken at, the id of the thread that took it,
// and the SFEN. Only call this when Enabled().
//
// Every record goes through one mutex, so a dump with Threads greater than 1
// is complete and uncorrupted but not reproducible: the helper threads search
// their own depths in their own order. Threads 1 is what makes two dumps of
// the same position comparable.
void Record(const Position& pos, int score, Site site, int ply, int worker);

// How many records have been written since the file was opened.
int64_t Count();

// How many evaluations have reached Record, stride included. Count() over this
// is the sampling rate that was actually achieved.
int64_t Seen();

}  // namespace luna::eval::dump
