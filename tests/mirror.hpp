#pragma once

#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "core/types.hpp"

namespace test {

inline std::vector<std::string> Split(const std::string& s, char sep) {
  std::vector<std::string> parts;
  size_t start = 0;
  while (true) {
    const size_t at = s.find(sep, start);
    parts.push_back(s.substr(start, at == std::string::npos ? at : at - start));
    if (at == std::string::npos) break;
    start = at + 1;
  }
  return parts;
}

inline char SwapCase(char c) {
  if (c >= 'a' && c <= 'z') return static_cast<char>(c - 'a' + 'A');
  if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
  return c;
}

// The same position with the two sides exchanged: the board rotated by 180
// degrees, every piece recoloured, the hands swapped and the other side to
// move. Everything an evaluation knows is defined relative to a side, so a
// position and its mirror have to look the same to whichever side is looking.
inline std::string MirrorSfen(const std::string& sfen) {
  const std::vector<std::string> fields = Split(sfen, ' ');
  REQUIRE(fields.size() >= 3);

  // One entry per square in SFEN reading order, "" for an empty square.
  std::vector<std::string> squares;
  for (const std::string& rank : Split(fields[0], '/')) {
    for (size_t i = 0; i < rank.size(); ++i) {
      const char c = rank[i];
      if (c >= '1' && c <= '9') {
        squares.insert(squares.end(), static_cast<size_t>(c - '0'), "");
      } else if (c == '+') {
        squares.push_back(std::string("+") + SwapCase(rank[++i]));
      } else {
        squares.push_back(std::string(1, SwapCase(c)));
      }
    }
  }
  REQUIRE(squares.size() == static_cast<size_t>(luna::kSquareNb));

  std::string board;
  int empty = 0;
  for (int i = luna::kSquareNb - 1; i >= 0; --i) {
    if (squares[static_cast<size_t>(i)].empty()) {
      ++empty;
    } else {
      if (empty > 0) board += static_cast<char>('0' + empty);
      empty = 0;
      board += squares[static_cast<size_t>(i)];
    }
    if (i % 9 == 0) {
      if (empty > 0) board += static_cast<char>('0' + empty);
      empty = 0;
      if (i > 0) board += '/';
    }
  }

  std::string hands;
  for (const char c : fields[2]) hands += SwapCase(c);

  return board + (fields[1] == "b" ? " w " : " b ") + hands + " 1";
}

}  // namespace test
