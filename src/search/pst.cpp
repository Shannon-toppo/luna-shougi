#include "search/pst.hpp"

#include <array>

namespace luna::eval {
namespace {

// Every table below is written as a board seen from black: the first row is
// rank a, which is the enemy's back rank, and the columns run from file 9 on
// the left to file 1 on the right. That is the layout of a shogi diagram and
// of the board part of an SFEN, so a table can be read against a real
// position without transposing anything in your head.
//
// The tables are left-right symmetric on purpose. Shogi openings are not —
// the rook starts on file 2 and the king castles the other way — but which
// side is the attacking side depends entirely on the opening, and a table
// that picked one would be wrong for every other.
using Table = std::array<int, kSquareNb>;

// Advancing a pawn takes space, and a pawn two ranks from promotion is a real
// threat. Nothing here comes near the 450 that promoting itself is worth; the
// table only decides which pawn to push when nothing tactical is going on.
constexpr Table kPawnTable = {
    // A pawn cannot legally sit on rank a unpromoted, so that row never reads.
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    22, 24, 26, 28, 28, 28, 26, 24, 22,  //
    16, 18, 20, 22, 22, 22, 20, 18, 16,  //
    10, 12, 14, 16, 16, 16, 14, 12, 10,  //
    6,  8,  10, 12, 12, 12, 10, 8,  6,   //
    2,  4,  5,  6,  6,  6,  5,  4,  2,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   // the rank pawns start on
    -4, -4, -4, -4, -4, -4, -4, -4, -4,  //
    -8, -8, -8, -8, -8, -8, -8, -8, -8,  //
};

// A lance is happiest at home behind its own pawn, where it holds the file
// without being attackable. Pushing it commits it to a single file for the
// rest of the game, so only the promotion zone pays.
constexpr Table kLanceTable = {
    8,  8,  8,  8,  8,  8,  8,  8,  8,   //
    12, 12, 12, 12, 12, 12, 12, 12, 12,  //
    10, 10, 10, 10, 10, 10, 10, 10, 10,  //
    2,  2,  2,  2,  2,  2,  2,  2,  2,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    2,  2,  2,  2,  2,  2,  2,  2,  2,   //
    4,  4,  4,  4,  4,  4,  4,  4,  4,   //
};

// A knight only ever moves forward and cannot be brought back, so a knight
// still at home is a knight doing nothing. One that reaches rank c or d forks
// two squares the opponent usually cannot both defend.
constexpr Table kKnightTable = {
    // Ranks a and b are illegal for an unpromoted knight.
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    12, 18, 20, 22, 22, 22, 20, 18, 12,  //
    10, 14, 16, 18, 18, 18, 16, 14, 10,  //
    4,  6,  8,  10, 10, 10, 8,  6,  4,   //
    0,  2,  2,  4,  4,  4,  2,  2,  0,   //
    -2, -2, -2, -2, -2, -2, -2, -2, -2,  //
    -2, -2, -2, -2, -2, -2, -2, -2, -2,  //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
};

// The silver is the piece that goes forward: it can retreat diagonally, which
// is what makes advancing one so much safer than advancing a gold. A silver
// left on the back rank is the classic wasted piece.
constexpr Table kSilverTable = {
    4,  6,  8,  8,  8,  8,  8,  6,  4,   //
    10, 12, 14, 16, 16, 16, 14, 12, 10,  //
    12, 14, 16, 18, 18, 18, 16, 14, 12,  //
    8,  10, 12, 14, 14, 14, 12, 10, 8,   //
    4,  6,  8,  10, 10, 10, 8,  6,  4,   //
    2,  4,  4,  6,  6,  6,  4,  4,  2,   //
    0,  2,  2,  2,  2,  2,  2,  2,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    -6, -4, -4, -4, -4, -4, -4, -4, -6,  //
};

// A gold cannot retreat diagonally, so pushing one forward is close to
// irreversible. The table stays almost flat and lets the king-proximity term
// decide whether a particular gold is defending or attacking.
constexpr Table kGoldTable = {
    0,  0,  2, 2, 2, 2, 2, 0,  0,   //
    2,  4,  6, 6, 6, 6, 6, 4,  2,   //
    4,  6,  8, 8, 8, 8, 8, 6,  4,   //
    2,  4,  6, 6, 6, 6, 6, 4,  2,   //
    0,  2,  4, 4, 4, 4, 4, 2,  0,   //
    0,  2,  2, 4, 4, 4, 2, 2,  0,   //
    0,  2,  2, 2, 2, 2, 2, 2,  0,   //
    2,  4,  4, 4, 4, 4, 4, 4,  2,   //
    -4, -2, 0, 0, 0, 0, 0, -2, -4,  //
};

// A bishop is about diagonal length, which peaks in the centre and collapses
// in the corners. Its own starting square is cramped until the diagonal opens.
constexpr Table kBishopTable = {
    -4, -2, 0,  2,  2,  2,  0,  -2, -4,  //
    0,  4,  8,  10, 10, 10, 8,  4,  0,   //
    2,  8,  12, 14, 14, 14, 12, 8,  2,   //
    4,  10, 14, 18, 18, 18, 14, 10, 4,   //
    4,  10, 14, 18, 20, 18, 14, 10, 4,   //
    2,  8,  12, 14, 14, 14, 12, 8,  2,   //
    0,  4,  8,  10, 10, 10, 8,  4,  0,   //
    0,  4,  6,  8,  8,  8,  6,  4,  0,   //
    -6, -2, 0,  2,  2,  2,  0,  -2, -6,  //
};

// A rook on rank b is the standard attacking formation: it rakes the rank the
// opponent's pieces sit on and promotes next move. Behind its own lines it is
// worth roughly the same anywhere, since it is the file that matters and this
// table cannot see files.
constexpr Table kRookTable = {
    6,  8,  10, 10, 10, 10, 10, 8,  6,   //
    12, 14, 16, 18, 18, 18, 16, 14, 12,  //
    10, 12, 14, 16, 16, 16, 14, 12, 10,  //
    4,  6,  8,  8,  8,  8,  8,  6,  4,   //
    2,  4,  6,  6,  6,  6,  6,  4,  2,   //
    2,  4,  6,  6,  6,  6,  6,  4,  2,   //
    0,  2,  4,  4,  4,  4,  4,  2,  0,   //
    2,  4,  6,  6,  8,  6,  6,  4,  2,   //
    0,  2,  2,  2,  2,  2,  2,  2,  0,   //
};

// The one table with real weight behind it. A king that leaves its own camp
// in a middlegame is lost, and a king still sitting on 5i has not castled —
// the whole point of every shogi opening is to move it off that square and
// into a corner. The numbers are large enough to be worth a tempo or two but
// not a piece.
//
// The cost is that this engine will never go for 入玉: marching the king up
// the board is a legitimate winning plan in some endgames and the table
// forbids it outright. Telling those endgames apart needs a game phase, which
// the handcrafted evaluation does not have.
constexpr Table kKingTable = {
    -100, -100, -100, -100, -100, -100, -100, -100, -100,  //
    -90,  -90,  -90,  -90,  -90,  -90,  -90,  -90,  -90,   //
    -80,  -80,  -80,  -80,  -80,  -80,  -80,  -80,  -80,   //
    -60,  -60,  -60,  -60,  -60,  -60,  -60,  -60,  -60,   //
    -40,  -40,  -40,  -40,  -40,  -40,  -40,  -40,  -40,   //
    -20,  -22,  -24,  -26,  -28,  -26,  -24,  -22,  -20,   //
    0,    -2,   -6,   -10,  -14,  -10,  -6,   -2,   0,     //
    14,   12,   6,    -2,   -8,   -2,   6,    12,   14,    //
    20,   18,   12,   2,    -6,   2,    12,   18,   20,    //
};

// と金 and the three other small promotions all move as a gold and all sit in
// or near the enemy camp, having promoted there. Their job from that point on
// is to walk into the opponent's castle, so unlike a real gold they want to be
// forward.
constexpr Table kPromotedGoldTable = {
    10, 12, 12, 14, 14, 14, 12, 12, 10,  //
    12, 14, 16, 16, 16, 16, 16, 14, 12,  //
    12, 14, 16, 16, 16, 16, 16, 14, 12,  //
    8,  10, 12, 12, 12, 12, 12, 10, 8,   //
    4,  6,  8,  8,  8,  8,  8,  6,  4,   //
    2,  2,  4,  4,  4,  4,  4,  2,  2,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
    0,  0,  0,  0,  0,  0,  0,  0,  0,   //
};

// 馬は自陣に引け. A horse pulled back to its own camp defends squares the
// opponent needs for mate and is almost impossible to trade off, which is why
// this table leans the opposite way to every other promoted piece.
constexpr Table kHorseTable = {
    0,  2,  4,  4,  4,  4,  4,  2,  0,   //
    2,  6,  8,  10, 10, 10, 8,  6,  2,   //
    4,  8,  12, 14, 14, 14, 12, 8,  4,   //
    6,  10, 14, 16, 16, 16, 14, 10, 6,   //
    8,  12, 16, 18, 18, 18, 16, 12, 8,   //
    10, 14, 16, 18, 18, 18, 16, 14, 10,  //
    12, 14, 16, 18, 18, 18, 16, 14, 12,  //
    12, 14, 16, 16, 16, 16, 16, 14, 12,  //
    8,  10, 12, 12, 12, 12, 12, 10, 8,   //
};

// A dragon is strong wherever it stands, so the spread is small. It still
// prefers the enemy camp, where it attacks the castle and cannot easily be
// driven away.
constexpr Table kDragonTable = {
    10, 12, 14, 14, 14, 14, 14, 12, 10,  //
    12, 16, 18, 20, 20, 20, 18, 16, 12,  //
    12, 16, 18, 20, 20, 20, 18, 16, 12,  //
    10, 12, 14, 16, 16, 16, 14, 12, 10,  //
    8,  10, 12, 14, 14, 14, 12, 10, 8,   //
    8,  10, 12, 12, 12, 12, 12, 10, 8,   //
    8,  10, 10, 12, 12, 12, 10, 10, 8,   //
    6,  8,  10, 10, 10, 10, 10, 8,  6,   //
    4,  6,  8,  8,  8,  8,  8,  6,  4,   //
};

constexpr std::array<Table, kPieceTypeNb> BuildTables() {
  std::array<Table, kPieceTypeNb> tables{};
  tables[kPawn] = kPawnTable;
  tables[kLance] = kLanceTable;
  tables[kKnight] = kKnightTable;
  tables[kSilver] = kSilverTable;
  tables[kBishop] = kBishopTable;
  tables[kRook] = kRookTable;
  tables[kGold] = kGoldTable;
  tables[kKing] = kKingTable;
  tables[kProPawn] = kPromotedGoldTable;
  tables[kProLance] = kPromotedGoldTable;
  tables[kProKnight] = kPromotedGoldTable;
  tables[kProSilver] = kPromotedGoldTable;
  tables[kHorse] = kHorseTable;
  tables[kDragon] = kDragonTable;
  return tables;
}

constexpr std::array<Table, kPieceTypeNb> kTables = BuildTables();

}  // namespace

int PstValue(Color c, PieceType pt, Square sq) {
  // White reads the same table through a 180-degree rotation of the board,
  // which for this square numbering is simply counting from the other end.
  return kTables[pt][c == kBlack ? sq : kSquareNb - 1 - sq];
}

}  // namespace luna::eval
