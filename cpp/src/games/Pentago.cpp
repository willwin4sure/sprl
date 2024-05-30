// #include "Pentago.hpp"

// #include <cassert>
// #include <iostream>

// namespace SPRL {

// Pentago::State Pentago::startState() const {
//     return State {};
// }

// ActionIdx Pentago::actionToActionIdx(const Pentago::Action& action) const {
//     assert(action.boardIdx >= 0);
//     assert(action.boardIdx < PTG_BOARD_SIZE);

//     return static_cast<int>(action.rotDirection) * PTG_BOARD_SIZE * PTG_NUM_QUADRS
//          + static_cast<int>(action.rotQuadrant) * PTG_BOARD_SIZE
//          + static_cast<int>(action.boardIdx);
// }

// Pentago::Action Pentago::actionIdxToAction(const ActionIdx actionIdx) const {
//     assert(actionIdx >= 0);
//     assert(actionIdx < PTG_NUM_ACTIONS);

//     const Pentago::Action action =  {
//         static_cast<RotationDirection>(actionIdx / (PTG_BOARD_SIZE * PTG_NUM_QUADRS)),
//         static_cast<RotationQuadrant>((actionIdx / PTG_BOARD_SIZE) % PTG_NUM_QUADRS),
//         static_cast<int8_t>(actionIdx % PTG_BOARD_SIZE)
//     };
//     return action;
// }

// Pentago::State Pentago::nextState(const State& state, const ActionIdx actionIdx) const {
//     assert(!state.isTerminal());
//     assert(actionMask(state)[actionIdx] == 1.0f);

//     State::Board newBoard = state.getBoard();  // The board that we return, a copy of the original

//     const Player player = state.getPlayer();
//     const Player newPlayer = 1 - player;

//     const Piece piece = static_cast<Piece>(player);

//     Action action = actionIdxToAction(actionIdx);

//     // Place the piece at the appropriate location
//     newBoard[action.boardIdx] = piece;

//     // Let's see if this wins the game immediately, then we don't need to rotate
//     bool placementWin = checkPlacementWin(newBoard, action.boardIdx / PTG_BOARD_WIDTH, action.boardIdx % PTG_BOARD_WIDTH, piece);

//     if (placementWin) {
//         return State { newBoard, newPlayer, player, true };
//     }

//     // Rotate the board

//     // 0  1  2     3  4  5
//     // 6  7  8     9  10 11
//     // 12 13 14    15 16 17

//     // 18 19 20    21 22 23
//     // 24 25 26    27 28 29
//     // 30 31 32    33 34 35

//     // Code is specific to the above case, for simplicity and efficiency

//     int subboardTopLeft = -1;

//     switch(action.rotQuadrant) {
//     case RotationQuadrant::topLeft:
//         subboardTopLeft = 0;

//         break;
//     case RotationQuadrant::topRight:
//         subboardTopLeft = 3;

//         break;
//     case RotationQuadrant::bottomLeft:
//         subboardTopLeft = 18;

//         break;
//     case RotationQuadrant::bottomRight:
//         subboardTopLeft = 21;

//         break;
//     default:
//         assert(false);
//     }

//     Piece temp = newBoard[subboardTopLeft];  // temporary location to store a piece

//     switch (action.rotDirection) {
//     case RotationDirection::clockwise:
//         newBoard[subboardTopLeft]      = newBoard[subboardTopLeft + 12];
//         newBoard[subboardTopLeft + 12] = newBoard[subboardTopLeft + 14];
//         newBoard[subboardTopLeft + 14] = newBoard[subboardTopLeft + 2];
//         newBoard[subboardTopLeft + 2]  = temp;

//         temp = newBoard[subboardTopLeft + 1];
//         newBoard[subboardTopLeft + 1]  = newBoard[subboardTopLeft + 6];
//         newBoard[subboardTopLeft + 6]  = newBoard[subboardTopLeft + 13];
//         newBoard[subboardTopLeft + 13] = newBoard[subboardTopLeft + 8];
//         newBoard[subboardTopLeft + 8]  = temp;

//         break;
//     case RotationDirection::counterClockwise: // counterclockwise
//         newBoard[subboardTopLeft]      = newBoard[subboardTopLeft + 2];
//         newBoard[subboardTopLeft + 2]  = newBoard[subboardTopLeft + 14];
//         newBoard[subboardTopLeft + 14] = newBoard[subboardTopLeft + 12];
//         newBoard[subboardTopLeft + 12] = temp;

//         temp = newBoard[subboardTopLeft + 1];
//         newBoard[subboardTopLeft + 1]  = newBoard[subboardTopLeft + 8];
//         newBoard[subboardTopLeft + 8]  = newBoard[subboardTopLeft + 13];
//         newBoard[subboardTopLeft + 13] = newBoard[subboardTopLeft + 6];
//         newBoard[subboardTopLeft + 6]  = temp;

//         break;
//     default:
//         assert(false);
//     }

//     // Check if the move wins the game, or if the game ends
//     Player winner = -1;
//     bool terminal = true;

//     std::pair<bool, bool> won = checkWin(newBoard);

//     if (won.first && won.second) {
//         // Game is a tie: leave winner as -1 and terminal as true

//     } else if (won.first) {
//         // Player 0 wins, leave terminal as true
//         winner = 0;

//     } else if (won.second) {
//         // Player 1 wins, leave terminal as true
//         winner = 1;

//     } else {
//         // Neither player has won: check if we can still place a piece to turn off terminal
//         terminal = std::find(std::begin(newBoard), std::end(newBoard), emptySquare) == std::end(newBoard);
//     }

//     return State { newBoard, newPlayer, winner, terminal };
// }

// Pentago::ActionDist Pentago::actionMask(const State& state) const {
//     ActionDist mask {};
//     mask.fill(0.0f);

//     const State::Board& board = state.getBoard();
//     for (int boardIdx = 0; boardIdx < PTG_BOARD_SIZE; ++boardIdx) {
//         // Check if the board has a piece in that location
//         if (board[boardIdx] == -1) {
//             // All rotations are legal
//             for (int i = 0; i < PTG_NUM_ACTIONS / PTG_BOARD_SIZE; ++i) {
//                 mask[boardIdx + i * PTG_BOARD_SIZE] = 1.0f;
//             }
//         }
//     }

//     return mask;
// }

// std::pair<Value, Value> Pentago::rewards(const State& state) const {
//     const Player winner = state.getWinner();
//     switch (winner) {
//     case 0:
//         return { 1.0f, -1.0f };
//     case 1:
//         return { -1.0f, 1.0f };
//     default:
//         return { 0.0f, 0.0f };
//     }
// }

// int Pentago::numSymmetries() const {
//     // Symmetry group is dihedral D_4: we have four rotations and four reflected rotations
//     //
//     // Mapping:
//     //     0: identity
//     //     1: single 90 deg cw rotation
//     //     2: full 180 deg rotation
//     //     3: single 90 deg ccw rotation
//     //     4: reflection across vertical axis
//     //     5: reflection across vertical axis followed by single 90 deg cw rotation
//     //     6: reflection across vertical axis followed by full 180 deg rotation
//     //     7: reflection across vertical axis followed by single 90 deg ccw rotation

//     return 8;
// }


// Symmetry Pentago::inverseSymmetry(const Symmetry& symmetry) const {
//     assert(symmetry >= 0);
//     assert(symmetry < 8);

//     static constexpr std::array<Symmetry, 8> inverseSymmetries = { 0, 3, 2, 1, 4, 5, 6, 7 };
//     return inverseSymmetries[symmetry];
// }

// Pentago::State Pentago::symmetrizeSingleState(const State& state, Symmetry symmetry) const {
//     switch (symmetry) {
//     case 0: {
//         // identity
//         return state;
//     }
    
//     case 1: {
//         // single 90 deg cw rotation
//         State::Board board = state.getBoard(); // Copy of the board
//         for (int row = 0; row < PTG_BOARD_WIDTH / 2; ++row) {
//             for (int col = 0; col < PTG_BOARD_WIDTH / 2; ++col) {
//                 Piece temp = board[row * PTG_BOARD_WIDTH + col];
//                 board[row * PTG_BOARD_WIDTH + col] = board[(PTG_BOARD_WIDTH - 1 - col) * PTG_BOARD_WIDTH + row];
//                 board[(PTG_BOARD_WIDTH - 1 - col) * PTG_BOARD_WIDTH + row] = board[(PTG_BOARD_WIDTH - 1 - row) * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - col)];
//                 board[(PTG_BOARD_WIDTH - 1 - row) * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - col)] = board[col * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - row)];
//                 board[col * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - row)] = temp;
//             }
//         }
//         return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
//     }

//     case 2: {
//         // full 180 deg rotation
//         State::Board board = state.getBoard(); // Copy of the board
//         for (int row = 0; row < PTG_BOARD_WIDTH / 2; ++row) {
//             for (int col = 0; col < PTG_BOARD_WIDTH; ++col) {
//                 std::swap(board[row * PTG_BOARD_WIDTH + col],
//                             board[(PTG_BOARD_WIDTH - 1 - row) * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - col)]);
//             }
//         }
//         return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
//     }

//     case 3: {
//         // single 90 deg ccw rotation
//         State::Board board = state.getBoard(); // Copy of the board
//         for (int row = 0; row < PTG_BOARD_WIDTH / 2; ++row) {
//             for (int col = 0; col < PTG_BOARD_WIDTH / 2; ++col) {
//                 Piece temp = board[row * PTG_BOARD_WIDTH + col];
//                 board[row * PTG_BOARD_WIDTH + col] = board[col * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - row)];
//                 board[col * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - row)] = board[(PTG_BOARD_WIDTH - 1 - row) * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - col)];
//                 board[(PTG_BOARD_WIDTH - 1 - row) * PTG_BOARD_WIDTH + (PTG_BOARD_WIDTH - 1 - col)] = board[(PTG_BOARD_WIDTH - 1 - col) * PTG_BOARD_WIDTH + row];
//                 board[(PTG_BOARD_WIDTH - 1 - col) * PTG_BOARD_WIDTH + row] = temp;
//             }
//         }
//         return State { board, state.getPlayer(), state.getWinner(), state.isTerminal() };
//     }

//     case 4: case 5: case 6: case 7: {
//         // first perform reflection across vertical axis
//         State::Board board = state.getBoard(); // Copy of the board
//         for (int row = 0; row < PTG_BOARD_WIDTH; ++row) {
//             for (int col = 0; col < PTG_BOARD_WIDTH / 2; ++col) {
//                 std::swap(board[row * PTG_BOARD_WIDTH + col], board[row * PTG_BOARD_WIDTH + PTG_BOARD_WIDTH - 1 - col]);
//             }
//         }

//         // then perform rotation by reusing the code above
//         return symmetrizeSingleState(State { board, state.getPlayer(), state.getWinner(), state.isTerminal() },
//                                      static_cast<Symmetry>(symmetry - 4));
//     }

//     default:
//         assert(false);
//     }

//     // Should not reach here.
//     return state;
// }

// std::vector<Pentago::State> Pentago::symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const {
//     std::vector<State> symmetrizedStates;
//     symmetrizedStates.reserve(symmetries.size());

//     for (const Symmetry& symmetry : symmetries) {
//         symmetrizedStates.push_back(symmetrizeSingleState(state, symmetry));
//     }

//     return symmetrizedStates;
// }

// Pentago::Action Pentago::symmetrizeSingleAction(const Action& action, Symmetry symmetry) const {

//     // 0 1
//     // 2 3

//     static constexpr std::array<int, 4> cwRotations = { 1, 3, 0, 2 };
//     static constexpr std::array<int, 4> fullRotations = { 3, 2, 1, 0 };
//     static constexpr std::array<int, 4> ccwRotations = { 2, 0, 3, 1 };

//     static constexpr std::array<int, 4> vertReflection = { 1, 0, 3, 2 };

//     switch (symmetry) {
//     case 0: {
//         // identity
//         return action;
//     }

//     case 1: {
//         // single 90 deg cw rotation
//         RotationQuadrant newQuadrant = static_cast<RotationQuadrant>(
//             cwRotations[static_cast<int>(action.rotQuadrant)]);

//         int row = action.boardIdx / PTG_BOARD_WIDTH;
//         int col = action.boardIdx % PTG_BOARD_WIDTH;

//         int newRow = col;
//         int newCol = PTG_BOARD_WIDTH - 1 - row;

//         return { action.rotDirection, newQuadrant, static_cast<int8_t>(newRow * PTG_BOARD_WIDTH + newCol) };
//     }

//     case 2: {
//         // full 180 deg rotation
//         RotationQuadrant newQuadrant = static_cast<RotationQuadrant>(
//             fullRotations[static_cast<int>(action.rotQuadrant)]);

//         int row = action.boardIdx / PTG_BOARD_WIDTH;
//         int col = action.boardIdx % PTG_BOARD_WIDTH;

//         int newRow = PTG_BOARD_WIDTH - 1 - row;
//         int newCol = PTG_BOARD_WIDTH - 1 - col;

//         return { action.rotDirection, newQuadrant, static_cast<int8_t>(newRow * PTG_BOARD_WIDTH + newCol) };
//     }

//     case 3: {
//         // single 90 deg ccw rotation
//         RotationQuadrant newQuadrant = static_cast<RotationQuadrant>(
//             ccwRotations[static_cast<int>(action.rotQuadrant)]);

//         int row = action.boardIdx / PTG_BOARD_WIDTH;
//         int col = action.boardIdx % PTG_BOARD_WIDTH;

//         int newRow = PTG_BOARD_WIDTH - 1 - col;
//         int newCol = row;

//         return { action.rotDirection, newQuadrant, static_cast<int8_t>(newRow * PTG_BOARD_WIDTH + newCol) };
//     }

//     case 4: case 5: case 6: case 7: {
//         // first perform reflection across vertical axis
//         int row = action.boardIdx / PTG_BOARD_WIDTH;
//         int col = action.boardIdx % PTG_BOARD_WIDTH;

//         int newRow = row;
//         int newCol = PTG_BOARD_WIDTH - 1 - col;

//         Action newAction = {
//             static_cast<RotationDirection>(1 - static_cast<int>(action.rotDirection)),
//             static_cast<RotationQuadrant>(vertReflection[static_cast<int>(action.rotQuadrant)]),
//             static_cast<int8_t>(newRow * PTG_BOARD_WIDTH + newCol)
//         };

//         // then perform rotation by reusing the code above
//         return symmetrizeSingleAction(newAction, static_cast<Symmetry>(symmetry - 4));
//     }

//     default:
//         assert(false);
//     }

//     // Should not reach here.
//     return action;
// }

// Pentago::ActionDist Pentago::symmetrizeSingleActionDist(const ActionDist& actionDist, Symmetry symmetry) const {
//     ActionDist newActionDist;
//     newActionDist.fill(-1.0f);

//     for (int boardIdx = 0; boardIdx < PTG_BOARD_SIZE; ++boardIdx) {
//         for (int rotQuadrant = 0; rotQuadrant < PTG_NUM_QUADRS; ++rotQuadrant) {
//             for (int rotDirection = 0; rotDirection < PTG_NUM_ROT_DIRS; ++rotDirection) {
//                 Action action = { static_cast<RotationDirection>(rotDirection),
//                                   static_cast<RotationQuadrant>(rotQuadrant),
//                                   static_cast<int8_t>(boardIdx) };

//                 ActionIdx actionIdx = actionToActionIdx(action);
//                 ActionIdx newActionIdx = actionToActionIdx(symmetrizeSingleAction(action, symmetry));

//                 newActionDist[newActionIdx] = actionDist[actionIdx];
//             }
//         }
//     }

//     return newActionDist;
// }

// std::vector<Pentago::ActionDist> Pentago::symmetrizeActionDist(const ActionDist& actionDist, const std::vector<Symmetry>& symmetries) const {
//     std::vector<ActionDist> symmetrizedActionDists;
//     symmetrizedActionDists.reserve(symmetries.size());

//     for (const Symmetry& symmetry : symmetries) {
//         symmetrizedActionDists.push_back(symmetrizeSingleActionDist(actionDist, symmetry));
//     }

//     return symmetrizedActionDists;
// }


// std::string Pentago::stateToString(const State& state) const {
//     std::string str = "";

//     const State::Board& board = state.getBoard();
//     for (int row = 0; row < PTG_BOARD_WIDTH; row++) {
//         str += std::to_string(row) + " ";
//         for (int col = 0; col < PTG_BOARD_WIDTH; col++) {
//             switch (board[row * PTG_BOARD_WIDTH + col]) {
//             case -1:
//                 str += ". ";
//                 break;
//             case 0:
//                 // colored red
//                 str += "\033[31mO\033[0m ";
//                 break;
//             case 1:
//                 // colored yellow
//                 str += "\033[33mX\033[0m ";
//                 break;
//             default:
//                 assert(false);
//             }
//         }
//         str += "\n";   
//     }

//     str += "  ";
//     for (int col = 0; col < PTG_BOARD_WIDTH; col++) {
//         str += ('A' + col);
//         str += " ";
//     }

//     return str;
// }

// Pentago::State Pentago::stringToState(const std::string& str) const {
//     State::Board board;
//     board.fill(emptySquare);

//     int row = 0;
//     int col = 0;

//     int numO = 0;
//     int numX = 0;

//     for (char c : str) {
//         if (c == 'O') {
//             board[row * PTG_BOARD_WIDTH + col] = 0;
//             col++;
//             numO++;
//         } else if (c == 'X') {
//             board[row * PTG_BOARD_WIDTH + col] = 1;
//             col++;
//             numX++;
//         } else if (c == '.') {
//             col++;
//         } else if (c == '\n') {
//             row++;
//             col = 0;
//         }
//     }

//     return State { board, static_cast<Player>((numO > numX) ? 1 : 0), -1, false };
// }

// /**
//  * Returns a bool indicating whether the single placement wins for that color. Used after placements.
// */
// bool Pentago::checkPlacementWin(const State::Board& board, int piece_row, int piece_col, const Piece piece) const {
//     // First, extend left and right
//     int row_count = 1;

//     int col = piece_col - 1;
//     while (col >= 0 && board[piece_row * PTG_BOARD_WIDTH + col] == piece) {
//         ++row_count;
//         --col;
//     }

//     col = piece_col + 1;
//     while (col < PTG_BOARD_WIDTH && board[piece_row * PTG_BOARD_WIDTH + col] == piece) {
//         ++row_count;
//         ++col;
//     }

//     if (row_count >= 5) {
//         return true;
//     }

//     // Next, extend up and down
//     int col_count = 1;

//     int row = piece_row - 1;
//     while (row >= 0 && board[row * PTG_BOARD_WIDTH + piece_col] == piece) {
//         ++col_count;
//         --row;
//     }

//     row = piece_row + 1;
//     while (row < PTG_BOARD_WIDTH && board[row * PTG_BOARD_WIDTH + piece_col] == piece) {
//         ++col_count;
//         ++row;
//     }

//     if (col_count >= 5) {
//         return true;
//     }

//     // Next, extend along the main diagonal
//     int main_diag_count = 1;

//     row = piece_row - 1;
//     col = piece_col - 1;
//     while (row >= 0 && col >= 0 && board[row * PTG_BOARD_WIDTH + col] == piece) {
//         ++main_diag_count;
//         --row;
//         --col;
//     }

//     row = piece_row + 1;
//     col = piece_col + 1;
//     while (row < PTG_BOARD_WIDTH && col < PTG_BOARD_WIDTH && board[row * PTG_BOARD_WIDTH + col] == piece) {
//         ++main_diag_count;
//         ++row;
//         ++col;
//     }

//     if (main_diag_count >= 5) {
//         return true;
//     }

//     // Finally, extend along the anti-diagonal
//     int anti_diag_count = 1;

//     row = piece_row - 1;
//     col = piece_col + 1;
//     while (row >= 0 && col < PTG_BOARD_WIDTH && board[row * PTG_BOARD_WIDTH + col] == piece) {
//         ++anti_diag_count;
//         --row;
//         ++col;
//     }

//     row = piece_row + 1;
//     col = piece_col - 1;
//     while (row < PTG_BOARD_WIDTH && col >= 0 && board[row * PTG_BOARD_WIDTH + col] == piece) {
//         ++anti_diag_count;
//         ++row;
//         --col;
//     }

//     return anti_diag_count >= 5;
// }

// /**
//  * Returns a pair of bools indicating whether each player wins. Used after rotations.
// */
// std::pair<bool, bool> Pentago::checkWin(const State::Board& board) const {
//     std::pair<bool, bool> won { false, false };
    
//     // Horizontal win
//     for (int row = 0; row < PTG_BOARD_WIDTH; ++row) {
//         // In order to win a row, you must control the piece in col 1
//         Piece piece = board[row * PTG_BOARD_WIDTH + 1];
//         if (piece == emptySquare) continue;

//         // You must also control columns 2 through 4
//         bool isRow = true;
//         for (int col = 2; col < 5; ++col) {
//             if (board[row * PTG_BOARD_WIDTH + col] != piece) {
//                 isRow = false;
//                 break;
//             }
//         }

//         // Finally, you must control a piece in col 0 or 5
//         if (isRow && (board[row * PTG_BOARD_WIDTH] == piece || board[row * PTG_BOARD_WIDTH + 5] == piece)) {
//             if (piece == 0) {
//                 won.first = true;
//             } else {
//                 won.second = true;
//             }
//         }
//     }

//     // Vertical win
//     for (int col = 0; col < PTG_BOARD_WIDTH; ++col) {
//         // In order to win a column, you must control the piece in row 1
//         Piece piece = board[1 * PTG_BOARD_WIDTH + col];
//         if (piece == emptySquare) continue;

//         // You must also control rows 2 through 4
//         bool isCol = true;
//         for (int row = 2; row < 5; ++row) {
//             if (board[row * PTG_BOARD_WIDTH + col] != piece) {
//                 isCol = false;
//                 break;
//             }
//         }

//         // Finally, you must control a piece in row 0 or 5
//         if (isCol && (board[col] == piece || board[5 * PTG_BOARD_WIDTH + col] == piece)) {
//             if (piece == 0) {
//                 won.first = true;
//             } else {
//                 won.second = true;
//             }
//         }
//     }

//     // Main-diagonal win: just check them all
//     for (int row = 0; row < 2; ++row) {
//         for (int col = 0; col < 2; ++col) {
//             Piece piece = board[row * PTG_BOARD_WIDTH + col];
//             if (piece == emptySquare) continue;

//             bool isDiag = true;
//             for (int i = 1; i < 5; ++i) {
//                 if (piece != board[(row + i) * PTG_BOARD_WIDTH + (col + i)]) {
//                     isDiag = false;
//                 }
//             }

//             if (isDiag) {
//                 if (piece == 0) {
//                     won.first = true;
//                 } else {
//                     won.second = true;
//                 }
//             }
//         }
//     }

//     // Anti-diagonal win: just check them all
//     for (int row = 0; row < 2; ++row) {
//         for (int col = PTG_BOARD_WIDTH - 2; col < PTG_BOARD_WIDTH; ++col) {
//             Piece piece = board[row * PTG_BOARD_WIDTH + col];
//             if (piece == emptySquare) continue;

//             bool isDiag = true;
//             for (int i = 1; i < 5; ++i) {
//                 if (piece != board[(row + i) * PTG_BOARD_WIDTH + (col - i)]) {
//                     isDiag = false;
//                 }
//             }

//             if (isDiag) {
//                 if (piece == 0) {
//                     won.first = true;
//                 } else {
//                     won.second = true;
//                 }
//             }
//         }
//     }

//     return won;
// }

// } // namespace SPRL

