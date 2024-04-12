#include "ConnectFour.hpp"

#include <cassert>

namespace SPRL {

ConnectFour::State ConnectFour::startState() const {
    return State {};
}

ConnectFour::State ConnectFour::nextState(const State& state, const ActionIdx action) const {
    assert(!isTerminal(state));
    assert(actionMask(state)[action] == 1.0f);

    State::Board newBoard = state.getBoard();  // The board that we return, a copy of the original.

    const Player player = state.getPlayer();
    const Piece piece = static_cast<Piece>(player);

    int col = action;

    // Find the lowest empty row in the column
    int row = C4_NUM_ROWS - 1;
    while (row >= 0 && newBoard[row * C4_NUM_COLS + col] != -1) {
        row--;
    }

    // Place the piece there
    newBoard[row * C4_NUM_COLS + col] = piece;

    // Check if the move wins the game
    Player winner = checkWin(newBoard, row, col, piece) ? player : -1;

    return State { newBoard, 1 - player, winner };
}

bool ConnectFour::isTerminal(const State& state) const {
    if (state.getWinner() != -1) {
        return true;
    }

    const State::Board& board = state.getBoard();
    for (int col = 0; col < C4_NUM_COLS; col++) {
        // Check if there is an empty space in the top row
        if (board[col] == -1) {
            return false;
        }
    }

    return true;
}

ConnectFour::ActionDist ConnectFour::actionMask(const State& state) const {
    ActionDist mask {};
    mask.fill(0.0f);

    const State::Board& board = state.getBoard();
    for (int col = 0; col < C4_NUM_COLS; col++) {
        // Check if the top row has a piece for each column
        if (board[col] == -1) {
            mask[col] = 1.0f;
        }
    }

    return mask;
}

std::pair<float, float> ConnectFour::rewards(const State& state) const {
    const Player winner = state.getWinner();
    switch (winner) {
    case -1:
        return { 0.0f, 0.0f };
    case 0:
        return { 1.0f, -1.0f };
    case 1:
        return { -1.0f, 1.0f };
    default:
        assert(false);
    }
}

int ConnectFour::numSymmetries() const {
    // The only symmetries are identity and flipping across the center vertical axis
    return 2;
}

Symmetry ConnectFour::inverseSymmetry(const Symmetry& symmetry) const {
    // All symmetries are involutions
    return symmetry;
}

std::vector<ConnectFour::State> ConnectFour::symmetrizeState(const State& state, const std::vector<Symmetry>& symmetries) const {
    std::vector<State> symmetrizedStates;
    symmetrizedStates.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        switch (symmetry) {
        case 0:
            symmetrizedStates.push_back(state);
            break;

        case 1:
            State::Board board = state.getBoard(); // Copy of the board
            for (int row = 0; row < C4_NUM_ROWS; row++) {
                for (int col = 0; col < C4_NUM_COLS / 2; col++) {
                    std::swap(board[row * C4_NUM_COLS + col], board[row * C4_NUM_COLS + C4_NUM_COLS - col - 1]);
                }
            }
            symmetrizedStates.push_back(State { board, state.getPlayer(), state.getWinner() });
            break;

        default:
            assert(false);
        }
    }

    return symmetrizedStates;
}

std::vector<ConnectFour::ActionDist> ConnectFour::symmetrizeActionSpace(const ActionDist& actionSpace, const std::vector<Symmetry>& symmetries) const {
    std::vector<ActionDist> symmetrizedActionSpaces;
    symmetrizedActionSpaces.reserve(symmetries.size());

    for (const Symmetry& symmetry : symmetries) {
        ActionDist symmetrizedActionSpace {};
        switch (symmetry) {
        case 0:
            symmetrizedActionSpaces.push_back(actionSpace);
            break;

        case 1:
            for (int col = 0; col < C4_NUM_COLS / 2; col++) {
                symmetrizedActionSpace[col] = actionSpace[C4_NUM_COLS - col - 1];
                symmetrizedActionSpace[C4_NUM_COLS - col - 1] = actionSpace[col];
            }
            symmetrizedActionSpaces.push_back(symmetrizedActionSpace);
            break;

        default:
            assert(false);
        }
    }

    return symmetrizedActionSpaces;
}


std::string ConnectFour::stateToString(const State& state) const {
    std::string str = "";

    const State::Board& board = state.getBoard();
    for (int row = 0; row < C4_NUM_ROWS; row++) {
        for (int col = 0; col < C4_NUM_COLS; col++) {
            switch (board[row * C4_NUM_COLS + col]) {
            case -1:
                str += ". ";
                break;
            case 0:
                // colored red
                str += "\033[31mO\033[0m ";
                break;
            case 1:
                // colored yellow
                str += "\033[33mX\033[0m ";
                break;
            default:
                assert(false);
            }
        }
        str += "\n";   
    }

    for (int col = 0; col < C4_NUM_COLS; col++) {
        str += std::to_string(col) + " ";
    }

    return str;
}


bool ConnectFour::checkWin(const State::Board& board, const int piece_row, const int piece_col, const Piece piece) const {
    // First, extend left and right
    int row_count = 1;

    int col = piece_col - 1;
    while (col >= 0 && board[piece_row * C4_NUM_COLS + col] == piece) {
        row_count++;
        col--;
    }

    col = piece_col + 1;
    while (col < C4_NUM_COLS && board[piece_row * C4_NUM_COLS + col] == piece) {
        row_count++;
        col++;
    }

    if (row_count >= 4) {
        return true;
    }

    // Next, extend up and down
    int col_count = 1;

    int row = piece_row - 1;
    while (row >= 0 && board[row * C4_NUM_COLS + piece_col] == piece) {
        col_count++;
        row--;
    }

    row = piece_row + 1;
    while (row < C4_NUM_ROWS && board[row * C4_NUM_COLS + piece_col] == piece) {
        col_count++;
        row++;
    }

    if (col_count >= 4) {
        return true;
    }

    // Extend along main diagonal
    int main_diag_count = 1;

    row = piece_row - 1;
    col = piece_col - 1;
    while (row >= 0 && col >= 0 && board[row * C4_NUM_COLS + col] == piece) {
        main_diag_count++;
        row--;
        col--;
    }

    row = piece_row + 1;
    col = piece_col + 1;
    while (row < C4_NUM_ROWS && col < C4_NUM_COLS && board[row * C4_NUM_COLS + col] == piece) {
        main_diag_count++;
        row++;
        col++;
    }

    if (main_diag_count >= 4) {
        return true;
    }

    // Extend along anti-diagonal
    int anti_diag_count = 1;

    row = piece_row - 1;
    col = piece_col + 1;
    while (row >= 0 && col < C4_NUM_COLS && board[row * C4_NUM_COLS + col] == piece) {
        anti_diag_count++;
        row--;
        col++;
    }

    row = piece_row + 1;
    col = piece_col - 1;
    while (row < C4_NUM_ROWS && col >= 0 && board[row * C4_NUM_COLS + col] == piece) {
        anti_diag_count++;
        row++;
        col--;
    }

    if (anti_diag_count >= 4) {
        return true;
    }

    return false;
}

} // namespace SPRL

