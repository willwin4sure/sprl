#include "ConnectFourNode.hpp"

#include <cassert>

namespace SPRL {

int ConnectFourNode::toIndex(int row, int col) {
    return row * C4_NUM_COLS + col;
}

void ConnectFourNode::setStartNode() {
    m_board.fill(Piece::NONE);
    m_player = Player::ZERO;
    m_winner = Player::NONE;
    m_isTerminal = false;
    m_actionMask = computeActionMask();
}

std::unique_ptr<ConnectFourNode::GNode> ConnectFourNode::getNextNode(ActionIdx action) const {
    assert(!m_isTerminal);
    assert(m_actionMask[action] > 0.0f);

    Board newBoard = m_board;  // The board that we return, a copy of the original
    assert(&newBoard != &m_board);

    const Player player = m_player;
    const Player newPlayer = otherPlayer(player);

    const Piece piece = static_cast<Piece>(player);

    int col = action;

    // Find the lowest empty row in the column
    int row = C4_NUM_ROWS - 1;
    while (row >= 0 && newBoard[toIndex(row, col)] != Piece::NONE) {
        row--;
    }

    // Place the piece there
    newBoard[toIndex(row, col)] = piece;

    // Check if the move wins the game
    Player winner = checkWin(newBoard, row, col, piece) ? player : Player::NONE;

    // See if the game has ended in a draw
    bool boardFilled = true;
    for (int col = 0; col < C4_NUM_COLS; col++) {
        if (newBoard[toIndex(0, col)] == Piece::NONE) {
            boardFilled = false;
            break;
        }
    }

    return std::make_unique<ConnectFourNode>(
        newBoard, newPlayer, winner, (winner != Player::NONE) || boardFilled);
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

std::pair<Value, Value> ConnectFour::rewards(const State& state) const {
    const Player winner = state.getWinner();
    switch (winner) {
    case 0:
        return { 1.0f, -1.0f };
    case 1:
        return { -1.0f, 1.0f };
    default:
        return { 0.0f, 0.0f };
    }
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

    return anti_diag_count >= 4;
}

} // namespace SPRL

