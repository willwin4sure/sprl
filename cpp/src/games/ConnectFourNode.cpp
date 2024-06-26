#include "ConnectFourNode.hpp"

#include <cassert>

namespace SPRL {

ActionIdx ConnectFourNode::toIndex(int row, int col) {
    assert(row >= 0 && row < C4_NUM_ROWS);
    assert(col >= 0 && col < C4_NUM_COLS);
    return row * C4_NUM_COLS + col;
}

void ConnectFourNode::setStartNodeImpl() {
    m_parent = nullptr;
    m_action = 0;
    m_actionMask.fill(1.0f);
    m_player = Player::ZERO;
    m_winner = Player::NONE;
    m_isTerminal = false;
    m_board.fill(Piece::NONE);
}

std::unique_ptr<ConnectFourNode> ConnectFourNode::getNextNodeImpl(ActionIdx action) {
    assert(!m_isTerminal);
    assert(m_actionMask[action] > 0.0f);

    Board newBoard = m_board;  // A copy of the original.
    ActionDist newActionMask = m_actionMask;  // A copy of the original.

    assert(&newBoard != &m_board);
    assert(&newActionMask != &m_actionMask);

    const Player player = m_player;
    const Player newPlayer = otherPlayer(player);

    const Piece piece = pieceFromPlayer(player);

    int col = action;

    // Find the lowest empty row in the column
    int row = C4_NUM_ROWS - 1;
    while (row >= 0 && newBoard[toIndex(row, col)] != Piece::NONE) {
        row--;
    }

    // Place the piece there
    newBoard[toIndex(row, col)] = piece;

    // Update the action mask if necessary
    if (row == 0) {
        newActionMask[col] = 0.0f;
    }

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

    // Whether the game has ended
    bool terminal = winner != Player::NONE || boardFilled;

    // If game ended, should be no legal actions
    if (terminal) {
        newActionMask.fill(0.0f);
    }

    std::unique_ptr<ConnectFourNode> newNode = std::make_unique<ConnectFourNode>(
        this, action, std::move(newActionMask), newPlayer, winner, terminal, std::move(newBoard));

    return std::move(newNode);
}

ConnectFourNode::State ConnectFourNode::getGameStateImpl() const {
    std::array<Board, C4_HISTORY_SIZE> history { m_board };
    return State { std::move(history), C4_HISTORY_SIZE, m_player };
}

std::array<Value, 2> ConnectFourNode::getRewardsImpl() const {
    switch (m_winner) {
    case Player::ZERO: return { 1.0f, -1.0f };
    case Player::ONE:  return { -1.0f, 1.0f };
    default:           return { 0.0f, 0.0f };
    }
}

std::string ConnectFourNode::toStringImpl() const {
    std::string str = "";

    bool shouldBold = true;
    for (int row = 0; row < C4_NUM_ROWS; row++) {
        for (int col = 0; col < C4_NUM_COLS; col++) {
            switch (m_board[row * C4_NUM_COLS + col]) {
            case Piece::NONE:
                str += ". ";
                break;
            case Piece::ZERO:
                // O, colored red. If the last move, then bold it as well.
                if (shouldBold && m_player == Player::ONE && m_action == col) {
                    str += "\x1b[31m\x1b[1mO\x1b[0m\033[0m ";
                    shouldBold = false;
                } else {
                    str += "\x1b[31mO\033[0m ";
                }
                break;
            case Piece::ONE:
                // X, colored yellow. If the last move, then bold it as well.
                if (shouldBold && m_player == Player::ZERO && m_action == col) {
                    str += "\x1b[33m\x1b[1mX\x1b[0m\033[0m ";
                    shouldBold = false;
                } else {
                    str += "\x1b[33mX\033[0m ";
                }
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

bool ConnectFourNode::checkWin(const Board& board, const int piece_row, const int piece_col, const Piece piece) {
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

