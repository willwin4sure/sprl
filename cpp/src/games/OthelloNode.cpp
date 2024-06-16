#include "OthelloNode.hpp"

#include <cassert>

namespace SPRL {

int OthelloNode::toIndex(int row, int col) {
    assert(row >= 0 && row < OTH_BOARD_WIDTH);
    assert(col >= 0 && col < OTH_BOARD_WIDTH);
    return row * OTH_BOARD_WIDTH + col;
}

bool OthelloNode::inBounds(int row, int col) {
    return 0 <= row && row < SPRL::OTH_BOARD_WIDTH
        && 0 <= col && col < SPRL::OTH_BOARD_WIDTH;
}

void OthelloNode::setStartNodeImpl() {
    m_parent = nullptr;
    m_action = 0;
    m_player = Player::ZERO;
    m_winner = Player::NONE;
    m_isTerminal = false;

    m_board.fill(Piece::NONE);
    m_board[toIndex(3, 3)] = Piece::ONE;
    m_board[toIndex(3, 4)] = Piece::ZERO;
    m_board[toIndex(4, 3)] = Piece::ZERO;
    m_board[toIndex(4, 4)] = Piece::ONE;

    m_actionMask = actionMask(m_board, m_player);
}

std::unique_ptr<OthelloNode> OthelloNode::getNextNodeImpl(ActionIdx action) {
    assert(!m_isTerminal);
    assert(m_actionMask[action] > 0.0f);

    Board newBoard = m_board;  // A copy of the original.
    ActionDist newActionMask = m_actionMask;  // A copy of the original.

    assert(&newBoard != &m_board);
    assert(&newActionMask != &m_actionMask);

    const Player player = m_player;
    const Player newPlayer = otherPlayer(player);

    const Piece piece = pieceFromPlayer(player);

    // Action index 64 is a pass.
    if (action != OTH_BOARD_SIZE) {
        // Place the piece there
        newBoard[action] = piece;

        int row = action / OTH_BOARD_WIDTH;
        int col = action % OTH_BOARD_WIDTH;

        // Perform all the captures
        for (const int idx : captures(newBoard, row, col, piece)) {
            newBoard[idx] = piece;
        }
    }

    Player winner = Player::NONE;
    const bool terminal = isTerminal(newBoard);

    if (terminal) {
        int count0 = 0;
        int count1 = 0;
        
        for (int i = 0; i < OTH_BOARD_SIZE; ++i) {
            if (newBoard[i] == Piece::ZERO) {
                count0++;
            }
            else if (newBoard[i] == Piece::ONE) {
                count1++;
            }
        }

        if (count0 > count1) winner = Player::ZERO;
        if (count1 > count0) winner = Player::ONE;
    }

    std::unique_ptr<OthelloNode> newNode = std::make_unique<OthelloNode>(
        this, action, actionMask(newBoard, newPlayer), newPlayer, winner, terminal, std::move(newBoard));

    return std::move(newNode);
}

OthelloNode::State OthelloNode::getGameStateImpl() const {
    std::array<Board, OTH_HISTORY_SIZE> history { m_board };
    return State { std::move(history), OTH_HISTORY_SIZE, m_player };
}

std::array<Value, 2> OthelloNode::getRewardsImpl() const {
    switch (m_winner) {
    case Player::ZERO: return { 1.0f, -1.0f };
    case Player::ONE:  return { -1.0f, 1.0f };
    default:           return { 0.0f, 0.0f };
    }
}

std::string OthelloNode::toStringImpl() const {
    std::string str = "";

    str += "  ";
    for (int col = 0; col < OTH_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";
    
    for (int row = 0; row < OTH_BOARD_WIDTH; row++) {
        str += std::to_string(row) + " ";
        for (int col = 0; col < OTH_BOARD_WIDTH; col++) {
            switch (m_board[toIndex(row, col)]) {
            case Piece::NONE:
                str += ". ";
                break;
                
            case Piece::ZERO:
                // O, colored red. If the last move, then bold it as well.
                if (m_action == toIndex(row, col)) {
                    str += "\x1b[31m\x1b[1mO\x1b[0m\033[0m ";
                } else {
                    str += "\x1b[31mO\033[0m ";
                }
                break;

            case Piece::ONE:
                // X, colored yellow. If the last move, then bold it as well.
                if (m_action == toIndex(row, col)) {
                    str += "\x1b[33m\x1b[1mX\x1b[0m\033[0m ";
                } else {
                    str += "\x1b[33mX\033[0m ";
                }
                break;

            default:
                assert(false);
            }
        }
        str += std::to_string(row);
        str += "\n";
    }

    str += "  ";
    for (int col = 0; col < OTH_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";

    return str;
}

OthelloNode::ActionDist OthelloNode::actionMask(const Board& board, const Player player) {
    ActionDist mask {};

    for (int row = 0; row < OTH_BOARD_WIDTH; ++row) {
        for (int col = 0; col < OTH_BOARD_WIDTH; ++col) {
            if (board[toIndex(row, col)] != Piece::NONE) continue;
            mask[toIndex(row, col)] = canCapture(board, row, col, pieceFromPlayer(player)) ? 1.0f : 0.0f;
        }
    }
    
    bool canPass = true;
    for (int i = 0; i < OTH_BOARD_SIZE; ++i) {
        if (mask[i] > 0.0f) {
            canPass = false;
            break;
        }
    }

    mask[OTH_BOARD_SIZE] = canPass ? 1.0f : 0.0f;

    return mask;
}

bool OthelloNode::isTerminal(const Board& board) {
    bool output = true;

    ActionDist mask0 = actionMask(board, Player::ZERO);

    // If you cannot pass, then you can move.
    if (mask0[OTH_BOARD_SIZE] == 0.0f) return false;

    ActionDist mask1 = actionMask(board, Player::ONE);

    // Returns true iff both players can only pass.
    return mask1[OTH_BOARD_SIZE] > 0.0f;
}

const std::vector<ActionIdx> OthelloNode::captures(
    const Board& board, const int row, const int col, const Piece piece) {

    std::vector<ActionIdx> output {};

    constexpr int NUM_DIRS = 8;
    constexpr std::array<int, NUM_DIRS> r_delta { 1, 1, 0, -1, -1, -1, 0, 1 };
    constexpr std::array<int, NUM_DIRS> c_delta { 0, 1, 1, 1, 0, -1, -1, -1 };

    const Piece opponentPiece = otherPiece(piece);

    for (int i = 0; i < NUM_DIRS; ++i) {
        int nextRow = row + r_delta[i];
        int nextCol = col + c_delta[i];

        while (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == opponentPiece) {
            nextRow += r_delta[i];
            nextCol += c_delta[i];
        }

        if (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == piece) {
            for (int r = row + r_delta[i], c = col + c_delta[i];
                 r != nextRow || c != nextCol;
                 r += r_delta[i], c += c_delta[i]) {

                output.push_back(toIndex(r, c));
            }
        }
    }

    return output;
}

bool OthelloNode::canCapture(const Board& board, int row, int col, const Piece piece) {
    constexpr int NUM_DIRS = 8;
    constexpr std::array<int, NUM_DIRS> r_delta {1, 1, 0, -1, -1, -1, 0, 1};
    constexpr std::array<int, NUM_DIRS> c_delta {0, 1, 1, 1, 0, -1, -1, -1};

    const Piece opponentPiece = otherPiece(piece);

    for (int i = 0; i < NUM_DIRS; ++i) {
        int nextRow = row + r_delta[i];
        int nextCol = col + c_delta[i];

        bool oppExists = false;

        while (inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == opponentPiece) {
            nextRow += r_delta[i];
            nextCol += c_delta[i];

            oppExists = true;
        }

        if (oppExists && inBounds(nextRow, nextCol) && board[toIndex(nextRow, nextCol)] == piece) {
            return true;
        }
    }

    return false;
}


} // namespace SPRL

