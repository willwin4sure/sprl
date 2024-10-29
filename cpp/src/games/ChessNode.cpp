#include "ChessNode.hpp"

#include "../utils/timer.hpp"

namespace SPRL {

/**
 * Flips the ranks of a UCI move.
 */
static std::string flipUCI(const std::string& uci) {
    assert(uci.size() == 4 || uci.size() == 5);

    std::string flipped = uci;
    flipped[1] = '8' - (uci[1] - '1');
    flipped[3] = '8' - (uci[3] - '1');

    return flipped;
}

/**
 * Drops the promotion suffix from a UCI move if it is the default 'q'.
 */
static std::string uciToMoveStr(const std::string& uci) {
    assert(uci.size() == 4 || uci.size() == 5);

    std::string moveStr = uci;
    if (moveStr.size() == 5 && moveStr[4] == 'q') {
        moveStr.pop_back();
    }

    return moveStr;
}

/**
 * Computes the action mask for the current board state, with a parameter
 * to flip the ranks (used for Black's actions).
 */
static std::pair<ChessNode::ActionDist, ChessNode::MoveStorage>
computeMaskStorage(const chess::Movelist& moves, bool flip) {
    ChessNode::ActionDist mask;
    ChessNode::MoveStorage storage;

    for (const chess::Move& move : moves) {
        std::string uciMove = chess::uci::moveToUci(move);
        if (flip) {
            // Flip the move for Black.
            uciMove = flipUCI(uciMove);
        }

        std::string moveStr = uciToMoveStr(uciMove);

        // Should always exist in map.
        ActionIdx action = CHESS_MOVE_STR_TO_IDX.at(moveStr);
        mask[action] = 1.0f;

        storage[action] = move;
    }

    return { mask, storage };
}

void ChessNode::setStartNodeImpl() {
    m_parent = nullptr;
    m_action = 0;
    m_player = Player::ZERO;
    m_winner = Player::NONE;
    m_isTerminal = false;

    m_depth = 0;
    m_board = chess::Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, m_board);

    auto maskStorage = computeMaskStorage(moves, false);  // White first.
    m_actionMask = std::move(maskStorage.first);
    m_moveStorage = std::move(maskStorage.second);
}


std::unique_ptr<ChessNode> ChessNode::getNextNodeImpl(ActionIdx actionIdx) {
    // Copy the board and make the appropriate move.
    chess::Board newBoard = m_board;

    std::string moveStr = CHESS_MOVE_STRS[actionIdx];
    if (m_player == Player::ONE) {
        // Flip the move for Black.
        moveStr = flipUCI(moveStr);
    }

    newBoard.makeMove(m_moveStorage[actionIdx]);

    Player winner = Player::NONE;
    bool isTerminal = false;

    // Check if the game ended.
    if (newBoard.isHalfMoveDraw()) {
        chess::Movelist movelist;
        chess::movegen::legalmoves(movelist, newBoard);

        if (movelist.empty() && newBoard.inCheck()) {
            // The player that just moved wins.
            winner = m_player;
        }

        // Regardless, the game ends.
        isTerminal = true;

    } else if (newBoard.isInsufficientMaterial()) {
        // Draw due to insufficient material.
        isTerminal = true;

    } else if (newBoard.isRepetition()) {
        // Draw due to threefold repetition.
        isTerminal = true;
    }

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, newBoard);

    if (moves.empty()) {
        if (newBoard.inCheck()) {
            // Checkmate. The player that just moved wins.
            winner = m_player;
        } else {
            // Stalemate.
        }

        // Regardless, the game ends.
        isTerminal = true;
    }

    if (m_depth + 1 >= CHESS_MAX_DEPTH) {
        // Draw due to maximum depth.
        isTerminal = true;
    }

    // Compute the new action mask and storage if necessary.
    // Flip the ranks for Black.
    ActionDist newActionMask = ActionDist {};
    MoveStorage newMoveStorage = MoveStorage {};

    if (!isTerminal) {
        auto maskStorage = computeMaskStorage(moves, otherPlayer(m_player) == Player::ONE);
        newActionMask = std::move(maskStorage.first);
        newMoveStorage = std::move(maskStorage.second);
    }

    return std::make_unique<ChessNode>(
        this,
        actionIdx,
        std::move(newActionMask),
        otherPlayer(m_player),
        winner,
        isTerminal,
        m_depth + 1,
        std::move(newBoard),
        std::move(newMoveStorage)
    );
}

ChessNode::State ChessNode::getGameStateImpl() const {
    return State {};
}

std::array<Value, 2> ChessNode::getRewardsImpl() const {
    switch (m_winner) {
    case Player::ZERO: return { 1.0f, -1.0f };
    case Player::ONE:  return { -1.0f, 1.0f };
    default:           return { 0.0f, 0.0f };
    }
}

std::string pieceToUnicode(const chess::Piece& piece) {
    char c = std::string(piece)[0];
    switch (c) {
        case 'P': return "♙";
        case 'N': return "♘";
        case 'B': return "♗";
        case 'R': return "♖";
        case 'Q': return "♕";
        case 'K': return "♔";
        case 'p': return "♟";
        case 'n': return "♞";
        case 'b': return "♝";
        case 'r': return "♜";
        case 'q': return "♛";
        case 'k': return "♚";
        default: return ".";
    }
}

std::string ChessNode::toStringImpl() const {
    std::string str = "";

    str += "Player: " + std::to_string(static_cast<int>(m_player)) + "\n";
    str += "Winner: " + std::to_string(static_cast<int>(m_winner)) + "\n";
    str += "IsTerminal: " + std::to_string(m_isTerminal) + "\n";
    str += "Action: " + std::to_string(m_action) + "\n";
    str += "Depth: " + std::to_string(m_depth) + "\n";

    str += "Board:\n";

    str += "  ";
    for (int col = 0; col < CHESS_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }
    str += "\n";

    std::string moveStr = CHESS_MOVE_STRS[m_action];
    std::string endPos = moveStr.substr(2, 2);

    int endRank = endPos[1] - '1';
    if (m_player == Player::ZERO) {
        endRank = CHESS_BOARD_WIDTH - 1 - endRank;
    }

    int endFile = endPos[0] - 'a';

    for (int r = 0; r < CHESS_BOARD_WIDTH; ++r) {
        str += std::to_string(CHESS_BOARD_WIDTH - r) + " ";

        // Flip ranks for Black.
        int rank = (m_player == Player::ZERO) ? CHESS_BOARD_WIDTH - 1 - r : r;
        for (int file = 0; file < CHESS_BOARD_WIDTH; ++file) {
            chess::Piece piece = m_board.at(
                chess::Square { chess::Rank { rank }, chess::File { file } });

            std::string unicodePiece = pieceToUnicode(piece);

            if (piece.color() == chess::Color::underlying::WHITE) {
                unicodePiece = "\033[37m" + unicodePiece + "\033[0m";

            } else if (piece.color() == chess::Color::underlying::BLACK) {
                unicodePiece = "\033[30m" + unicodePiece + "\033[0m";
            }

            if (m_depth != 0 && rank == endRank && file == endFile) {
                // Bolded
                unicodePiece = "\033[1m" + unicodePiece + "\033[0m";
            } 

            str += unicodePiece + " ";
        }

        str += " " + std::to_string(CHESS_BOARD_WIDTH - r) + "\n";
    }

    str += "  ";
    for (int col = 0; col < CHESS_BOARD_WIDTH; col++) {
        str += ('A' + col);
        str += " ";
    }

    return str;
}

} // namespace SPRL