#ifndef SPRL_CONNECT_FOUR_NODE_HPP
#define SPRL_CONNECT_FOUR_NODE_HPP

#include "GridState.hpp"

namespace SPRL {

constexpr int C4_NUM_ROWS = 6;
constexpr int C4_NUM_COLS = 7;

constexpr int C4_BOARD_SIZE = C4_NUM_ROWS * C4_NUM_COLS;
constexpr int C4_ACTION_SIZE = C4_NUM_COLS;
constexpr int C4_HISTORY_SIZE = 1;

/**
 * Implementation of the classic Connect Four game.
 * 
 * See https://en.wikipedia.org/wiki/Connect_Four for details.
*/
class ConnectFourNode : public GameNode<ConnectFourNode, GridState<C4_BOARD_SIZE, C4_HISTORY_SIZE>, C4_ACTION_SIZE> {
public:
    using Board = GridBoard<C4_BOARD_SIZE>;
    using State = GridState<C4_BOARD_SIZE, C4_HISTORY_SIZE>;

    /**
     * Constructs a new Connect Four game node in the initial state (for root).
    */
    ConnectFourNode() {
        setStartNode();
    }

    /**
     * Constructs a new Connect Four game node with given parameters.
     * Large mutable objects need to be moved in.
     * 
     * @param parent The parent node.
     * @param action The action taken to reach the new node.
     * @param actionMask The action mask at the new node.
     * @param player The new player to move.
     * @param winner The new winner of the game, if any.
     * @param isTerminal Whether the game has ended.
     * @param board The new board state.
    */
    ConnectFourNode(ConnectFourNode* parent, ActionIdx action, ActionDist&& actionMask,
                    Player player, Player winner, bool isTerminal, Board&& board)
        : GameNode<ConnectFourNode, State, C4_ACTION_SIZE> { parent, action, std::move(actionMask), player, winner, isTerminal },
          m_board { std::move(board) } {

    }

private:
    void setStartNodeImpl();
    std::unique_ptr<ConnectFourNode> getNextNodeImpl(ActionIdx action);
    
    State getGameStateImpl() const;
    std::array<Value, 2> getRewardsImpl() const;

    std::string toStringImpl() const;

private:
    static int toIndex(int row, int col);
    static bool checkWin(const Board& board, const int row, const int col, const Piece piece);

    Board m_board;

    friend class GameNode<ConnectFourNode, State, C4_ACTION_SIZE>;
    friend class ConnectFourNetwork;
    friend class ConnectFourSymmetrizer;
};

} // namespace SPRL

#endif