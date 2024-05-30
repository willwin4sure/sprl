#ifndef SPRL_CONNECT_FOUR_NODE_HPP
#define SPRL_CONNECT_FOUR_NODE_HPP

#include "GridState.hpp"

namespace SPRL {

constexpr int C4_NUM_ROWS = 6;
constexpr int C4_NUM_COLS = 7;

constexpr int C4_BS = C4_NUM_ROWS * C4_NUM_COLS;
constexpr int C4_AS = C4_NUM_COLS;

/**
 * Implementation of the classic Connect Four game.
 * 
 * See https://en.wikipedia.org/wiki/Connect_Four for details.
*/
class ConnectFourNode : public GameNode<GridState<C4_BS>, C4_AS> {
public:
    using Board = GridBoard<C4_BS>;
    using State = GridState<C4_BS>;
    using GNode = GameNode<State, C4_AS>;

    /**
     * Constructs a new Connect Four game node in the initial state (for root).
    */
    ConnectFourNode() {
        setStartNode();
    }

    /**
     * Constructs a new Connect Four game node with given parameters.
     * 
     * @param parent The parent node.
     * @param action The action taken to reach the new node.
     * @param actionMask The action mask at the new node.
     * @param player The new player to move.
     * @param winner The new winner of the game, if any.
     * @param isTerminal Whether the game had ended.
     * @param board The new board state.
    */
    ConnectFourNode(ConnectFourNode* parent, ActionIdx action, const ActionDist& actionMask,
                    Player player, Player winner, bool isTerminal, const Board& board)
        : GNode { parent, action, actionMask, player, winner, isTerminal },
          m_board { board } {

    }

    void setStartNode() override;
    std::unique_ptr<GNode> getNextNode(ActionIdx action) override;
    
    State getGameState() const override;
    std::array<Value, 2> getRewards() const override;

    std::string toString() const override;

private:
    static int toIndex(int row, int col);
    static bool checkWin(const Board& board, const int row, const int col, const Piece piece);

    Board m_board;
};

} // namespace SPRL

#endif