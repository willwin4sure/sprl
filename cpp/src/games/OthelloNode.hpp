#ifndef SPRL_OTHELLO_NODE_HPP
#define SPRL_OTHELLO_NODE_HPP

#include "GridState.hpp"

namespace SPRL {

constexpr int OTH_BOARD_WIDTH = 8;
constexpr int OTH_BOARD_SIZE = OTH_BOARD_WIDTH * OTH_BOARD_WIDTH;
constexpr int OTH_ACTION_SIZE = OTH_BOARD_SIZE + 1;  // Last index represents pass.
constexpr int OTH_HISTORY_SIZE = 1;

/**
 * Implementation of the game of Othello, also known as Reversi.
 * 
 * See https://en.wikipedia.org/wiki/Reversi for details.
*/
class OthelloNode : public GameNode<OthelloNode, GridState<OTH_BOARD_SIZE, OTH_HISTORY_SIZE>, OTH_ACTION_SIZE> {
public:
    using Board = GridBoard<OTH_BOARD_SIZE>;
    using State = GridState<OTH_BOARD_SIZE, OTH_HISTORY_SIZE>;

    /**
     * Constructs a new Othello game node in the initial state (for root).
    */
    OthelloNode() {
        setStartNode();
    }

    /**
     * Constructs a new Othello game node with given parameters.
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
    OthelloNode(OthelloNode* parent, ActionIdx action, ActionDist&& actionMask,
                Player player, Player winner, bool isTerminal, Board&& board)
        : GameNode<OthelloNode, State, OTH_ACTION_SIZE> { parent, action, std::move(actionMask), player, winner, isTerminal },
          m_board { std::move(board) } {

    }

private:
    void setStartNodeImpl();
    std::unique_ptr<OthelloNode> getNextNodeImpl(ActionIdx action);
    
    State getGameStateImpl() const;
    std::array<Value, 2> getRewardsImpl() const;

    std::string toStringImpl() const;

private:
    static int toIndex(int row, int col);
    static bool inBounds(int row, int col);

    /**
     * @returns The action mask for a particular player on the given board.
     */
    static ActionDist actionMask(const Board& board, const Player player);

    /**
     * @returns If the board has no legal actions by either player.
     */
    static bool isTerminal(const Board& board);

    /**
     * @returns The indices of the pieces that would be captured by placing a piece at the given position.
     */
    static const std::vector<ActionIdx> captures(const Board& board, const int row, const int col, const Piece piece);

    /**
     * @returns Whether the given piece can capture any pieces by placing it at the given position.
     */
    static bool canCapture(const Board& board, int row, int col, const Piece piece);

    Board m_board;

    friend class GameNode<OthelloNode, State, OTH_ACTION_SIZE>;
    friend class OthelloHeuristic;
};

} // namespace SPRL

#endif