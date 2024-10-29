#ifndef SPRL_CHESS_NODE_HPP
#define SPRL_CHESS_NODE_HPP

#include "GameNode.hpp"
#include "ChessState.hpp"

namespace SPRL {

constexpr int CHESS_BOARD_WIDTH = 8;
constexpr int CHESS_BOARD_SIZE = CHESS_BOARD_WIDTH * CHESS_BOARD_WIDTH;
constexpr int CHESS_HISTORY_SIZE = 8;
constexpr int CHESS_MAX_DEPTH = 500;

// can use chess::constants::MAX_MOVES to bound length of game

class ChessNode : public GameNode<ChessNode, ChessState, CHESS_ACTION_SIZE> {
public:
    using State = ChessState;
    using MoveStorage = std::unordered_map<ActionIdx, chess::Move>;

    /**
     * Constructs a new Chess game node in the initial state (for root).
     */
    ChessNode() {
        setStartNode();
    }

    /**
     * Constructs a new Chess game node with given parameters.
     * Large mutable objects need to be moved in.
     */
    ChessNode(ChessNode* parent, ActionIdx action, ActionDist&& actionMask,
              Player player, Player winner, bool isTerminal, int depth,
              chess::Board&& board, MoveStorage&& moveStorage)
        : GameNode<ChessNode, ChessState, CHESS_ACTION_SIZE>
          { parent, action, std::move(actionMask), player, winner, isTerminal },
          m_depth { depth },
          m_board { std::move(board) },
          m_moveStorage { std::move(moveStorage) } {
            
    }

private:
    void setStartNodeImpl();
    std::unique_ptr<ChessNode> getNextNodeImpl(ActionIdx action);

    State getGameStateImpl() const;
    std::array<Value, 2> getRewardsImpl() const;

    std::string toStringImpl() const;

private:
    int m_depth;
    chess::Board m_board;
    MoveStorage m_moveStorage;

    friend class GameNode<ChessNode, State, CHESS_ACTION_SIZE>;
};

} // namespace SPRL

#endif