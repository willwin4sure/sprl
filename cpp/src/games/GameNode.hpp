#ifndef SPRL_GAME_NODE_HPP
#define SPRL_GAME_NODE_HPP

#include "GameState.hpp"

#include <cassert>
#include <memory>
#include <string>

namespace SPRL {

/// Type alias for the action index.
using ActionIdx = int16_t;

/// Type alias for the relative value of a position, a float in the range `[-1, 1]`.
using Value = float;

/**
 * Represents a node in the game tree.
 * 
 * Used to implement games that hold more state than just the information
 * available on the current game board.
 * 
 * @tparam BS The size of the board.
 * @tparam AS The size of the action space.
*/
template <int BS, int AS>
class GameNode {
public:
    using Board = GameBoard<BS>;
    using State = GameState<BS>;
    using ActionDist = GameActionDist<AS>;

    /**
     * Constructs a new game node with the initial state of the game.
    */
    GameNode()
        : m_parent { nullptr }, m_action { 0 }, m_board {},
          m_player { Player::ZERO }, m_winner { Player::NONE }, m_isTerminal { false } {

        setStartNode();
    }

    /**
     * Constructs a new game node with given parameters.
     * 
     * @param parent Raw pointer to the parent node.
     * @param action The action taken to reach the new node.
     * @param board The state of the board, will be copied.
     * @param player The player to move in the new node.
     * @param winner The winner of the game in the new node.
     * @param isTerminal Whether the new node is terminal.
    */
    GameNode(GameNode* parent, ActionIdx action, const Board& board,
             Player player, Player winner, bool isTerminal)

        : m_parent { parent }, m_action { action }, m_board { board },
          m_player { player }, m_winner { winner }, m_isTerminal { isTerminal } {

    }

    virtual ~GameNode() { }

    /**
     * @returns A raw pointer to the parent of the current node.
    */
    GameNode* getParent() const {
        return m_parent;
    }

    /**
     * @param action The action to take from the given state, must be legal.
     * 
     * @returns A raw pointer to the child of the current non-terminal node.
     * 
     * @note If the child does not already exist, creates it.
    */
    GameNode* getAddChild(ActionIdx action) {
        assert(!m_isTerminal);
        assert(m_actionMask[action] > 0.0f);

        if (m_children[action] == nullptr) {
            m_children[action] = getNextNode(action);
        }

        return m_children[action].get();
    }

    /**
     * Prunes away all the children of the node (and their descendants)
     * except for the one corresponding to the given action.
     * 
     * Can be used on any non-terminal node, for subtree reuse.
    */
    void pruneChildrenExcept(ActionIdx action) {
        assert(!m_isTerminal);
        assert(m_actionMask[action] > 0.0f);

        for (ActionIdx i = 0; i < AS; ++i) {
            if (i != action) {
                m_children[i] = nullptr;
            }
        }
    }

    /**
     * @returns Whether the node is terminal.
    */
    bool isTerminal() const {
        return m_isTerminal;
    }

    /**
     * @returns A readonly reference to the action mask at this node.
    */
    const ActionDist& getActionMask() const {
        return m_actionMask;
    }
    
    /**
     * @returns The game state at this node, i.e. a short history of board states
     * fed into the neural network.
    */
    virtual State getGameState() const = 0;

    /**
     * @returns The rewards for the two players at this node.
    */
    virtual std::array<Value, 2> getRewards() const = 0;

    /**
     * @returns A string representation of the node, for display purposes.
    */
    virtual std::string toString() const = 0;

private:
    /**
     * Sets the node to the initial state of the game.
    */
    virtual void setStartNode() = 0;

    /**
     * @returns The node that would result from taking the given action.
     * 
     * @param action The action to take. Must be legal.
    */
    virtual std::unique_ptr<GameNode> getNextNode(ActionIdx action) const = 0;

    GameNode* m_parent;  // Raw pointer to the parent, nullptr if root.
    std::array<std::unique_ptr<GameNode>, AS> m_children;  // Parent owns children.

    ActionIdx m_action;       // Action index taken into this node, 0 if root.
    Board m_board;            // Current state of the pieces on the board.
    ActionDist m_actionMask;  // Current action mask of legal moves.

    Player m_player;    // The player to move.
    Player m_winner;    // The winner of the game.
    bool m_isTerminal;  // Whether the current state is terminal.
};

} // namespace SPRL

#endif