#ifndef SPRL_GAME_NODE_HPP
#define SPRL_GAME_NODE_HPP

#include "GameActionDist.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>

namespace SPRL {

/**
 * Represents a player in the game.
*/
enum class Player : int8_t {
    NONE = -1,
    ZERO = 0,
    ONE  = 1
};

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
 * @tparam State The state of the game.
 * @tparam AS The size of the action space.
*/
template <typename State, int AS>
class GameNode {
public:
    using ActionDist = GameActionDist<AS>;

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
     * @returns The player to move at this node.
    */
    Player getPlayer() const {
        return m_player;
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
     * @returns The game state at this node, e.g. a short history of board states
     * that can be fed into the neural network.
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
     * Mutates this node to the initial state of the game.
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
    ActionDist m_actionMask;  // Current action mask of legal moves.

    Player m_player;    // The player to move.
    Player m_winner;    // The winner of the game.
    bool m_isTerminal;  // Whether the current state is terminal.
};

} // namespace SPRL

#endif