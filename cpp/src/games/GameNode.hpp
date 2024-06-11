#ifndef SPRL_GAME_NODE_HPP
#define SPRL_GAME_NODE_HPP

#include "GameActionDist.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace SPRL {

/**
 * Represents a player in the game.
*/
enum class Player : int8_t {
    NONE = -1,
    ZERO = 0,
    ONE  = 1
};

/**
 * @returns The other player.
*/
constexpr Player otherPlayer(Player player) {
    switch (player) {
    case Player::ZERO: return Player::ONE;
    case Player::ONE:  return Player::ZERO;
    default:           return Player::NONE;
    }
}

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
 * @tparam ACTION_SIZE The size of the action space.
 * @tparam ImplNode The implementation of the game node, e.g. GoNode, for purposes of CRTP.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class GameNode {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    /**
     * Constructs a new game node.
    */
    GameNode()
        : m_parent { nullptr }, m_action { 0 },
          m_player { Player::ZERO }, m_winner { Player::NONE }, m_isTerminal { false } {
    }

    /**
     * Constructs a new game node with given parameters.
     * 
     * @param parent Raw pointer to the parent node.
     * @param action The action taken to reach the new node.
     * @param actionMask The action mask at the new node.
     * @param player The player to move in the new node.
     * @param winner The winner of the game in the new node.
     * @param isTerminal Whether the new node is terminal.
    */
    GameNode(ImplNode* parent, ActionIdx action, ActionDist&& actionMask,
             Player player, Player winner, bool isTerminal)

        : m_parent { parent }, m_action { action }, m_actionMask { std::move(actionMask) },
          m_player { player }, m_winner { winner }, m_isTerminal { isTerminal } {
    }

    virtual ~GameNode() { }

    /**
     * @returns A raw pointer to the parent of the current node.
    */
    ImplNode* getParent() const {
        return m_parent;
    }

    /**
     * @param action The action to take from the given state, must be legal.
     * 
     * @returns A raw pointer to the child of the current non-terminal node.
     * 
     * @note If the child does not already exist, creates it.
    */
    ImplNode* getAddChild(ActionIdx action) {
        assert(!m_isTerminal);
        assert(m_actionMask[action] > 0.0f);

        if (m_children[action] == nullptr) {
            m_children[action] = std::move(getNextNode(action));
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

        for (ActionIdx i = 0; i < ACTION_SIZE; ++i) {
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
     * @returns The winner of the game at this node.
    */
    Player getWinner() const {
        return m_winner;
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
    State getGameState() const {
        return static_cast<const ImplNode*>(this)->getGameStateImpl();
    }

    /**
     * @returns The rewards for the two players at this node.
    */
    std::array<Value, 2> getRewards() const {
        return static_cast<const ImplNode*>(this)->getRewardsImpl();
    }

    /**
     * @returns A string representation of the node, for display purposes.
    */
    std::string toString() const {
        return static_cast<const ImplNode*>(this)->toStringImpl();
    }

protected:
    /**
     * Mutates this node to the initial state of the game.
    */
    void setStartNode() {
        return static_cast<ImplNode*>(this)->setStartNodeImpl();
    }

    /**
     * @returns The node that would result from taking the given action.
     * 
     * @param action The action to take. Must be legal.
    */
    std::unique_ptr<ImplNode> getNextNode(ActionIdx action) {
        return static_cast<ImplNode*>(this)->getNextNodeImpl(action);
    }

    ImplNode* m_parent;  // Raw pointer to the parent, nullptr if root.
    std::array<std::unique_ptr<ImplNode>, ACTION_SIZE> m_children;  // Parent owns children.

    ActionIdx m_action;       // Action index taken into this node, 0 if root.
    ActionDist m_actionMask;  // Current action mask of legal moves.

    Player m_player;    // The player to move.
    Player m_winner;    // The winner of the game.
    bool m_isTerminal;  // Whether the current state is terminal.
};

} // namespace SPRL

#endif