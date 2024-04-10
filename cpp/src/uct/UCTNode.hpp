#ifndef UCT_NODE_HPP
#define UCT_NODE_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <memory>

namespace SPRL { 

template<int BOARD_SIZE, int ACTION_SIZE>
class UCTTree;

/**
 * Class representing a node in the tree for the UCT algorithm.
*/
template<int BOARD_SIZE, int ACTION_SIZE>
class UCTNode {
public:
    /**
     * Holds statistics for the edges coming out of this node in the UCT tree.
    */
    struct EdgeStatistics {
        /// Prior probability of selecting edges, used in computing U.
        std::array<float, ACTION_SIZE> m_childPriors {};

        /// Total Q value accumulated on this edge.
        std::array<float, ACTION_SIZE> m_totalValues {};

        /// Number of times this edge has been traversed.
        std::array<int32_t, ACTION_SIZE> m_numberVisits {};

        EdgeStatistics() {
            m_childPriors.fill(0.0);
            m_totalValues.fill(0.0);
            m_numberVisits.fill(0);
        }
    };

    // Constructor for parent node.
    UCTNode(EdgeStatistics* edgeStats, Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state)
        : m_game { game }, m_state { state } {

        m_isTerminal = game->isTerminal(state);
        m_actionMask = game->actionMask(state);

        m_parentEdgeStatistics = edgeStats;
    }

    // Constructor for child nodes.
    UCTNode(UCTNode* parent, ActionIdx action, Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state)
        : m_parent { parent }, m_action { action }, m_game { game }, m_state { state } {

        m_isTerminal = game->isTerminal(state);
        m_actionMask = game->actionMask(state);

        m_parentEdgeStatistics = &parent->m_edgeStatistics;
    }

    const EdgeStatistics* getEdgeStatistics() const {
        return &m_edgeStatistics;
    }

    /**
     * Gets a reference to the current number of visits to the node.
    */
    int& N() { return m_parentEdgeStatistics->m_numberVisits[m_action]; }

    /**
     * Gets a reference to the current total value of the node.
    */
    float& W() { return m_parentEdgeStatistics->m_totalValues[m_action]; }

    /**
     * Number of visits to a particular child.
    */
    int child_N(ActionIdx action) const { return m_edgeStatistics.m_numberVisits[action]; }

    /**
     * Total value of a particular child.
    */
    float child_W(ActionIdx action) const { return m_edgeStatistics.m_totalValues[action]; }

    /**
     * Prior probability of selecting a particular child.
    */
    float child_P(ActionIdx action) const { return m_edgeStatistics.m_childPriors[action]; }

    /**
     * Average action value of a particular child, as described in UCT.
    */
    float child_Q(ActionIdx action) {
        return child_W(action) / (1 + child_N(action));
    }

    /**
     * Uncertainty value of a particular child, as described in the UCT algorithm.
    */
    float child_U(ActionIdx action) {
        return child_P(action) * std::sqrt(static_cast<float>(N())) / (1 + child_N(action));
    }

    /**
     * Returns the action index of the best move according to the UCT algorithm.
    */
    ActionIdx bestAction(float exploration) {
        assert(!m_isTerminal && m_isExpanded && m_isNetworkEvaluated);

        ActionIdx bestAction = -1;
        float bestValue = -std::numeric_limits<float>::infinity();

        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip
                continue;
            }

            const float value = child_Q(action) + exploration * child_U(action);

            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        return bestAction;
    }

    /**
     * Returns a raw pointer to the child.
     * 
     * If child does not already exist, creates it.
    */
    UCTNode* getAddChild(ActionIdx action) {
        assert(!m_isTerminal && m_isExpanded && m_isNetworkEvaluated);

        if (m_children[action] == nullptr) {
            m_children[action] = std::make_unique<UCTNode<BOARD_SIZE, ACTION_SIZE>>(
                this, action, m_game, m_game->nextState(m_state, action));
        }

        return m_children[action].get();
    }

    /**
     * Updates the cached network evaluations.
    */
    void updateNetworkOutput(const std::array<float, ACTION_SIZE>& networkPolicy, float valueEstimate, bool addNoise = true) {
        assert(!m_isNetworkEvaluated);
        m_networkPolicy = networkPolicy;
        m_networkValue = valueEstimate;
        m_isNetworkEvaluated = true;
    }

    /**
     * Expands a node, as in the UCT algorithm.
     * 
     * Only adds on the new network estimates if necessary, otherwise ignored.
    */
    void expand(const std::array<float, ACTION_SIZE>& networkPolicy, bool addNoise = true) {
        assert(!m_isTerminal && !m_isExpanded && m_isNetworkEvaluated);

        m_isExpanded = true;

        // Current init type is ZERO
        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip
                continue;
            }

            m_edgeStatistics.m_childPriors[action] = m_networkPolicy[action];
            // TODO: add initialization type options

            if (addNoise) {
                // TODO: add Dirichlet noise
            }
        }
    }


private:
    /// Raw pointer to the parent, nullptr if you are the root.
    UCTNode* m_parent { nullptr };

    /// Unique pointers to the children; the parent owns them.
    std::array<std::unique_ptr<UCTNode>, ACTION_SIZE> m_children { nullptr };

    /// The action index taken into this node, 0 if you are the root.
    ActionIdx m_action { 0 };

    /// Whether or not the node has been expanded by the UCT algorithm.
    bool m_isExpanded { false };


    /// A raw pointer to an instance of the game we are playing.
    Game<BOARD_SIZE, ACTION_SIZE>* m_game;

    /// Current state of the game.
    GameState<BOARD_SIZE> m_state;

    /// Action mask for legal moves from the current position.
    GameActionDist<ACTION_SIZE> m_actionMask;

    /// Whether the current state is terminal.
    bool m_isTerminal;


    /// Whether this node has already been evaluated by the network.
    bool m_isNetworkEvaluated { false };

    /// Cached network policy evaluation.
    std::array<float, ACTION_SIZE> m_networkPolicy {};

    /// Cached value estimate.
    float m_networkValue { 0.0 };


    /// Current statistics for the edges coming out of this node.
    EdgeStatistics m_edgeStatistics {};

    /// Current statistics for the edges coming out of the parent, populated even if root.
    EdgeStatistics* m_parentEdgeStatistics { nullptr };

    friend class UCTTree<BOARD_SIZE, ACTION_SIZE>;
};

} // namespace SPRL

#endif