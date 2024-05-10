#ifndef UCT_NODE_HPP
#define UCT_NODE_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"

#include "../random/Random.hpp"

#include "../constants.hpp"

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
template <int BOARD_SIZE, int ACTION_SIZE>
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
            reset();
        }

        void reset() {
            m_childPriors.fill(0.0);
            m_totalValues.fill(0.0);
            m_numberVisits.fill(0);
        }
    };

    // Constructor for parent node.
    UCTNode(EdgeStatistics* edgeStats, Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state, bool useParentQ = true)
        : m_parent { nullptr }, m_action { 0 }, m_game { game }, m_state { state }, m_useParentQ { useParentQ } {

        m_isTerminal = state.isTerminal();
        m_actionMask = game->actionMask(state);

        m_parentEdgeStatistics = edgeStats;
    }

    // Constructor for child nodes.
    UCTNode(UCTNode* parent, ActionIdx action, Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state, bool useParentQ = true)
        : m_parent { parent }, m_action { action }, m_game { game }, m_state { state }, m_useParentQ { useParentQ } {

        m_isTerminal = state.isTerminal();
        m_actionMask = game->actionMask(state);

        m_parentEdgeStatistics = &parent->m_edgeStatistics;
    }

    /**
     * Returns a readonly pointer to the edge statistics of this node.
    */
    const EdgeStatistics* getEdgeStatistics() const {
        return &m_edgeStatistics;
    }

    /**
     * Returns a readonly reference to the state of the node.
    */
    const GameState<BOARD_SIZE>& getState() const {
        return m_state;
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
     * Average action value of the current node, as described in UCT.
    */
    float Q() {
        return W() / (1 + static_cast<float>(N()));  // add 1 to avoid division by zero
    }

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
        return child_W(action) / (1 + static_cast<float>(child_N(action)));  // add 1 to avoid division by zero
    }

    /**
     * Uncertainty value of a particular child, as described in the UCT algorithm.
    */
    float child_U(ActionIdx action) {
        return child_P(action) * std::sqrt(static_cast<float>(N()))
                               / (1 + static_cast<float>(child_N(action)));  // add 1 to avoid division by zero
    }

    /**
     * Returns the action index of the best move according to the UCT algorithm.
     * 
     * Can only be applied on active nodes.
    */
    ActionIdx bestAction(float exploration) {
        assert(!m_isTerminal);
        assert(m_isExpanded);
        assert(m_isNetworkEvaluated);

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

        assert(bestAction != -1);

        return bestAction;
    }

    /**
     * Returns a raw pointer to the child of the current non-terminal node.
     * 
     * If the child does not already exist, creates it.
    */
    UCTNode* getAddChild(ActionIdx action) {
        assert(!m_isTerminal);

        if (m_children[action] == nullptr) {
            // Child doesn't exist, so we create it.
            m_children[action] = std::make_unique<UCTNode<BOARD_SIZE, ACTION_SIZE>>(
                this, action, m_game, m_game->nextState(m_state, action), m_useParentQ);

            // Parent Q-initialization
            if (m_isNetworkEvaluated) {
                m_edgeStatistics.m_totalValues[action] = m_useParentQ ? m_networkValue : 0.0f;           
            }
        }

        return m_children[action].get();
    }

    /**
     * Caches the network output. Converts empty nodes into gray nodes.
    */
    void addNetworkOutput(const std::array<float, ACTION_SIZE>& networkPolicy, float valueEstimate) {
        assert(!m_isTerminal);
        assert(!m_isNetworkEvaluated);
        assert(!m_isExpanded);

        m_isNetworkEvaluated = true;

        m_networkPolicy = networkPolicy;
        m_networkValue = valueEstimate;
    }

    /**
     * Expands a node, as in the UCT algorithm. Converts gray nodes into active nodes.
     * 
     * Only adds on the new network estimates if necessary, otherwise ignored.
    */
    void expand(bool addNoise = true) {
        assert(!m_isTerminal);
        assert(!m_isExpanded);
        assert(m_isNetworkEvaluated);

        m_isExpanded = true;

        int numLegal = 0;
        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip
                continue;
            }

            m_edgeStatistics.m_childPriors[action] = m_networkPolicy[action];
            ++numLegal;
        }

        if (addNoise && m_parent == nullptr) {
            // Only add noise if you are the root node.
            std::vector<float> noise (numLegal);
            GetRandom().Dirichlet(DIRICHLET_ALPHA, noise);

            int readIdx = 0;
            for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
                if (m_actionMask[action] == 0.0f) {
                    // Illegal action, skip
                    continue;
                }

                m_edgeStatistics.m_childPriors[action] = (1 - DIRICHLET_EPSILON) * m_edgeStatistics.m_childPriors[action]
                                                             + DIRICHLET_EPSILON * noise[readIdx];
                                                        
                ++readIdx;
            }
        }
    }

    /**
     * Prunes away all children of the node except for the one corresponding to the given action.
     * 
     * Can be used on any non-terminal node. Used in rerooting.
    */
    void pruneChildrenExcept(ActionIdx action) {
        assert(!m_isTerminal);

        for (ActionIdx i = 0; i < ACTION_SIZE; ++i) {
            if (i != action) {
                m_children[i] = nullptr;
            }
        }
    }

private:
    /// Raw pointer to the parent, nullptr if you are the root.
    UCTNode* m_parent;

    /// Unique pointers to the children; the parent owns them.
    std::array<std::unique_ptr<UCTNode>, ACTION_SIZE> m_children {};

    /// The action index taken into this node, 0 if you are the root.
    ActionIdx m_action;

    /// Whether or not the node has been expanded by the UCT algorithm.
    bool m_isExpanded { false };


    /// A raw pointer to an instance of the game we are playing.
    Game<BOARD_SIZE, ACTION_SIZE>* m_game;

    /// Current state of the game.
    GameState<BOARD_SIZE> m_state;

    // Whether to use parent Q-initialization.
    bool m_useParentQ;

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
