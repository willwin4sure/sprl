#ifndef SPRL_UCT_NODE_HPP
#define SPRL_UCT_NODE_HPP

#include "../games/GameNode.hpp"

#include "../random/Random.hpp"

#include "../constants.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <memory>

namespace SPRL { 

template<typename State, int AS>
class UCTTree;

/**
 * Supported methods of initializing the Q values of the nodes.
*/
enum class InitQ {
    ZERO,    // Always initialize to zero.
    PARENT,  // Initialize to the network output of the parent, if available.
};

/**
 * Class representing a node in the tree for the UCT algorithm.
 * 
 * @tparam State The state of the game.
 * @tparam AS The size of the action space.
*/
template <typename State, int AS>
class UCTNode {
public:
    using ActionDist = GameActionDist<AS>;
    using GNode = GameNode<State, AS>;

    /**
     * Holds statistics for the edges coming out of this node in the UCT tree.
    */
    struct EdgeStatistics {

        ActionDist m_childPriors {};  // Prior from network, used to compute U.
        ActionDist m_totalValues {};  // Total Q value accumulated on each edge.
        ActionDist m_numVisits {};    // Number of times each edge has been traversed.

        EdgeStatistics() {
            reset();
        }

        void reset() {
            m_childPriors.fill(0.0);
            m_totalValues.fill(0.0);
            m_numVisits.fill(0.0);
        }
    };

    /**
     * Constructor for root UCT node.
     * 
     * @param edgeStats Pointer to the edge statistics of the virtual "parent" node, held by `UCTTree`.
     * @param gameNode The root game node, also held by `UCTTree`.
     * @param dirEps The epsilon parameter for Dirichlet noise.
     * @param dirAlpha The alpha parameter for Dirichlet noise.
     * @param initQMethod The method to use for initializing the Q values of the nodes.
    */
    UCTNode(EdgeStatistics* edgeStats, GNode* gameNode,
            float dirEps = 0.25f, float dirAlpha = 0.1f, InitQ initQMethod = InitQ::PARENT)
        : m_gameNode { gameNode }, m_parentEdgeStatistics { edgeStats },
          m_dirEps { dirEps }, m_dirAlpha { dirAlpha }, m_initQMethod { initQMethod } {
          
        m_children.fill(nullptr);

        m_isTerminal = m_gameNode->isTerminal();
        m_actionMask = m_gameNode->getActionMask();
    }

    /**
     * Constructor for child UCT nodes.
     * 
     * @param parent Pointer to the parent UCT node.
     * @param action The action taken to reach this node.
     * @param gameNode The game node corresponding to this UCT node.
     * @param dirEps The epsilon parameter for Dirichlet noise.
     * @param dirAlpha The alpha parameter for Dirichlet noise.
     * @param initQMethod The method to use for initializing the Q values of the nodes.
    */
    UCTNode(UCTNode* parent, ActionIdx action, GNode* gameNode,
            float dirEps = 0.25f, float dirAlpha = 0.1f, InitQ initQMethod = InitQ::PARENT)
        : m_parent { parent }, m_action { action }, m_gameNode { gameNode },
          m_dirEps { dirEps }, m_dirAlpha { dirAlpha }, m_initQMethod { initQMethod } {
          

        m_children.fill(nullptr);

        m_isTerminal = m_gameNode->isTerminal();
        m_actionMask = m_gameNode->getActionMask();

        m_parentEdgeStatistics = &parent->m_edgeStatistics;
    }

    /**
     * @returns A readonly pointer to the edge statistics of this node.
    */
    const EdgeStatistics* getEdgeStatistics() const {
        return &m_edgeStatistics;
    }

    /**
     * @returns The player to move at this node.
    */
    Player getPlayer() const {
        return m_gameNode->getPlayer();
    }

    /**
     * @returns Whether the node is terminal.
    */
    bool isTerminal() const {
        return m_isTerminal;
    }

    /**
     * @returns The game state of the underlying game node.
    */
    State getGameState() const {
        return m_gameNode->getGameState();
    }

    /**
     * @returns The rewards of the underlying game node.
    */
    std::array<Value, 2> getRewards() const {
        return m_gameNode->getRewards();
    }

    /**
     * @returns A reference to the current number of visits to this node.
    */
    float& N() { return m_parentEdgeStatistics->m_numberVisits[m_action]; }

    /**
     * @returns A reference to the current total value of this node.
    */
    float& W() { return m_parentEdgeStatistics->m_totalValues[m_action]; }

    /**
     * @returns The current average action value of this node, as described in UCT.
    */
    float Q() {
        return W() / (1 + N());  // Adding 1 avoids division by zero.
    }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The number of visits to a particular child.
    */
    float child_N(ActionIdx action) const { return m_edgeStatistics.m_numberVisits[action]; }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The total value of a particular child.
    */
    float child_W(ActionIdx action) const { return m_edgeStatistics.m_totalValues[action]; }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The prior probability of selecting a particular child.
    */
    float child_P(ActionIdx action) const { return m_edgeStatistics.m_childPriors[action]; }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The average action value of a particular child, as described in UCT.
    */
    float child_Q(ActionIdx action) {
        return child_W(action) / (1 + child_N(action));  // Adding 1 avoids division by zero.
    }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The uncertainty value of a particular child, as described in the UCT algorithm.
    */
    float child_U(ActionIdx action) {
        return child_P(action) * std::sqrt(N()) / (1 + child_N(action));  // Adding 1 avoids division by zero.
    }

    /**
     * @param uWeight The weighting of the U value compared to the Q value.
     * 
     * @returns The action index of the best move according to the UCT algorithm.
     * 
     * @note Can only be applied on active nodes, i.e.
     * non-terminals that are evaluated and expanded.
    */
    ActionIdx bestAction(float uWeight) {
        assert(!m_isTerminal);

        assert(m_isExpanded);
        assert(m_isNetworkEvaluated);

        std::vector<ActionIdx> bestActions;
        float bestValue = -std::numeric_limits<float>::infinity();

        for (ActionIdx action = 0; action < AS; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip
                continue;
            }

            const float value = child_Q(action) + uWeight * child_U(action);

            if (value > bestValue) {
                bestValue = value;
                bestActions.clear();
                
                bestActions.push_back(action);

            } else if (value == bestValue) {
                bestActions.push_back(action);
            }
        }

        assert(!bestActions.empty());
        return bestActions[GetRandom().UniformInt(0, bestActions.size() - 1)];
    }

    /**
     * @returns A raw pointer to the child of the current non-terminal node.
     * 
     * @note If the child does not already exist, creates it.
    */
    UCTNode* getAddChild(ActionIdx action) {
        assert(!m_isTerminal);

        if (m_children[action] == nullptr) {
            // Child doesn't exist, so we create it.
            m_children[action] = std::make_unique<UCTNode>(
                this, action, m_gameNode->getAddChild(action), m_initQMethod);

            // Handle Q-initialization based on the method.
            switch (m_initQMethod) {
            case InitQ::ZERO:
                m_edgeStatistics.m_totalValues[action] = 0.0f;
                break;
            case InitQ::PARENT:
                m_edgeStatistics.m_totalValues[action] = m_isNetworkEvaluated ? m_networkValue : 0.0f;
                break;
            }
        }

        return m_children[action].get();
    }

    /**
     * Caches the network output. Converts empty nodes into gray nodes.
     * 
     * @param networkPolicy The policy output of the network.
     * @param valueEstimate The value output of the network.
    */
    void addNetworkOutput(const ActionDist& networkPolicy, Value valueEstimate) {
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
     * 
     * @param addNoise Whether to add Dirichlet noise to the priors.
     * It is the caller's responsibility to set this to true when expanding
     * the current decision node during training.
    */
    void expand(bool addNoise) {
        assert(!m_isTerminal);
        assert(!m_isExpanded);
        assert(m_isNetworkEvaluated);

        m_isExpanded = true;

        int numLegal = 0;
        for (ActionIdx action = 0; action < AS; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip
                continue;
            }

            m_edgeStatistics.m_childPriors[action] = m_networkPolicy[action];
            ++numLegal;
        }

        if (addNoise) {
            std::vector<float> noise (numLegal);
            GetRandom().Dirichlet(m_dirAlpha, noise);

            int readIdx = 0;
            for (ActionIdx action = 0; action < AS; ++action) {
                if (m_actionMask[action] == 0.0f) {
                    // Illegal action, skip
                    continue;
                }

                m_edgeStatistics.m_childPriors[action]
                    = (1.0 - m_dirEps) * m_edgeStatistics.m_childPriors[action]
                            + m_dirEps * noise[readIdx];
                                                        
                ++readIdx;
            }
        }
    }

    /**
     * Prunes away all children of the node except for
     * the one corresponding to the given action.
     * 
     * Can be used on any non-terminal node. Used in rerooting.
    */
    void pruneChildrenExcept(ActionIdx action) {
        assert(!m_isTerminal);

        for (ActionIdx i = 0; i < AS; ++i) {
            if (i != action) {
                m_children[i] = nullptr;
            }
        }

        m_gameNode->pruneChildrenExcept(action);
    }

private:
    UCTNode* m_parent { nullptr };  // Raw pointer to the parent, nullptr if root.
    std::array<std::unique_ptr<UCTNode>, AS> m_children {};  // Parent owns children.

    ActionIdx m_action { 0 };        // Action index taken into this node, 0 if root.
    GNode* m_gameNode;                // Pointer to current game node.
    bool m_isTerminal;               // Whether the current node is terminal.
    const ActionDist& m_actionMask;  // Mask of legal actions.

    bool m_isExpanded { false };          // Whether node has been expanded.
    bool m_isNetworkEvaluated { false };  // Whether node has been evaluated by the network.

    ActionDist m_networkPolicy {};  // Cached network policy output.
    float m_networkValue {};        // Cached network value output.

    EdgeStatistics m_edgeStatistics {};         // Edge stats out of this node.
    EdgeStatistics* m_parentEdgeStatistics {};  // Pointer to edge stats out of parent.

    float m_dirEps {};                      // Dirichlet noise epsilon.
    float m_dirAlpha {};                    // Dirichlet noise alpha.
    InitQ m_initQMethod { InitQ::PARENT };  // Method to use for initializing Q values.

    friend class UCTTree<State, AS>;
};

} // namespace SPRL

#endif
