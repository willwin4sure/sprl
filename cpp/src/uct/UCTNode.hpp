#ifndef SPRL_UCT_NODE_HPP
#define SPRL_UCT_NODE_HPP

#include "../games/GameNode.hpp"
#include "../uct/UCTOptions.hpp"
#include "../utils/random.hpp"

#include "../constants.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <memory>

namespace SPRL { 

// Forward declaration of the UCT tree class.
template<typename ImplNode, typename State, int ACTION_SIZE>
class UCTTree;

/**
 * Class representing a node in the tree for the UCT algorithm.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam State The state of the game, e.g. `GridState`.
 * @tparam ACTION_SIZE The size of the action space.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class UCTNode {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    /**
     * Holds statistics for the edges coming out of this node in the UCT tree.
    */
    struct EdgeStatistics {
        
        ActionDist m_childPriors {};  // Prior from network, *after* Dirichlet noise, for `U` value.
        ActionDist m_totalValues {};  // Total `Q` value accumulated on each edge.
        ActionDist m_numVisits {};    // Number of times each edge has been traversed.

        EdgeStatistics() {
            reset();
        }

        void reset() {
            m_childPriors.fill(0.0f);
            m_totalValues.fill(0.0f);
            m_numVisits.fill(0.0f);
        }

    };

    /**
     * Constructor for root UCT node.
     * 
     * @param nodeOptions The options for the UCT node.
     * @param edgeStats Pointer to the edge statistics of the virtual "parent" node, held by `UCTTree`.
     * @param gameNode The game node corresponding to this UCT node.
    */
    UCTNode(NodeOptions nodeOptions,
            EdgeStatistics* edgeStats,
            GameNode<ImplNode, State, ACTION_SIZE>* gameNode)
        : m_gameNode { gameNode }, m_parentEdgeStatistics { edgeStats }, m_nodeOptions { nodeOptions },
          m_isTerminal { m_gameNode->isTerminal() }, m_actionMask { m_gameNode->getActionMask() } {
            
    }

    /**
     * Constructor for child UCT nodes.
     * 
     * @param parent Pointer to the parent UCT node.
     * @param action The action taken to reach this node.
     * @param gameNode The game node corresponding to this UCT node.
    */
    UCTNode(UCTNode* parent, ActionIdx action, GameNode<ImplNode, State, ACTION_SIZE>* gameNode)
        : m_parent { parent }, m_action { action }, m_gameNode { gameNode }, m_nodeOptions { parent->m_nodeOptions },
          m_isTerminal { m_gameNode->isTerminal() }, m_actionMask { m_gameNode->getActionMask() },
          m_parentEdgeStatistics { &parent->m_edgeStatistics } {

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
     * @returns A string representation of the underlying game node.
     */
    std::string getGameNodeString() const {
        return m_gameNode->toString();
    }

    /**
     * @returns A reference to the current number of visits to this node.
    */
    float& N() const { return m_parentEdgeStatistics->m_numVisits[m_action]; }

    /**
     * @returns A reference to the current total value of this node.
    */
    float& W() const { return m_parentEdgeStatistics->m_totalValues[m_action]; }

    /**
     * @returns The current average action value of this node, as described in UCT.
    */
    float Q() const {
        if (N() == 0.0f) {
            if (m_nodeOptions.initQMethod == InitQ::PARENT_LIVE_Q) {
                // Return the parent `Q` value!
                if (m_parent == nullptr) return 0.0f;
                return m_parent->Q();
            }
            return W();  // The standard behavior: `W / (N + 1) = W`.
            
        } else {
            if (m_nodeOptions.takeTrueQAvg) return W() / N();
            return W() / (N() + 1);
        }
    }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The number of visits to a particular child.
    */
    float child_N(ActionIdx action) const { return m_edgeStatistics.m_numVisits[action]; }

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
    float child_Q(ActionIdx action) const {
        if (child_N(action) == 0.0f) {
            if (m_nodeOptions.initQMethod == InitQ::PARENT_LIVE_Q) {
                // Return this node's `Q` value!
                return Q();
            }
            return child_W(action);  // The standard behavior: `W / (N + 1) = W`.

        } else {
            if (m_nodeOptions.takeTrueQAvg) {
                return child_W(action) / child_N(action);
            }
            return child_W(action) / (child_N(action) + 1);
        }
    }

    /**
     * @param action The action index of the child to query.
     * 
     * @returns The uncertainty value of a particular child, as described in the UCT algorithm.
    */
    float child_U(ActionIdx action) const {
        return child_P(action) * std::sqrt(N()) / (1 + child_N(action));  // Adding 1 avoids division by zero.
    }

    /**
     * @param action The action index of the child to query.
     *
     * @returns The number of visits to a particular child that are forced.
    */
    float child_N_forced(ActionIdx action) {
        return std::sqrt(2 * child_P(action) * (N() - 1));
    }

    /**
     * @returns The policy target for the UCT algorithm.
     */
    ActionDist getPrunedPolicyTarget() const {
        // Subtract 1 because I'm talking about the number of child playouts.
        float total_N = N() - 1;

        ActionDist all_Q {}; // All 0s.

        float v_max = 0;
        float v_min = 0;
        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip.
                continue;
            }

            all_Q[action] = child_Q(action);
            float value = child_Q(action) + m_nodeOptions.uWeight * child_U(action);

            if (value > v_max) {
                v_max = value;
            }
            if (value < v_min) {
                v_min = value;
            }
        }
        
        v_max *= 2;
        v_min *= 2;

        // binary search between v_min and v_max
        float v = (v_max + v_min) / 2;
        float epsilon = 0.0001;

        // Avoid bugs in rare cases where v_max and v_min are too close.
        if (v_max - v_min < epsilon) {
            v_max += epsilon;
            v_min -= epsilon;
        }
        epsilon = std::min(epsilon, (v_max - v_min) / 100.0f);
        // In case epsilon is too large.
        // TODO: figure out what magnitudes are appropriate.

        ActionDist inverse_N = {};
        while (v_max - v_min > epsilon) {
            float sum = 0.0f;
            for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
                if (m_actionMask[action] == 0.0f) {
                    // Illegal action, skip.
                    continue;
                }
                if(v < child_Q(action)) {
                    inverse_N[action] = 0.0f;
                } else {
                    inverse_N[action] = std::max(0.0f, m_nodeOptions.uWeight * m_networkPolicy[action] * sqrt(total_N) / (v - child_Q(action)) - 1);
                }
                sum += inverse_N[action];
            }

            if (sum > total_N) {
                // If sum is too big, we should *increase* v.
                // This causes the inverse_N to go *down*.
                v_min = v;
            } else {
                v_max = v;
            }

            v = (v_max + v_min) / 2;
        }

        return inverse_N;
    }


    /**
     * @returns The action index of the best move according to the UCT algorithm.
     * 
     * @param forced Whether to force the selection of a move that has not been explored enough.
     * 
     * @note Can only be applied on active nodes, i.e.
     * non-terminals that are evaluated and expanded.
    */
    ActionIdx bestAction(bool forced) {
        assert(!m_isTerminal);

        assert(m_isExpanded);
        assert(m_isNetworkEvaluated);

        std::vector<ActionIdx> bestActions;
        float bestValue = -std::numeric_limits<float>::infinity();

        if (forced) {
            for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
                if (m_actionMask[action] == 0.0f) {
                    // Illegal action, skip.
                    continue;
                }

                if (child_N(action) < child_N_forced(action)) {
                    bestActions.push_back(action);
                }
            }

            if (bestActions.size() > 0) {
                return bestActions[GetRandom().UniformInt(0, bestActions.size() - 1)];
            }
        }
        
        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip.
                continue;
            }

            const float value = child_Q(action) + m_nodeOptions.uWeight * child_U(action);

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
                this, action, m_gameNode->getAddChild(action));

            // Handle Q-initialization based on the method.
            switch (m_nodeOptions.initQMethod) {
            case InitQ::ZERO:
                m_edgeStatistics.m_totalValues[action] = 0.0f;
                break;
            case InitQ::PARENT_NN_EVAL:
                m_edgeStatistics.m_totalValues[action] = m_isNetworkEvaluated ? m_networkValue : 0.0f;
                break;
            case InitQ::PARENT_LIVE_Q:
                // This value is never used! Set to 0, so that
                // after the child is expanded and its network eval is computed,
                // it increments to that correct value.
                m_edgeStatistics.m_totalValues[action] = 0.0f;
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
        for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
            if (m_actionMask[action] == 0.0f) {
                // Illegal action, skip.
                continue;
            }

            m_edgeStatistics.m_childPriors[action] = m_networkPolicy[action];

            ++numLegal;
        }

        if (addNoise) {
            std::vector<float> noise(numLegal);
            GetRandom().Dirichlet(m_nodeOptions.dirAlpha, noise);

            int readIdx = 0;
            for (ActionIdx action = 0; action < ACTION_SIZE; ++action) {
                if (m_actionMask[action] == 0.0f) {
                    // Illegal action, skip.
                    continue;
                }

                m_edgeStatistics.m_childPriors[action]
                    = (1.0 - m_nodeOptions.dirEps) * m_edgeStatistics.m_childPriors[action]
                          + (m_nodeOptions.dirEps) * noise[readIdx];
                                                        
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

        for (ActionIdx i = 0; i < ACTION_SIZE; ++i) {
            if (i != action) {
                m_children[i] = nullptr;
            }
        }

        m_gameNode->pruneChildrenExcept(action);
    }

private:
    UCTNode* m_parent { nullptr };  // Raw pointer to the parent, nullptr if root.
    std::array<std::unique_ptr<UCTNode>, ACTION_SIZE> m_children {};  // Parent owns children.

    ActionIdx m_action { 0 };                            // Action index taken into this node, 0 if root.
    GameNode<ImplNode, State, ACTION_SIZE>* m_gameNode;  // Pointer to current game node.
    bool m_isTerminal;                                   // Whether the current node is terminal.
    const ActionDist& m_actionMask;                      // Mask of legal actions.

    bool m_isExpanded { false };          // Whether node has been expanded.
    bool m_isNetworkEvaluated { false };  // Whether node has been evaluated by the network.

    ActionDist m_networkPolicy {};  // Cached network policy output.
    float m_networkValue {};        // Cached network value output.

    EdgeStatistics m_edgeStatistics {};         // Edge stats out of this node.
    EdgeStatistics* m_parentEdgeStatistics {};  // Pointer to edge stats out of parent.

    NodeOptions m_nodeOptions;  // Options for the UCT node.

    friend class UCTTree<ImplNode, State, ACTION_SIZE>;
};

} // namespace SPRL

#endif
