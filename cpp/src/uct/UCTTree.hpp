#ifndef SPRL_UCT_TREE_HPP
#define SPRL_UCT_TREE_HPP

#include "../games/GameNode.hpp"
#include "../networks/Network.hpp"
#include "../symmetry/ISymmetrizer.hpp"

#include "UCTNode.hpp"

#include <algorithm>
#include <queue>


namespace SPRL {

template <typename ImplNode, typename State, int AS>
class UCTTree {
public:
    using UNode = UCTNode<ImplNode, State, AS>;

    /**
     * Constructs a UCT tree rooted at the initial state of the game.
    */
    UCTTree(std::unique_ptr<ImplNode> gameRoot,
            float dirEps, float dirAlpha, InitQ initQMethod,
            ISymmetrizer<State, AS>* symmetrizer, bool addNoise = true)

        : m_edgeStatistics {},
          m_gameRoot { std::move(gameRoot) },
          m_uctRoot { std::make_unique<UNode>(
            &m_edgeStatistics, m_gameRoot.get(), dirEps, dirAlpha, initQMethod ) },
          m_decisionNode { m_uctRoot.get() },
          m_dirEps { dirEps },
          m_dirAlpha { dirAlpha },
          m_initQMethod { initQMethod },
          m_addNoise { addNoise },
          m_symmetrizer { symmetrizer } {

    }

    /**
     * Returns a readonly pointer to the decision node.
    */
    const UNode* getDecisionNode() {
        return m_decisionNode;
    }

    /**
     * Performs many iterations of search by repeatedly selecting leaves,
     * applying virtual losses during downward traversals.
     * 
     * When leaves are terminal or gray, immediately backpropagates the result.
     * When leaves are empty, appends them to a vector for batched NN evaluation.
     * 
     * Returns this batch of empty leaves, as well as the number of leaf selections performed.
    */
    std::pair<std::vector<UNode*>, int> searchAndGetLeaves(
        int maxTraversals, int maxQueueSize, Network<State, AS>* network, float uWeight = 1.0f) {

        std::vector<UNode*> leaves;

        int iter = 0;

        while (iter < maxTraversals) {
            ++iter;
            UNode* leaf = selectLeaf(uWeight);  // Must be terminal, empty, or gray

            if (leaf->m_isTerminal) {
                // Terminal case: compute the exact value and backpropagate immediately
                std::array<Value, 2> rewards = leaf->getRewards();
                Value value = rewards[static_cast<int>(leaf->getPlayer())];

                backup(leaf, value);
                continue;

            } else if (leaf->m_isNetworkEvaluated) {
                // Gray case: expand the node to active and backpropagate the network value estimate
                leaf->expand(m_addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.

                backup(leaf, leaf->m_networkValue);
                continue;

            } else {
                // Empty case: append the node to the queue and do expansion and backup step after batched NN evaluation
                leaves.push_back(leaf);
            }

            // Once we have collected enough leaves, exit. Can also exit from hitting max traversal count.
            if (leaves.size() >= maxQueueSize) {
                break;
            }
        }

        return { leaves, iter };
    }

    /**
     * Takes in queued leaves and evaluates them with the network, then backpropagates the results.
     * 
     * Inputs must be leaves that are all empty, as in the return value from searchAndGetLeaves.
    */
    void evaluateAndBackpropLeaves(const std::vector<UNode*>& leaves, Network<State, AS>* network) {
        int numLeaves = leaves.size();

        assert(numLeaves > 0);

        // Assemble a vector of states and masks for input into the NN
        std::vector<State> states;
        std::vector<GameActionDist<AS>> masks;

        states.reserve(numLeaves);
        masks.reserve(numLeaves);

        for (int i = 0; i < numLeaves; ++i) {
            states.push_back(leaves[i]->getGameState());
            masks.push_back(leaves[i]->m_actionMask);
        }

        // Generate symmetrizations for the states, if necessary
        std::vector<SymmetryIdx> symmetries(numLeaves, 0);
        if (m_symmetrizer != nullptr) {
            int numSymmetries = m_symmetrizer->numSymmetries();
            for (int i = 0; i < numLeaves; ++i) {
                symmetries[i] = static_cast<SymmetryIdx>(GetRandom().UniformInt(0, numSymmetries - 1));
                states[i] = m_symmetrizer->symmetrizeState(states[i], { symmetries[i] })[0];
            }
        }

        // Perform batched evaluation of the states
        std::vector<std::pair<GameActionDist<AS>, Value>> outputs = network->evaluate(states, masks);

        for (int i = 0; i < numLeaves; ++i) {
            UNode* leaf = leaves[i];
            std::pair<GameActionDist<AS>, Value> output = outputs[i];

            GameActionDist policy = output.first;
            Value value = output.second;

            // Undo the symmetrization
            if (m_symmetrizer != nullptr) {
                policy = m_symmetrizer->symmetrizeActionDist(policy, { m_symmetrizer->inverseSymmetry(symmetries[i]) })[0];
            }

            // Note that the same leaf could occur multiple times in the output.
            // We cannot easily remove duplicates since we still need to remove the virtual losses,
            // but code could be written to optimize this by not passing them all into the 
            // network and instead backing up directly.

            if (!leaf->m_isNetworkEvaluated) {
                // Update the cached network values, making the leaf gray
                leaf->addNetworkOutput(policy, value);
            }

            if (!leaf->m_isExpanded) {
                // Expand the node, making the leaf active
                leaf->expand(m_addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.
            }
            
            // Backpropagate the network value estimate
            backup(leaf, leaf->m_networkValue);
        }
    }

    /**
     * Advances the decision node to the child corresponding to the given action.
     * 
     * The decision node must be non-terminal and the action must be legal.
     * 
     * Clears all the statistics and expanded bits in the subtree,
     * but leaves the network evaluations intact. In particular, all
     * active nodes are turned gray.
    */
    void advanceDecision(ActionIdx action) {
        assert(!m_decisionNode->m_isTerminal);
        assert(m_decisionNode->m_actionMask[action] != 0.0f);

        // Destroy all children except for the one we are rerooting to
        m_decisionNode->pruneChildrenExcept(action);

        // Clear all edges statistics of the new subtree, and turn all active nodes gray
        UNode* child = m_decisionNode->getAddChild(action);
        clearSubtree(child);

        // Set the new decision node
        m_decisionNode = m_decisionNode->m_children[action].get();
    }

private:
    /**
     * Deterministically select the next leaf based on the best path
     * through the current active nodes from the root.
     * 
     * Adds virtual losses while traveling down the tree, to all nodes
     * from the root to the leaf, inclusive.
     * 
     * @returns A node that is terminal, empty, or gray. Must be the first
     * such node along the path down from the root.
    */
    UNode* selectLeaf(float uWeight) {
        UNode* current = m_decisionNode;

        while (current->m_isExpanded && !current->m_isTerminal) {
            // Keep selecting down active nodes.
            ActionIdx bestAction = current->bestAction(uWeight);

            // Record a virtual loss to discount retracing the same path again
            current->N()++;
            current->W()--;

            assert(current->m_isNetworkEvaluated);

            current = current->getAddChild(bestAction);
        }

        // Record a virtual loss to discount retracing the same path again
        current->N()++;
        current->W()--;

        // Reached a terminal, gray, or empty node.
        assert(current->m_isTerminal || !current->m_isExpanded);

        return current;
    }

    /**
     * Propagates the value estimate of a given node back up along the path to the root.
     * 
     * Undoes the virtual loss penalty from the node to the root, inclusive.
     * 
     * The node at the bottom must be terminal or active.
    */
    void backup(UNode* node, float valueEstimate) {
        assert(node->m_isTerminal || (node->m_isNetworkEvaluated && node->m_isExpanded));

        // Value is negated since they are stored from the perspective of the parent
        float estimate = -valueEstimate * ((node->getPlayer() == Player::ZERO) ? 1 : -1);
        UNode* current = node;
        while (current != m_decisionNode->m_parent) {
            // Extra +1 due to reverting the virtual losses
            current->W() += 1 + estimate * ((current->getPlayer() == Player::ZERO) ? 1 : -1);

            current = current->m_parent;
        }
    }

    /**
     * Clears all the nodes in the subtree of the node by resetting edge statistics,
     * as well as setting them all to un-expanded (but keeping the network evaluation).
     * 
     * Turns all active nodes to gray.
    */
    void clearSubtree(UNode* node) {
        if (!node->m_isExpanded) {
            return;
        }

        // Reset the edge statistics and turn off the expanded bit
        node->m_edgeStatistics.reset();
        node->m_isExpanded = false;

        // Recursively call on the children
        for (const std::unique_ptr<UNode>& child : node->m_children) {
            if (child != nullptr) {
                clearSubtree(child.get());
            }
        }
    }

    /// Edge statistics of a virtual "parent" of the root, for accessing N() at the root.
    UNode::EdgeStatistics m_edgeStatistics {};

    /// A unique pointer to the root node of the game tree; we own it.
    std::unique_ptr<ImplNode> m_gameRoot;

    /// A unique pointer to the root node of the UCT tree; we own it.
    std::unique_ptr<UNode> m_uctRoot;

    /// The current node in the tree, i.e. our decision point for the next action.
    UNode* m_decisionNode;

    InitQ m_initQMethod { InitQ::PARENT };
    float m_dirAlpha {};
    float m_dirEps {};

    bool m_addNoise { true };

    ISymmetrizer<State, AS>* m_symmetrizer { nullptr };
};

} // namespace SPRL

#endif
