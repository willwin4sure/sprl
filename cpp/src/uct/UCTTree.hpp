#ifndef SPRL_UCT_TREE_HPP
#define SPRL_UCT_TREE_HPP

#include "../games/GameNode.hpp"
#include "../networks/INetwork.hpp"
#include "../symmetry/ISymmetrizer.hpp"
#include "../uct/UCTOptions.hpp"

#include "UCTNode.hpp"

#include <algorithm>
#include <queue>


namespace SPRL {

/**
 * Class representing a UCT tree for a game.
 * 
 * @tparam ImplNode The implementation of the game node.
 * @tparam State The state of the game.
 * @tparam ACTION_SIZE The number of actions in the game.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class UCTTree {
public:
    using UNode = UCTNode<ImplNode, State, ACTION_SIZE>;

    /**
     * Constructs a UCT tree rooted at the initial state of the game.
     * 
     * @param treeOptions The options for the UCT tree.
     * @param symmetrizer The symmetrizer for the game state.
    */
    UCTTree(TreeOptions treeOptions,
            ISymmetrizer<State, ACTION_SIZE>* symmetrizer = nullptr)
        : m_edgeStatistics {}, m_treeOptions { treeOptions }, m_symmetrizer { symmetrizer } {
        
        // Create the root of the game tree.
        m_gameRoot = std::make_unique<ImplNode>();
        
        // Create the root of the UCT tree.
        NodeOptions nodeOptions = treeOptions.nodeOptions;
        m_uctRoot = std::make_unique<UNode>(
            nodeOptions, &m_edgeStatistics, m_gameRoot.get());

        // Set the decision node to the root.
        m_decisionNode = m_uctRoot.get();
    }

    /**
     * @returns A readonly pointer to the decision node.
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
     * @param maxBatchSize The maximum number of traversals to perform.
     * @param maxQueueSize The maximum number of leaves to evaluate in a batch.
     * @param forced Whether to force the selection of a move that has not been explored enough.
     * @param network The network to evaluate the leaves with.
     * 
     * @returns This batch of empty leaves, as well as the number of leaf selections performed.
    */
    std::pair<std::vector<UNode*>, int> searchAndGetLeaves(
        int maxBatchSize, int maxQueueSize, bool forced, INetwork<State, ACTION_SIZE>* network) {

        // Collected leaves for NN evaluation.
        std::vector<UNode*> leaves;

        int traversals = 0;
        while (traversals < maxBatchSize) {
            ++traversals;
            
            // Selected leaf must be terminal, empty, or gray.
            UNode* leaf = selectLeaf(forced);

            if (leaf->m_isTerminal) {
                // Terminal case: compute the exact value and backpropagate immediately.
                std::array<Value, 2> rewards = leaf->getRewards();
                Value value = rewards[static_cast<int>(leaf->getPlayer())];

                backup(leaf, value);
                continue;

            } else if (leaf->m_isNetworkEvaluated) {
                // Gray case: expand the node to active and backpropagate the network value estimate.
                leaf->expand(m_treeOptions.addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.

                backup(leaf, leaf->m_networkValue);
                continue;

            } else {
                // Empty case: append the node to the queue and do expansion and backup step after batched NN evaluation.
                leaves.push_back(leaf);
            }

            // Once we have collected enough leaves, exit. Can also exit from hitting max batch size.
            if (leaves.size() >= maxQueueSize) break;
        }

        return { leaves, traversals };
    }

    /**
     * Takes in queued leaves and evaluates them with the network, then backpropagates the results.
     * 
     * Requires that leaves are all empty, as in the return value from searchAndGetLeaves.
     * 
     * @param leaves The leaves to evaluate and backpropagate.
     * @param network The network to evaluate the leaves with.
    */
    void evaluateAndBackpropLeaves(const std::vector<UNode*>& leaves, INetwork<State, ACTION_SIZE>* network) {
        int numLeaves = leaves.size();
        assert(numLeaves > 0);

        // Assemble a vector of states and masks for input into the NN.
        std::vector<State> states;
        std::vector<GameActionDist<ACTION_SIZE>> masks;

        states.reserve(numLeaves);
        masks.reserve(numLeaves);

        for (int i = 0; i < numLeaves; ++i) {
            states.push_back(leaves[i]->getGameState());
            masks.push_back(leaves[i]->m_actionMask);
        }

        // Generate symmetrizations for the states, if necessary.
        std::vector<SymmetryIdx> symmetries(numLeaves, 0);
        if (m_treeOptions.symmetrizeState && m_symmetrizer != nullptr) {
            int numSymmetries = m_symmetrizer->numSymmetries();
            for (int i = 0; i < numLeaves; ++i) {
                symmetries[i] = static_cast<SymmetryIdx>(GetRandom().UniformInt(0, numSymmetries - 1));
                states[i] = m_symmetrizer->symmetrizeState(states[i], { symmetries[i] })[0];
            }
        }

        // Perform batched evaluation of the states.
        std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> outputs = network->evaluate(states, masks);

        for (int i = 0; i < numLeaves; ++i) {
            UNode* leaf = leaves[i];
            std::pair<GameActionDist<ACTION_SIZE>, Value> output = outputs[i];

            GameActionDist policy = output.first;
            Value value = output.second;

            // Undo the symmetrization.
            if (m_treeOptions.symmetrizeState && m_symmetrizer != nullptr) {
                policy = m_symmetrizer->symmetrizeActionDist(policy, { m_symmetrizer->inverseSymmetry(symmetries[i]) })[0];
            }

            // Note that the same leaf could occur multiple times in the output.
            // We cannot easily remove duplicates since we still need to remove the virtual losses,
            // but code could be written to optimize this by not passing them all into the 
            // network and instead backing up directly.

            if (!leaf->m_isNetworkEvaluated) {
                // Update the cached network values, making the leaf gray.
                leaf->addNetworkOutput(policy, value);
            }

            if (!leaf->m_isExpanded) {
                // Expand the node, making the leaf active.
                leaf->expand(m_treeOptions.addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.
            }
            
            // Backpropagate the network value estimate.
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
     * 
     * @param action The action to advance the decision node using.
    */
    void advanceDecision(ActionIdx action) {
        assert(!m_decisionNode->m_isTerminal);
        assert(m_decisionNode->m_actionMask[action] > 0.0f);

        // Destroy all children except for the one we are rerooting to.
        m_decisionNode->pruneChildrenExcept(action);

        // Clear all edges statistics of the new subtree, and turn all active nodes gray.
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
     * @param forced Whether to force the selection of a move that has not been explored enough.
     * 
     * @returns A pointer to a node that is terminal, empty, or gray. Must be the first
     * such node along the path down from the root.
    */
    UNode* selectLeaf(bool forced) {
        UNode* current = m_decisionNode;

        while (current->m_isExpanded && !current->m_isTerminal) {
            // Keep selecting down active nodes.
            ActionIdx bestAction = current->bestAction(forced);

            // Record a virtual loss to discount retracing the same path again.
            current->N()++;
            current->W()--;

            assert(current->m_isNetworkEvaluated);

            current = current->getAddChild(bestAction);
        }

        // Record a virtual loss to discount retracing the same path again.
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
     * 
     * @param node The node to backpropagate from.
     * @param valueEstimate The value estimate to backpropagate.
    */
    void backup(UNode* node, float valueEstimate) {
        assert(node->m_isTerminal || (node->m_isNetworkEvaluated && node->m_isExpanded));

        // Value is negated since they are stored from the perspective of the parent.
        float estimate = -valueEstimate * ((node->getPlayer() == Player::ZERO) ? 1 : -1);
        UNode* current = node;
        while (current != m_decisionNode->m_parent) {
            // Extra +1 due to reverting the virtual losses.
            current->W() += 1 + estimate * ((current->getPlayer() == Player::ZERO) ? 1 : -1);

            current = current->m_parent;
        }
    }

    /**
     * Clears all the nodes in the subtree of the node by resetting edge statistics,
     * as well as setting them all to un-expanded (but keeping the network evaluation).
     * 
     * Turns all active nodes to gray.
     * 
     * @param node The node to clear the subtree of.
    */
    void clearSubtree(UNode* node) {
        if (!node->m_isExpanded) return;

        // Reset the edge statistics and turn off the expanded bit.
        node->m_edgeStatistics.reset();
        node->m_isExpanded = false;

        // Recursively call on the children.
        for (const std::unique_ptr<UNode>& child : node->m_children) {
            if (child != nullptr) {
                clearSubtree(child.get());
            }
        }
    }

    /// Edge statistics of a virtual "parent" of the root, for accessing N() at the root.
    UNode::EdgeStatistics m_edgeStatistics {};

    /// A unique pointer to the root node of the game tree; we own it.
    std::unique_ptr<GameNode<ImplNode, State, ACTION_SIZE>> m_gameRoot;

    /// A unique pointer to the root node of the UCT tree; we own it.
    std::unique_ptr<UNode> m_uctRoot;

    /// The current node in the tree, i.e. our decision point for the next action.
    UNode* m_decisionNode;

    /// The options for the UCT tree.
    TreeOptions m_treeOptions;

    ISymmetrizer<State, ACTION_SIZE>* m_symmetrizer { nullptr };
};

} // namespace SPRL

#endif
