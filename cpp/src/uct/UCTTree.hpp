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
     * @param forced Whether to force the selection of a move that has not been explored enough.
     * 
     * @returns This batch of empty leaves, as well as the number of leaf selections performed.
    */
    UNode* searchAndGetLeaf(
        bool forced
    ) {

        // Selected leaf must be terminal, empty, or gray.
        UNode* leaf = selectLeaf(forced);

        if (leaf->m_isTerminal) {
            // Terminal case: compute the exact value and backpropagate immediately.
            std::array<Value, 2> rewards = leaf->getRewards();
            Value value = rewards[static_cast<int>(leaf->getPlayer())];

            backup(leaf, value);
            return nullptr;

        } else if (leaf->m_isNetworkEvaluated) {
            // Gray case: expand the node to active and
            // backpropagate the network value estimate.
            leaf->expand(m_treeOptions.addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.

            backup(leaf, leaf->m_networkValue);
            return nullptr;

        } else {
            // Empty case: append the node to the queue
            // and do expansion and backup step after batched NN evaluation.
            return leaf;
        }
    }

    /**
     * Take a leaf, and apply a random symmetry to it.
     * 
     * @param leaf The leaf to apply the symmetry to.
     * 
     * @returns A tuple of the rotated state, rotated mask, and the symmetry index.
     */
    std::tuple<State, GameActionDist<ACTION_SIZE>, SymmetryIdx> applyRandomSymmetry(UNode* leaf) {
        if (!m_treeOptions.symmetrizeState || m_symmetrizer == nullptr) {
            return { leaf->getGameState(), leaf->m_actionMask, 0 };
        }

        // Generate a random symmetry.
        int numSymmetries = m_symmetrizer->numSymmetries();
        SymmetryIdx symmetry = static_cast<SymmetryIdx>(GetRandom().UniformInt(0, numSymmetries - 1));

        // Apply the symmetry to the state and mask.
        State rotatedState = m_symmetrizer->symmetrizeState(leaf->getGameState(), { symmetry })[0];
        GameActionDist<ACTION_SIZE> rotatedMask = m_symmetrizer->symmetrizeActionDist(leaf->m_actionMask, { symmetry })[0];

        return { rotatedState, rotatedMask, symmetry };
    }

    /**
     * Take a neural network evaluation, and apply the inverse symmetry to it,
     * then backpropagate the result.
     * 
     * @param leaf The leaf to backpropagate from.
     * @param policy The policy output of the network.
     * @param value The value output of the network.
     * @param symmetry The symmetry index to undo.
     */
    void applyInverseSymmetryAndBackpropagate(UNode* leaf, GameActionDist<ACTION_SIZE> policy, Value value, SymmetryIdx symmetry) {
        GameActionDist<ACTION_SIZE> inversePolicy = policy;
        if (m_treeOptions.symmetrizeState && m_symmetrizer != nullptr) {
            // Apply the inverse symmetry to the policy.
            inversePolicy = m_symmetrizer->symmetrizeActionDist(policy, { m_symmetrizer->inverseSymmetry(symmetry) })[0];
        }

        // Note that the same leaf could occur multiple times in the output.
        // We cannot easily remove duplicates since we still need to remove the virtual losses,
        // but code could be written to optimize this by not passing them all into the 
        // network and instead backing up directly.

        // Update the cached network values, making the leaf gray.
        if (!leaf->m_isNetworkEvaluated) {
            leaf->addNetworkOutput(inversePolicy, value);
        }

        if (!leaf->m_isExpanded) {
            // Expand the node, making the leaf active.
            leaf->expand(m_treeOptions.addNoise && (leaf == m_decisionNode));  // Only add noise if decision node.
        }

        // Backpropagate the network value estimate.
        backup(leaf, leaf->m_networkValue);
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
     * @param clearStatistics Whether to clear the statistics of the new subtree.
    */
    void advanceDecision(ActionIdx action, bool clearStatistics = true) {
        assert(!m_decisionNode->m_isTerminal);
        assert(m_decisionNode->m_actionMask[action] > 0.0f);

        // Destroy all children except for the one we are rerooting to.
        m_decisionNode->pruneChildrenExcept(action);

        // Clear all edges statistics of the new subtree, and turn all active nodes gray.
        UNode* child = m_decisionNode->getAddChild(action);
        if (clearStatistics) clearSubtree(child);

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
