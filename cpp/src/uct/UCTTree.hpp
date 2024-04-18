#ifndef UCT_TREE_HPP
#define UCT_TREE_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"
#include "../networks/Network.hpp"

#include "UCTNode.hpp"

#include <algorithm>
#include <queue>


namespace SPRL {

template <int BOARD_SIZE, int ACTION_SIZE>
class UCTTree {
public:
    using Node = UCTNode<BOARD_SIZE, ACTION_SIZE>;

    // Constructor for the tree.
    UCTTree(Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state, bool addNoise = true, bool symmetrize = true)
        : m_addNoise { addNoise }, m_symmetrize { symmetrize }, m_edgeStatistics {}, m_game { game }, 
          m_root { std::make_unique<Node>(&m_edgeStatistics, game, state) } {}


    /**
     * Returns a readonly pointer to the root.
    */
    const Node* getRoot() {
        return m_root.get();
    }

    /**
     * Performs many iterations of search by repeatedly selecting leaves,
     * applying virtual losses during downward traversals.
     * 
     * When leaves are terminal or gray, immediately backpropagates the result.
     * When leaves are empty, appends them to a vector for batched NN evaluation.
     * 
     * Returns this batch of empty leaves, as well as the number of leaf selections performed.
     * 
     * TODO: If we multithread, the interface between leaf traversals and NN eval is a queue,
     * and we need to adjust the inputs and outputs appropriately
    */
    std::pair<std::vector<Node*>, int> searchAndGetLeaves(int maxTraversals, int maxQueueSize, Network<BOARD_SIZE, ACTION_SIZE>* network, float exploration = 1.0f) {
        std::vector<Node*> leaves;

        int iter = 0;

        while (iter < maxTraversals) {
            ++iter;
            Node* leaf = selectLeaf(exploration);  // must be terminal, empty, or gray

            if (leaf->m_isTerminal) {
                // Terminal case: compute the exact value and backpropagate immediately
                std::pair<Value, Value> rewards = m_game->rewards(leaf->m_state);
                Value value = (leaf->m_state.getPlayer() == 0) ? rewards.first : rewards.second;

                backup(leaf, value);
                continue;

            } else if (leaf->m_isNetworkEvaluated) {
                // Gray case: expand the node to active and backpropagate the network value estimate
                leaf->expand(m_addNoise);

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
    void evaluateAndBackpropLeaves(const std::vector<Node*>& leaves, Network<BOARD_SIZE, ACTION_SIZE>* network) {
        int numLeaves = leaves.size();

        assert(numLeaves > 0);

        // Assemble a vector of states for input into the NN
        std::vector<GameState<BOARD_SIZE>> states;
        for (int i = 0; i < numLeaves; ++i) {
            states.push_back(leaves[i]->m_state);
        }

        // Generate symmetrizations for the states, if necessary
        std::vector<Symmetry> symmetries(numLeaves, 0);
        if (m_symmetrize) {
            int numSymmetries = m_game->numSymmetries();
            for (int i = 0; i < numLeaves; ++i) {
                symmetries[i] = GetRandom().UniformInt(0, numSymmetries - 1);
                states[i] = m_game->symmetrizeState(states[i], { symmetries[i] })[0];
            }
        }

        // Perform batched evaluation of the states
        std::vector<std::pair<std::array<float, ACTION_SIZE>, float>> outputs = network->evaluate(m_game, states);

        for (int i = 0; i < numLeaves; ++i) {
            Node* leaf = leaves[i];
            std::pair<std::array<float, ACTION_SIZE>, float> output = outputs[i];

            std::array<float, ACTION_SIZE> policy = output.first;
            float value = output.second;

            // Undo the symmetrization
            if (m_symmetrize) {
                policy = m_game->symmetrizeActionDist(policy, { m_game->inverseSymmetry(symmetries[i]) })[0];
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
                leaf->expand(m_addNoise);
            }
            
            // Backpropagate the network value estimate
            backup(leaf, leaf->m_networkValue);
        }
    }

    /**
     * Performs a single iteration of search.
     * 
     * Uses unbatched NN evaluations. Should be replaced by calls to
     * searchAndGetLeaves and evaluateAndBackpropLeaves for batching
     * efficiency.
    */
    void searchIteration(Network<BOARD_SIZE, ACTION_SIZE>* network, float exploration = 1.0f) {
        Node* leaf = selectLeaf(exploration);

        float valueEstimate;

        if (leaf->m_isTerminal) {
            std::pair<Value, Value> rewards = m_game->rewards(leaf->m_state);
            valueEstimate = (leaf->m_state.getPlayer() == 0) ? rewards.first : rewards.second;
            
        } else {
            if (!leaf->m_isNetworkEvaluated) {
                std::pair<std::array<float, ACTION_SIZE>, float> output = network->evaluate(m_game, {leaf->m_state})[0];

                std::array<float, ACTION_SIZE> policy = output.first;
                float value = output.second;

                leaf->addNetworkOutput(policy, value);
            }

            leaf->expand(m_addNoise);

            valueEstimate = leaf->m_networkValue;
        }

        backup(leaf, valueEstimate);
    }

    /**
     * Reroots the tree by taking an action, moving the root to one of its children.
     * 
     * The root node must be non-terminal.
     * 
     * Clears all the statistics and expanded bits in the subtree,
     * but leaves the network evaluations intact. In particular, all
     * active nodes are turned gray.
    */
    void rerootTree(ActionIdx action) {
        assert(!m_root->m_isTerminal);
        assert(m_root->m_actionMask[action] != 0.0f);

        // Destroy all children except for the one we are rerooting to
        m_root->pruneChildrenExcept(action);

        // Clear all edges statistics of the new subtree, and turn all active nodes gray
        Node* child = m_root->getAddChild(action);
        clearSubtree(child);

        // Move the child into the root position and set its parent to null
        m_root = std::move(m_root->m_children[action]);
        m_root->m_parent = nullptr;

        // Reset the virtual parent edge statistics and set it for the new root
        m_edgeStatistics.reset();
        m_root->m_parentEdgeStatistics = &m_edgeStatistics;
    }

private:
    /**
     * Deterministically select the next leaf based on the best path
     * through the current active nodes from the root.
     * 
     * Adds virtual losses while traveling down the tree, to all nodes
     * from the root to the leaf, inclusive.
     * 
     * Returns a node that is terminal, empty, or gray. Must be the first
     * such node along the path down from the root.
    */
    Node* selectLeaf(float exploration) {
        Node* current = m_root.get();

        while (current->m_isExpanded && !current->m_isTerminal) {
            ActionIdx bestAction = current->bestAction(exploration);

            // Record a virtual loss to discount retracing the same path again
            current->N()++;
            current->W()--;

            assert(current->m_isNetworkEvaluated);

            current = current->getAddChild(bestAction);
        }

        // Record a virtual loss to discount retracing the same path again
        current->N()++;
        current->W()--;

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
    void backup(Node* node, float valueEstimate) {
        assert(node->m_isTerminal || (node->m_isNetworkEvaluated && node->m_isExpanded));

        // Value is negated since they are stored from the perspective of the parent
        float estimate = -valueEstimate * ((node->m_state.getPlayer() == 0) ? 1 : -1);
        Node* current = node;
        while (current != nullptr) {
            // Extra +1 due to reverting the virtual losses
            current->W() += 1 + estimate * ((current->m_state.getPlayer() == 0) ? 1 : -1);

            current = current->m_parent;
        }
    }

    /**
     * Clears all the nodes in the subtree of the node by resetting edge statistics,
     * as well as setting them all to un-expanded (but keeping the network evaluation).
     * 
     * Turns all active nodes to gray.
    */
    void clearSubtree(Node* node) {
        if (!node->m_isExpanded) {
            return;
        }

        // Reset the edge statistics and turn off the expanded bit
        node->m_edgeStatistics.reset();
        node->m_isExpanded = false;

        // Recursively call on the children
        for (const std::unique_ptr<Node>& child : node->m_children) {
            if (child != nullptr) {
                clearSubtree(child.get());
            }
        }
    }

    bool m_addNoise { true };

    bool m_symmetrize { true };

    /// Edge statistics of a virtual "parent" of the root, for accessing N() at the root.
    Node::EdgeStatistics m_edgeStatistics {};

    /// A raw pointer to an instance of the game we are playing.
    Game<BOARD_SIZE, ACTION_SIZE>* m_game;

    /// A unique pointer to the root node of the UCT tree; we own it.
    std::unique_ptr<Node> m_root;
};

} // namespace SPRL

#endif
