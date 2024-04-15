#ifndef UCT_TREE_HPP
#define UCT_TREE_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"
#include "../networks/Network.hpp"

#include "UCTNode.hpp"

#include <algorithm>
#include <queue>


namespace SPRL {

template<int BOARD_SIZE, int ACTION_SIZE>
class UCTTree {
public:
    using Node = UCTNode<BOARD_SIZE, ACTION_SIZE>;

    // Constructor for the tree.
    UCTTree(Game<BOARD_SIZE, ACTION_SIZE>* game, const GameState<BOARD_SIZE>& state)
        : m_edgeStatistics {},
          m_game { game },
          m_root { std::make_unique<Node>(&m_edgeStatistics, game, state) } {}

    const Node* getRoot() {
        return m_root.get();
    }

    /**
     * Deterministically select the next leaf based on the best path.
    */
    Node* selectLeaf(float exploration) {
        Node* current = m_root.get();

        while (current->m_isExpanded && !current->m_isTerminal) {
            ActionIdx bestAction = current->bestAction(exploration);

            current->N()++;
            current->W()--;

            current = current->getAddChild(bestAction);
        }

        current->N()++;
        current->W()--;

        return current;
    }

    /**
     * Propagates the value estimate of a given node back up along the path.
    */
    void backup(Node* node, float valueEstimate) {
        assert(node->m_isTerminal || node->m_isNetworkEvaluated);

        // value is negated since they are stored from the perspective of the parent
        float estimate = -valueEstimate * ((node->m_state.getPlayer() == 0) ? 1 : -1);
        Node* current = node;
        while (current != nullptr) {
            current->W() += 1 + estimate * ((current->m_state.getPlayer() == 0) ? 1 : -1);

            current = current->m_parent;
        }
    }

    /**
     * Performs lots of iterations of search, and returns a vector of leaves.
     * 
     * TODO: if we multithread, the interface between leave traversals and NN eval is a queue,
     * and we need to adjust the inputs and outputs appropriately
    */
    std::vector<Node*> searchAndGetLeaves(int max_traversals, int max_queue_size, Network<BOARD_SIZE, ACTION_SIZE>* network, float exploration = 1.0f) {
        std::vector<Node*> leaves;

        for (int i = 0; i < max_traversals; ++i) {
            Node* leaf = selectLeaf(exploration);

            if (leaf->m_isTerminal) {
                int valueEstimate;
                auto rewards = m_game->rewards(leaf->m_state);
                if (leaf->m_state.getPlayer() == 0) {
                    valueEstimate = rewards.first;
                } else {
                    valueEstimate = rewards.second;
                }
                backup(leaf, valueEstimate);
                continue;

            } else if (leaf->m_isNetworkEvaluated) {
                leaf->expand(leaf->m_networkPolicy);
                backup(leaf, leaf->m_networkValue);
                continue;

            } else {
                leaves.push_back(leaf);
            }

            if (leaves.size() >= max_queue_size) {
                break;
            }
        }

        return leaves;
    }

    void evaluateAndBackpropLeaves(const std::vector<Node*>& leaves, Network<BOARD_SIZE, ACTION_SIZE>* network) {
        std::vector<GameState<BOARD_SIZE>> states;
        for (Node* leaf : leaves) {
            states.push_back(leaf->m_state);
        }

        std::vector<std::pair<std::array<float, ACTION_SIZE>, float>> outputs = network->evaluate(m_game, states);

        for (int i = 0; i < leaves.size(); ++i) {
            Node* leaf = leaves[i];
            std::pair<std::array<float, ACTION_SIZE>, float> output = outputs[i];

            if (!leaf->m_isNetworkEvaluated) {
                leaf->updateNetworkOutput(output.first, output.second);
            }

            if (!leaf->m_isExpanded) {
                leaf->expand(output.first);
            }
            
            backup(leaf, output.second);
        }
    }

    /**
     * Performs a single iteration of search.
    */
    // void searchIteration(Network<BOARD_SIZE, ACTION_SIZE>* network, float exploration = 1.0f) {
    //     Node* leaf = selectLeaf(exploration);

    //     float valueEstimate;

    //     if (leaf->m_isTerminal) {
    //         auto rewards = m_game->rewards(leaf->m_state);
    //         if (leaf->m_state.getPlayer() == 0) {
    //             valueEstimate = rewards.first;
    //         } else {
    //             valueEstimate = rewards.second;
    //         }
            
    //     } else {
    //         if (!leaf->m_isNetworkEvaluated) {
    //             std::pair<std::array<float, ACTION_SIZE>, float> output = network->evaluate(m_game, {leaf->m_state})[0];

    //             std::array<float, ACTION_SIZE> policy = output.first;
    //             float value = output.second;

    //             leaf->updateNetworkOutput(policy, value);
    //         }

    //         leaf->expand(leaf->m_networkPolicy);

    //         valueEstimate = leaf->m_networkValue;
    //     }

    //     backup(leaf, valueEstimate);
    // }

    /**
     * Reroots the tree by taking an action. Clears all the statistics and expanded bits,
     * but leaves the network evaluations intact.
    */
    void rerootTree(ActionIdx action) {
        m_root->m_edgeStatistics.reset();
        m_root->pruneChildrenExcept(action);

        std::cout << "rerooting tree" << std::endl;

        Node* child = m_root->getAddChild(action);
        child->clearSubtree();

        m_root = std::move(m_root->m_children[action]);
        m_root->m_parent = nullptr;
    }

private:
    /// Edge statistics of a virtual "parent" of the root, for accessing N() at the root.
    Node::EdgeStatistics m_edgeStatistics {};

    /// A raw pointer to an instance of the game we are playing.
    Game<BOARD_SIZE, ACTION_SIZE>* m_game;

    /// A unique pointer to the root node of the UCT tree; we own it.
    std::unique_ptr<Node> m_root;
};

} // namespace SPRL

#endif