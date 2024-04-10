#ifndef UCT_TREE_HPP
#define UCT_TREE_HPP

#include "../games/GameState.hpp"
#include "../games/Game.hpp"
#include "../networks/Network.hpp"

#include "UCTNode.hpp"

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
            current = current->getAddChild(bestAction);
        }

        return current;
    }

    /**
     * Propagates the value estimate of a given node back up along the path.
    */
    void backup(Node* node, float valueEstimate) {
        assert(node->m_isTerminal || node->m_isNetworkEvaluated);

        // value is negated since they are stored from the perspective of the parent
        float estimate = -valueEstimate * (std::pow(-1, node->m_state.getPlayer()));
        Node* current = node;
        while (current != nullptr) {
            ++current->N();
            current->W() += estimate * (std::pow(-1, current->m_state.getPlayer()));

            current = current->m_parent;
        }
    }

    /**
     * Performs a single iteration of search.
    */
    void searchIteration(Network<BOARD_SIZE, ACTION_SIZE>* network, float exploration = 1.0f) {
        Node* leaf = selectLeaf(exploration);

        float valueEstimate;

        if (leaf->m_isTerminal) {
            auto rewards = m_game->rewards(leaf->m_state);
            if (leaf->m_state.getPlayer() == 0) {
                valueEstimate = rewards.first;
            } else {
                valueEstimate = rewards.second;
            }
            
        } else {
            std::pair<std::array<float, ACTION_SIZE>, float> output = network->evaluate(leaf->m_state);

            std::array<float, ACTION_SIZE> policy = output.first;
            float value = output.second;

            leaf->updateNetworkOutput(policy, value);
            leaf->expand(policy);

            valueEstimate = value;
        }

        backup(leaf, valueEstimate);
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