#ifndef SPRL_UCT_NETWORK_AGENT_HPP
#define SPRL_UCT_NETWORK_AGENT_HPP

#include "Agent.hpp"
#include "../networks/Network.hpp"
#include "../uct/UCTTree.hpp"

#include <iostream>

namespace SPRL {

/**
 * Agent that uses the UCT algorithm with a neural network for action selection.
 * 
 * @tparam ImplNode The implementation of the game node, e.g. GoNode.
 * @tparam State The state of the game, e.g. GoState.
 * @tparam ACTION_SIZE The number of possible actions in the game.
*/
template <typename ImplNode, typename State, int ACTION_SIZE>
class UCTNetworkAgent : public Agent<ImplNode, State, ACTION_SIZE> {
public:
    using ActionDist = GameActionDist<ACTION_SIZE>;

    /**
     * Constructs a new UCT network agent.
     * 
     * @param network The neural network to use for action selection.
     * @param tree The UCT tree to use for action selection.
     * @param numIters The number of iterations to run the UCT algorithm.
     * @param maxTraversals The maximum number of traversals to run per batch.
     * @param maxQueueSize The maximum number of leaves to evaluate per batch.
    */
    UCTNetworkAgent(Network<State, ACTION_SIZE>* network,
                    UCTTree<ImplNode, State, ACTION_SIZE>* tree,
                    int numIters, int maxTraversals, int maxQueueSize)
        : m_network(network), m_tree(tree), m_numIters(numIters),
          m_maxTraversals(maxTraversals), m_maxQueueSize(maxQueueSize) {}


    ActionIdx act(const GameNode<ImplNode, State, ACTION_SIZE>* gameNode,
                  bool verbose = false) const override {
        
        int iters = 0;
        while (iters < m_numIters) {
            // Greedily search and collect leaves, expanding the tree iteratively.
            auto [leaves, iter] = m_tree->searchAndGetLeaves(
                m_maxTraversals, m_maxQueueSize, m_network);

            // If we have leaves, evaluate them using the NN and backpropagate.
            if (leaves.size() > 0) {
                m_tree->evaluateAndBackpropLeaves(leaves, m_network);
            }

            iters += iter;
        }

        // Get the priors, values, and visits for the root node.
        auto priors = m_tree->getDecisionNode()->getEdgeStatistics()->m_childPriors;
        auto values = m_tree->getDecisionNode()->getEdgeStatistics()->m_totalValues;
        auto visits = m_tree->getDecisionNode()->getEdgeStatistics()->m_numVisits;

        if (verbose) {
            std::cout << "Priors: ";
            for (const auto& prior : priors) {
                std::cout << prior << ' ';
            }

            std::cout << "\nValues: ";
            for (const auto& value : values) {
                std::cout << value << ' ';
            }

            std::cout << "\nVisits: ";
            for (const auto& visit : visits) {
                std::cout << visit << ' ';
            }

            std::cout << "\nAverage values: ";
            for (int i = 0; i < ACTION_SIZE; ++i) {
                std::cout << values[i] / (1 + visits[i]) << ' ';
            }

            std::cout << '\n';
        }

        // Sample action with most visits
        ActionIdx action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));

        if (verbose) {
            std::cout << "Action: " << action << '\n';
            std::cout << "Action prior: " << priors[action] << '\n';
            std::cout << "Action visits: " << visits[action] << '\n';
            std::cout << "Action average value: " << values[action] / (1 + visits[action]) << '\n';
        }

        // Advance the tree to the next decision node.
        m_tree->advanceDecision(action);

        return action;
    }

    void opponentAct(const ActionIdx action) const override {
        m_tree->advanceDecision(action);
    }

private:
    Network<State, ACTION_SIZE>* m_network;
    UCTTree<ImplNode, State, ACTION_SIZE>* m_tree;
    int m_numIters {};
    int m_maxTraversals {};
    int m_maxQueueSize {};
};

} // namespace SPRL

#endif
