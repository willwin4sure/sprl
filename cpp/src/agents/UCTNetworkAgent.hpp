#ifndef SPRL_UCT_NETWORK_AGENT_HPP
#define SPRL_UCT_NETWORK_AGENT_HPP

#include "Agent.hpp"
#include "../networks/Network.hpp"
#include "../uct/UCTTree.hpp"

#include <iostream>

namespace SPRL {

template <typename ImplNode, typename State, int AS>
class UCTNetworkAgent : public Agent<ImplNode, State, AS> {
public:
    using ActionDist = GameActionDist<AS>;

    UCTNetworkAgent(Network<State, AS>* network,
                    UCTTree<ImplNode, State, AS>* tree,
                    int numIters, int maxTraversals, int maxQueueSize)
        : m_network(network), m_tree(tree), m_numIters(numIters),
          m_maxTraversals(maxTraversals), m_maxQueueSize(maxQueueSize) {}


    ActionIdx act(const ImplNode* gameNode,
                  bool verbose = false) const override {
        
        int iters = 0;
        while (iters < m_numIters) {
            auto [leaves, iter] = m_tree->searchAndGetLeaves(
                m_maxTraversals, m_maxQueueSize, m_network);

            if (leaves.size() > 0) {
                m_tree->evaluateAndBackpropLeaves(leaves, m_network);
            }

            iters += iter;
        }

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
            for (int i = 0; i < AS; ++i) {
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

        m_tree->advanceDecision(action);

        return action;
    }

    void opponentAct(const ActionIdx action) const override {
        m_tree->advanceDecision(action);
    }

private:
    Network<State, AS>* m_network;
    UCTTree<ImplNode, State, AS>* m_tree;
    int m_numIters {};
    int m_maxTraversals {};
    int m_maxQueueSize {};
};

} // namespace SPRL

#endif
