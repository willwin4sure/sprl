#ifndef UCT_NETWORK_AGENT_HPP
#define UCT_NETWORK_AGENT_HPP

#include "Agent.hpp"
#include "../networks/Network.hpp"
#include "../uct/UCTTree.hpp"

#include <iostream>

namespace SPRL {

template <int BOARD_SIZE, int ACTION_SIZE>
class UCTNetworkAgent : public Agent<BOARD_SIZE, ACTION_SIZE> {
public:
    using State = GameState<BOARD_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    UCTNetworkAgent(Network<BOARD_SIZE, ACTION_SIZE>* network,
                    UCTTree<BOARD_SIZE, ACTION_SIZE>* tree,
                    int numIters, int maxTraversals, int maxQueueSize)
        : m_network(network), m_tree(tree), m_numIters(numIters),
          m_maxTraversals(maxTraversals), m_maxQueueSize(maxQueueSize) {}


    ActionIdx act(Game<BOARD_SIZE, ACTION_SIZE>* game,
                  const State& state,
                  const ActionDist& actionMask,
                  bool verbose = false) const override {
        
        int iters = 0;
        while (iters < m_numIters) {
            auto [leaves, iter] = m_tree->searchAndGetLeaves(m_maxTraversals, m_maxQueueSize, m_network);

            if (leaves.size() > 0) {
                m_tree->evaluateAndBackpropLeaves(leaves, m_network);
            }

            iters += iter;
        }

        auto priors = m_tree->getRoot()->getEdgeStatistics()->m_childPriors;
        auto values = m_tree->getRoot()->getEdgeStatistics()->m_totalValues;
        auto visits = m_tree->getRoot()->getEdgeStatistics()->m_numberVisits;

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

        // sample action with most visits
        ActionIdx action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));

        if (verbose) {
            std::cout << "Action: " << action << '\n';
            std::cout << "Action prior: " << priors[action] << '\n';
            std::cout << "Action visits: " << visits[action] << '\n';
            std::cout << "Action average value: " << values[action] / (1 + visits[action]) << '\n';
        }

        m_tree->rerootTree(action);

        return action;
    }

    void opponentAct(const ActionIdx action) override {
        m_tree->rerootTree(action);
    }

private:
    Network<BOARD_SIZE, ACTION_SIZE>* m_network;
    UCTTree<BOARD_SIZE, ACTION_SIZE>* m_tree;
    int m_numIters {};
    int m_maxTraversals {};
    int m_maxQueueSize {};
};

} // namespace SPRL

#endif
