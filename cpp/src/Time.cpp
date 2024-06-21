#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"

#include "networks/INetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/GridNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

#include "utils/npy.hpp"
#include "utils/tqdm.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./Time.exe <modelPath> <numTraversals> <maxBatchSize> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int numTraversals = std::stoi(argv[2]);
    int maxBatchSize = std::stoi(argv[3]);
    int maxQueueSize = std::stoi(argv[4]);

    SPRL::GridNetwork<SPRL::C4_NUM_ROWS, SPRL::C4_NUM_COLS, SPRL::C4_HISTORY_SIZE, SPRL::C4_ACTION_SIZE> network { modelPath };

    float totalTime = 0.0f;
    Timer t {};

    SPRL::UCTTree<SPRL::ConnectFourNode, SPRL::GridState<SPRL::C4_BOARD_SIZE, SPRL::C4_HISTORY_SIZE>, SPRL::C4_ACTION_SIZE> tree { 
        std::make_unique<SPRL::ConnectFourNode>(),
        0.25f,
        0.3f,
        SPRL::InitQ::PARENT_NN_EVAL,
        nullptr,
        false
    };

    SPRL::ConnectFourNode rootNode {};
    SPRL::ConnectFourNode* currentNode = &rootNode;

    while (!currentNode->isTerminal()) {
        t.reset();

        int traversals = 0;
        while (traversals < numTraversals) {
            auto [leaves, trav] = tree.searchAndGetLeaves(maxBatchSize, maxQueueSize, &network);

            if (leaves.size() > 0) {
                tree.evaluateAndBackpropLeaves(leaves, &network);
            }

            traversals += trav;
        }

        auto priors = tree.getDecisionNode()->getEdgeStatistics()->m_childPriors;
        auto values = tree.getDecisionNode()->getEdgeStatistics()->m_totalValues;
        auto visits = tree.getDecisionNode()->getEdgeStatistics()->m_numVisits;

        SPRL::ActionIdx action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));

        currentNode = static_cast<SPRL::ConnectFourNode*>(currentNode->getAddChild(action));

        tree.advanceDecision(action);

        totalTime += t.elapsed();

        std::cout << "Action: " << action << '\n';
        std::cout << "State:\n" << currentNode->toString() << '\n';
    }

    std::cout << "Total time: " << totalTime << '\n';

    return 0;
}
