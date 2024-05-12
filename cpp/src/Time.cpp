#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/ConnectFourNetwork.hpp"

#include "tqdm/tqdm.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./Time.exe <modelPath> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int numIters = std::stoi(argv[2]);
    int maxTraversals = std::stoi(argv[3]);
    int maxQueueSize = std::stoi(argv[4]);

    auto game = std::make_unique<SPRL::ConnectFour>();

    SPRL::ConnectFourNetwork network { modelPath };

    float totalTime = 0.0f;
    Timer t {};

    SPRL::GameState<42> state = game->startState();
    SPRL::UCTTree<42, 7> tree { game.get(), state, false }; 

    while (!state.isTerminal()) {
        SPRL::GameActionDist<7> actionMask = game->actionMask(state);

        t.reset();

        int iters = 0;
        while (iters < numIters) {
            auto [leaves, iter] = tree.searchAndGetLeaves(maxTraversals, maxQueueSize, &network);

            if (leaves.size() > 0) {
                tree.evaluateAndBackpropLeaves(leaves, &network);
            }

            iters += iter;
        }

        auto priors = tree.getRoot()->getEdgeStatistics()->m_childPriors;
        auto values = tree.getRoot()->getEdgeStatistics()->m_totalValues;
        auto visits = tree.getRoot()->getEdgeStatistics()->m_numberVisits;

        SPRL::ActionIdx action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));

        state = game->nextState(state, action);

        // tree.rerootTree(action);
        tree = SPRL::UCTTree<42, 7> { game.get(), state, false };

        totalTime += t.elapsed();

        std::cout << "Action: " << action << '\n';
        std::cout << "State:\n" << game->stateToString(state) << '\n';
    }

    std::cout << "Total time: " << totalTime << '\n';

    return 0;
}
