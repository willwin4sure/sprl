#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "agents/HumanAgent.hpp"
#include "agents/HumanPentagoAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"
#include "games/Pentago.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/PentagoHeuristic.hpp"
#include "networks/PentagoNetwork.hpp"

#include "tqdm/tqdm.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


int main(int argc, char* argv[]) {
    if (argc != 11) {
        std::cerr << "Usage: ./Evaluate.exe <modelPath0> <modelPath1> <numGames> <numIters> <maxTraversals> <maxQueueSize> <model0UseSymmetrize> <model0UseParentQ> <model1UseSymmetrize> <model1UseParentQ>" << std::endl;
        return 1;
    }

    std::string modelPath0 = argv[1];
    std::string modelPath1 = argv[2];
    int numGames = std::stoi(argv[3]);
    int numIters = std::stoi(argv[4]);
    int maxTraversals = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);
    bool model0UseSymmetrize = std::stoi(argv[7]) > 0;
    bool model0UseParentQ = std::stoi(argv[8]) > 0;
    bool model1UseSymmetrize = std::stoi(argv[9]) > 0;
    bool model1UseParentQ = std::stoi(argv[10]) > 0;

    auto game = std::make_unique<SPRL::Pentago>();

    SPRL::Network<36, 288>* network0;
    SPRL::Network<36, 288>* network1;

    SPRL::RandomNetwork<36, 288> randomNetwork {};

    SPRL::PentagoNetwork neuralNetwork0 { modelPath0 };
    SPRL::PentagoNetwork neuralNetwork1 { modelPath1 };

    if (modelPath0 == "random") {
        std::cout << "Using random network..." << std::endl;
        network0 = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network0 = &neuralNetwork0;
    }

    if (modelPath1 == "random") {
        std::cout << "Using random network..." << std::endl;
        network1 = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network1 = &neuralNetwork1;
    }

    int numWins0 = 0;
    int numWins1 = 0;

    auto pbar = tq::trange(numGames);

    for (int t : pbar) {
        SPRL::GameState<36> state0 = game->startState();
        SPRL::GameState<36> state1 = game->startState();

        SPRL::UCTTree<36, 288> tree0 { game.get(), state0, false, model0UseSymmetrize, model0UseParentQ }; 
        SPRL::UCTTree<36, 288> tree1 { game.get(), state1, false, model1UseSymmetrize, model0UseParentQ };

        SPRL::UCTNetworkAgent<36, 288> networkAgent0 { network0, &tree0, numIters, maxTraversals, maxQueueSize };
        SPRL::UCTNetworkAgent<36, 288> networkAgent1 { network1, &tree1, numIters, maxTraversals, maxQueueSize };

        std::array<SPRL::Agent<36, 288>*, 2> agents;

        if (t % 2 == 0) {
            agents = { &networkAgent0, &networkAgent1 };
        } else {
            agents = { &networkAgent1, &networkAgent0 };
        }

        int winner = SPRL::playGame(game.get(), game->startState(), agents, false);

        if (winner == 0) {
            if (t % 2 == 0) {
                numWins0++;
            } else {
                numWins1++;
            }
        } else if (winner == 1) {
            if (t % 2 == 0) {
                numWins1++;
            } else {
                numWins0++;
            }
        }

        pbar << "Player 0 wins: " << numWins0 << ", Player 1 wins: " << numWins1 << ", Draws: " << t + 1 - numWins0 - numWins1;
    }

    return 0;
}
