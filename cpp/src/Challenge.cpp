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
#include "networks/ConnectFourNetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/PentagoHeuristic.hpp"
#include "networks/PentagoNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

constexpr int BOARD_SIZE = 36;
constexpr int ACTION_SIZE = 288;

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: ./Challenge.exe <modelPath> <player> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int player = std::stoi(argv[2]);
    int numIters = std::stoi(argv[3]);
    int maxTraversals = std::stoi(argv[4]);
    int maxQueueSize = std::stoi(argv[5]);

    auto game = std::make_unique<SPRL::Pentago>();

    SPRL::Network<BOARD_SIZE, ACTION_SIZE>* network;

    SPRL::RandomNetwork<BOARD_SIZE, ACTION_SIZE> randomNetwork {};
    SPRL::PentagoNetwork neuralNetwork { modelPath };

    if (modelPath == "random") {
        std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    SPRL::GameState<BOARD_SIZE> state = game->startState();

    SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game.get(), state, false }; 
    SPRL::UCTNetworkAgent<BOARD_SIZE, ACTION_SIZE> networkAgent { network, &tree, numIters, maxTraversals, maxQueueSize };

    SPRL::HumanPentagoAgent humanAgent {};

    std::array<SPRL::Agent<BOARD_SIZE, ACTION_SIZE>*, 2> agents;

    if (player == 0) {
        agents = { &humanAgent, &networkAgent };
    } else {
        agents = { &networkAgent, &humanAgent };
    }

    SPRL::playGame(game.get(), state, agents, true);

    return 0;
}
