#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "agents/HumanAgent.hpp"
#include "agents/HumanGoAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/GoNode.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/ConnectFourNetwork.hpp"
#include "networks/RandomNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

constexpr int BS = SPRL::GO_BOARD_SIZE;
constexpr int ACTION_SIZE = SPRL::GO_ACTION_SIZE;

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

    SPRL::Network<SPRL::GridState<BS>, ACTION_SIZE>* network;

    SPRL::RandomNetwork<SPRL::GridState<BS>, ACTION_SIZE> randomNetwork {};
    // SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    // if (modelPath == "random") {
    //     std::cout << "Using random network..." << std::endl;
    //     network = &randomNetwork;
    // } else {
    //     std::cout << "Using traced PyTorch network..." << std::endl;
    //     network = &neuralNetwork;
    // }

    network = &randomNetwork;

    SPRL::UCTTree<SPRL::GoNode, SPRL::GridState<BS>, ACTION_SIZE> tree {
        std::make_unique<SPRL::GoNode>(),
        0.25,
        0.1,
        SPRL::InitQ::PARENT,
        nullptr,
        true
    };

    SPRL::UCTNetworkAgent<SPRL::GoNode, SPRL::GridState<BS>, ACTION_SIZE> networkAgent {
        network,
        &tree,
        numIters,
        maxTraversals,
        maxQueueSize
    };

    // SPRL::HumanAgent<SPRL::GridState<BS>, ACTION_SIZE> humanAgent {};
    SPRL::HumanGoAgent humanAgent {};

    std::array<SPRL::Agent<SPRL::GoNode, SPRL::GridState<BS>, ACTION_SIZE>*, 2> agents;

    if (player == 0) {
        agents = { &humanAgent, &networkAgent };
    } else {
        agents = { &networkAgent, &humanAgent };
    }

    SPRL::GoNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}