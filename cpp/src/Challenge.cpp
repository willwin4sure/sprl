#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "agents/HumanAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/ConnectFourNetwork.hpp"
#include "networks/RandomNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

constexpr int BS = SPRL::C4_BS;
constexpr int AS = SPRL::C4_AS;

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


    SPRL::Network<SPRL::GridState<BS>, AS>* network;

    SPRL::RandomNetwork<SPRL::GridState<BS>, AS> randomNetwork {};
    SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    if (modelPath == "random") {
        std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    SPRL::UCTTree<SPRL::GridState<BS>, AS> tree {
        std::make_unique<SPRL::ConnectFourNode>(),
        0.25,
        0.3,
        SPRL::InitQ::PARENT,
        nullptr,
        true
    };

    SPRL::UCTNetworkAgent<SPRL::GridState<BS>, AS> networkAgent {
        network,
        &tree,
        numIters,
        maxTraversals,
        maxQueueSize
    };

    SPRL::HumanAgent<SPRL::GridState<BS>, AS> humanAgent {};

    std::array<SPRL::Agent<SPRL::GridState<BS>, AS>*, 2> agents;

    if (player == 0) {
        agents = { &humanAgent, &networkAgent };
    } else {
        agents = { &networkAgent, &humanAgent };
    }

    SPRL::ConnectFourNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}
