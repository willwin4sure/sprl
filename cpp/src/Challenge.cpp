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

#include "symmetry/ConnectFourSymmetrizer.hpp"
#include "symmetry/D4GridSymmetrizer.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

constexpr int BOARD_SIZE = SPRL::C4_BOARD_SIZE;
constexpr int ACTION_SIZE = SPRL::C4_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::C4_HISTORY_SIZE;

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: ./Challenge.exe <modelPath> <player> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    using State = SPRL::GridState<BOARD_SIZE, HISTORY_SIZE>;
    using ImplNode = SPRL::ConnectFourNode;

    std::string modelPath = argv[1];
    int player = std::stoi(argv[2]);
    int numIters = std::stoi(argv[3]);
    int maxTraversals = std::stoi(argv[4]);
    int maxQueueSize = std::stoi(argv[5]);

    SPRL::Network<State, ACTION_SIZE>* network;

    SPRL::RandomNetwork<State, ACTION_SIZE> randomNetwork {};
    SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    if (modelPath == "random") {
        std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    // network = &randomNetwork;
    // SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE> symmetrizer {};
    
    SPRL::ConnectFourSymmetrizer symmetrizer {};

    SPRL::UCTTree<ImplNode, State, ACTION_SIZE> tree {
        std::make_unique<ImplNode>(),
        0.25,
        0.1,
        SPRL::InitQ::PARENT,
        &symmetrizer,
        true
    };

    SPRL::UCTNetworkAgent<ImplNode, State, ACTION_SIZE> networkAgent {
        network,
        &tree,
        numIters,
        maxTraversals,
        maxQueueSize
    };

    SPRL::HumanAgent<ImplNode, State, ACTION_SIZE> humanAgent {};
    // SPRL::HumanGoAgent humanAgent {};

    std::array<SPRL::Agent<ImplNode, State, ACTION_SIZE>*, 2> agents;

    if (player == 0) {
        agents = { &humanAgent, &networkAgent };
    } else {
        agents = { &networkAgent, &humanAgent };
    }

    ImplNode rootNode {};

    SPRL::playGame(&rootNode, agents, true);

    return 0;
}