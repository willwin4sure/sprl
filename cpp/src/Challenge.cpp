#include "agents/HumanAgent.hpp"
#include "agents/HumanGridAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "selfplay/SelfPlay.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/OthelloNode.hpp"
#include "games/GoNode.hpp"

#include "networks/INetwork.hpp"
#include "networks/GridNetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/OthelloHeuristic.hpp"

#include "symmetry/ConnectFourSymmetrizer.hpp"
#include "symmetry/D4GridSymmetrizer.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

constexpr int NUM_ROWS = SPRL::OTH_BOARD_WIDTH;
constexpr int NUM_COLS = SPRL::OTH_BOARD_WIDTH;
constexpr int BOARD_SIZE = NUM_ROWS * NUM_COLS;

constexpr int ACTION_SIZE = SPRL::OTH_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::OTH_HISTORY_SIZE;

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: ./Challenge.exe <modelPath> <player> <numTraversals> <maxBatchSize> <maxQueueSize>" << std::endl;
        return 1;
    }

    using State = SPRL::GridState<BOARD_SIZE, HISTORY_SIZE>;
    using ImplNode = SPRL::OthelloNode;

    std::string modelPath = argv[1];
    int player = std::stoi(argv[2]);
    int numTraversals = std::stoi(argv[3]);
    int maxBatchSize = std::stoi(argv[4]);
    int maxQueueSize = std::stoi(argv[5]);

    SPRL::INetwork<State, ACTION_SIZE>* network;

    SPRL::OthelloHeuristic randomNetwork {};
    SPRL::GridNetwork<NUM_ROWS, NUM_COLS, HISTORY_SIZE, ACTION_SIZE> neuralNetwork { modelPath };

    if (modelPath == "random") {
        std::cout << "Using random network..." << std::endl;
        network = &randomNetwork;
    } else {
        std::cout << "Using traced PyTorch network..." << std::endl;
        network = &neuralNetwork;
    }

    // SPRL::ConnectFourSymmetrizer symmetrizer {};
    SPRL::D4GridSymmetrizer<SPRL::OTH_BOARD_WIDTH, HISTORY_SIZE> symmetrizer {};
    
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
        numTraversals,
        maxBatchSize,
        maxQueueSize
    };

    // SPRL::HumanAgent<ImplNode, State, ACTION_SIZE> humanAgent {};
    SPRL::HumanGridAgent<ImplNode, NUM_ROWS, NUM_COLS, HISTORY_SIZE> humanAgent {};

    std::array<SPRL::IAgent<ImplNode, State, ACTION_SIZE>*, 2> agents;

    if (player == 0) {
        agents = { &humanAgent, &networkAgent };
    } else {
        agents = { &networkAgent, &humanAgent };
    }

    ImplNode rootNode {};
    SPRL::playGame(&rootNode, agents, true);

    return 0;
}