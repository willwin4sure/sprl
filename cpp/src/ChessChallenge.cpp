#include "agents/HumanAgent.hpp"
#include "agents/HumanChessAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "selfplay/SelfPlay.hpp"

#include "games/GameNode.hpp"
#include "games/ChessNode.hpp"

#include "networks/RandomNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTOptions.hpp"
#include "uct/UCTTree.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#endif

constexpr int NUM_ROWS = SPRL::CHESS_BOARD_WIDTH;
constexpr int NUM_COLS = SPRL::CHESS_BOARD_WIDTH;
constexpr int BOARD_SIZE = NUM_ROWS * NUM_COLS;

constexpr int ACTION_SIZE = SPRL::CHESS_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::CHESS_HISTORY_SIZE;

int main(int argc, char* argv[]) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    if (argc != 7) {
        std::cerr << "Usage: ./Challenge.exe <modelPath> <optionsPath> <player> <numTraversals> <maxBatchSize> <maxQueueSize>" << std::endl;
        // example: ./Challenge /home/gridsan/rzhong/sprl/data/models/quail_gamma/traced_quail_gamma_iteration_213.pt /home/gridsan/rzhong/sprl/data/configs/quail_gamma_config_uct.json 0 2048 8 16
        return 1;
    }

    using State = SPRL::ChessState;
    using ImplNode = SPRL::ChessNode;

    std::string modelPath = argv[1];
    std::string optionsPath = argv[2];
    int player = std::stoi(argv[3]);
    int numTraversals = std::stoi(argv[4]);
    int maxBatchSize = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);

    SPRL::INetwork<State, ACTION_SIZE>* network;

    SPRL::RandomNetwork<State, ACTION_SIZE> randomNetwork {};

    std::cout << "Using random network..." << std::endl;
    network = &randomNetwork;

    SPRL::UCTOptionsParser uctParser {};
    
    SPRL::TreeOptions treeOptions;
    uctParser.parse(optionsPath, treeOptions);

    std::cout << "Using UCT options:" << std::endl;
    std::cout << uctParser.toString(treeOptions) << std::endl;

    SPRL::UCTTree<ImplNode, State, ACTION_SIZE> tree { treeOptions, nullptr };

    SPRL::UCTNetworkAgent<ImplNode, State, ACTION_SIZE> networkAgent {
        network,
        &tree,
        numTraversals,
        maxBatchSize,
        maxQueueSize
    };

    SPRL::HumanChessAgent humanAgent {};

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