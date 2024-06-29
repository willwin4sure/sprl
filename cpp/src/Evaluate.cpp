#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "agents/HumanAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/ConnectFourNode.hpp"
#include "games/OthelloNode.hpp"
#include "games/GoNode.hpp"

#include "networks/INetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/GridNetwork.hpp"
#include "networks/OthelloHeuristic.hpp"

#include "symmetry/ConnectFourSymmetrizer.hpp"
#include "symmetry/D4GridSymmetrizer.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"

#include "utils/npy.hpp"
#include "utils/tqdm.hpp"

constexpr int NUM_ROWS = SPRL::GO_BOARD_WIDTH;
constexpr int NUM_COLS = SPRL::GO_BOARD_WIDTH;
constexpr int BOARD_SIZE = NUM_ROWS * NUM_COLS;

constexpr int ACTION_SIZE = SPRL::GO_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::GO_HISTORY_SIZE;

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./Evaluate.exe <modelPath0> <modelPath1> <numGames> <numTraversals> <maxBatchSize> <maxQueueSize>" << std::endl;
        // <model0UseSymmetrize> <model0UseParentQ> <model1UseSymmetrize> <model1UseParentQ>

        // Example:
        // ./Evaluate /home/gridsan/rzhong/sprl/data/models/panda_gamma_slower/traced_panda_gamma_slower_iteration_179.pt /home/gridsan/rzhong/sprl/data/models/panda_gamma_fast/traced_panda_gamma_fast_iteration_199.pt 100 1024 16 8
        return 1;
    }

    std::string modelPath0 = argv[1];
    std::string modelPath1 = argv[2];
    int numGames = std::stoi(argv[3]);
    int numTraversals = std::stoi(argv[4]);
    int maxBatchSize = std::stoi(argv[5]);
    int maxQueueSize = std::stoi(argv[6]);
    // bool model0UseSymmetrize = std::stoi(argv[7]) > 0;
    // bool model0UseParentQ = std::stoi(argv[8]) > 0;
    // bool model1UseSymmetrize = std::stoi(argv[9]) > 0;
    // bool model1UseParentQ = std::stoi(argv[10]) > 0;

    using State = SPRL::GridState<BOARD_SIZE, HISTORY_SIZE>;
    using ImplNode = SPRL::GoNode;

    SPRL::INetwork<State, ACTION_SIZE>* network0;
    SPRL::INetwork<State, ACTION_SIZE>* network1;

    SPRL::RandomNetwork<State, ACTION_SIZE> randomNetwork {};
    // SPRL::OthelloHeuristic heuristicNetwork {};

    // network0 = &randomNetwork;
    // network1 = &heuristicNetwork;

    SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, HISTORY_SIZE> symmetrizer {};
    // SPRL::ConnectFourSymmetrizer symmetrizer {};

    SPRL::GridNetwork<NUM_ROWS, NUM_COLS, HISTORY_SIZE, ACTION_SIZE> neuralNetwork0 { modelPath0 };
    SPRL::GridNetwork<NUM_ROWS, NUM_COLS, HISTORY_SIZE, ACTION_SIZE> neuralNetwork1 { modelPath1 };

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
        SPRL::UCTTree<ImplNode, State, ACTION_SIZE> tree0 {
            std::make_unique<ImplNode>(),
            0.25,
            0.1,
            SPRL::InitQ::PARENT_LIVE_Q,
            true,
            &symmetrizer,
            true
        };

        SPRL::UCTTree<ImplNode, State, ACTION_SIZE> tree1 {
            std::make_unique<ImplNode>(),
            0.25,
            0.1,
            SPRL::InitQ::PARENT_LIVE_Q,
            true,
            &symmetrizer,
            true
        };

        SPRL::UCTNetworkAgent<ImplNode, State, ACTION_SIZE> networkAgent0 {
            network0,
            &tree0,
            numTraversals,
            maxBatchSize,
            maxQueueSize
        };

        SPRL::UCTNetworkAgent<ImplNode, State, ACTION_SIZE> networkAgent1 {
            network1,
            &tree1,
            numTraversals,
            maxBatchSize,
            maxQueueSize
        };

        std::array<SPRL::IAgent<ImplNode, State, ACTION_SIZE>*, 2> agents;

        if (t % 2 == 0) {
            agents = { &networkAgent0, &networkAgent1 };
        } else {
            agents = { &networkAgent1, &networkAgent0 };
        }

        ImplNode rootNode {};
        SPRL::Player winner = SPRL::playGame(&rootNode, agents, true);

        if (winner == SPRL::Player::ZERO) {
            if (t % 2 == 0) {
                numWins0++;
            } else {
                numWins1++;
            }
        } else if (winner == SPRL::Player::ONE) {
            if (t % 2 == 0) {
                numWins1++;
            } else {
                numWins0++;
            }
        }

        pbar << "Player 0 wins: " << numWins0 << ", Player 1 wins: " << numWins1 << ", Draws: " << t + 1 - numWins0 - numWins1;

        // int x;
        // std::cin >> x;
    }

    return 0;
}
