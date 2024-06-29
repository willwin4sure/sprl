#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <filesystem>
#include <fstream>

#include "agents/IAgent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameNode.hpp"
#include "games/GoNode.hpp"

#include "networks/INetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/GridNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


// struct Player {
//     std::string modelPath;
//     bool useSymmetrize;
//     bool useParentQ;
// };

constexpr int NUM_ROWS = SPRL::GO_BOARD_WIDTH;
constexpr int NUM_COLS = SPRL::GO_BOARD_WIDTH;
constexpr int BOARD_SIZE = NUM_ROWS * NUM_COLS;

constexpr int ACTION_SIZE = SPRL::GO_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::GO_HISTORY_SIZE;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: ./RobinWorker.exe <task_id> <num_tasks> <num_players> (<modelPath>)+";
        return 1;
    }

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);
    int numPlayers = std::stoi(argv[3]);

    // Check if the right number of players.
    if (argc != 4 + numPlayers) {
        std::cerr << "Usage: ./RobinWorker.exe <task_id> <num_tasks> <num_players> (<modelPath>)+";
        return 1;
    }

    int myGroup = myTaskId / (numTasks / 4);
    std::cout << "I am task " << myTaskId << " of " << numTasks << " in group " << myGroup << std::endl;

    std::string runName = "panda_fight";
    std::string saveDir = "data/robin/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

    // Make the directory if it doesn't exist.
    try {
        bool result = std::filesystem::create_directories(saveDir);
        if (result) {
            std::cout << "Created directory: " << saveDir << std::endl;
        } else {
            std::cout << "Directory already exists: " << saveDir << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return 1;
    }

    // Setup the players.
    std::vector<std::string> modelPaths(numPlayers);

    for (int i = 0; i < numPlayers; ++i) {
        modelPaths[i] = argv[4 + i];

        std::cout << "Player " << i << " has model path: " << modelPaths[i] << std::endl;
    }

    // A win is worth 2 points; a draw is worth 1 point.
    std::vector<std::vector<int>> points(numPlayers, std::vector<int>(numPlayers, 0));

    using State = SPRL::GridState<BOARD_SIZE, HISTORY_SIZE>;
    using ImplNode = SPRL::GoNode;

    // Setup the networks.
    std::vector<std::unique_ptr<SPRL::INetwork<State, ACTION_SIZE>>> networks(numPlayers);

    for (int i = 0; i < numPlayers; ++i) {
        if (modelPaths[i] == "random") {
            std::cout << "Using random network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::RandomNetwork<State, ACTION_SIZE>>();
        } else {
            std::cout << "Using traced PyTorch network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::GridNetwork<NUM_ROWS, NUM_COLS, HISTORY_SIZE, ACTION_SIZE>>(modelPaths[i]);
        }
    }

    // Write game results to a log.
    std::string logPath = saveDir + "/log.txt";
    std::ofstream logFile(logPath);

    if (!logFile.is_open()) {
        std::cerr << "Error opening file: " << logPath << std::endl;
        return 1;
    }

    SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, HISTORY_SIZE> symmetrizer {};

    // Play two games between each pair of players.
    for (int k = 0; k < numPlayers; ++k) {
        for (int j = 0; j < numPlayers; ++j) {
            int i = (k + myTaskId) % numPlayers;
            if (i == j) continue;

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
                networks[i].get(),
                &tree0,
                128,
                16,
                8
            };

            SPRL::UCTNetworkAgent<ImplNode, State, ACTION_SIZE> networkAgent1 {
                networks[j].get(),
                &tree1,
                128,
                16,
                8
            };

            std::array<SPRL::IAgent<ImplNode, State, ACTION_SIZE>*, 2> agents { &networkAgent0, &networkAgent1 };

            ImplNode rootNode {};
            SPRL::Player winner = SPRL::playGame(&rootNode, agents, false);

            logFile << i << " " << j << " " << static_cast<int>(winner) << std::endl;

            if (winner == SPRL::Player::ZERO) {
                // Player i wins
                points[i][j] += 2;

            } else if (winner == SPRL::Player::ONE) {
                // Player j wins
                points[j][i] += 2;

            } else {
                // Draw
                points[i][j] += 1;
                points[j][i] += 1;
            }
        }
    }

    logFile.close();

    // Write the table to a file using fstream.
    std::string tableSavePath = saveDir + "/points.txt";
    std::ofstream tableFile(tableSavePath);

    if (!tableFile.is_open()) {
        std::cerr << "Error opening file: " << tableSavePath << std::endl;
        return 1;
    }

    for (int i = 0; i < numPlayers; ++i) {
        for (int j = 0; j < numPlayers; ++j) {
            tableFile << points[i][j] << " ";
        }
        tableFile << std::endl;
    }

    tableFile.close();

    return 0;
}
