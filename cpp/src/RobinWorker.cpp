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

#include "agents/Agent.hpp"
#include "agents/UCTNetworkAgent.hpp"

#include "evaluate/play.hpp"

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/Othello.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/OthelloNetwork.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


struct Player {
    std::string modelPath;
    bool useSymmetrize;
    bool useParentQ;
};


int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: ./RobinWorker.exe <task_id> <num_tasks> <num_players> (<modelPath> <useSymmetrize> <useParentQ>)+";
        return 1;
    }

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);
    int numPlayers = std::stoi(argv[3]);

    // Check if the right number of players.
    if (argc != 4 + 3 * numPlayers) {
        std::cerr << "Usage: ./RobinWorker.exe <task_id> <num_tasks> <num_players> (<modelPath> <useSymmetrize> <useParentQ>)+";
        return 1;
    }

    int myGroup = myTaskId / (numTasks / 4);
    std::cout << "I am task " << myTaskId << " of " << numTasks << " in group " << myGroup << std::endl;

    std::string runName = "manatee";
    std::string saveDir = "data/robin2/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

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
    std::vector<Player> players;

    for (int i = 0; i < numPlayers; ++i) {
        Player player {};
        player.modelPath = argv[4 + 3 * i];
        player.useSymmetrize = std::stoi(argv[5 + 3 * i]) > 0;
        player.useParentQ = std::stoi(argv[6 + 3 * i]) > 0;

        std::cout << "Player " << i << " has model path: " << player.modelPath << ", useSymmetrize: " << player.useSymmetrize << ", useParentQ: " << player.useParentQ << std::endl;

        players.push_back(player);
    }

    // A win is worth 2 points; a draw is worth 1 point.
    std::vector<std::vector<int>> points(numPlayers, std::vector<int>(numPlayers, 0));

    auto game = std::make_unique<SPRL::Othello>();

    // Setup the networks.
    std::vector<std::unique_ptr<SPRL::Network<64, 65>>> networks(numPlayers);

    for (int i = 0; i < numPlayers; ++i) {
        if (players[i].modelPath == "random") {
            std::cout << "Using random network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::RandomNetwork<64, 65>>();
        } else {
            std::cout << "Using traced PyTorch network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::OthelloNetwork>(players[i].modelPath);
        }
    }

    // Write game results to a log.
    std::string logPath = saveDir + "/log.txt";
    std::ofstream logFile(logPath);

    if (!logFile.is_open()) {
        std::cerr << "Error opening file: " << logPath << std::endl;
        return 1;
    }

    // Play two games between each pair of players.
    for (int i = 0; i < numPlayers; ++i) {
        for (int j = 0; j < numPlayers; ++j) {
            if (i == j) continue;

            SPRL::GameState<64> state0 = game->startState();
            SPRL::GameState<64> state1 = game->startState();

            SPRL::UCTTree<64, 65> tree0 { game.get(), state0, false, players[i].useSymmetrize, players[i].useParentQ };
            SPRL::UCTTree<64, 65> tree1 { game.get(), state1, false, players[j].useSymmetrize, players[j].useParentQ };

            SPRL::UCTNetworkAgent<64, 65> networkAgent0 { networks[i].get(), &tree0, 100, 8, 4 };
            SPRL::UCTNetworkAgent<64, 65> networkAgent1 { networks[j].get(), &tree1, 100, 8, 4 };

            std::array<SPRL::Agent<64, 65>*, 2> agents { &networkAgent0, &networkAgent1 };

            int winner = SPRL::playGame(game.get(), game->startState(), agents, false);

            logFile << i << " " << j << " " << winner << std::endl;

            if (winner == 0) {
                // Player i wins
                points[i][j] += 2;

            } else if (winner == 1) {
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
