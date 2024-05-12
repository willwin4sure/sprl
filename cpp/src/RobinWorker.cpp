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
#include "networks/ConnectFourNetwork.hpp"

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

    std::string runName = "gorilla_ablation";
    std::string saveDir = "data/robin/" + runName + "/" + std::to_string(myGroup);

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

    auto game = std::make_unique<SPRL::ConnectFour>();

    // Setup the networks.
    std::vector<std::unique_ptr<SPRL::Network<42, 7>>> networks(numPlayers);

    for (int i = 0; i < numPlayers; ++i) {
        if (players[i].modelPath == "random") {
            std::cout << "Using random network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::RandomNetwork<42, 7>>();
        } else {
            std::cout << "Using traced PyTorch network for player " << i << "..." << std::endl;
            networks[i] = std::make_unique<SPRL::ConnectFourNetwork>(players[i].modelPath);
        }
    }

    // Play two games between each pair of players.
    for (int i = 0; i < numPlayers; ++i) {
        for (int j = 0; j < numPlayers; ++j) {
            if (i == j) continue;

            SPRL::GameState<42> state0 = game->startState();
            SPRL::GameState<42> state1 = game->startState();

            SPRL::UCTTree<42, 7> tree0 { game.get(), state0, false, players[i].useSymmetrize, players[i].useParentQ };
            SPRL::UCTTree<42, 7> tree1 { game.get(), state1, false, players[j].useSymmetrize, players[j].useParentQ };

            SPRL::UCTNetworkAgent<42, 7> networkAgent0 { networks[i].get(), &tree0, 1000, 8, 4 };
            SPRL::UCTNetworkAgent<42, 7> networkAgent1 { networks[j].get(), &tree1, 1000, 8, 4 };

            std::array<SPRL::Agent<42, 7>*, 2> agents { &networkAgent0, &networkAgent1 };

            int winner = SPRL::playGame(game.get(), game->startState(), agents, false);

            std::cout << "Game between players " << i << " and " << j << " ended with winner " << winner << std::endl;

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

    // Write the results to a file using fstream.
    std::string savePath = saveDir + "/points.txt";
    std::ofstream file(savePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << savePath << std::endl;
        return 1;
    }

    for (int i = 0; i < numPlayers; ++i) {
        for (int j = 0; j < numPlayers; ++j) {
            file << points[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();

    return 0;
}
