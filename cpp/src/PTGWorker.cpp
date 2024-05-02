/**
 * PTGWorker.cpp
*/

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/Pentago.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/PentagoNetwork.hpp"

#include "selfplay/SelfPlay.hpp"

#include "constants.hpp"

#include <iostream>
#include <string>
#include <thread>
#include <filesystem>

std::string waitModelPath(int iteration, const std::string& runName) {
    // Gets the model path when the model is actually ready
    if (iteration == -1) {
        return "random";
    }

    std::string modelPath;

    do {
        modelPath = "data/models/" + runName + "/traced_" + runName + "_iteration_" + std::to_string(iteration) + ".pt";

        if (!std::filesystem::exists(modelPath)) {
            std::cout << "Spinning on traced model from iteration " << iteration << "..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(10));
        }
    } while (!std::filesystem::exists(modelPath));

    std::this_thread::sleep_for(std::chrono::seconds(2));

    return modelPath;
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./PTGWorker.exe <task_id> <num_tasks>" << std::endl;
        return 1;
    }

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);

    assert(numTasks == NUM_WORKER_TASKS);

    int myGroup = myTaskId / (NUM_WORKER_TASKS / NUM_GROUPS);

    // Log who I am.
    std::cout << "Task " << myTaskId << " of " << numTasks << ", in group " << myGroup << "." << std::endl;

    std::string runName = "jaguar_mini_two";
    std::string saveDir = "data/games/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

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

    SPRL::Pentago game;
    SPRL::Network<36, 288>* network;
    SPRL::RandomNetwork<36, 288> randomNetwork {};

    for (int iteration = 0; iteration < NUM_ITERS; ++iteration) {
        std::cout << "Starting iteration " << iteration << "..." << std::endl;

        std::string modelPath = waitModelPath(iteration - 1, runName);
        std::string savePath = saveDir + "/" + runName + "_iteration_" + std::to_string(iteration);
        int numGames = (iteration == 0) ? INIT_NUM_GAMES_PER_WORKER : NUM_GAMES_PER_WORKER;
        int numIters = (iteration == 0) ? INIT_UCT_ITERATIONS : UCT_ITERATIONS;
        int maxTraversals = (iteration == 0) ? INIT_MAX_TRAVERSALS : MAX_TRAVERSALS;
        int maxQueueSize = (iteration == 0) ? INIT_MAX_QUEUE_SIZE : MAX_QUEUE_SIZE;

        SPRL::PentagoNetwork neuralNetwork { modelPath };

        if (modelPath == "random") {
            std::cout << "Using random network..." << std::endl;
            network = &randomNetwork;
        } else {
            std::cout << "Using traced PyTorch network..." << std::endl;
            network = &neuralNetwork;
        }

        auto [states, distributions, outcomes] = SPRL::runIteration(&game, network, numGames, numIters, maxTraversals, maxQueueSize);

        std::vector<float> embeddedStates;

        for (const SPRL::GameState<36>& state : states) {
            int player = state.getPlayer();

            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) {
                    if (state.getBoard()[row * 6 + col] == player) {
                        embeddedStates.push_back(1.0f);
                    } else {
                        embeddedStates.push_back(0.0f);
                    }
                }
            }

            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) {
                    if (state.getBoard()[row * 6 + col] == (1 - player)) {
                        embeddedStates.push_back(1.0f);
                    } else {
                        embeddedStates.push_back(0.0f);
                    }
                }
            }

            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) {
                    embeddedStates.push_back((state.getPlayer() == 0) ? 1.0f : 0.0f);
                }
            }
        }

        npy::npy_data_ptr<float> stateData {};
        stateData.data_ptr = embeddedStates.data();
        stateData.shape = { static_cast<unsigned long>(states.size()), 3, 6, 6 };

        npy::write_npy(savePath + "_states.npy", stateData);

        std::vector<float> embeddedDistributions;
        for (const SPRL::GameActionDist<288>& dist : distributions) {
            for (int i = 0; i < 288; ++i) {
                embeddedDistributions.push_back(dist[i]);
            }
        }

        npy::npy_data_ptr<float> distData {};
        distData.data_ptr = embeddedDistributions.data();
        distData.shape = { static_cast<unsigned long>(distributions.size()), 288 };

        npy::write_npy(savePath + "_distributions.npy", distData);

        npy::npy_data_ptr<float> outcomeData {};
        outcomeData.data_ptr = outcomes.data();
        outcomeData.shape = { static_cast<unsigned long>(outcomes.size()) };

        npy::write_npy(savePath + "_outcomes.npy", outcomeData);
    }

    return 0;
}   
