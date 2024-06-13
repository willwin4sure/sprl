#ifndef SPRL_GRID_WORKER_HPP
#define SPRL_GRID_WORKER_HPP

#include "../games/GridState.hpp"

#include "../interface/npy.hpp"

#include "../networks/Network.hpp"
#include "../networks/GridNetwork.hpp"
#include "../networks/RandomNetwork.hpp"

#include "../selfplay/SelfPlay.hpp"

#include "../constants.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace SPRL {

/**
 * Blocks the current thread until the model file for the given iteration exists,
 * and then returns the path to the model file.
 * 
 * @param iteration The iteration to get the model file for. If -1, returns "random" immediately.
 * @param runName The name of the run, defining the model file path.
 * 
 * @returns The path to the model file for the given iteration.
 */
std::string waitModelPath(int iteration, const std::string& runName) {
    if (iteration == -1) {
        return "random";
    }

    std::string modelPath;

    do {
        modelPath = "data/models/" + runName + "/traced_" + runName + "_iteration_" + std::to_string(iteration) + ".pt";

        if (!std::filesystem::exists(modelPath)) {
            std::cout << "Spinning on traced model from iteration " << iteration << "..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(30));
        }
        
    } while (!std::filesystem::exists(modelPath));

    std::this_thread::sleep_for(std::chrono::seconds(2));

    return modelPath;
}

template <typename NeuralNetwork, typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
void runWorker(std::string runName, std::string saveDir,
               Network<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* initialNetwork,
               ISymmetrizer<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* symmetrizer) {

    using State = GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    // Make the save directory if it doesn't exist.
    try {
        bool result = std::filesystem::create_directories(saveDir);
        if (result) {
            std::cout << "Created directory: " << saveDir << std::endl;
        } else {
            std::cout << "Directory already exists: " << saveDir << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return;
    }

    Network<State, ACTION_SIZE>* network;  // Holds the current network.

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        std::cout << "Starting iteration " << iter << "..." << std::endl;

        std::string modelPath = waitModelPath(iter - 1, runName);
        std::string savePath = saveDir + "/" + runName + "_iteration_" + std::to_string(iter);

        int numGames = (iter == 0) ? INIT_NUM_GAMES_PER_WORKER : NUM_GAMES_PER_WORKER;
        int numIters = (iter == 0) ? INIT_UCT_ITERATIONS : UCT_ITERATIONS;
        int maxTraversals = (iter == 0) ? INIT_MAX_TRAVERSALS : MAX_TRAVERSALS;
        int maxQueueSize = (iter == 0) ? INIT_MAX_QUEUE_SIZE : MAX_QUEUE_SIZE;

        NeuralNetwork neuralNetwork { modelPath };

        if (modelPath == "random") {
            std::cout << "Using initial network..." << std::endl;
            network = initialNetwork;
        } else {
            std::cout << "Using traced PyTorch network..." << std::endl;
            network = &neuralNetwork;
        }

        auto [states, distributions, outcomes] = runIteration<ImplNode, State, ACTION_SIZE>(
            network,
            numGames,
            numIters,
            maxTraversals,
            maxQueueSize,
            DIRICHLET_EPSILON,
            DIRICHLET_ALPHA,
            InitQ::PARENT,
            symmetrizer,
            true
        );

        std::vector<float> embeddedStates;

        for (const State& state : states) {
            Piece ourPiece = pieceFromPlayer(state.getPlayer());

            // Stone bitmasks.
            for (int t = 0; t < state.size(); ++t) {
                for (Piece piece : {ourPiece, otherPiece(ourPiece)}) {
                    for (int row = 0; row < NUM_ROWS; ++row) {
                        for (int col = 0; col < NUM_COLS; ++col) {
                            if (state.getHistory()[t][row * NUM_COLS + col] == piece) {
                                embeddedStates.push_back(1.0f);
                            } else {
                                embeddedStates.push_back(0.0f);
                            }
                        }
                    }
                }
            }

            // Pad the history using zeros.
            embeddedStates.resize(embeddedStates.size() + 2 * NUM_ROWS * NUM_COLS * (HISTORY_SIZE - state.size()), 0.0f);

            // Color channel.
            embeddedStates.resize(embeddedStates.size() + NUM_ROWS * NUM_COLS, (state.getPlayer() == Player::ZERO) ? 1.0f : 0.0f);
        }

        npy::npy_data_ptr<float> stateData {};
        stateData.data_ptr = embeddedStates.data();
        stateData.shape = { static_cast<unsigned long>(states.size()), 2 * HISTORY_SIZE + 1, NUM_ROWS, NUM_COLS };

        npy::write_npy(savePath + "_states.npy", stateData);

        std::vector<float> embeddedDistributions;
        for (const ActionDist& dist : distributions) {
            for (int i = 0; i < ACTION_SIZE; ++i) {
                embeddedDistributions.push_back(dist[i]);
            }
        }

        npy::npy_data_ptr<float> distData {};
        distData.data_ptr = embeddedDistributions.data();
        distData.shape = { static_cast<unsigned long>(distributions.size()), ACTION_SIZE };

        npy::write_npy(savePath + "_distributions.npy", distData);

        npy::npy_data_ptr<float> outcomeData {};
        outcomeData.data_ptr = outcomes.data();
        outcomeData.shape = { static_cast<unsigned long>(outcomes.size()) };

        npy::write_npy(savePath + "_outcomes.npy", outcomeData);
    }
}

} // namespace SPRL

#endif