#ifndef SPRL_GRID_WORKER_HPP
#define SPRL_GRID_WORKER_HPP

#include "../games/GridState.hpp"

#include "../networks/INetwork.hpp"
#include "../networks/GridNetwork.hpp"
#include "../networks/RandomNetwork.hpp"

#include "../selfplay/SelfPlay.hpp"
#include "../selfplay/SelfPlayOptions.hpp"

#include "../uct/UCTOptions.hpp"

#include "../utils/npy.hpp"
#include "../utils/timer.hpp"
#include "../constants.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace SPRL {

constexpr int MODEL_PATH_WAIT_INTERVAL = 30;  // Seconds to wait between checking for the model file.


std::string getTracedModelPath(const std::string& runName, int iteration) {
    if (iteration == -1) {
        return "random";
    }
    return "data/models/" + runName + "/traced_" + runName + "_iteration_" + std::to_string(iteration) + ".pt";
}

std::string getStatesPath(const std::string& saveDir, const std::string& runName, int iteration) {
    return saveDir + "/" + runName + "_iteration_" + std::to_string(iteration) + "_states.npy";
}
std::string getDistsPath(const std::string& saveDir, const std::string& runName, int iteration) {
    return saveDir + "/" + runName + "_iteration_" + std::to_string(iteration) + "_distributions.npy";
}
std::string getOutcomesPath(const std::string& saveDir, const std::string& runName, int iteration) {
    return saveDir + "/" + runName + "_iteration_" + std::to_string(iteration) + "_outcomes.npy";
}

/**
 * If sync is true:
 * iteration MUST be -1 (default). Finds the most recent model file that exists, and returns the path to it.
 * 
 * If sync is false:
 * Blocks the current thread until the model file for the given iteration exists,
 * and then returns the path to the model file.
 * 
 * @param runName The name of the run, defining the model file path.
 * @param sync Whether to wait for the model file to exist.
 * @param iteration The iteration to get the model file for, if sync is false.
 * 
 * @returns The current iteration (useful if sync is true).
*/
int waitModelPath(const std::string& runName, bool sync, int iteration = -1) {
    if (sync) {
        if (iteration == -1) {
            return -1;
        }

        std::string modelPath;
        Timer t {};
        t.reset();
        modelPath = getTracedModelPath(runName, iteration);
        while (!std::filesystem::exists(modelPath)) {
            std::this_thread::sleep_for(std::chrono::seconds(MODEL_PATH_WAIT_INTERVAL));
        }
        double elapsed = t.elapsed();
        std::cout << "Found traced model in " << elapsed << " seconds." << std::endl;

        return iteration;
    } else {
        int iteration = -1;
        while (true) {
            std::string modelPath = getTracedModelPath(runName, iteration + 1);
            if (!std::filesystem::exists(modelPath)) {
                break;
            }
            iteration++;
        }

        if (iteration == -1) {
            std::cout << "No models found, using random network..." << std::endl;
            return -1;
        }
        std::cout << "Using traced model from iteration " << iteration << "..." << std::endl;
        return iteration;
    }
}

/**
 * Determine which iteration it is, when the GoWorker is initialized for the first time.
 * Look for the last iteration which has states.npy, distributions.npy, and outcomes.npy files.
 * E.g., returns 0 iff not all of 0_states, 0_distributions, and 0_outcomes exist.
 */
int determineIteration(const std::string& saveDir, std::string& runName) {
    int iteration = 0;
    while (true) {
        if (!std::filesystem::exists(getStatesPath(saveDir, runName, iteration))
         || !std::filesystem::exists(getDistsPath(saveDir, runName, iteration))
         || !std::filesystem::exists(getOutcomesPath(saveDir, runName, iteration))) {
            break;
        }

        iteration++;
    }

    return iteration;
}

/**
 * Runs the worker process for the given run name and save directory.
 * 
 * @tparam NeuralNetwork The type of the neural network, e.g. `GridNetwork`.
 *                       Must have a constructor that takes a model file path.
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam NUM_ROWS The number of rows in the grid.
 * @tparam NUM_COLS The number of columns in the grid.
 * @tparam HISTORY_SIZE The number of previous states to include in the state.
 * @tparam ACTION_SIZE The number of actions in the action space.
 * 
 * @param workerOptions The options for the worker process.
 * @param treeOptions The options for the UCT tree.
 * @param initialNetwork The network to use for the first iteration.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data.
 * @param saveDir The directory to save the self-play data to.
 */
template <typename NeuralNetwork, typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
void runWorker(SPRL::WorkerOptions workerOptions,
               SPRL::TreeOptions treeOptions,
               INetwork<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* initialNetwork,
               ISymmetrizer<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* symmetrizer,
               const std::string& saveDir) {
    
    using State = GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    // if workeroptions sync is set to false, make sure that iterationOptions.numGamesPerWorker == 1.
    if (!workerOptions.sync) {
        if (workerOptions.iterationOptions.numGamesPerWorker != 1) {
            std::cerr << "Error: iterationOptions.numGamesPerWorker must be 1 when sync is false." << std::endl;
            return;
        }
        if (workerOptions.initIterationOptions.numGamesPerWorker != 1) {
            std::cerr << "Error: initIterationOptions.numGamesPerWorker must be 1 when sync is false." << std::endl;
            return;
        }
    }

    Timer total_t {};
    total_t.reset();
    std::string runName = workerOptions.modelName + "_" + workerOptions.modelVariant;
    
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

    INetwork<State, ACTION_SIZE>* network;  // Holds the current network.

    // Check which iteration it is.
    int iter = determineIteration(saveDir, runName);
    std::cout << "I now believe it is iteration " << iter << "." << std::endl;

    while (true) {
        if (workerOptions.sync && iter >= workerOptions.numIters) break;
        Timer t {};
        t.reset();

        std::cout << "Starting iteration " << iter << "..." << std::endl;

        // Block until the model file for the previous iteration exists.
        int modelIter = waitModelPath(runName, workerOptions.sync, iter - 1);

        if (!workerOptions.sync && modelIter >= workerOptions.numIters - 1) break;

        std::string modelPath = getTracedModelPath(runName, modelIter);
        std::string savePath = saveDir + "/" + runName + "_iteration_" + std::to_string(iter);

        IterationOptions iterationOptions = workerOptions.iterationOptions;
        if (modelPath == "random") {
            iterationOptions = workerOptions.initIterationOptions;
        }

        NeuralNetwork neuralNetwork = NeuralNetwork(modelPath);

        if (modelPath == "random") {
            std::cout << "Using initial network..." << std::endl;
            network = initialNetwork;
        } else {
            std::cout << "Using traced PyTorch network..." << std::endl;
            network = &neuralNetwork;
        }

        auto [states, distributions, outcomes] = runIteration<ImplNode, State, ACTION_SIZE>(
            iterationOptions,
            treeOptions,
            network,
            symmetrizer
        );

        std::vector<float> embeddedStates;

        for (const State& state : states) {
            Piece ourPiece = pieceFromPlayer(state.getPlayer());

            // Stone bitmasks. The iteration order is important; must match input to network.
            for (int t = 0; t < state.size(); ++t) {
                for (Piece piece : { ourPiece, otherPiece(ourPiece) }) {
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
        std::cout << "Games collected in " << t.elapsed() << " seconds." << std::endl;
    
        iter++;
    }

    std::cout << "Worker process completed in " << total_t.elapsed() << " seconds." << std::endl;

}

} // namespace SPRL

#endif