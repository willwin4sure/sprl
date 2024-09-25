#ifndef SPRL_GRID_WORKER_HPP
#define SPRL_GRID_WORKER_HPP

#include "../games/GridState.hpp"

#include "../selfplay/SelfPlay.hpp"
#include "../selfplay/SelfPlayOptions.hpp"
#include "../selfplay/WorkerUtils.hpp"

#include "../uct/UCTOptions.hpp"

#include "../utils/npy.hpp"
#include "../utils/timer.hpp"

#include "../constants.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace SPRL {
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
 * @param mctsWorkerIdx The index of the worker process.
 * @param numWorkers The total number of worker processes.
 * @param workerOptions The options for the worker process.
 * @param treeOptions The options for the UCT tree.
 * @param initialNetwork The network to use for the first iteration.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data.
 * @param saveDir The directory to save the self-play data to.
 * @param workQueue The queue to receive queries from the CPU threads.
 * @param resultQueue The queue to send the results back to the CPU threads.
 */
template <typename NeuralNetwork, typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
void runWorker(int mctsWorkerIdx,
               SPRL::WorkerOptions workerOptions,
               SPRL::TreeOptions treeOptions,
               INetwork<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* initialNetwork,
               ISymmetrizer<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* symmetrizer,
               const std::string& saveDir,
               WorkQueue<State, ACTION_SIZE>& workQueue,
               ResultQueue<ACTION_SIZE>& resultQueue) {
    
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
    int iter = determineGameIteration(saveDir, runName);
    std::cout << "I now believe it is iteration " << iter << "." << std::endl;

    while (true) {
        if (workerOptions.sync && iter >= workerOptions.numIters) break;
        
        Timer t {};
        t.reset();

        std::cout << "Starting iteration " << iter << "..." << std::endl;

        int modelIter;
        if (workerOptions.sync) {
            // Block until the model file for the previous iteration exists.
            modelIter = iter - 1;
            waitModelPath(runName, modelIter);

        } else {
            // Stop if the controllers have finished training the models.
            modelIter = determineModelIteration(saveDir, runName);
            if (modelIter >= workerOptions.numIters - 1) break;
        }
        
        std::string modelPath = getTracedModelPath(runName, modelIter);
        std::string savePath = saveDir + "/" + runName + "_iteration_" + std::to_string(iter);

        IterationOptions iterationOptions = workerOptions.iterationOptions;
        if (modelPath == "random") {
            iterationOptions = workerOptions.initIterationOptions;
        }

        auto [states, distributions, outcomes] = runIteration<ImplNode, State, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, ACTION_SIZE>(
            mctsWorkerIdx,
            iterationOptions,
            treeOptions,
            symmetrizer,
            workQueue,
            resultQueue
        );

        // Embed and save the states.
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

        // Embed and save the distributions.
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

        // Save the outcomes.
        npy::npy_data_ptr<float> outcomeData {};
        outcomeData.data_ptr = outcomes.data();
        outcomeData.shape = { static_cast<unsigned long>(outcomes.size()) };
        
        npy::write_npy(savePath + "_outcomes.npy", outcomeData);

        std::cout << "Games collected in " << t.elapsed() << " seconds." << std::endl;
    
        ++iter;
    }

    std::cout << "Worker process completed in " << total_t.elapsed() << " seconds." << std::endl;

}

} // namespace SPRL

#endif