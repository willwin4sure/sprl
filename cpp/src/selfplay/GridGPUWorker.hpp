#ifndef SPRL_GRID_GPU_WORKER_HPP
#define SPRL_GRID_GPU_WORKER_HPP

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
 * @param worker_idx The index of the worker process.
 * @param workerOptions The options for the worker process.
 * @param treeOptions The options for the UCT tree.
 * @param initialNetwork The network to use for the first iteration.
 * @param symmetrizer The symmetrizer to use for symmetrizing the network and data.
 * @param saveDir The directory to save the self-play data to.
 */
template <typename NeuralNetwork, typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
void runGPUWorker(moodycamel::ConcurrentQueue<std::tuple<int, int, SPRL::GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>, SPRL::GameActionDist<ACTION_SIZE>>>& queue,
    std::vector<moodycamel::ConcurrentQueue<std::tuple<int, SPRL::GameActionDist<ACTION_SIZE>, float>>>& resultQueues) {
    
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
    
    INetwork<State, ACTION_SIZE>* network;  // Holds the current network.

    // Check which iteration it is.
    int iter = determineIteration(saveDir, runName);
    std::cout << "I now believe it is iteration " << iter << "." << std::endl;

    int oldModelIter = -2; // This value is different from modelIter no matter what.
    int modelIter; // This value doesn't matter because it is immediately set.

    IterationOptions iterationOptions = workerOptions.iterationOptions;
    NeuralNetwork neuralNetwork = NeuralNetwork("random");

    float active_time = 0.0f;

    int total_states_processed = 0;

    while (true) {
        if (workerOptions.sync && iter >= workerOptions.numIters) break;
        Timer t {};
        t.reset();

        std::cout << "Starting iteration " << iter << "..." << std::endl;

        // Block until the model file for the previous iteration exists.
        modelIter = waitModelPath(runName, workerOptions.sync, iter - 1);

        if (!workerOptions.sync && modelIter >= workerOptions.numIters - 1) break;

        if (modelIter != oldModelIter){
            std::string modelPath = getTracedModelPath(runName, modelIter);
            std::string savePath = saveDir + "/" + runName + "_iteration_" + std::to_string(iter);
            if (modelPath == "random") {
                iterationOptions = workerOptions.initIterationOptions;
            } else {
                iterationOptions = workerOptions.iterationOptions;
                neuralNetwork = NeuralNetwork(modelPath);
            }
            
            if (modelPath == "random") {
                std::cout << "Using initial network..." << std::endl;
                network = initialNetwork;
            } else {
                std::cout << "Using traced PyTorch network..." << std::endl;
                network = &neuralNetwork;
            }
        }

        // Pull states from the queue and process them using the network.
        int numStates = 0;
        std::vector<int> workerIds;
        std::vector<int> leafTaskIds;
        std::vector<State> states;
        std::vector<GameActionDist<ACTION_SIZE>> masks;

        std::tuple<int, int, State, ActionDist> tmp;
        while(queue.try_dequeue(tmp) && numStates < iterationOptions.maxBatchSize) {
            auto [workerId, leafTaskId, state, mask] = tmp;
            workerIds.push_back(workerId);
            leafTaskIds.push_back(leafTaskId);
            states.push_back(state);
            masks.push_back(mask);
            numStates++;
        }

        if (numStates != 0) {
            active_time -= t.elapsed();
            std::vector<std::pair<GameActionDist<ACTION_SIZE>, Value>> outputs = network->evaluate(states, masks);
            active_time += t.elapsed();

            for (int i = 0; i < numStates; i++) {
                resultQueues[workerIds[i]].enqueue({leafTaskIds[i], outputs[i].first, outputs[i].second});
            }
        }
        total_states_processed += numStates;
        std::cout << "Iteration " << iter << " completed in " << t.elapsed() << " seconds, active time: " << active_time << " seconds." << std::endl;
        std::cout << "Throughput: " << total_states_processed / active_time << " states/second, uptime: " << active_time / t.elapsed() << std::endl;
    }

    std::cout << "GPU Worker process completed in " << total_t.elapsed() << " seconds." << std::endl;

}

} // namespace SPRL

#endif