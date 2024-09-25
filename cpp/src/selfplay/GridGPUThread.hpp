#ifndef SPRL_GRID_GPU_WORKER_HPP
#define SPRL_GRID_GPU_WORKER_HPP

#include "../games/GridState.hpp"

#include "../networks/INetwork.hpp"
#include "../networks/GridNetwork.hpp"
#include "../networks/RandomNetwork.hpp"

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

constexpr int QUEUE_PULL_TIMEOUT_SECS = 1;

/**
 * Runs the GPU thread for the self-play process.
 * 
 * @tparam NeuralNetwork The type of the neural network, e.g. `GridNetwork`.
 *                       Must have a constructor that takes a model file path.
 * @tparam ImplNode The implementation of the game node, e.g. `GoNode`.
 * @tparam NUM_ROWS The number of rows in the grid.
 * @tparam NUM_COLS The number of columns in the grid.
 * @tparam HISTORY_SIZE The number of previous states to include in the state.
 * @tparam ACTION_SIZE The number of actions in the action space.
 * 
 * @param initialNetwork The network to use for the first iteration.
 * @param saveDir The directory to save the self-play data to.
 * @param workQueue The queue to pull work items from.
 * @param resultQueues The queues to push result items to.
 */
template <typename NeuralNetwork, typename ImplNode, int NUM_ROWS, int NUM_COLS, int HISTORY_SIZE, int ACTION_SIZE>
void runGPUThread(
    INetwork<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>* initialNetwork,
    const std::string& saveDir,
    WorkQueue<GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>, ACTION_SIZE>& workQueue,
    std::vector<ResultQueue<ACTION_SIZE>>& resultQueues) {
    
    using State = GridState<NUM_ROWS * NUM_COLS, HISTORY_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    Timer totalTimer {};
    totalTimer.reset();

    std::string runName = workerOptions.modelName + "_" + workerOptions.modelVariant;
    
    // Holds the current network to use for inference.
    INetwork<State, ACTION_SIZE>* network;

    int oldModelIter = -2;
    int modelIter = -1;

    IterationOptions iterationOptions = workerOptions.initIterationOptions;
    NeuralNetwork neuralNetwork = NeuralNetwork("random");

    float active_time = 0.0f;  // Time spent running the network.
    int total_states_processed = 0;

    Timer t {};

    while (true) {
        // Grab the newest model iteration, no matter what.
        modelIter = waitModelPath(runName, false);

        // If the last model has been produced, kill the thread.
        if (modelIter >= workerOptions.numIters - 1) break;

        // If the model iteration has changed, load the new model.
        if (modelIter != oldModelIter) {
            std::string modelPath = getTracedModelPath(runName, modelIter);

            if (modelPath == "random") {
                printf("Using initial network...\n");
                iterationOptions = workerOptions.initIterationOptions;
                network = initialNetwork;

            } else {
                printf("Using traced PyTorch network...\n");
                iterationOptions = workerOptions.iterationOptions;
                neuralNetwork = NeuralNetwork(modelPath);
                network = &neuralNetwork;
            }
        }

        // Pull states from the queue and process them using the network.
        int numStates = 0;

        std::vector<int> workerIds;
        std::vector<int> leafTaskIds;
        std::vector<State> states;
        std::vector<ActionDist> masks;

        WorkItem tmp;

        // Keep pulling until we hit batch size or we fail to pull before timeout.
        bool pulled = true;
        while (pulled && numStates < iterationOptions.maxBatchSize) {
            pulled = workQueue.wait_dequeue_timed(tmp, std::chrono::seconds(QUEUE_PULL_TIMEOUT_SECS));

            if (pulled) {
                workerIds.push_back(tmp.m_workerId);
                leafTaskIds.push_back(tmp.m_leafTaskId);
                states.push_back(tmp.m_state);
                masks.push_back(tmp.m_mask);

                numStates++;
            }
        }

        if (numStates != 0) {
            active_time -= t.elapsed();
            std::vector<std::pair<ActionDist, Value>> outputs = network->evaluate(states, masks);
            active_time += t.elapsed();

            for (int i = 0; i < numStates; i++) {
                resultQueues[workerIds[i]].enqueue({ leafTaskIds[i], outputs[i].first, outputs[i].second });
            }
        }

        oldModelIter = modelIter;
        total_states_processed += numStates;


        std::cout << "Iteration " << iter << " completed in " << t.elapsed() << " seconds, Active time: " << active_time << " seconds." << std::endl;
        std::cout << "Throughput: " << total_states_processed / active_time << " states/second, Uptime: " << active_time / t.elapsed() << std::endl;
    }

    std::cout << "GPU worker process completed in " << totalTimer.elapsed() << " seconds." << std::endl;
}

} // namespace SPRL

#endif