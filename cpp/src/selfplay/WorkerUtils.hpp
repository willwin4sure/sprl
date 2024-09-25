#ifndef SPRL_WORKER_UTILS_HPP
#define SPRL_WORKER_UTILS_HPP


#include "../games/GridState.hpp"

#include "../networks/INetwork.hpp"
#include "../networks/GridNetwork.hpp"
#include "../networks/RandomNetwork.hpp"

#include "../selfplay/SelfPlay.hpp"
#include "../selfplay/SelfPlayOptions.hpp"

#include "../uct/UCTOptions.hpp"

#include "../utils/blockingconcurrentqueue.h"
#include "../utils/npy.hpp"
#include "../utils/timer.hpp"

#include "../constants.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace SPRL {

/**
 * Work item from CPU thread to GPU thread to evaluate a state
 * using the neural network.
 * 
 * @tparam State The state of the game.
 * @tparam ACTION_SIZE The number of possible actions in the game.
 */
template <typename State, int ACTION_SIZE>
struct WorkItem {
    int m_workerId;
    int m_leafTaskId;
    State m_state;
    GameActionDist<ACTION_SIZE> m_mask;
};


/**
 * Result item for GPU thread to send back to the CPU thread.
 * 
 * @tparam ACTION_SIZE The number of possible actions in the game.
 */
template <int ACTION_SIZE>
struct ResultItem {
    int m_leafTaskId;
    GameActionDist<ACTION_SIZE> m_policy;
    Value m_value;
};


/**
 * Work queue for CPU threads to push work items to the GPU thread.
 */
template <typename State, int ACTION_SIZE>
using WorkQueue = moodycamel::BlockingConcurrentQueue<WorkItem<State, ACTION_SIZE>>;


/**
 * Result queue for the GPU thread to push result items to the CPU threads.
 */
template <int ACTION_SIZE>
using ResultQueue = moodycamel::BlockingConcurrentQueue<ResultItem<ACTION_SIZE>>;


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
std::string getDistrsPath(const std::string& saveDir, const std::string& runName, int iteration) {
    return saveDir + "/" + runName + "_iteration_" + std::to_string(iteration) + "_distributions.npy";
}
std::string getOutcosPath(const std::string& saveDir, const std::string& runName, int iteration) {
    return saveDir + "/" + runName + "_iteration_" + std::to_string(iteration) + "_outcomes.npy";
}

/**
 * Blocks the current thread until the model file for the given iteration exists,
 * and then returns the path to the model file.
 *
 * @param runName The name of the run, defining the model file path.
 * @param sync Whether to wait for the model file to exist.
 * @param iteration The iteration to get the model file for, if sync is true.
*/
void waitModelPath(const std::string& runName, int iteration) {
    if (iteration == -1) return;

    Timer t {};
    t.reset();
    std::string modelPath = getTracedModelPath(runName, iteration);
    while (!std::filesystem::exists(modelPath)) {
        std::this_thread::sleep_for(std::chrono::seconds(MODEL_PATH_WAIT_INTERVAL));
    }
    double elapsed = t.elapsed();
    std::cout << "Found traced model in " << elapsed << " seconds." << std::endl;
}

/**
 * Determine the latest iteration of model that already exists,
 * or -1 if none exist.
 */
int determineModelIteration(const std::string& saveDir, std::string& runName, int startIteration = -1) {
    int iteration = startIteration;
    while (std::filesystem::exists(getTracedModelPath(runName, iteration + 1))) {
        iteration++;
    }
    return iteration;
}

/**
 * Determine the next iteration of games to generate for this process,
 * e.g. if there is no data at all yet, returns 0.
 * 
 * Returns an iteration number that is at least `startIteration`.
 */
int determineGameIteration(const std::string& saveDir, std::string& runName, int startIteration = 0) {
    int iteration = startIteration;
    while (std::filesystem::exists(getStatesPath(saveDir, runName, iteration))
        && std::filesystem::exists(getDistrsPath(saveDir, runName, iteration))
        && std::filesystem::exists(getOutcosPath(saveDir, runName, iteration))) {

        iteration++;
    }

    return iteration;
}

} // namespace SPRL


#endif