#ifndef SPRL_WORKER_UTILS_HPP
#define SPRL_WORKER_UTILS_HPP


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
} // namespace SPRL


#endif