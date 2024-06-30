#ifndef SPRL_SELF_PLAY_OPTIONS_HPP
#define SPRL_SELF_PLAY_OPTIONS_HPP

#include "../uct/UCTOptions.hpp"

namespace SPRL {

/**
 * Options for a single iteration of self-play.
*/
struct IterationOptions {
    int numGamesPerWorker;  // How many games each worker players per iteration.
    int uctTraversals;      // How many UCT traversals to perform per move.
    int maxBatchSize;       // The maximum number of traversals per batch of search.
    int maxQueueSize;       // The maximum number of states to evaluate per batch of search.

    bool symmetrizeData;       // Whether to symmetrize the generated self-play data.
    float fastPlayoutProb;     // Chance of using a fast playout each move, in `[0, 1]`.
    float fastPlayoutFactor;   // Multiplier for number of traversals in fast playouts, in `[0, 1]`.
    bool policyTargetPruning;  // Whether to implement policy target pruning in data generation.
    bool forcedPlayouts;       // Whether to force some actions to be played enough times.

    int earlyGameCutoff;  // The number of moves after which to swap temperature from early to rest.
    float earlyGameExp;   // The inverse temperature for the early game.
    float restGameExp;    // The inverse temperature for the rest of the game.
};

/**
 * Options for the behavior of self-play workers.
*/
struct WorkerOptions {
    std::string modelName;     // The full model path is `modelName + "_" + modelVariant`.
    std::string modelVariant;  // The full model path is `modelName + "_" + modelVariant`.

    int numGroups;       // Supercloud: number of groups to split the workers into.
    int numWorkerTasks;  // Supercloud: number of total worker threads.
    int numIters;        // The total number of iterations to run.

    IterationOptions initIterationOptions;  // Iteration options for the first iteration.
    IterationOptions iterationOptions;      // Iteration options for all subsequent iterations.
};

/**
 * Parses self-play worker options from a JSON file.
*/
class SelfPlayOptionsParser {
public:
    SelfPlayOptionsParser() {
        sm::reg(&IterationOptions::numGamesPerWorker, "numGamesPerWorker", sm::Required {});
        sm::reg(&IterationOptions::uctTraversals, "uctTraversals", sm::Required {});
        sm::reg(&IterationOptions::maxBatchSize, "maxBatchSize", sm::Required {});
        sm::reg(&IterationOptions::maxQueueSize, "maxQueueSize", sm::Required {});

        sm::reg(&IterationOptions::symmetrizeData, "symmetrizeData", sm::Default { true });
        sm::reg(&IterationOptions::fastPlayoutProb, "fastPlayoutProb", sm::Bounds { 0.0f, 1.0f }, sm::Default { 0.0f });
        sm::reg(&IterationOptions::fastPlayoutFactor, "fastPlayoutFactor", sm::Bounds { 0.0f, 1.0f }, sm::Default { 1.0f });
        sm::reg(&IterationOptions::policyTargetPruning, "policyTargetPruning", sm::Default { false });
        sm::reg(&IterationOptions::forcedPlayouts, "forcedPlayouts", sm::Default { false });

        sm::reg(&IterationOptions::earlyGameCutoff, "earlyGameCutoff", sm::Default { 15 });
        sm::reg(&IterationOptions::earlyGameExp, "earlyGameExp", sm::Default { 0.98f });
        sm::reg(&IterationOptions::restGameExp, "restGameExp", sm::Default { 10.0f });


        sm::reg(&WorkerOptions::modelName, "modelName", sm::Required {});
        sm::reg(&WorkerOptions::modelVariant, "modelVariant", sm::Default { "base" });

        sm::reg(&WorkerOptions::numGroups, "numGroups", sm::Required {});
        sm::reg(&WorkerOptions::numWorkerTasks, "numWorkerTasks", sm::Required {});
        sm::reg(&WorkerOptions::numIters, "numIters", sm::Required {});

        sm::reg(&WorkerOptions::initIterationOptions, "initIterationOptions", sm::Required {});
        sm::reg(&WorkerOptions::iterationOptions, "iterationOptions", sm::Required {});
    }

    /**
     * Parses self-play worker options from a JSON file.
     * 
     * @param path The path to the JSON file to parse.
     * @param options The worker options to populate.
    */
    void parse(const std::string& path, WorkerOptions& options) {
        auto stream = std::ifstream(path);
        sm::map_json_to_struct(options, stream);
    }

    /**
     * Converts self-play worker options to a JSON string.
     * 
     * @param options The worker options to convert.
     * @returns The JSON string representation of the worker options.
     * 
     * @note The input parameter cannot be const.
    */
    std::string toString(WorkerOptions& options) {
        std::ostringstream out_json_data;
        sm::map_struct_to_json(options, out_json_data, "    ");
        return out_json_data.str();
    }
};

} // namespace SPRL

#endif