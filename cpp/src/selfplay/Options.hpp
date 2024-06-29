#ifndef SPRL_OPTIONS_HPP
#define SPRL_OPTIONS_HPP

#include <string>

namespace SPRL {

// The options classes are hierarchical, reflecting the
// structure of the codebase.

/**
 * Supported methods of initializing the Q values of the nodes.
*/
enum class InitQ {
    ZERO,    // Always initialize to zero.
    PARENT_NN_EVAL,  // Initialize to the network output of the parent, if available.
    PARENT_LIVE_Q     // Todo: write description.
};

// See `options.md` for a description of each option.
struct NodeOptions {
    float dirEps;
    float dirAlpha;
    float uWeight;
    InitQ initQMethod;
    bool dropParent;
};

// See `options.md` for a description of each option.
struct TreeOptions {
    bool addNoise;
    NodeOptions nodeOptions;
};

struct IterationOptions {
    int NUM_GAMES_PER_WORKER;
    int UCT_TRAVERSALS;
    int MAX_BATCH_SIZE;
    int MAX_QUEUE_SIZE;
    int EARLY_GAME_CUTOFF;
    float EARLY_GAME_EXP;
    float REST_GAME_EXP;
    TreeOptions treeOptions;
};


// See `options.md` for a description of each option.
// Implementation detail: model_name and model_variant
// are char[] instead of std::string, because std::string
// is not literal, and we want to make everything constexpr.
struct WorkerOptions {
    int NUM_GROUPS;
    int NUM_WORKER_TASKS;
    int numIters;

    IterationOptions iterationOptions;
    IterationOptions initIterationOptions;

    char model_name[32];
    char model_variant[32];
};



} // namespace SPRL

#endif