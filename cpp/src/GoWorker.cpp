#include "games/GoNode.hpp"

#include "networks/GridNetwork.hpp"

#include "selfplay/GridWorker.hpp"

#include "symmetry/D4GridSymmetrizer.hpp" 

#include "selfplay/Options.hpp"


constexpr SPRL::NodeOptions nodeOptions = {
    .dirEps = 0.25f,
    .dirAlpha = 0.2f,
    .initQMethod = SPRL::InitQ::PARENT_LIVE_Q,
    .dropParent = true
};


constexpr SPRL::TreeOptions treeOptions = {
    .addNoise = true,
    .nodeOptions = nodeOptions
};

constexpr SPRL::IterationOptions iterationOptions = {
    .NUM_GAMES_PER_WORKER = 5,
    .UCT_TRAVERSALS = 4096,
    .MAX_BATCH_SIZE = 16,
    .MAX_QUEUE_SIZE = 8,
    .treeOptions = treeOptions
};

constexpr SPRL::IterationOptions initIterationOptions = {
    .NUM_GAMES_PER_WORKER = 5,
    .UCT_TRAVERSALS = 16384,
    .MAX_BATCH_SIZE = 1,
    .MAX_QUEUE_SIZE = 1,
    .treeOptions = treeOptions
};

constexpr SPRL::WorkerOptions workerOptions = {
    .NUM_GROUPS = 4,
    .NUM_WORKER_TASKS = 192,
    .numIters = 200,
    .iterationOptions = iterationOptions,
    .initIterationOptions = initIterationOptions,
    .model_name = "panda_gamma",
    .model_variant = "slow"
};



int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./GoWorker.exe <task_id> <num_tasks>" << std::endl;
        return 1;
    }
    std::string runName = std::string(workerOptions.model_name) + "_" + workerOptions.model_variant;

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);

    assert(numTasks == workerOptions.NUM_WORKER_TASKS);

    int myGroup = myTaskId / (workerOptions.NUM_WORKER_TASKS / workerOptions.NUM_GROUPS);

    // Log who I am.
    std::cout << "Task " << myTaskId << " of " << numTasks << ", in group " << myGroup << "." << std::endl;

    std::string saveDir = "data/games/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

    // SPRL::OthelloHeuristic heuristicNetwork {};
    SPRL::RandomNetwork<SPRL::GridState<SPRL::GO_BOARD_SIZE, SPRL::GO_HISTORY_SIZE>, SPRL::GO_ACTION_SIZE> randomNetwork {};
    SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE> symmetrizer {};

    SPRL::runWorker<SPRL::GridNetwork<SPRL::GO_BOARD_WIDTH, SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE, SPRL::GO_ACTION_SIZE>,
                    SPRL::GoNode,
                    SPRL::GO_BOARD_WIDTH,
                    SPRL::GO_BOARD_WIDTH,
                    SPRL::GO_HISTORY_SIZE,
                    SPRL::GO_ACTION_SIZE>(
        workerOptions, &randomNetwork, &symmetrizer, saveDir
    );

    return 0;
}   
