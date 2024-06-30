#include "games/GoNode.hpp"

#include "networks/GridNetwork.hpp"

#include "selfplay/GridWorker.hpp"

#include "symmetry/D4GridSymmetrizer.hpp" 

#include "selfplay/Options.hpp"


constexpr SPRL::NodeOptions goNodeOptions = {
    .dirEps = 0.25f,
    .dirAlpha = 0.2f,
    .uWeight = 1.1f,
    .initQMethod = SPRL::InitQ::PARENT_LIVE_Q,
    .dropParent = true,
    .forcedPlayouts = false
};


constexpr SPRL::TreeOptions goTreeOptions = {
    .addNoise = true,
    .nodeOptions = goNodeOptions
};

constexpr SPRL::IterationOptions goIterationOptions = {
    .NUM_GAMES_PER_WORKER = 4,
    .UCT_TRAVERSALS = 4096,
    .MAX_BATCH_SIZE = 16,
    .MAX_QUEUE_SIZE = 8,
    .FAST_PLAY_PROBABILITY = 0.75f,
    .USE_PTP = false,
    .treeOptions = goTreeOptions
};

constexpr SPRL::IterationOptions goInitIterationOptions = {
    .NUM_GAMES_PER_WORKER = 4,
    .UCT_TRAVERSALS = 16384,
    .MAX_BATCH_SIZE = 1,
    .MAX_QUEUE_SIZE = 1,
    .EARLY_GAME_CUTOFF = 15,
    .EARLY_GAME_EXP = 0.98f,
    .REST_GAME_EXP = 10.0f,
    .treeOptions = goTreeOptions
};

const SPRL::ControllerOptions goControllerOptions = {
    .WORKER_TIME_TO_KILL = 1200,
    .WORKER_DATA_WAIT_INTERVAL = 30,
    .MODEL_NUM_BLOCKS = 6,
    .MODEL_NUM_CHANNELS = 64,
    .RESET_NETWORK = false,
    .LINEAR_WEIGHTING = true,
    .NUM_PAST_ITERS_TO_TRAIN = 5,
    .MAX_GROUPS = 10,
    .EPOCHS_PER_GROUP = 10,
    .BATCH_SIZE = 1024,
    .LR_INIT = 0.001,
    .LR_DECAY_FACTOR = 0.1,
    .LR_MILESTONE_ITERS = {50, 100}
};

const SPRL::WorkerOptions goWorkerOptions = {
    .NUM_GROUPS = 4,
    .NUM_WORKER_TASKS = 192,
    .numIters = 200,
    .iterationOptions = goIterationOptions,
    .initIterationOptions = goInitIterationOptions,
    .controllerOptions = goControllerOptions,
    .model_name = "panda",
    .model_variant = "delta_fast_equiv"
};



int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./GoWorker.exe <task_id> <num_tasks>" << std::endl;
        return 1;
    }
    std::string runName = goWorkerOptions.model_name + "_" + goWorkerOptions.model_variant;

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);

    assert(numTasks == goWorkerOptions.NUM_WORKER_TASKS);

    int myGroup = myTaskId / (goWorkerOptions.NUM_WORKER_TASKS / goWorkerOptions.NUM_GROUPS);

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
        goWorkerOptions, &randomNetwork, &symmetrizer, saveDir
    );

    return 0;
}   
