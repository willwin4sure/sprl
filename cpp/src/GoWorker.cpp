#include "games/GoNode.hpp"

#include "networks/GridNetwork.hpp"

#include "selfplay/GridWorker.hpp"

#include "symmetry/D4GridSymmetrizer.hpp" 

// Parameters controlling the training run.

constexpr int NUM_GROUPS = 4;
constexpr int NUM_WORKER_TASKS = 384;

constexpr int NUM_ITERS = 100;

constexpr int INIT_NUM_GAMES_PER_WORKER = 5;
constexpr int INIT_UCT_TRAVERSALS = 16384;
constexpr int INIT_MAX_BATCH_SIZE = 1;
constexpr int INIT_MAX_QUEUE_SIZE = 1;

constexpr int NUM_GAMES_PER_WORKER = 5;
constexpr int UCT_TRAVERSALS = 4096;
constexpr int MAX_BATCH_SIZE = 16;
constexpr int MAX_QUEUE_SIZE = 8;

constexpr float DIRICHLET_EPSILON = 0.25f;
constexpr float DIRICHLET_ALPHA = 0.2f;


int main(int argc, char *argv[]) {
    std::string runName = "panda_beta";  // Change me too!

    if (argc != 3) {
        std::cerr << "Usage: ./GoWorker.exe <task_id> <num_tasks>" << std::endl;
        return 1;
    }

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);

    assert(numTasks == NUM_WORKER_TASKS);

    int myGroup = myTaskId / (NUM_WORKER_TASKS / NUM_GROUPS);

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

        runName, saveDir, &randomNetwork, &symmetrizer,
        NUM_ITERS,
        INIT_NUM_GAMES_PER_WORKER, INIT_UCT_TRAVERSALS, INIT_MAX_BATCH_SIZE, INIT_MAX_QUEUE_SIZE,
        NUM_GAMES_PER_WORKER, UCT_TRAVERSALS, MAX_BATCH_SIZE, MAX_QUEUE_SIZE,
        DIRICHLET_EPSILON, DIRICHLET_ALPHA
    );

    return 0;
}   
