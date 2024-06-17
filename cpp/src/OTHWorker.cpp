#include "games/OthelloNode.hpp"

#include "networks/GridNetwork.hpp"
#include "networks/OthelloHeuristic.hpp"

#include "selfplay/GridWorker.hpp"

#include "symmetry/D4GridSymmetrizer.hpp" 

// Parameters controlling the training run.

constexpr int NUM_GROUPS = 4;
constexpr int NUM_WORKER_TASKS = 384;

constexpr int NUM_ITERS = 50;

constexpr int INIT_NUM_GAMES_PER_WORKER = 3;
constexpr int INIT_UCT_TRAVERSALS = 131072;
constexpr int INIT_MAX_BATCH_SIZE = 1;
constexpr int INIT_MAX_QUEUE_SIZE = 1;

constexpr int NUM_GAMES_PER_WORKER = 3;
constexpr int UCT_TRAVERSALS = 8192;
constexpr int MAX_BATCH_SIZE = 8;
constexpr int MAX_QUEUE_SIZE = 4;

constexpr float DIRICHLET_EPSILON = 0.25f;
constexpr float DIRICHLET_ALPHA = 0.3f;


int main(int argc, char *argv[]) {
    std::string runName = "orangutan_alpha";  // Change me too!

    if (argc != 3) {
        std::cerr << "Usage: ./OTHWorker.exe <task_id> <num_tasks>" << std::endl;
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
    SPRL::RandomNetwork<SPRL::GridState<SPRL::OTH_BOARD_SIZE, SPRL::OTH_HISTORY_SIZE>, SPRL::OTH_ACTION_SIZE> randomNetwork {};
    SPRL::D4GridSymmetrizer<SPRL::OTH_BOARD_WIDTH, SPRL::OTH_HISTORY_SIZE> symmetrizer {};

    SPRL::runWorker<SPRL::GridNetwork<SPRL::OTH_BOARD_WIDTH, SPRL::OTH_BOARD_WIDTH, SPRL::OTH_HISTORY_SIZE, SPRL::OTH_ACTION_SIZE>,
                    SPRL::OthelloNode,
                    SPRL::OTH_BOARD_WIDTH,
                    SPRL::OTH_BOARD_WIDTH,
                    SPRL::OTH_HISTORY_SIZE,
                    SPRL::OTH_ACTION_SIZE>(

        runName, saveDir, &randomNetwork, &symmetrizer,
        NUM_ITERS,
        INIT_NUM_GAMES_PER_WORKER, INIT_UCT_TRAVERSALS, INIT_MAX_BATCH_SIZE, INIT_MAX_QUEUE_SIZE,
        NUM_GAMES_PER_WORKER, UCT_TRAVERSALS, MAX_BATCH_SIZE, MAX_QUEUE_SIZE,
        DIRICHLET_EPSILON, DIRICHLET_ALPHA
    );

    return 0;
}   
