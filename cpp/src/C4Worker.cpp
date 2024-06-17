#include "games/ConnectFourNode.hpp"

#include "networks/GridNetwork.hpp"

#include "selfplay/GridWorker.hpp"

#include "symmetry/ConnectFourSymmetrizer.hpp" 

// Parameters controlling the training run.

constexpr int NUM_GROUPS = 4;
constexpr int NUM_WORKER_TASKS = 384;

constexpr int NUM_ITERS = 25;

constexpr int INIT_NUM_GAMES_PER_WORKER = 4;
constexpr int INIT_UCT_TRAVERSALS = 32768;
constexpr int INIT_MAX_BATCH_SIZE = 1;
constexpr int INIT_MAX_QUEUE_SIZE = 1;

constexpr int NUM_GAMES_PER_WORKER = 2;
constexpr int UCT_TRAVERSALS = 512;
constexpr int MAX_BATCH_SIZE = 8;
constexpr int MAX_QUEUE_SIZE = 4;

constexpr float DIRICHLET_EPSILON = 0.25f;
constexpr float DIRICHLET_ALPHA = 0.5f;


int main(int argc, char *argv[]) {
    std::string runName = "narwhal_alpha";  // Change me too!

    if (argc != 3) {
        std::cerr << "Usage: ./C4Worker.exe <task_id> <num_tasks>" << std::endl;
        return 1;
    }

    int myTaskId = std::stoi(argv[1]);
    int numTasks = std::stoi(argv[2]);

    assert(numTasks == NUM_WORKER_TASKS);

    int myGroup = myTaskId / (NUM_WORKER_TASKS / NUM_GROUPS);

    // Log who I am.
    std::cout << "Task " << myTaskId << " of " << numTasks << ", in group " << myGroup << "." << std::endl;

    std::string saveDir = "data/games/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

    SPRL::RandomNetwork<SPRL::GridState<SPRL::C4_BOARD_SIZE, SPRL::C4_HISTORY_SIZE>, SPRL::C4_ACTION_SIZE> randomNetwork {};
    SPRL::ConnectFourSymmetrizer symmetrizer {};

    SPRL::runWorker<SPRL::GridNetwork<SPRL::C4_NUM_ROWS, SPRL::C4_NUM_COLS, SPRL::C4_HISTORY_SIZE, SPRL::C4_ACTION_SIZE>,
                    SPRL::ConnectFourNode,
                    SPRL::C4_NUM_ROWS,
                    SPRL::C4_NUM_COLS,
                    SPRL::C4_HISTORY_SIZE,
                    SPRL::C4_ACTION_SIZE>(

        runName, saveDir, &randomNetwork, &symmetrizer,
        NUM_ITERS,
        INIT_NUM_GAMES_PER_WORKER, INIT_UCT_TRAVERSALS, INIT_MAX_BATCH_SIZE, INIT_MAX_QUEUE_SIZE,
        NUM_GAMES_PER_WORKER, UCT_TRAVERSALS, MAX_BATCH_SIZE, MAX_QUEUE_SIZE,
        DIRICHLET_EPSILON, DIRICHLET_ALPHA
    );

    return 0;
}   
