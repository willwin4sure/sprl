/**
 * @file GoWorker.cpp
*/

#include "games/GoNode.hpp"
#include "networks/GridNetwork.hpp"
#include "selfplay/GridWorker.hpp"
#include "symmetry/D4GridSymmetrizer.hpp"

int main(int argc, char *argv[]) {
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

    std::string runName = "test_go";
    std::string saveDir = "data/games/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);

    SPRL::RandomNetwork<SPRL::GridState<SPRL::GO_BOARD_SIZE, SPRL::GO_HISTORY_SIZE>, SPRL::GO_ACTION_SIZE> randomNetwork {};
    SPRL::D4GridSymmetrizer<SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE> symmetrizer {};

    SPRL::runWorker<SPRL::GridNetwork<SPRL::GO_BOARD_WIDTH, SPRL::GO_BOARD_WIDTH, SPRL::GO_HISTORY_SIZE, SPRL::GO_ACTION_SIZE>,
                    SPRL::GoNode,
                    SPRL::GO_BOARD_WIDTH,
                    SPRL::GO_BOARD_WIDTH,
                    SPRL::GO_HISTORY_SIZE,
                    SPRL::GO_ACTION_SIZE>(

        runName, saveDir, &randomNetwork, &symmetrizer
    );

    return 0;
}   
