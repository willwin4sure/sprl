#include "games/GoNode.hpp"

#include "networks/GridNetwork.hpp"

#include "selfplay/GridWorker.hpp"
#include "selfplay/SelfPlayOptions.hpp"

#include "symmetry/D4GridSymmetrizer.hpp" 
#include "symmetry/ISymmetrizer.hpp"

#include "distributed/concurrentqueue/concurrentqueue.h"
#include "uct/UCTOptions.hpp"


constexpr int BOARD_WIDTH = SPRL::GO_BOARD_WIDTH;
constexpr int BOARD_SIZE = SPRL::GO_BOARD_SIZE;
constexpr int ACTION_SIZE = SPRL::GO_ACTION_SIZE;
constexpr int HISTORY_SIZE = SPRL::GO_HISTORY_SIZE;


int main(int argc, char *argv[]) {
    using State = GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>;
    using ActionDist = GameActionDist<ACTION_SIZE>;

    if (argc != 1) {
        std::cerr << "Usage: ./GoWorker.exe" << std::endl;
        return 1;
    }

    SPRL::WorkerOptions workerOptions {};
    SPRL::SelfPlayOptionsParser selfPlayParser {};

    // Parse the self-play options from hard-coded path.
    selfPlayParser.parse("config/config_selfplay.json", workerOptions);

    SPRL::TreeOptions treeOptions {};
    SPRL::UCTOptionsParser uctParser {};

    // Parse the UCT options from hard-coded path.
    uctParser.parse("config/config_uct.json", treeOptions);

    std::string runName = workerOptions.modelName + "_" + workerOptions.modelVariant;

    // create a ConcurrentQueue for the GPU thread to receive queries from the CPU threads.
    // Elements of this queue take the form {int worker_id, State state, ActionDist mask}.
    moodycamel::ConcurrentQueue<std::tuple<int, State, ActionDist>> queue;

    for (int i = 0; i < workerOptions.numWorkerTasks; i++) {
        std::thread workerThread(startWorker, i, workerOptions.numWorkerTasks, runName, workerOptions, treeOptions);
        workerThread.detach();
    }
}

void startWorker(int myTaskId, int numTasks, const std::string& runName,
    WorkerOptions workerOptions, TreeOptions treeOptions
) {
    assert(numTasks == workerOptions.numWorkerTasks);

    int myGroup = myTaskId / (workerOptions.numWorkerTasks / workerOptions.numGroups);

    // Log who I am.
    std::cout << "Task " << myTaskId << " of " << numTasks << ", in group " << myGroup << "." << std::endl;

    std::string saveDir = "data/games/" + runName + "/" + std::to_string(myGroup) + "/" + std::to_string(myTaskId);


    using State = SPRL::GridState<BOARD_SIZE, HISTORY_SIZE>;
    using Node = SPRL::GoNode;

    SPRL::RandomNetwork<State, ACTION_SIZE> randomNetwork {};
    SPRL::D4GridSymmetrizer<BOARD_WIDTH, HISTORY_SIZE> symmetrizer {};

    SPRL::runWorker<SPRL::GridNetwork<BOARD_WIDTH, BOARD_WIDTH, HISTORY_SIZE, ACTION_SIZE>,
                    Node, BOARD_WIDTH, BOARD_WIDTH, HISTORY_SIZE, ACTION_SIZE>(
        workerOptions, treeOptions, &randomNetwork, &symmetrizer, saveDir
    );

    return 0;
}   
