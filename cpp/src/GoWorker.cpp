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
    
    using State = SPRL::GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

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
    // Elements of this queue take the form {int worker_id, int leaf_task_id, State state, ActionDist mask}.
    moodycamel::ConcurrentQueue<std::tuple<int, int, State, ActionDist>> queue;

    // for each worker, create a ConcurrentQueue which will be used to send the results
    // of the network evaluation back to the CPU thread.
    // Elements of this queue take the form {int worker_id, ActionDist actionDist, float value}.
    std::vector<moodycamel::ConcurrentQueue<std::tuple<int, ActionDist, SPRL::Value>>> resultQueues;
    for (int i = 0; i < workerOptions.numWorkerTasks; i++) {
        resultQueues.push_back(moodycamel::ConcurrentQueue<std::tuple<int, ActionDist, float>>());
    }

    for (int i = 0; i < workerOptions.numWorkerTasks; i++) {
        std::thread workerThread(startWorker,
            i,
            workerOptions.numWorkerTasks,
            runName,
            workerOptions,
            treeOptions,
            std::ref(queue),
            std::ref(resultQueues[i])
        );
        workerThread.detach();
    }

    // Start GPU thread.
    std::thread gpuThread(startGPUWorker, std::ref(queue), std::ref(resultQueues));
    return 0;
}

void startGPUWorker(moodycamel::ConcurrentQueue<std::tuple<int, int, SPRL::GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>, SPRL::GameActionDist<ACTION_SIZE>>>& queue,
    std::vector<moodycamel::ConcurrentQueue<std::tuple<int, SPRL::GameActionDist<ACTION_SIZE>, SPRL::Value>>>& resultQueues
) {
    SPRL::runGPUWorker<SPRL::GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>, SPRL::GameActionDist<ACTION_SIZE>>(queue, resultQueues);
}


void startWorker(int myTaskId, int numTasks, const std::string& runName,
    SPRL::WorkerOptions workerOptions, SPRL::TreeOptions treeOptions,
    moodycamel::ConcurrentQueue<std::tuple<int, int, SPRL::GridState<BOARD_WIDTH * BOARD_WIDTH, HISTORY_SIZE>, SPRL::GameActionDist<ACTION_SIZE>>>& queue,
    moodycamel::ConcurrentQueue<std::tuple<int, SPRL::GameActionDist<ACTION_SIZE>, SPRL::Value>>& resultQueue
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
                        myTaskId, numTasks, workerOptions, treeOptions, &randomNetwork, &symmetrizer, saveDir,
                        queue, resultQueue
    );
}

