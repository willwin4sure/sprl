#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"
#include "games/Pentago.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/ConnectFourNetwork.hpp"
#include "networks/RandomNetwork.hpp"
#include "networks/PentagoHeuristic.hpp"

#include "uct/UCTNode.hpp"
#include "uct/UCTTree.hpp"


template <int ACTION_SIZE>
int getHumanAction(const SPRL::GameActionDist<ACTION_SIZE>& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a valid action: ";
        std::cin >> action;
    }

    return action;
}

// https://www.learncpp.com/cpp-tutorial/timing-your-code/
class Timer {
private:
    using Clock = std::chrono::steady_clock;
    using Second = std::chrono::duration<double, std::ratio<1>>;

    std::chrono::time_point<Clock> m_beg { Clock::now() };

public:
    void reset() {
        m_beg = Clock::now();
    }

    double elapsed() const {
        return std::chrono::duration_cast<Second>(Clock::now() - m_beg).count();
    }
};

int getPentagoAction(SPRL::Pentago::ActionDist& actionSpace) {
    int action = -1;
    while (action < 0 || action >= actionSpace.size() || actionSpace[action] != 1.0f) {
        std::cout << "Enter a square, followed by a quadrant, then a rotation (e.g. C5 0 1): ";
        char file, rank;
        int quadrant, rotation;
        std::cin >> file >> rank >> quadrant >> rotation;
        
        int boardIdx = (file - 'A') + (rank - '0') * 6;
        action = boardIdx + quadrant * 36 + rotation * 36 * 4;
    }

    return action;
}

template <int BOARD_SIZE, int ACTION_SIZE>
void play(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game,
          SPRL::Network<BOARD_SIZE, ACTION_SIZE>* network,
          std::string modelPath, int player, int numIters, int maxTraversals, int maxQueueSize) {

    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    State state = game->startState();
    SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state, false };

    Timer t{};

    float totalTime = 0.0f;

    int moves = 0;
    while (!state.isTerminal()) {
        std::cout << game->stateToString(state) << '\n';

        ActionDist actionMask = game->actionMask(state);
        std::cout << "Action mask: ";
        for (auto& action : actionMask) {
            std::cout << action << ' ';
        }
        std::cout << '\n';

        int action;
        if (moves % 2 == player) {
            action = getPentagoAction(actionMask);
        } else {
            // SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state };
            t.reset();
            int iters = 0;
            while (iters < numIters) {
                // tree.searchIteration(&network);
                // ++iters;

                auto [leaves, iter] = tree.searchAndGetLeaves(maxTraversals, maxQueueSize, network);
                if (leaves.size() > 0) {
                    tree.evaluateAndBackpropLeaves(leaves, network);
                }
                iters += iter;
            }
            std::cout << "Time taken: " << t.elapsed() << "s\n";
            totalTime += t.elapsed();
            std::cout << "Total number of evaluations: " << network->getNumEvals() << '\n';
            
            auto priors = tree.getRoot()->getEdgeStatistics()->m_childPriors;
            auto values = tree.getRoot()->getEdgeStatistics()->m_totalValues;
            auto visits = tree.getRoot()->getEdgeStatistics()->m_numberVisits;

            std::cout << "Priors: ";
            for (const auto& prior : priors) {
                std::cout << prior << ' ';
            }

            std::cout << "\nValues: ";
            for (const auto& value : values) {
                std::cout << value << ' ';
            }

            std::cout << "\nVisits: ";
            for (const auto& visit : visits) {
                std::cout << visit << ' ';
            }

            std::cout << "\nAverage values: ";
            for (int i = 0; i < ACTION_SIZE; ++i) {
                std::cout << values[i] / (1 + visits[i]) << ' ';
            }

            std::cout << '\n';

            // sample action with most visits
            action = std::distance(visits.begin(), std::max_element(visits.begin(), visits.end()));
        }

        tree.rerootTree(action);
        state = game->nextState(state, action);

        ++moves;
    }

    std::cout << "Game over!" << '\n';
    std::cout << game->stateToString(state) << '\n';

    std::cout << "The winner is Player " << static_cast<int>(state.getWinner()) << '\n';
    std::cout << "The rewards are " << game->rewards(state).first << " and " << game->rewards(state).second << '\n';
    std::cout << "Total time taken: " << totalTime << "s\n";
}

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: ./challenge.exe <modelPath> <player> <numIters> <maxTraversals> <maxQueueSize>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    int player = std::stoi(argv[2]);
    int numIters = std::stoi(argv[3]);
    int maxTraversals = std::stoi(argv[4]);
    int maxQueueSize = std::stoi(argv[5]);

    auto game = std::make_unique<SPRL::Pentago>();

    SPRL::Network<36, 288>* network;

    SPRL::PentagoHeuristic pentagoHeuristic {};

    network = &pentagoHeuristic;
    // SPRL::ConnectFourNetwork neuralNetwork { modelPath };

    // if (modelPath == "random") {
    //     std::cout << "Using random network..." << std::endl;
    //     network = &randomNetwork;
    // } else {
    //     std::cout << "Using traced PyTorch network..." << std::endl;
    //     network = &neuralNetwork;
    // }

    play(game.get(), network, modelPath, player, numIters, maxTraversals, maxQueueSize);

    return 0;
}
