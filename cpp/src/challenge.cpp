#include <torch/torch.h>
#include <torch/script.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>

#include "games/GameState.hpp"
#include "games/Game.hpp"
#include "games/ConnectFour.hpp"

#include "interface/npy.hpp"

#include "networks/Network.hpp"
#include "networks/ConnectFourNetwork.hpp"

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


template <int BOARD_SIZE, int ACTION_SIZE>
void play(SPRL::Game<BOARD_SIZE, ACTION_SIZE>* game, std::string modelPath, int player, int numIters, int maxTraversals, int maxQueueSize) {
    using State = SPRL::GameState<BOARD_SIZE>;
    using ActionDist = SPRL::GameActionDist<ACTION_SIZE>;

    std::cout << "CUDA available: ";
    std::cout << torch::cuda::is_available() << std::endl;
    std::cout << torch::cuda::cudnn_is_available() << std::endl;
    
    SPRL::ConnectFourNetwork network { modelPath };

    State state = game->startState();
    SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state };

    Timer t{};

    float totalTime = 0.0f;

    int moves = 0;
    while (!game->isTerminal(state)) {
        std::cout << game->stateToString(state) << '\n';

        ActionDist actionMask = game->actionMask(state);
        std::cout << "Action mask: ";
        for (auto& action : actionMask) {
            std::cout << action << ' ';
        }
        std::cout << '\n';

        int action;
        if (moves % 2 == player) {
            action = getHumanAction(actionMask);
        } else {
            // SPRL::UCTTree<BOARD_SIZE, ACTION_SIZE> tree { game, state };
            t.reset();
            int iters = 0;
            while (iters < numIters) {
                // tree.searchIteration(&network);
                // ++iters;

                auto [leaves, iter] = tree.searchAndGetLeaves(maxTraversals, maxQueueSize, &network);
                if (leaves.size() > 0) {
                    tree.evaluateAndBackpropLeaves(leaves, &network);
                }
                iters += iter;
            }
            std::cout << "Time taken: " << t.elapsed() << "s\n";
            totalTime += t.elapsed();
            std::cout << "Total number of evaluations: " << network.m_numEvals << '\n';
            
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

    auto game = std::make_unique<SPRL::ConnectFour>();
    play(game.get(), modelPath, player, numIters, maxTraversals, maxQueueSize);

    return 0;
}
